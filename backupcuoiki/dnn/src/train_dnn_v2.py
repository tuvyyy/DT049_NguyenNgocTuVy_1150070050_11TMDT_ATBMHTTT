#!/usr/bin/env python3
"""
Supervised Binary Feedforward DNN Training & Validation Pipeline (Experiment V2)
================================================================================
Reference Paper:
  Kuchar & Fujdiak (2025), Computers & Security, Vol. 153, 104395
  DOI: 10.1016/j.cose.2025.104395 (Section 5.5 / Listing 1)

Verified Listing 1 Hidden Activations:
  ["relu", "tanh", "relu", "tanh", "tanh", "tanh"]
  with two Dropout(0.3) layers after Dense(60) and Dense(10).

Protocol Guarantees:
1. Train split:      test1 + test2 (162,002 rows; 4,078 attack rows; 25 attack events)
2. Validation split: test3         (108,001 rows; 1,535 attack rows; 8 attack events)
3. Held-out Eval V2: test4 + test5 (NEVER accessed or imported in this script)
4. Scaler:           Frozen `models/preprocessing/scaler.joblib` (fit on train1+train2 = 442,802 rows)
5. Class weights:    Computed strictly from Train (`test1 + test2`), `pos_weight_sqrt = 6.223009`
6. Model selection:  Validation PR-AUC (`average_precision_score`) on `test3`, max 250 epochs, early stopping
7. Thresholding:     Calibrated strictly on labeled Validation `test3` (distinct from V1 Q99.9 on normal train3)
"""

import argparse
import copy
import hashlib
import json
import logging
import random
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dnn_dataset_v2 import (
    CANONICAL_EVAL_SESSIONS,
    CANONICAL_TRAIN_SESSIONS,
    CANONICAL_VAL_SESSIONS,
    DEFAULT_CONFIG_PATH,
    build_v2_dataloaders,
    load_v2_config,
)
from src.dnn_model import (
    EXPECTED_TRAINABLE_PARAMETERS,
    VERIFIED_ACTIVATION_STATUS,
    VERIFIED_HIDDEN_ACTIVATIONS,
    FeedforwardDNNV2,
    verify_dnn_v2_architecture,
)


def get_peak_rss_mb() -> float:
    """Returns peak resident set size in MB for the current process."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_dnn_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_deterministic_seed(seed: int) -> torch.Generator:
    """Sets Python, NumPy, and PyTorch CPU seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def assert_no_heldout_eval_access(bundle: Dict) -> None:
    """
    Runtime guard verifying that `test4` and `test5` are completely absent from
    the training and validation data bundle.
    """
    if "eval_dataset" in bundle or "eval_loader" in bundle:
        raise RuntimeError("Leakage Guard Violation: eval_dataset/eval_loader present during training!")

    train_sessions = set(bundle["train_dataset"].sessions)
    val_sessions = set(bundle["val_dataset"].sessions)
    forbidden_eval = set(CANONICAL_EVAL_SESSIONS)

    if train_sessions.intersection(forbidden_eval) or val_sessions.intersection(forbidden_eval):
        raise RuntimeError(
            f"Leakage Guard Violation: Held-out sessions {forbidden_eval} found in "
            f"train={train_sessions} or val={val_sessions}!"
        )
    if train_sessions != set(CANONICAL_TRAIN_SESSIONS):
        raise RuntimeError(f"Expected Train sessions {CANONICAL_TRAIN_SESSIONS}, got {train_sessions}")
    if val_sessions != set(CANONICAL_VAL_SESSIONS):
        raise RuntimeError(f"Expected Validation sessions {CANONICAL_VAL_SESSIONS}, got {val_sessions}")


# =============================================================================
# EVENT-LEVEL VALIDATION HELPERS (Pure functions on Validation `test3`, zero V1 side effects)
# =============================================================================

def _extract_validation_gt_events(
    session_name: str,
    timestamps: np.ndarray,
    attack: np.ndarray,
    start_idx_alignment: int = 30,
) -> List[Dict]:
    """
    Extracts contiguous ground-truth attack intervals where `attack == 1`
    and aligns them to the common evaluation window `t >= start_idx_alignment`.
    """
    is_atk = (attack == 1).astype(np.int8)
    diff = np.diff(np.pad(is_atk, (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    events = []
    for idx, (s, e) in enumerate(zip(starts, ends), 1):
        evaluable = bool(e >= start_idx_alignment)
        eff_s = max(int(s), int(start_idx_alignment)) if evaluable else None
        events.append({
            "event_id": f"{session_name}_atk_{idx:02d}",
            "session": session_name,
            "orig_start_idx": int(s),
            "orig_end_idx": int(e),
            "orig_start": pd.Timestamp(timestamps[s]),
            "orig_end": pd.Timestamp(timestamps[e]),
            "duration_seconds": int(e - s + 1),
            "evaluable": evaluable,
            "effective_start_idx": eff_s,
            "effective_start": pd.Timestamp(timestamps[eff_s]) if evaluable else None,
        })
    return events


def _extract_predicted_alarms(
    session_name: str,
    scores_common: np.ndarray,
    threshold: float,
    target_timestamps: np.ndarray,
) -> List[Dict]:
    """Extracts maximal contiguous runs of `scores_common > threshold`."""
    is_alarm = (scores_common > threshold).astype(np.int8)
    diff = np.diff(np.pad(is_alarm, (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    alarms = []
    for idx, (s, e) in enumerate(zip(starts, ends), 1):
        alarms.append({
            "alarm_id": f"{session_name}_alm_{idx:04d}",
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time": pd.Timestamp(target_timestamps[s]),
            "end_time": pd.Timestamp(target_timestamps[e]),
            "duration_seconds": int(e - s + 1),
        })
    return alarms


def _count_normal_fa_events(is_alarm_common: np.ndarray, is_attack_common: np.ndarray) -> int:
    """Counts maximal contiguous runs of `alarm_t AND (attack_t == 0)`."""
    normal_alarm = (is_alarm_common & (is_attack_common == 0)).astype(np.int8)
    diff = np.diff(np.pad(normal_alarm, (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    return int(len(starts))


def _match_events_one_to_one_val(
    evaluable_attacks: List[Dict],
    predicted_alarms: List[Dict],
    is_alarm_common: np.ndarray,
    target_timestamps: np.ndarray,
    start_idx_alignment: int = 30,
) -> Dict:
    """Deterministic 1-to-1 chronological event matching on validation session."""
    sorted_attacks = sorted(evaluable_attacks, key=lambda a: a["effective_start"])
    sorted_alarms = sorted(predicted_alarms, key=lambda e: e["start_time"])

    matched_attack_ids = set()
    matched_alarm_ids = set()
    delays = {}

    for atk in sorted_attacks:
        a_start = atk["effective_start"]
        a_end = atk["orig_end"]

        earliest_match = None
        for alm in sorted_alarms:
            if alm["alarm_id"] in matched_alarm_ids:
                continue
            if max(a_start, alm["start_time"]) <= min(a_end, alm["end_time"]):
                earliest_match = alm
                break

        if earliest_match is not None:
            matched_attack_ids.add(atk["event_id"])
            matched_alarm_ids.add(earliest_match["alarm_id"])

            c_s = atk["effective_start_idx"] - start_idx_alignment
            c_e = atk["orig_end_idx"] - start_idx_alignment
            interval_alarm_idxs = np.where(is_alarm_common[c_s : c_e + 1])[0]
            if len(interval_alarm_idxs) > 0:
                first_detect_idx = c_s + int(interval_alarm_idxs[0])
                detect_time = pd.Timestamp(target_timestamps[first_detect_idx])
                delays[atk["event_id"]] = max(0.0, (detect_time - a_start).total_seconds())
            else:
                delays[atk["event_id"]] = 0.0
        else:
            delays[atk["event_id"]] = np.nan

    tp = len(matched_attack_ids)
    fn = len(sorted_attacks) - tp
    fp = len(sorted_alarms) - len(matched_alarm_ids)
    return {
        "tp_count": tp,
        "fn_count": fn,
        "fp_count": fp,
        "delays": delays,
    }


def calibrate_threshold_on_test3(
    val_probs_full: np.ndarray,
    val_labels_full: np.ndarray,
    val_timestamps_full: np.ndarray,
    calibration_cfg: Dict,
    session_name: str = "test3",
) -> Dict:
    """
    Evaluates probability thresholds on Validation (`test3`) at `t >= alignment_start_idx`
    (default 30) and selects the V2 threshold using the pre-defined criterion and
    deterministic tie-breaking order.
    """
    start_idx = int(calibration_cfg.get("alignment_start_idx", 30))
    grid_start = float(calibration_cfg.get("candidate_grid_start", 0.05))
    grid_end = float(calibration_cfg.get("candidate_grid_end", 0.95))
    grid_step = float(calibration_cfg.get("candidate_grid_step", 0.05))
    extra_cands = [float(x) for x in calibration_cfg.get("additional_candidates", [0.97, 0.99])]
    val_quantiles = [
        float(q)
        for q in calibration_cfg.get(
            "validation_quantiles",
            [0.90, 0.95, 0.97, 0.98, 0.985, 0.99, 0.991, 0.992, 0.995, 0.999],
        )
    ]
    fa_cap = float(calibration_cfg.get("max_normal_fa_events_per_hour_cap", 5.0))
    max_point_fpr_cap = float(calibration_cfg.get("max_point_fpr_cap", 0.05))
    min_recall_floor = float(calibration_cfg.get("min_event_recall_floor", 0.75))

    probs_common = val_probs_full[start_idx:]
    atk_common = (val_labels_full[start_idx:] == 1).astype(np.int8)
    ts_common = val_timestamps_full[start_idx:]

    base_grid = [round(x, 6) for x in np.arange(grid_start, grid_end + 1e-9, grid_step).tolist()]
    quantile_cands = [round(float(np.quantile(probs_common, q)), 6) for q in val_quantiles]
    candidates = sorted(set(base_grid + [round(x, 6) for x in extra_cands] + quantile_cands + [0.50]))

    normal_mask = (atk_common == 0)
    normal_samples = int(np.sum(normal_mask))
    normal_hours = normal_samples / 3600.0
    attack_samples = int(np.sum(atk_common == 1))

    gt_events = _extract_validation_gt_events(
        session_name=session_name,
        timestamps=val_timestamps_full,
        attack=val_labels_full.astype(np.int8),
        start_idx_alignment=start_idx,
    )
    evaluable_attacks = [a for a in gt_events if a["evaluable"]]

    records = []
    for tau in candidates:
        is_alarm = (probs_common > tau)
        pred_alarms = _extract_predicted_alarms(session_name, probs_common, tau, ts_common)
        norm_fa_events = _count_normal_fa_events(is_alarm, atk_common)
        norm_alarm_sec = int(np.sum(is_alarm & normal_mask))

        fa_events_per_h = norm_fa_events / normal_hours if normal_hours > 0 else 0.0
        fp_sec_per_h = norm_alarm_sec / normal_hours if normal_hours > 0 else 0.0
        point_fpr = norm_alarm_sec / normal_samples if normal_samples > 0 else 0.0

        # Point-level classification metrics
        pt_tp = int(np.sum(is_alarm & (atk_common == 1)))
        pt_fp = norm_alarm_sec
        pt_fn = int(attack_samples - pt_tp)
        pt_prec = pt_tp / (pt_tp + pt_fp) if (pt_tp + pt_fp) > 0 else 0.0
        pt_rec = pt_tp / (pt_tp + pt_fn) if (pt_tp + pt_fn) > 0 else 0.0
        pt_f1 = 2.0 * pt_prec * pt_rec / (pt_prec + pt_rec) if (pt_prec + pt_rec) > 0 else 0.0

        # Event-level 1-to-1 matching
        mres = _match_events_one_to_one_val(
            evaluable_attacks=evaluable_attacks,
            predicted_alarms=pred_alarms,
            is_alarm_common=is_alarm,
            target_timestamps=ts_common,
            start_idx_alignment=start_idx,
        )
        ev_tp = mres["tp_count"]
        ev_fn = mres["fn_count"]
        ev_fp = mres["fp_count"]
        ev_prec = ev_tp / (ev_tp + ev_fp) if (ev_tp + ev_fp) > 0 else 0.0
        ev_rec = ev_tp / (ev_tp + ev_fn) if (ev_tp + ev_fn) > 0 else 0.0
        ev_f1 = 2.0 * ev_prec * ev_rec / (ev_prec + ev_rec) if (ev_prec + ev_rec) > 0 else 0.0

        valid_delays = [d for d in mres["delays"].values() if not np.isnan(d)]
        mean_delay = float(np.mean(valid_delays)) if valid_delays else None
        median_delay = float(np.median(valid_delays)) if valid_delays else None

        non_degenerate = bool(point_fpr <= max_point_fpr_cap and ev_tp > 0)
        meets_primary = bool(
            non_degenerate
            and fa_events_per_h <= fa_cap
            and ev_rec >= min_recall_floor
        )

        records.append({
            "threshold": round(float(tau), 6),
            "meets_fa_cap": meets_primary,
            "non_degenerate_fpr": non_degenerate,
            "evaluable_attacks": len(evaluable_attacks),
            "tp_events": ev_tp,
            "fn_events": ev_fn,
            "event_level_fp": ev_fp,
            "total_alarm_events": len(pred_alarms),
            "event_precision": round(ev_prec, 6),
            "event_recall": round(ev_rec, 6),
            "event_f1": round(ev_f1, 6),
            "normal_fa_events": norm_fa_events,
            "normal_fa_events_per_hour": round(fa_events_per_h, 4),
            "fp_alarm_seconds_per_hour": round(fp_sec_per_h, 4),
            "point_precision": round(pt_prec, 6),
            "point_recall": round(pt_rec, 6),
            "point_f1": round(pt_f1, 6),
            "point_fpr": round(point_fpr, 8),
            "mean_delay_sec": round(mean_delay, 2) if mean_delay is not None else None,
            "median_delay_sec": round(median_delay, 2) if median_delay is not None else None,
        })

    # Deterministic selection & tie-breaking:
    # 1. Primary pool: meets_fa_cap (point_fpr <= 0.05, fa_events_per_h <= 5.0, event_recall >= 0.75)
    # 2. Fallback 1: non_degenerate_fpr & fa_events_per_h <= 5.0
    # 3. Fallback 2: non_degenerate_fpr
    # 4. Fallback 3: all records
    feasible_primary = [r for r in records if r["meets_fa_cap"]]
    feasible_fa = [r for r in records if r["non_degenerate_fpr"] and r["normal_fa_events_per_hour"] <= fa_cap]
    feasible_non_deg = [r for r in records if r["non_degenerate_fpr"]]
    pool = feasible_primary or feasible_fa or feasible_non_deg or records

    def _sort_key(r: Dict) -> Tuple:
        return (
            round(r["event_f1"], 8),
            -round(r["normal_fa_events_per_hour"], 8),
            round(r["point_f1"], 8),
            -round(abs(r["threshold"] - 0.50), 8),
        )

    selected = max(pool, key=_sort_key)
    unconstrained_best = max(feasible_non_deg or records, key=_sort_key)
    ref_050 = next((r for r in records if abs(r["threshold"] - 0.50) < 1e-6), records[0])

    return {
        "validation_session": session_name,
        "alignment_start_idx": start_idx,
        "scored_rows": int(len(probs_common)),
        "normal_exposure_hours": round(normal_hours, 4),
        "evaluable_attacks": len(evaluable_attacks),
        "selection_criterion": calibration_cfg.get(
            "primary_selection_criterion", "max_validation_event_f1_under_fa_and_fpr_caps"
        ),
        "max_normal_fa_events_per_hour_cap": fa_cap,
        "max_point_fpr_cap": max_point_fpr_cap,
        "min_event_recall_floor": min_recall_floor,
        "feasible_under_fa_cap": bool(len(feasible_primary) > 0),
        "selected_operating_point": selected,
        "unconstrained_max_event_f1_operating_point": unconstrained_best,
        "reference_0_50_operating_point": ref_050,
        "all_candidates": records,
    }


# =============================================================================
# CORE TRAINING & VALIDATION FUNCTIONS
# =============================================================================

def train_one_epoch(
    model: FeedforwardDNNV2,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: Optional[float] = 1.0,
) -> Dict:
    """Runs one epoch of supervised binary training on `test1 + test2`."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    max_observed_grad_norm = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device, non_blocking=False)
        y_batch = y_batch.to(device, non_blocking=False).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite training loss encountered: {loss.item()}")

        loss.backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            max_observed_grad_norm = max(max_observed_grad_norm, float(grad_norm))

        optimizer.step()

        bsz = int(x_batch.size(0))
        total_loss += float(loss.item()) * bsz
        total_samples += bsz

    mean_loss = total_loss / total_samples if total_samples > 0 else float("nan")
    return {
        "train_bce_loss": mean_loss,
        "samples_seen": total_samples,
        "max_grad_norm_before_clip": round(max_observed_grad_norm, 6),
    }


@torch.no_grad()
def evaluate_validation(
    model: FeedforwardDNNV2,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    """Evaluates the model on Validation (`test3`), returning loss, PR-AUC, ROC-AUC, and probabilities."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device, non_blocking=False)
        y_col = y_batch.to(device, non_blocking=False).unsqueeze(1)

        logits = model(x_batch)
        loss = criterion(logits, y_col)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite validation loss encountered: {loss.item()}")

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        bsz = int(x_batch.size(0))
        total_loss += float(loss.item()) * bsz
        total_samples += bsz

        all_probs.append(probs)
        all_labels.append(y_batch.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0).astype(np.int8)

    mean_loss = total_loss / total_samples if total_samples > 0 else float("nan")
    if len(np.unique(labels_arr)) > 1:
        pr_auc = float(average_precision_score(labels_arr, probs_arr))
        roc_auc = float(roc_auc_score(labels_arr, probs_arr))
    else:
        pr_auc = 0.0
        roc_auc = 0.5

    pred_05 = (probs_arr > 0.50).astype(np.int8)
    tp_05 = int(np.sum((pred_05 == 1) & (labels_arr == 1)))
    fp_05 = int(np.sum((pred_05 == 1) & (labels_arr == 0)))
    fn_05 = int(np.sum((pred_05 == 0) & (labels_arr == 1)))
    prec_05 = tp_05 / (tp_05 + fp_05) if (tp_05 + fp_05) > 0 else 0.0
    rec_05 = tp_05 / (tp_05 + fn_05) if (tp_05 + fn_05) > 0 else 0.0
    f1_05 = 2.0 * prec_05 * rec_05 / (prec_05 + rec_05) if (prec_05 + rec_05) > 0 else 0.0

    return {
        "val_bce_loss": mean_loss,
        "val_pr_auc": pr_auc,
        "val_roc_auc": roc_auc,
        "val_point_f1_at_0_50": f1_05,
        "val_point_precision_at_0_50": prec_05,
        "val_point_recall_at_0_50": rec_05,
        "probs": probs_arr,
        "labels": labels_arr,
    }


def _is_better_epoch(
    cand_pr_auc: float,
    cand_loss: float,
    best_pr_auc: float,
    best_loss: float,
    min_delta: float = 1e-5,
) -> bool:
    """
    Deterministic checkpoint selection & tie-breaking:
    1. Higher Validation PR-AUC by > min_delta
    2. If within min_delta, lower Validation BCE loss by > 1e-6
    3. Otherwise keep earlier epoch.
    """
    if cand_pr_auc > best_pr_auc + min_delta:
        return True
    if abs(cand_pr_auc - best_pr_auc) <= min_delta and cand_loss < best_loss - 1e-6:
        return True
    return False


def train_single_seed(
    seed: int,
    cfg: Dict,
    bundle: Dict,
    hidden_activations: Sequence[str],
    activation_status: str,
    max_epochs_override: Optional[int] = None,
    save_checkpoint: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Trains and validates `FeedforwardDNNV2` for a single seed (`42`, `43`, or `44`).
    """
    assert_no_heldout_eval_access(bundle)

    if save_checkpoint and activation_status != VERIFIED_ACTIVATION_STATUS:
        raise RuntimeError(
            f"Refusing to save official checkpoint while activation_status != '{VERIFIED_ACTIVATION_STATUS}'."
        )

    t_cfg = cfg["training_protocol"]
    device = torch.device(t_cfg.get("device", "cpu"))
    torch.set_num_threads(int(t_cfg.get("cpu_threads", 4)))

    gen = set_deterministic_seed(seed)

    model = FeedforwardDNNV2(
        hidden_activations=hidden_activations,
        input_size=int(cfg["features"]["input_dim"]),
        hidden_dims=cfg["reference_architecture"]["hidden_layers"],
        dropout_rate=float(cfg["reference_architecture"]["dropout_rate"]),
        output_size=int(cfg["reference_architecture"]["v2_adapted_output_units"]),
        activation_status=activation_status,
    ).to(device)

    arch_audit = verify_dnn_v2_architecture(model, EXPECTED_TRAINABLE_PARAMETERS)

    # Class weight strictly from Train (test1 + test2)
    cw_info = bundle["train_class_weights"]
    strategy = t_cfg["loss"].get("pos_weight_strategy", "sqrt_balanced_from_train")
    if strategy == "sqrt_balanced_from_train":
        pos_weight_val = float(cw_info["pos_weight_sqrt"])
    elif strategy == "balanced_from_train":
        pos_weight_val = float(cw_info["pos_weight_balanced"])
    else:
        pos_weight_val = 1.0

    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    lr = float(t_cfg["optimizer"]["learning_rate"])
    weight_decay = float(t_cfg["optimizer"]["weight_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    batch_size = int(t_cfg.get("batch_size", 512))
    eval_batch_size = int(t_cfg.get("eval_batch_size", 1024))
    max_epochs = int(max_epochs_override if max_epochs_override is not None else t_cfg.get("max_epochs", 250))
    patience = int(t_cfg["model_selection"].get("early_stopping_patience", 15))
    min_delta = float(t_cfg["model_selection"].get("early_stopping_min_delta", 1e-5))
    max_grad_norm = float(t_cfg["gradient_clipping"].get("max_norm", 1.0))

    train_loader = DataLoader(
        bundle["train_dataset"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=gen,
    )
    val_loader = DataLoader(
        bundle["val_dataset"],
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    best_epoch = 0
    best_train_loss_at_best = float("inf")
    best_val_pr_auc = -1.0
    best_val_loss = float("inf")
    best_val_roc_auc = 0.0
    best_state_dict = None
    best_val_probs = None
    epochs_no_improve = 0
    stop_reason = "max_epochs_reached"

    history: List[Dict] = []
    t_start = time.time()

    if logger:
        logger.info(
            f"[Seed {seed}] Starting training: params={arch_audit['trainable_parameters']}, "
            f"activations={list(hidden_activations)}, pos_weight={pos_weight_val:.6f}, "
            f"max_epochs={max_epochs}, patience={patience}"
        )

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        train_res = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        val_res = evaluate_validation(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )
        ep_sec = time.time() - ep_t0

        cand_pr_auc = val_res["val_pr_auc"]
        cand_loss = val_res["val_bce_loss"]

        improved = _is_better_epoch(
            cand_pr_auc=cand_pr_auc,
            cand_loss=cand_loss,
            best_pr_auc=best_val_pr_auc,
            best_loss=best_val_loss,
            min_delta=min_delta,
        )

        if improved:
            best_epoch = epoch
            best_train_loss_at_best = train_res["train_bce_loss"]
            best_val_pr_auc = cand_pr_auc
            best_val_loss = cand_loss
            best_val_roc_auc = val_res["val_roc_auc"]
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_probs = val_res["probs"].copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_bce_loss": round(train_res["train_bce_loss"], 6),
            "val_bce_loss": round(cand_loss, 6),
            "val_pr_auc": round(cand_pr_auc, 6),
            "val_roc_auc": round(val_res["val_roc_auc"], 6),
            "val_point_f1_at_0_50": round(val_res["val_point_f1_at_0_50"], 6),
            "max_grad_norm": train_res["max_grad_norm_before_clip"],
            "is_best": bool(improved),
            "epoch_duration_sec": round(ep_sec, 3),
        })

        if logger:
            marker = "*BEST*" if improved else f"wait={epochs_no_improve}/{patience}"
            logger.info(
                f"[Seed {seed} | Ep {epoch:03d}/{max_epochs}] "
                f"train_loss={train_res['train_bce_loss']:.6f} | "
                f"val_loss={cand_loss:.6f} | val_pr_auc={cand_pr_auc:.6f} | "
                f"val_roc_auc={val_res['val_roc_auc']:.6f} | "
                f"grad_norm={train_res['max_grad_norm_before_clip']:.3f} | "
                f"{ep_sec:.2f}s | {marker}"
            )

        if epochs_no_improve >= patience:
            stop_reason = f"early_stopping_patience_{patience}"
            if logger:
                logger.info(
                    f"[Seed {seed}] Early stopping triggered at epoch {epoch} "
                    f"(best epoch={best_epoch}, best val_pr_auc={best_val_pr_auc:.6f})"
                )
            break

    total_duration_sec = time.time() - t_start
    model.load_state_dict(best_state_dict)
    model.eval()

    # Calibrate threshold strictly on Validation `test3`
    val_ds = bundle["val_dataset"]
    val_ts_full = val_ds.timestamp_arrays[0][val_ds.start_index :]
    val_atk_full = val_ds.label_arrays[0][val_ds.start_index :]

    calibration_res = calibrate_threshold_on_test3(
        val_probs_full=best_val_probs,
        val_labels_full=val_atk_full,
        val_timestamps_full=val_ts_full,
        calibration_cfg=t_cfg["threshold_calibration"],
        session_name="test3",
    )

    ckpt_path_str = None
    ckpt_sha256 = None
    if save_checkpoint:
        model_dir = PROJECT_ROOT / cfg["directories"]["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = model_dir / f"dnn_v2_seed{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "epoch": best_epoch,
                "stop_epoch": len(history),
                "stop_reason": stop_reason,
                "model_state_dict": best_state_dict,
                "hidden_activations": list(hidden_activations),
                "activation_status": activation_status,
                "trainable_parameters": arch_audit["trainable_parameters"],
                "pos_weight": pos_weight_val,
                "train_bce_loss_at_best": best_train_loss_at_best,
                "val_pr_auc": best_val_pr_auc,
                "val_bce_loss": best_val_loss,
                "val_roc_auc": best_val_roc_auc,
                "calibrated_threshold_test3": calibration_res["selected_operating_point"]["threshold"],
                "reference_0_50_threshold": 0.50,
            },
            ckpt_path,
        )
        ckpt_path_str = str(ckpt_path.relative_to(PROJECT_ROOT))
        ckpt_sha256 = sha256_file(ckpt_path)

    return {
        "seed": seed,
        "activation_status": activation_status,
        "hidden_activations": list(hidden_activations),
        "trainable_parameters": arch_audit["trainable_parameters"],
        "pos_weight_strategy": strategy,
        "pos_weight_value": pos_weight_val,
        "best_epoch": best_epoch,
        "stop_epoch": len(history),
        "stop_reason": stop_reason,
        "train_bce_loss_at_best": round(best_train_loss_at_best, 6),
        "best_val_pr_auc": round(best_val_pr_auc, 6),
        "best_val_roc_auc": round(best_val_roc_auc, 6),
        "best_val_bce_loss": round(best_val_loss, 6),
        "threshold_calibration_test3": calibration_res,
        "duration_sec": round(total_duration_sec, 2),
        "peak_rss_mb": round(get_peak_rss_mb(), 2),
        "checkpoint_path": ckpt_path_str,
        "checkpoint_sha256": ckpt_sha256,
        "history": history,
    }


def export_aggregated_v2_artifacts(
    cfg: Dict,
    bundle: Dict,
    results_dir: Path,
    model_dir: Path,
) -> None:
    """
    Aggregates all completed per-seed JSON records in `results/v2_dnn/v2_seed_*_run.json`
    into unified CSVs and `models/v2_dnn/dnn_v2_metadata.json`.
    """
    seed_files = sorted(results_dir.glob("v2_seed_*_run.json"))
    runs_by_seed = {}
    for sf in seed_files:
        with open(sf, "r", encoding="utf-8") as f:
            r = json.load(f)
            runs_by_seed[int(r["seed"])] = r

    ordered_runs = [runs_by_seed[s] for s in sorted(runs_by_seed.keys())]
    if not ordered_runs:
        return

    all_history_rows = []
    summary_rows = []
    calibration_rows = []
    checkpoint_hashes = {}

    for r in ordered_runs:
        seed = r["seed"]
        all_history_rows.extend(r["history"])
        if r.get("checkpoint_path") and r.get("checkpoint_sha256"):
            checkpoint_hashes[Path(r["checkpoint_path"]).name] = r["checkpoint_sha256"]

        cal = r["threshold_calibration_test3"]
        sel = cal["selected_operating_point"]
        unc = cal["unconstrained_max_event_f1_operating_point"]
        ref05 = cal["reference_0_50_operating_point"]

        for cand in cal["all_candidates"]:
            row_c = {"seed": seed, **cand}
            row_c["is_selected_calibrated"] = bool(abs(cand["threshold"] - sel["threshold"]) < 1e-6)
            row_c["is_unconstrained_best_f1"] = bool(abs(cand["threshold"] - unc["threshold"]) < 1e-6)
            row_c["is_reference_0_50"] = bool(abs(cand["threshold"] - 0.50) < 1e-6)
            calibration_rows.append(row_c)

        summary_rows.append({
            "seed": seed,
            "trainable_parameters": r["trainable_parameters"],
            "pos_weight": r["pos_weight_value"],
            "best_epoch": r["best_epoch"],
            "stop_epoch": r["stop_epoch"],
            "stop_reason": r["stop_reason"],
            "train_bce_loss_at_best": r["train_bce_loss_at_best"],
            "val_bce_loss": r["best_val_bce_loss"],
            "val_pr_auc": r["best_val_pr_auc"],
            "val_roc_auc": r["best_val_roc_auc"],
            "calibrated_threshold_test3": sel["threshold"],
            "val_tp_events_calibrated": sel["tp_events"],
            "val_fn_events_calibrated": sel["fn_events"],
            "val_fp_events_calibrated": sel["event_level_fp"],
            "val_event_precision_calibrated": sel["event_precision"],
            "val_event_recall_calibrated": sel["event_recall"],
            "val_event_f1_calibrated": sel["event_f1"],
            "val_normal_fa_per_h_calibrated": sel["normal_fa_events_per_hour"],
            "val_point_f1_calibrated": sel["point_f1"],
            "val_median_delay_sec_calibrated": sel["median_delay_sec"],
            "unconstrained_best_threshold_test3": unc["threshold"],
            "val_event_f1_unconstrained_best": unc["event_f1"],
            "val_normal_fa_per_h_unconstrained_best": unc["normal_fa_events_per_hour"],
            "val_event_f1_at_0_50": ref05["event_f1"],
            "val_event_recall_at_0_50": ref05["event_recall"],
            "val_normal_fa_per_h_at_0_50": ref05["normal_fa_events_per_hour"],
            " val_point_f1_at_0_50": ref05["point_f1"],
            "duration_sec": r["duration_sec"],
            "peak_rss_mb": r["peak_rss_mb"],
            "checkpoint_path": r["checkpoint_path"],
        })

    pd.DataFrame(all_history_rows).to_csv(results_dir / "v2_dnn_training_history.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(results_dir / "v2_dnn_validation_summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(results_dir / "v2_dnn_threshold_calibration_test3.csv", index=False)

    pr_aucs = [r["best_val_pr_auc"] for r in ordered_runs]
    val_losses = [r["best_val_bce_loss"] for r in ordered_runs]
    cal_f1s = [r["threshold_calibration_test3"]["selected_operating_point"]["event_f1"] for r in ordered_runs]

    meta_payload = {
        "model_name": "feedforward_dnn_v2",
        "protocol_version": cfg["protocol_version"],
        "experiment_id": cfg["experiment_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "device": "cpu",
        "reference_paper": {
            "authors": "Karel Kuchar & Radek Fujdiak (2025)",
            "doi": cfg["reference_architecture"]["source_paper_doi"],
            "section": cfg["reference_architecture"]["source_section"],
            "adaptation_disclaimer": cfg["reference_architecture"]["adaptation_disclaimer"],
        },
        "architecture": {
            "input_size": 79,
            "hidden_layers": cfg["reference_architecture"]["hidden_layers"],
            "hidden_activations": cfg["reference_architecture"]["hidden_activations"],
            "hidden_activations_verification_status": cfg["reference_architecture"]["hidden_activations_verification_status"],
            "dropout_rate": cfg["reference_architecture"]["dropout_rate"],
            "dropout_after_hidden_layers_1based": cfg["reference_architecture"]["dropout_after_hidden_layer_indices_1based"],
            "output_units": 1,
            "output_type": "logit",
            "inference_activation": "sigmoid",
            "batch_norm": False,
            "layer_norm": False,
            "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        },
        "data_protocol": {
            "scaler_path": cfg["dataset"]["scaler_path"],
            "scaler_refit": False,
            "scaler_fit_sample_count": 442802,
            "train_sessions": ["test1", "test2"],
            "train_rows": 162002,
            "train_attack_rows": 4078,
            "validation_sessions": ["test3"],
            "validation_rows": 108001,
            "validation_attack_rows": 1535,
            "heldout_eval_sessions_accessed_during_training": False,
        },
        "train_class_weights": bundle["train_class_weights"],
        "training_hyperparameters": cfg["training_protocol"],
        "multiseed_validation_summary": {
            "completed_seeds": [r["seed"] for r in ordered_runs],
            "mean_val_pr_auc": round(float(np.mean(pr_aucs)), 6),
            "std_val_pr_auc": round(float(np.std(pr_aucs)), 6),
            "mean_val_bce_loss": round(float(np.mean(val_losses)), 6),
            "std_val_bce_loss": round(float(np.std(val_losses)), 6),
            "mean_val_calibrated_event_f1": round(float(np.mean(cal_f1s)), 6),
            "std_val_calibrated_event_f1": round(float(np.std(cal_f1s)), 6),
        },
        "checkpoint_sha256": checkpoint_hashes,
        "runs": [
            {k: v for k, v in r.items() if k != "history"}
            for r in ordered_runs
        ],
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "dnn_v2_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train Supervised Binary Feedforward DNN V2 (HAI 21.03)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to configs/dnn_v2_protocol.yaml",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional override of seeds to run (default from config: 42 43 44)",
    )
    args = parser.parse_args()

    cfg = load_v2_config(Path(args.config))
    ref_arch = cfg["reference_architecture"]
    ver_status = ref_arch.get("hidden_activations_verification_status", "")
    hidden_acts = ref_arch.get("hidden_activations", None)

    if ver_status != VERIFIED_ACTIVATION_STATUS or tuple(hidden_acts or ()) != VERIFIED_HIDDEN_ACTIVATIONS:
        print("=========================================================================", file=sys.stderr)
        print("[BLOCKER - PHẦN A INTEGRITY GATE]: UNVERIFIED LISTING 1 ACTIVATIONS", file=sys.stderr)
        print("=========================================================================", file=sys.stderr)
        sys.exit(2)

    results_dir = PROJECT_ROOT / cfg["directories"]["results_dir"]
    model_dir = PROJECT_ROOT / cfg["directories"]["model_dir"]
    logger = setup_logger(results_dir / "v2_dnn_training.log")

    bundle = build_v2_dataloaders(
        config_path=Path(args.config),
        batch_size=int(cfg["training_protocol"]["batch_size"]),
        include_eval=False,
    )
    assert_no_heldout_eval_access(bundle)

    seeds = args.seeds if args.seeds is not None else cfg["training_protocol"]["seeds"]

    for seed in seeds:
        logger.info("=========================================================================")
        logger.info(f">>> Training Official DNN V2 | Seed {seed} <<<")
        logger.info("=========================================================================")
        run_res = train_single_seed(
            seed=int(seed),
            cfg=cfg,
            bundle=bundle,
            hidden_activations=hidden_acts,
            activation_status=VERIFIED_ACTIVATION_STATUS,
            save_checkpoint=True,
            logger=logger,
        )
        seed_json_path = results_dir / f"v2_seed_{seed}_run.json"
        with open(seed_json_path, "w", encoding="utf-8") as f:
            json.dump(run_res, f, indent=2)

        sel_op = run_res["threshold_calibration_test3"]["selected_operating_point"]
        ref_05 = run_res["threshold_calibration_test3"]["reference_0_50_operating_point"]
        logger.info(
            f"[Seed {seed} DONE] Best Epoch={run_res['best_epoch']}/{run_res['stop_epoch']} ({run_res['stop_reason']}) "
            f"| Train Loss@Best={run_res['train_bce_loss_at_best']:.6f} "
            f"| Val BCE={run_res['best_val_bce_loss']:.6f} "
            f"| Val PR-AUC={run_res['best_val_pr_auc']:.6f} "
            f"| Val ROC-AUC={run_res['best_val_roc_auc']:.6f} "
            f"| Calibrated Tau={sel_op['threshold']:.6f} "
            f"(Event F1={sel_op['event_f1']:.4f}, Recall={sel_op['event_recall']:.4f}, FA/h={sel_op['normal_fa_events_per_hour']:.4f}) "
            f"| Tau=0.50 (Event F1={ref_05['event_f1']:.4f}, FA/h={ref_05['normal_fa_events_per_hour']:.4f}) "
            f"| Time={run_res['duration_sec']:.2f}s | Peak RSS={run_res['peak_rss_mb']:.2f} MB"
        )

    export_aggregated_v2_artifacts(cfg, bundle, results_dir, model_dir)
    logger.info(f"[OK] Exported aggregated V2 summaries to {results_dir} and {model_dir / 'dnn_v2_metadata.json'}")


if __name__ == "__main__":
    main()
