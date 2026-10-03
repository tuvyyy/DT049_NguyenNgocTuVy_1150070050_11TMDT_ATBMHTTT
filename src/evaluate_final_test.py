#!/usr/bin/env python3
"""
HAI 21.03 Official Final Test Evaluation Pipeline (DT049)
=========================================================
Executes the FIRST and ONLY frozen final test evaluation across official test1-test5.

Scientific integrity & freeze constraints:
- Development freeze commit: 6ce11fb8bcb682c10e125219d0b964f394b8a5d2
- Protocol commit: 6a0613b87a0a9d89c89b6ae848350b44c1aa1740
- Evaluates:
  * Baseline 0: Persistence (frozen Q99.9 = 1.6805225610733032)
  * Baseline 1: Ridge alpha=300 (frozen Q99.9 = 1.4009277820587158)
  * Primary GRU: W=30, H=64, 1 layer (32,975 parameters)
    - Seed 42: canonical primary (frozen Q99.9 = 1.0585819482803345)
    - Seed 43: stability replicate 1 (frozen Q99.9 = 1.1958686113357544)
    - Seed 44: stability replicate 2 (frozen Q99.9 = 1.0768321752548218)
- Common timeline: target row index >= 30 within each test file independently.
- Deterministic 1-to-1 chronological event matching.
- Separation of Event-level FP (in Precision/F1) vs Normal False Alarms (operational burden).
- Output tables: final_test_summary.csv, final_test_multiseed.csv, final_attack_events.csv,
  final_alarm_events.csv, final_detection_delays.csv, final_per_file.csv.
- Execution metadata: final_test_execution.json.
- Publication figures under results/figures/final_test/.
"""

import os
import sys
import gc
import json
import yaml
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import psutil
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gru_model import OneStepGRU
from gru_dataset import SessionWindowDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "final_evaluation.yaml"
MANIFEST_PATH = REPO_ROOT / "results" / "audit" / "pretest_freeze_manifest.json"
PROCESSED_TEST_DIR = REPO_ROOT / "data" / "processed" / "hai_21_03" / "final_test"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "final_test"
AUDIT_DIR = REPO_ROOT / "results" / "audit"
MODELS_DIR = REPO_ROOT / "models"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("HAI2103_FinalTestEval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_file = REPO_ROOT / "results" / "logs" / "evaluate_final_test.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def score_persistence(feats: np.ndarray) -> np.ndarray:
    """
    Persistence: x_hat_t = x_{t-1}.
    Evaluated for target index t >= 30.
    Returns float32 RMSE array of length len(feats) - 30.
    """
    targets = feats[30:]           # shape: (N - 30, 79)
    preds = feats[29:-1]           # shape: (N - 30, 79)
    diffs = targets.astype(np.float64) - preds.astype(np.float64)
    scores = np.sqrt(np.mean(diffs ** 2, axis=1)).astype(np.float32)
    return scores


def score_ridge(feats: np.ndarray, ridge_model) -> np.ndarray:
    """
    Ridge: x_hat_t = x_{t-1} W^T + b^T.
    Evaluated for target index t >= 30.
    Returns float32 RMSE array of length len(feats) - 30.
    """
    inputs = feats[29:-1]          # shape: (N - 30, 79)
    targets = feats[30:]           # shape: (N - 30, 79)
    preds = ridge_model.predict(inputs).astype(np.float64)
    diffs = targets.astype(np.float64) - preds
    scores = np.sqrt(np.mean(diffs ** 2, axis=1)).astype(np.float32)
    return scores


def score_gru(feats: np.ndarray, model: nn.Module, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """
    GRU: input rows [t-30 : t], target row t.
    Evaluated for target index t >= 30.
    Returns float32 RMSE array of length len(feats) - 30.
    """
    ds = SessionWindowDataset([feats], window_size=30, stride=1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    scores_list = []

    model.eval()
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            preds = model(bx).cpu().numpy().astype(np.float64)
            targets = by.numpy().astype(np.float64)
            diffs = targets - preds
            batch_scores = np.sqrt(np.mean(diffs ** 2, axis=1)).astype(np.float32)
            scores_list.append(batch_scores)

    return np.concatenate(scores_list)


# =============================================================================
# EVENT EXTRACTION & MATCHING
# =============================================================================

def extract_ground_truth_events(session_name: str, timestamps: np.ndarray, atk: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    """
    Extracts maximal contiguous runs of attack == 1.
    Evaluates common timeline target index >= 30.
    """
    diff = np.diff(np.pad(atk, (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    events = []
    for idx, (s, e) in enumerate(zip(starts, ends)):
        event_id = f"{session_name}_atk_{idx+1:02d}"
        orig_start = pd.Timestamp(timestamps[s])
        orig_end = pd.Timestamp(timestamps[e])
        dur = int(e - s + 1)

        p1_active = bool(np.any(p1[s : e + 1] == 1))
        p2_active = bool(np.any(p2[s : e + 1] == 1))
        p3_active = bool(np.any(p3[s : e + 1] == 1))

        pset = []
        if p1_active: pset.append("P1")
        if p2_active: pset.append("P2")
        if p3_active: pset.append("P3")

        # Common timeline check (target row index >= 30)
        if e < 30:
            evaluable = False
            left_censored = False
            eff_start = None
            eff_start_idx = None
            status = "not_evaluable_before_common_timeline"
        elif s < 30 and e >= 30:
            evaluable = True
            left_censored = True
            eff_start = pd.Timestamp(timestamps[30])
            eff_start_idx = 30
            status = "evaluable_left_censored"
        else:
            evaluable = True
            left_censored = False
            eff_start = orig_start
            eff_start_idx = s
            status = "evaluable"

        events.append({
            "session_name": session_name,
            "event_id": event_id,
            "orig_start_idx": int(s),
            "orig_end_idx": int(e),
            "orig_start": orig_start,
            "orig_end": orig_end,
            "duration_seconds": dur,
            "attack_P1": p1_active,
            "attack_P2": p2_active,
            "attack_P3": p3_active,
            "process_set": pset,
            "evaluable": evaluable,
            "left_censored": left_censored,
            "effective_start": eff_start,
            "effective_start_idx": eff_start_idx,
            "status": status,
        })

    return events


def extract_predicted_alarm_events(session_name: str, model_id: str, scores: np.ndarray, threshold: float, target_timestamps: np.ndarray):
    """
    Extracts maximal contiguous runs of score > threshold on the common timeline.
    No smoothing, no hysteresis, no alarm merging, no min duration.
    """
    is_alarm = (scores > threshold)
    diff = np.diff(np.pad(is_alarm.astype(np.int8), (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    events = []
    for idx, (s, e) in enumerate(zip(starts, ends)):
        alarm_id = f"{session_name}_{model_id}_alm_{idx+1:04d}"
        s_time = pd.Timestamp(target_timestamps[s])
        e_time = pd.Timestamp(target_timestamps[e])
        dur = int(e - s + 1)
        sub_scores = scores[s : e + 1]

        events.append({
            "session_name": session_name,
            "model_id": model_id,
            "alarm_id": alarm_id,
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time": s_time,
            "end_time": e_time,
            "duration_seconds": dur,
            "peak_score": float(np.max(sub_scores)),
            "mean_score": float(np.mean(sub_scores)),
        })

    return events


def extract_normal_false_alarm_events(session_name: str, is_alarm_common: np.ndarray, is_attack_common: np.ndarray, target_timestamps: np.ndarray):
    """
    Constructs normal_alarm_t = alarm_t AND (attack_t == 0).
    A normal false-alarm event is a maximal contiguous run of normal_alarm_t == True.
    Attack timestamps strictly break normal runs.
    """
    normal_alarm = is_alarm_common & (is_attack_common == 0)
    diff = np.diff(np.pad(normal_alarm.astype(np.int8), (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    events = []
    for idx, (s, e) in enumerate(zip(starts, ends)):
        dur = int(e - s + 1)
        events.append({
            "session_name": session_name,
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time": pd.Timestamp(target_timestamps[s]),
            "end_time": pd.Timestamp(target_timestamps[e]),
            "duration_seconds": dur,
        })
    return events


def match_events_one_to_one(evaluable_attacks: list, predicted_alarms: list, is_alarm_common: np.ndarray, target_timestamps: np.ndarray):
    """
    Deterministic 1-to-1 chronological matching within a test file:
    1. Sort attacks by effective_start.
    2. Sort alarms by start_time.
    3. In chronological order, match earliest unmatched alarm with non-zero overlap.
    Returns:
      matched_attacks, matched_alarms, unmatched_attacks (FN), unmatched_alarms (FP),
      detection_delays_dict: {event_id: delay_seconds}
    """
    sorted_attacks = sorted(evaluable_attacks, key=lambda a: a["effective_start"])
    sorted_alarms = sorted(predicted_alarms, key=lambda e: e["start_time"])

    matched_attack_ids = set()
    matched_alarm_ids = set()
    matched_pairs = []
    delays = {}

    for atk in sorted_attacks:
        a_start = atk["effective_start"]
        a_end = atk["orig_end"]

        earliest_match = None
        for alm in sorted_alarms:
            if alm["alarm_id"] in matched_alarm_ids:
                continue

            # Temporal overlap: max(start) <= min(end)
            if max(a_start, alm["start_time"]) <= min(a_end, alm["end_time"]):
                earliest_match = alm
                break

        if earliest_match is not None:
            matched_attack_ids.add(atk["event_id"])
            matched_alarm_ids.add(earliest_match["alarm_id"])
            matched_pairs.append((atk, earliest_match))

            # Detection delay: first alarm-positive timestamp in [effective_start, end]
            # Since target_timestamps is index-aligned with is_alarm_common:
            # common indices correspond to index in target_timestamps:
            # effective_start_idx in full file is atk["effective_start_idx"], so in common: atk["effective_start_idx"] - 30
            c_s = atk["effective_start_idx"] - 30
            c_e = atk["orig_end_idx"] - 30
            interval_alarm_idxs = np.where(is_alarm_common[c_s : c_e + 1])[0]

            if len(interval_alarm_idxs) > 0:
                first_detect_idx = c_s + interval_alarm_idxs[0]
                detect_time = pd.Timestamp(target_timestamps[first_detect_idx])
                delay_sec = max(0.0, (detect_time - a_start).total_seconds())
                delays[atk["event_id"]] = delay_sec
            else:
                delays[atk["event_id"]] = 0.0  # fallback (should not occur if overlap > 0)
        else:
            delays[atk["event_id"]] = np.nan

    unmatched_attacks = [a for a in sorted_attacks if a["event_id"] not in matched_attack_ids]
    unmatched_alarms = [e for e in sorted_alarms if e["alarm_id"] not in matched_alarm_ids]

    return {
        "matched_pairs": matched_pairs,
        "tp_count": len(matched_pairs),
        "fn_count": len(unmatched_attacks),
        "fp_count": len(unmatched_alarms),
        "unmatched_attacks": unmatched_attacks,
        "unmatched_alarms": unmatched_alarms,
        "delays": delays,
    }


# =============================================================================
# MAIN EVALUATION SCRIPT
# =============================================================================

def main():
    logger = setup_logger()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    start_rss = get_memory_mb()
    peak_rss = start_rss

    logger.info("=================================================================")
    logger.info("Executing FIRST OFFICIAL FINAL TEST EVALUATION (DT049)")
    logger.info(f"Start Time (UTC): {start_time.isoformat()}")
    logger.info(f"Initial Memory Usage: {start_rss:.2f} MB")
    logger.info("=================================================================")

    # Load configuration and manifest
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    freeze_commit = manifest["repository"]["development_freeze_commit"]
    protocol_commit = "6a0613b87a0a9d89c89b6ae848350b44c1aa1740"
    git_tag = "dt049-pretest-freeze"
    logger.info(f"Development Freeze Commit: {freeze_commit}")
    logger.info(f"Protocol Commit:           {protocol_commit}")
    logger.info(f"Git Tag:                   {git_tag}")

    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    # 1. Load Frozen Models
    logger.info("Loading Frozen Model Checkpoints...")

    # Ridge
    ridge_path = MODELS_DIR / "baselines" / "ridge.joblib"
    ridge_model = joblib.load(ridge_path)
    logger.info(f"Loaded Ridge baseline (alpha=300) from {ridge_path}")

    # GRU Models (seeds 42, 43, 44)
    gru_models = {}
    for seed in [42, 43, 44]:
        ckpt_path = MODELS_DIR / "primary" / f"gru_w30_seed{seed}.pt"
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        m = OneStepGRU(input_size=79, hidden_size=64, num_layers=1).to(device)
        m.load_state_dict(checkpoint["model_state_dict"])
        m.eval()
        gru_models[seed] = m
        logger.info(f"Loaded GRU seed {seed} from {ckpt_path} (epoch {checkpoint['epoch']})")

    # Thresholds
    thresholds = {
        "persistence": float(manifest["frozen_models"]["persistence_baseline_0"]["frozen_thresholds"]["Q99.9_primary"]),
        "ridge": float(manifest["frozen_models"]["ridge_baseline_1"]["frozen_thresholds"]["Q99.9_primary"]),
        "gru_seed42": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["frozen_thresholds"]["Q99.9_primary"]),
        "gru_seed43": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["frozen_thresholds"]["Q99.9_primary"]),
        "gru_seed44": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["frozen_thresholds"]["Q99.9_primary"]),
    }
    logger.info(f"Frozen Primary Q99.9 Thresholds: {thresholds}")

    # Model identifiers
    model_ids = ["persistence", "ridge", "gru_seed42", "gru_seed43", "gru_seed44"]
    model_labels = {
        "persistence": "Baseline 0: Persistence",
        "ridge": "Baseline 1: Ridge (alpha=300)",
        "gru_seed42": "Primary GRU (Seed 42, Canonical)",
        "gru_seed43": "Primary GRU (Seed 43, Replicate 1)",
        "gru_seed44": "Primary GRU (Seed 44, Replicate 2)",
    }

    test_sessions = ["test1", "test2", "test3", "test4", "test5"]

    # Storage for detailed evaluation results
    all_attack_events = []
    all_alarm_events = []
    all_detection_delays = []
    session_data_dict = {}

    per_file_records = []
    model_session_scores = {mid: {} for mid in model_ids}

    # 2. Iterate through test sessions and evaluate
    logger.info("=================================================================")
    logger.info("Executing Evaluation Across All 5 Test Sessions...")
    logger.info("=================================================================")

    for sname in test_sessions:
        logger.info(f">>> Evaluating Session: {sname} <<<")
        feats = np.load(PROCESSED_TEST_DIR / f"{sname}_features.npy", mmap_mode="r")
        ts = np.load(PROCESSED_TEST_DIR / f"{sname}_timestamps.npy", mmap_mode="r")
        atk = np.load(PROCESSED_TEST_DIR / f"{sname}_attack.npy", mmap_mode="r")
        p1 = np.load(PROCESSED_TEST_DIR / f"{sname}_attack_P1.npy", mmap_mode="r")
        p2 = np.load(PROCESSED_TEST_DIR / f"{sname}_attack_P2.npy", mmap_mode="r")
        p3 = np.load(PROCESSED_TEST_DIR / f"{sname}_attack_P3.npy", mmap_mode="r")

        total_rows = len(feats)
        common_rows = total_rows - 30
        target_ts = ts[30:]
        target_atk = atk[30:]

        logger.info(f"Total Rows: {total_rows:,} | Common Evaluated Rows (target >= 30): {common_rows:,}")

        # Ground-truth attacks
        gt_attacks = extract_ground_truth_events(sname, ts, atk, p1, p2, p3)
        all_attack_events.extend(gt_attacks)
        evaluable_atks = [a for a in gt_attacks if a["evaluable"]]
        logger.info(f"Identified {len(gt_attacks)} ground-truth attack events (Evaluable: {len(evaluable_atks)})")

        # Normal exposure
        normal_mask = (target_atk == 0)
        normal_samples = int(np.sum(normal_mask))
        normal_hours = normal_samples / 3600.0

        session_data_dict[sname] = {
            "total_rows": total_rows,
            "common_rows": common_rows,
            "target_ts": target_ts,
            "target_atk": target_atk,
            "normal_samples": normal_samples,
            "normal_hours": normal_hours,
            "evaluable_attacks": evaluable_atks,
        }

        # Compute Anomaly Scores on common region
        # 1. Persistence
        t0 = datetime.now()
        scores_pers = score_persistence(feats)
        model_session_scores["persistence"][sname] = scores_pers
        logger.info(f"[{sname}] Persistence scored in {(datetime.now() - t0).total_seconds():.2f}s")

        # 2. Ridge
        t0 = datetime.now()
        scores_ridge = score_ridge(feats, ridge_model)
        model_session_scores["ridge"][sname] = scores_ridge
        logger.info(f"[{sname}] Ridge scored in {(datetime.now() - t0).total_seconds():.2f}s")

        # 3. GRU Seeds 42, 43, 44
        for seed in [42, 43, 44]:
            t0 = datetime.now()
            scores_g = score_gru(feats, gru_models[seed], device, batch_size=512)
            model_session_scores[f"gru_seed{seed}"][sname] = scores_g
            logger.info(f"[{sname}] GRU seed {seed} scored in {(datetime.now() - t0).total_seconds():.2f}s")

        peak_rss = max(peak_rss, get_memory_mb())

        # Evaluate each model on this session
        for mid in model_ids:
            scores = model_session_scores[mid][sname]
            thresh = thresholds[mid]
            is_alarm = (scores > thresh)

            # Predicted alarm events
            pred_alarms = extract_predicted_alarm_events(sname, mid, scores, thresh, target_ts)
            all_alarm_events.extend(pred_alarms)

            # Normal false alarm events
            norm_fa_events = extract_normal_false_alarm_events(sname, is_alarm, target_atk, target_ts)
            norm_fa_count = len(norm_fa_events)
            norm_alarm_samples = int(np.sum(is_alarm & normal_mask))

            fa_events_per_h = norm_fa_count / normal_hours if normal_hours > 0 else 0.0
            fp_alarm_sec_per_h = norm_alarm_samples / normal_hours if normal_hours > 0 else 0.0
            point_fpr = norm_alarm_samples / normal_samples if normal_samples > 0 else 0.0

            # Match events 1-to-1
            match_res = match_events_one_to_one(evaluable_atks, pred_alarms, is_alarm, target_ts)
            tp = match_res["tp_count"]
            fn = match_res["fn_count"]
            fp_ev = match_res["fp_count"]

            prec = tp / (tp + fp_ev) if (tp + fp_ev) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            valid_delays = [d for d in match_res["delays"].values() if not np.isnan(d)]
            mean_d = float(np.mean(valid_delays)) if valid_delays else np.nan
            med_d = float(np.median(valid_delays)) if valid_delays else np.nan

            per_file_records.append({
                "session": sname,
                "model_id": mid,
                "model_name": model_labels[mid],
                "threshold": thresh,
                "total_rows": total_rows,
                "scored_transitions": common_rows,
                "normal_exposure_hours": round(normal_hours, 4),
                "evaluable_attacks": len(evaluable_atks),
                "tp_events": tp,
                "fn_events": fn,
                "event_level_fp": fp_ev,
                "total_alarm_events": len(pred_alarms),
                "event_precision": round(prec, 6),
                "event_recall": round(rec, 6),
                "event_f1": round(f1, 6),
                "normal_fa_events": norm_fa_count,
                "normal_fa_events_per_hour": round(fa_events_per_h, 4),
                "fp_alarm_seconds_per_hour": round(fp_alarm_sec_per_h, 4),
                "point_fpr": round(point_fpr, 8),
                "mean_delay_sec": round(mean_d, 2) if not np.isnan(mean_d) else np.nan,
                "median_delay_sec": round(med_d, 2) if not np.isnan(med_d) else np.nan,
            })

            logger.info(
                f"[{sname} | {mid:10s}] TP={tp:2d}/{len(evaluable_atks)} | FN={fn:2d} | FP_ev={fp_ev:3d} | "
                f"F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | Normal FA={fa_events_per_h:.3f} ev/h | "
                f"Med Delay={med_d:.1f}s"
            )

    # 3. Overall Aggregation across All 5 Test Sessions
    logger.info("=================================================================")
    logger.info("Computing Global Aggregates Across Full 5-Session Official Test...")
    logger.info("=================================================================")

    per_file_df = pd.DataFrame(per_file_records)
    per_file_csv_path = METRICS_DIR / "final_per_file.csv"
    per_file_df.to_csv(per_file_csv_path, index=False)
    logger.info(f"Saved: {per_file_csv_path}")

    # Compute attack-by-attack detection table
    # Ground truth total attacks = 50
    total_gt_count = len(all_attack_events)
    logger.info(f"Total Ground-Truth Attacks across All Sessions: {total_gt_count}")
    assert total_gt_count == 50, f"Expected 50 attacks, got {total_gt_count}!"

    # Build comprehensive attack-by-attack table
    attack_catalog = []
    for atk in all_attack_events:
        sname = atk["session_name"]
        aid = atk["event_id"]
        evaluable = atk["evaluable"]

        rec = {
            "attack_event_id": aid,
            "session": sname,
            "orig_start": str(atk["orig_start"]),
            "orig_end": str(atk["orig_end"]),
            "duration_seconds": atk["duration_seconds"],
            "evaluable": evaluable,
            "left_censored": atk["left_censored"],
            "effective_start": str(atk["effective_start"]) if atk["effective_start"] is not None else "",
            "process_set": ";".join(atk["process_set"]),
            "attack_P1": atk["attack_P1"],
            "attack_P2": atk["attack_P2"],
            "attack_P3": atk["attack_P3"],
        }

        # Check detection by each model
        for mid in model_ids:
            scores = model_session_scores[mid][sname]
            thresh = thresholds[mid]
            is_alarm = (scores > thresh)
            target_ts = session_data_dict[sname]["target_ts"]
            pred_alarms = extract_predicted_alarm_events(sname, mid, scores, thresh, target_ts)
            match_res = match_events_one_to_one([atk], pred_alarms, is_alarm, target_ts)

            detected = (match_res["tp_count"] == 1)
            d_val = match_res["delays"].get(aid, np.nan)

            rec[f"{mid}_detected"] = detected
            rec[f"{mid}_delay_seconds"] = round(d_val, 2) if not np.isnan(d_val) else np.nan

        attack_catalog.append(rec)

    attack_catalog_df = pd.DataFrame(attack_catalog)
    attack_events_csv_path = METRICS_DIR / "final_attack_events.csv"
    attack_catalog_df.to_csv(attack_events_csv_path, index=False)
    logger.info(f"Saved: {attack_events_csv_path}")

    # Build all alarm events CSV
    alarm_events_df = pd.DataFrame(all_alarm_events)
    alarm_events_csv_path = METRICS_DIR / "final_alarm_events.csv"
    alarm_events_df.to_csv(alarm_events_csv_path, index=False)
    logger.info(f"Saved: {alarm_events_csv_path}")

    # Build Global Summary Metrics
    total_evaluable_attacks = sum(len(session_data_dict[s]["evaluable_attacks"]) for s in test_sessions)
    total_normal_hours = sum(session_data_dict[s]["normal_hours"] for s in test_sessions)
    total_normal_samples = sum(session_data_dict[s]["normal_samples"] for s in test_sessions)
    total_scored_transitions = sum(session_data_dict[s]["common_rows"] for s in test_sessions)

    summary_records = []
    multiseed_records = []
    delays_records = []

    for mid in model_ids:
        # Sum TP, FN, FP_ev, alarms across sessions
        sub_df = per_file_df[per_file_df["model_id"] == mid]
        tp_total = int(sub_df["tp_events"].sum())
        fn_total = int(sub_df["fn_events"].sum())
        fp_ev_total = int(sub_df["event_level_fp"].sum())
        alarms_total = int(sub_df["total_alarm_events"].sum())
        norm_fa_events_total = int(sub_df["normal_fa_events"].sum())
        norm_alarm_samples_total = int(np.sum([
            np.sum((model_session_scores[mid][s] > thresholds[mid]) & (session_data_dict[s]["target_atk"] == 0))
            for s in test_sessions
        ]))

        prec = tp_total / (tp_total + fp_ev_total) if (tp_total + fp_ev_total) > 0 else 0.0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        norm_fa_rate = norm_fa_events_total / total_normal_hours
        fp_sec_per_h = norm_alarm_samples_total / total_normal_hours
        point_fpr = norm_alarm_samples_total / total_normal_samples

        # Collect all valid detection delays
        model_delays = attack_catalog_df[f"{mid}_delay_seconds"].dropna().tolist()

        d_mean = float(np.mean(model_delays)) if model_delays else np.nan
        d_med = float(np.median(model_delays)) if model_delays else np.nan
        d_std = float(np.std(model_delays)) if model_delays else np.nan
        d_min = float(np.min(model_delays)) if model_delays else np.nan
        d_max = float(np.max(model_delays)) if model_delays else np.nan
        d_p95 = float(np.percentile(model_delays, 95)) if model_delays else np.nan

        summary_records.append({
            "model_id": mid,
            "model_name": model_labels[mid],
            "operating_threshold": thresholds[mid],
            "total_scored_transitions": total_scored_transitions,
            "total_normal_hours": round(total_normal_hours, 4),
            "evaluable_attacks": total_evaluable_attacks,
            "tp_detected": tp_total,
            "fn_missed": fn_total,
            "event_level_fp": fp_ev_total,
            "total_predicted_alarms": alarms_total,
            "event_precision": round(prec, 6),
            "event_recall": round(rec, 6),
            "event_f1": round(f1, 6),
            "normal_fa_events": norm_fa_events_total,
            "normal_fa_events_per_hour": round(norm_fa_rate, 4),
            "false_positive_alarm_seconds_per_hour": round(fp_sec_per_h, 4),
            "point_fpr": round(point_fpr, 8),
            "delay_mean_sec": round(d_mean, 2),
            "delay_median_sec": round(d_med, 2),
            "delay_std_sec": round(d_std, 2),
            "delay_min_sec": round(d_min, 2),
            "delay_max_sec": round(d_max, 2),
            "delay_p95_sec": round(d_p95, 2),
            "detected_count": len(model_delays),
        })

        if "gru" in mid:
            seed_val = int(mid.replace("gru_seed", ""))
            multiseed_records.append({
                "seed": seed_val,
                "model_id": mid,
                "operating_threshold": thresholds[mid],
                "event_precision": prec,
                "event_recall": rec,
                "event_f1": f1,
                "tp_detected": tp_total,
                "fn_missed": fn_total,
                "event_level_fp": fp_ev_total,
                "normal_fa_events_per_hour": norm_fa_rate,
                "false_positive_alarm_seconds_per_hour": fp_sec_per_h,
                "point_fpr": point_fpr,
                "delay_mean_sec": d_mean,
                "delay_median_sec": d_med,
                "delay_p95_sec": d_p95,
            })

        for val in model_delays:
            delays_records.append({"model_id": mid, "delay_seconds": val})

    # Save summary tables
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = METRICS_DIR / "final_test_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved Primary Test Summary Table: {summary_csv_path}")

    multiseed_df = pd.DataFrame(multiseed_records)
    multiseed_csv_path = METRICS_DIR / "final_test_multiseed.csv"
    multiseed_df.to_csv(multiseed_csv_path, index=False)
    logger.info(f"Saved Final Test Multi-Seed Table: {multiseed_csv_path}")

    delays_df = pd.DataFrame(delays_records)
    delays_csv_path = METRICS_DIR / "final_detection_delays.csv"
    delays_df.to_csv(delays_csv_path, index=False)
    logger.info(f"Saved Detection Delays Table: {delays_csv_path}")

    # Compute 3-Seed Aggregate Statistics
    gru_f1s = multiseed_df["event_f1"].values
    gru_rec = multiseed_df["event_recall"].values
    gru_prec = multiseed_df["event_precision"].values
    gru_fa_rates = multiseed_df["normal_fa_events_per_hour"].values
    gru_fp_secs = multiseed_df["false_positive_alarm_seconds_per_hour"].values
    gru_point_fprs = multiseed_df["point_fpr"].values
    gru_delays_mean = multiseed_df["delay_mean_sec"].values
    gru_delays_med = multiseed_df["delay_median_sec"].values

    logger.info("=================================================================")
    logger.info("PRIMARY GRU THREE-SEED STABILITY AGGREGATE (Seeds 42, 43, 44):")
    logger.info(f"Event F1:                 {np.mean(gru_f1s):.4f} +/- {np.std(gru_f1s):.4f} (range: {np.min(gru_f1s):.4f} - {np.max(gru_f1s):.4f})")
    logger.info(f"Event Recall:             {np.mean(gru_rec):.4f} +/- {np.std(gru_rec):.4f} (range: {np.min(gru_rec):.4f} - {np.max(gru_rec):.4f})")
    logger.info(f"Event Precision:          {np.mean(gru_prec):.4f} +/- {np.std(gru_prec):.4f} (range: {np.min(gru_prec):.4f} - {np.max(gru_prec):.4f})")
    logger.info(f"Normal FA Events / Hour:  {np.mean(gru_fa_rates):.4f} +/- {np.std(gru_fa_rates):.4f} ev/h")
    logger.info(f"FP Alarm Seconds / Hour:  {np.mean(gru_fp_secs):.4f} +/- {np.std(gru_fp_secs):.4f} s/h")
    logger.info(f"Point FPR:                {np.mean(gru_point_fprs):.6f} +/- {np.std(gru_point_fprs):.6f}")
    logger.info(f"Mean Delay:               {np.mean(gru_delays_mean):.2f} +/- {np.std(gru_delays_mean):.2f}s")
    logger.info(f"Median Delay:             {np.mean(gru_delays_med):.2f} +/- {np.std(gru_delays_med):.2f}s")
    logger.info("=================================================================")

    # 4. Generate Publication Figures
    logger.info("Rendering Publication Quality Final Test Figures...")

    # Figure 1: Model Comparison (F1, Precision, Recall, FA Rate, Median Delay)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    models_to_plot = ["persistence", "ridge", "gru_seed42", "gru_seed43", "gru_seed44"]
    bar_labels = ["Persistence", "Ridge", "GRU s42\n(Canonical)", "GRU s43", "GRU s44"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#8c564b", "#9467bd"]

    # Plot 1: F1, Precision, Recall
    x = np.arange(len(models_to_plot))
    width = 0.25
    f1_vals = [summary_df[summary_df["model_id"] == m]["event_f1"].values[0] for m in models_to_plot]
    rec_vals = [summary_df[summary_df["model_id"] == m]["event_recall"].values[0] for m in models_to_plot]
    prec_vals = [summary_df[summary_df["model_id"] == m]["event_precision"].values[0] for m in models_to_plot]

    axes[0].bar(x - width, prec_vals, width, label="Precision", color="#17becf")
    axes[0].bar(x, rec_vals, width, label="Recall", color="#ff7f0e")
    axes[0].bar(x + width, f1_vals, width, label="F1 Score", color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bar_labels, fontsize=9)
    axes[0].set_ylabel("Score (0.0 - 1.0)")
    axes[0].set_title("(A) Event Precision, Recall & F1", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].legend(loc="upper left")
    axes[0].set_ylim(0, 1.05)

    # Plot 2: Normal False Alarm Rate (events / hour)
    fa_vals = [summary_df[summary_df["model_id"] == m]["normal_fa_events_per_hour"].values[0] for m in models_to_plot]
    bars2 = axes[1].bar(x, fa_vals, width=0.5, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bar_labels, fontsize=9)
    axes[1].set_ylabel("Normal False Alarm Events / Hour")
    axes[1].set_title("(B) Operational False-Alarm Rate (Normal Hours)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar, val in zip(bars2, fa_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Plot 3: Median Detection Delay
    med_delays = [summary_df[summary_df["model_id"] == m]["delay_median_sec"].values[0] for m in models_to_plot]
    bars3 = axes[2].bar(x, med_delays, width=0.5, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(bar_labels, fontsize=9)
    axes[2].set_ylabel("Median Detection Delay (seconds)")
    axes[2].set_title("(C) Detection Delay (Median)", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar, val in zip(bars3, med_delays):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}s", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "fig_01_final_model_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig1_path}")

    # Figure 2: Test Attack Timeline (Canonical Seed 42)
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), dpi=300, sharey=True)
    thresh_42 = thresholds["gru_seed42"]

    for idx, sname in enumerate(test_sessions):
        ax = axes[idx]
        ts_plot = session_data_dict[sname]["target_ts"]
        scores_plot = model_session_scores["gru_seed42"][sname]
        attacks_s = [a for a in all_attack_events if a["session_name"] == sname]

        # Convert to datetime for plotting
        dts = pd.to_datetime(ts_plot)
        ax.plot(dts, scores_plot, color="#1f77b4", linewidth=0.6, label="GRU Seed 42 Residual Score", alpha=0.8)
        ax.axhline(thresh_42, color="crimson", linestyle="--", linewidth=1.2, label=f"Frozen Q99.9 Threshold ({thresh_42:.3f})")

        # Shading attacks
        first_atk = True
        for a in attacks_s:
            lbl = "Ground-Truth Attack Interval" if (first_atk and idx == 0) else None
            ax.axvspan(pd.to_datetime(a["orig_start"]), pd.to_datetime(a["orig_end"]), color="salmon", alpha=0.45, label=lbl)
            first_atk = False

        ax.set_title(f"{sname.upper()} Timeline ({len(attacks_s)} Ground-Truth Attacks)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Residual RMSE")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "fig_02_test_attack_timeline_seed42.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig2_path}")

    # Figure 3: Detection Delay Distribution
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    delay_data = [attack_catalog_df[f"{m}_delay_seconds"].dropna().values for m in models_to_plot]
    bp = ax.boxplot(delay_data, patch_artist=True, tick_labels=bar_labels, showmeans=True, meanline=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set(color="black", linewidth=1.5)
    for mean in bp["means"]:
        mean.set(color="blue", linestyle="--", linewidth=1.5)

    ax.set_ylabel("Detection Delay (seconds)")
    ax.set_title("Distribution of Detection Delays for Detected Attacks (DT049 Official Test)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "fig_03_detection_delay_distribution.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig3_path}")

    # Figure 4: Attack-by-Attack Detection Matrix (50 attacks x 5 models)
    fig, ax = plt.subplots(figsize=(10, 14), dpi=300)
    det_matrix = np.zeros((50, len(models_to_plot)), dtype=int)
    for row_idx, r in attack_catalog_df.iterrows():
        for col_idx, m in enumerate(models_to_plot):
            det_matrix[row_idx, col_idx] = 1 if r[f"{m}_detected"] else 0

    im = ax.imshow(det_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(models_to_plot)))
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_yticks(np.arange(50))
    y_labels = [f"{r['attack_event_id']} ({r['duration_seconds']}s, {r['process_set']})" for _, r in attack_catalog_df.iterrows()]
    ax.set_yticklabels(y_labels, fontsize=6.5)

    # Annotate detection delay on green cells
    for row_idx, r in attack_catalog_df.iterrows():
        for col_idx, m in enumerate(models_to_plot):
            is_det = r[f"{m}_detected"]
            delay = r[f"{m}_delay_seconds"]
            if is_det:
                txt = f"{int(delay)}s" if not np.isnan(delay) else "OK"
                ax.text(col_idx, row_idx, txt, ha="center", va="center", color="black", fontsize=6, fontweight="bold")
            else:
                ax.text(col_idx, row_idx, "MISS", ha="center", va="center", color="white", fontsize=5.5)

    ax.set_title("Attack Detection Matrix (50 HAI 21.03 Test Attacks Across Detectors)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "fig_04_attack_detection_matrix.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig4_path}")

    # Figure 5: Normal False Alarm Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    # Subplot A: Per-file False Alarm Rate
    x_sess = np.arange(len(test_sessions))
    w = 0.16
    for idx_m, m in enumerate(models_to_plot):
        sub_fa = per_file_df[per_file_df["model_id"] == m]["normal_fa_events_per_hour"].values
        axes[0].bar(x_sess + (idx_m - 2) * w, sub_fa, w, label=bar_labels[idx_m].replace("\n", " "), color=colors[idx_m])
    axes[0].set_xticks(x_sess)
    axes[0].set_xticklabels([s.upper() for s in test_sessions])
    axes[0].set_ylabel("Normal False Alarm Events / Hour")
    axes[0].set_title("(A) Operational False Alarm Rate by Session", fontsize=10, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].legend(fontsize=7.5, loc="upper right")

    # Subplot B: Duration Distribution of Normal Alarm Events (Canonical GRU seed 42)
    s42_norm_alarms = []
    for s in test_sessions:
        is_a = (model_session_scores["gru_seed42"][s] > thresholds["gru_seed42"])
        target_a = session_data_dict[s]["target_atk"]
        alms = extract_normal_false_alarm_events(s, is_a, target_a, session_data_dict[s]["target_ts"])
        s42_norm_alarms.extend([e["duration_seconds"] for e in alms])

    dur_counts = pd.Series(s42_norm_alarms).value_counts().sort_index()
    axes[1].bar(dur_counts.index[:10], dur_counts.values[:10], color="#2ca02c", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Duration (seconds)")
    axes[1].set_ylabel("Count of Normal False Alarm Events")
    axes[1].set_title(f"(B) GRU Seed 42 Normal False Alarm Duration Profile (Total {len(s42_norm_alarms):,} events)", fontsize=10, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1].set_xticks(np.arange(1, 11))
    for x_val, y_val in zip(dur_counts.index[:10], dur_counts.values[:10]):
        axes[1].text(x_val, y_val + 2, str(y_val), ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    fig5_path = FIGURES_DIR / "fig_05_normal_false_alarm_analysis.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig5_path}")

    # Figure 6: Representative Detected Attacks Detail
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300)
    rep_detected_examples = [
        ("test1", "test1_atk_01", 2111, 2302),
        ("test2", "test2_atk_02", 9480, 9901),
        ("test3", "test3_atk_01", 49866, 49997),
        ("test5", "test5_atk_01", 1560, 1731),
    ]

    for idx, (sess, aid, s_idx, e_idx) in enumerate(rep_detected_examples):
        r_idx = idx // 2
        c_idx = idx % 2
        ax = axes[r_idx, c_idx]

        # Extract time slice around attack (+/- 120s buffer)
        buf = 120
        c_s = max(30, s_idx - buf)
        c_e = min(session_data_dict[sess]["total_rows"] - 1, e_idx + buf)

        sl_ts = pd.to_datetime(session_data_dict[sess]["target_ts"][c_s - 30 : c_e - 30 + 1])
        sl_score_gru = model_session_scores["gru_seed42"][sess][c_s - 30 : c_e - 30 + 1]
        sl_score_ridge = model_session_scores["ridge"][sess][c_s - 30 : c_e - 30 + 1]
        sl_score_pers = model_session_scores["persistence"][sess][c_s - 30 : c_e - 30 + 1]

        ax.plot(sl_ts, sl_score_gru, color="#2ca02c", label="GRU Seed 42", linewidth=1.2)
        ax.plot(sl_ts, sl_score_ridge, color="#1f77b4", label="Ridge", linewidth=0.9, alpha=0.7)
        ax.plot(sl_ts, sl_score_pers, color="#7f7f7f", label="Persistence", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axhline(thresholds["gru_seed42"], color="crimson", linestyle="--", linewidth=1.0, label="GRU Threshold")

        atk_start_t = pd.to_datetime(session_data_dict[sess]["target_ts"][s_idx - 30])
        atk_end_t = pd.to_datetime(session_data_dict[sess]["target_ts"][e_idx - 30])
        ax.axvspan(atk_start_t, atk_end_t, color="salmon", alpha=0.35, label="Attack Interval")

        delay_val = attack_catalog_df[attack_catalog_df["attack_event_id"] == aid]["gru_seed42_delay_seconds"].values[0]
        ax.set_title(f"Detected Attack: {aid} (Duration: {e_idx - s_idx + 1}s | GRU Delay: {delay_val:.0f}s)", fontsize=9, fontweight="bold")
        ax.set_ylabel("Residual Score")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig6_path = FIGURES_DIR / "fig_06_representative_detected_attacks.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig6_path}")

    # Figure 7: Representative Missed Attacks Detail (if missed attacks exist)
    missed_attacks_seed42 = attack_catalog_df[attack_catalog_df["gru_seed42_detected"] == False]
    logger.info(f"Number of attacks missed by canonical GRU seed 42: {len(missed_attacks_seed42)}")

    if len(missed_attacks_seed42) > 0:
        fig, axes = plt.subplots(min(4, len(missed_attacks_seed42)), 1, figsize=(14, 2.5 * min(4, len(missed_attacks_seed42))), dpi=300)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for idx, (_, r) in enumerate(missed_attacks_seed42.head(4).iterrows()):
            ax = axes[idx]
            sess = r["session"]
            aid = r["attack_event_id"]
            atk_obj = next(a for a in all_attack_events if a["event_id"] == aid)
            s_idx = atk_obj["orig_start_idx"]
            e_idx = atk_obj["orig_end_idx"]

            buf = 120
            c_s = max(30, s_idx - buf)
            c_e = min(session_data_dict[sess]["total_rows"] - 1, e_idx + buf)

            sl_ts = pd.to_datetime(session_data_dict[sess]["target_ts"][c_s - 30 : c_e - 30 + 1])
            sl_score_gru = model_session_scores["gru_seed42"][sess][c_s - 30 : c_e - 30 + 1]

            ax.plot(sl_ts, sl_score_gru, color="#2ca02c", label="GRU Seed 42 Score", linewidth=1.2)
            ax.axhline(thresholds["gru_seed42"], color="crimson", linestyle="--", linewidth=1.0, label="Frozen Q99.9 Threshold")

            atk_start_t = pd.to_datetime(session_data_dict[sess]["target_ts"][s_idx - 30])
            atk_end_t = pd.to_datetime(session_data_dict[sess]["target_ts"][e_idx - 30])
            ax.axvspan(atk_start_t, atk_end_t, color="salmon", alpha=0.4, label="Missed Attack Interval")

            ax.set_title(f"Missed Attack: {aid} ({sess.upper()}, Dur: {r['duration_seconds']}s, Process: {r['process_set']})", fontsize=9, fontweight="bold")
            ax.set_ylabel("Residual Score")
            ax.grid(True, linestyle=":", alpha=0.6)
            if idx == 0:
                ax.legend(fontsize=7, loc="upper right")

        plt.tight_layout()
        fig7_path = FIGURES_DIR / "fig_07_representative_missed_attacks.png"
        plt.savefig(fig7_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {fig7_path}")
    else:
        logger.info("Zero missed attacks for canonical GRU seed 42! Figure 7 skipped.")

    # 5. Sensitivity Analysis on Frozen Q99.5 and Q99.99
    logger.info("Executing Predetermined Sensitivity Analysis on Frozen Q99.5 and Q99.99...")
    sensitivity_records = []
    quantiles_to_test = ["Q99.5", "Q99.9", "Q99.99"]

    # Sensitivity thresholds from manifest
    sens_thresh_dict = {
        "persistence": {
            "Q99.5": float(manifest["frozen_models"]["persistence_baseline_0"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": thresholds["persistence"],
            "Q99.99": float(manifest["frozen_models"]["persistence_baseline_0"]["frozen_thresholds"]["Q99.99"]),
        },
        "ridge": {
            "Q99.5": float(manifest["frozen_models"]["ridge_baseline_1"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": thresholds["ridge"],
            "Q99.99": float(manifest["frozen_models"]["ridge_baseline_1"]["frozen_thresholds"]["Q99.99"]),
        },
        "gru_seed42": {
            "Q99.5": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": thresholds["gru_seed42"],
            "Q99.99": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["frozen_thresholds"]["Q99.99"]),
        },
        "gru_seed43": {
            "Q99.5": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": thresholds["gru_seed43"],
            "Q99.99": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["frozen_thresholds"]["Q99.99"]),
        },
        "gru_seed44": {
            "Q99.5": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": thresholds["gru_seed44"],
            "Q99.99": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["frozen_thresholds"]["Q99.99"]),
        },
    }

    for mid in model_ids:
        for q in quantiles_to_test:
            q_thresh = sens_thresh_dict[mid][q]
            tp_total = 0
            fn_total = 0
            fp_ev_total = 0
            norm_fa_events_total = 0
            norm_alarm_samples_total = 0

            for s in test_sessions:
                sc = model_session_scores[mid][s]
                tgt_atk = session_data_dict[s]["target_atk"]
                tgt_ts = session_data_dict[s]["target_ts"]
                ev_atks = session_data_dict[s]["evaluable_attacks"]
                norm_m = (tgt_atk == 0)

                is_alm = (sc > q_thresh)
                pred_alms = extract_predicted_alarm_events(s, mid, sc, q_thresh, tgt_ts)
                norm_fa_evs = extract_normal_false_alarm_events(s, is_alm, tgt_atk, tgt_ts)

                match_res = match_events_one_to_one(ev_atks, pred_alms, is_alm, tgt_ts)
                tp_total += match_res["tp_count"]
                fn_total += match_res["fn_count"]
                fp_ev_total += match_res["fp_count"]
                norm_fa_events_total += len(norm_fa_evs)
                norm_alarm_samples_total += int(np.sum(is_alm & norm_m))

            prec = tp_total / (tp_total + fp_ev_total) if (tp_total + fp_ev_total) > 0 else 0.0
            rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            norm_fa_rate = norm_fa_events_total / total_normal_hours

            sensitivity_records.append({
                "model_id": mid,
                "model_name": model_labels[mid],
                "operating_point": q,
                "threshold_value": round(q_thresh, 6),
                "is_primary": (q == "Q99.9"),
                "tp_detected": tp_total,
                "fn_missed": fn_total,
                "event_level_fp": fp_ev_total,
                "event_precision": round(prec, 6),
                "event_recall": round(rec, 6),
                "event_f1": round(f1, 6),
                "normal_fa_events_per_hour": round(norm_fa_rate, 4),
            })

    sens_df = pd.DataFrame(sensitivity_records)
    sens_csv_path = METRICS_DIR / "final_test_sensitivity.csv"
    sens_df.to_csv(sens_csv_path, index=False)
    logger.info(f"Saved Sensitivity Analysis Table: {sens_csv_path}")

    # 6. Save Execution Audit JSON
    elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    peak_rss = max(peak_rss, get_memory_mb())

    audit_payload = {
        "execution_name": "final_test_evaluation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "development_freeze_commit": freeze_commit,
        "protocol_commit": protocol_commit,
        "git_tag": git_tag,
        "software_environment": manifest["software_environment"],
        "scaler_sha256": manifest["preprocessing_and_feature_pipeline"]["scaler_sha256"],
        "model_hashes": {
            "ridge": manifest["frozen_models"]["ridge_baseline_1"]["model_sha256"],
            "gru_seed42": manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["sha256"],
            "gru_seed43": manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["sha256"],
            "gru_seed44": manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["sha256"],
        },
        "frozen_primary_thresholds": thresholds,
        "test_source_files": [f"test{i}.csv.gz" for i in range(1, 6)],
        "common_timeline_rule": "target row index >= 30 within every official test recording independently",
        "total_test_rows": sum(session_data_dict[s]["total_rows"] for s in test_sessions),
        "total_scored_transitions": total_scored_transitions,
        "total_normal_hours": round(total_normal_hours, 4),
        "total_evaluable_attacks": total_evaluable_attacks,
        "execution_duration_seconds": round(elapsed_time, 2),
        "measured_peak_rss_mb": round(peak_rss, 2),
    }

    audit_json_path = AUDIT_DIR / "final_test_execution.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)
    logger.info(f"Saved Execution Audit Record: {audit_json_path}")

    logger.info("=================================================================")
    logger.info("OFFICIAL FINAL TEST EVALUATION COMPLETED SUCCESSFULLY")
    logger.info(f"Elapsed Time: {elapsed_time:.2f} seconds | Peak RSS: {peak_rss:.2f} MB")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
