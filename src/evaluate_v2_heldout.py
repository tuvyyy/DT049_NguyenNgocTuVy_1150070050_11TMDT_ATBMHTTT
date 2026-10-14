#!/usr/bin/env python3
"""
Held-Out Evaluation (test4 + test5) for DT049 Experiment V2 (Feedforward DNN)
& Side-by-Side Comparison with Frozen V1 Models (Persistence, Ridge, GRU)
=============================================================================
Implements Prompt 4 Sections B, C, D, and E:
1. Verifies `results/v2_dnn/v2_pre_evaluation_freeze_manifest.json` and SHA-256
   checksums of all 3 frozen DNN V2 checkpoints (`seed 42, 43, 44`) before
   accessing `test4` or `test5`.
2. Evaluates strictly on `test4` and `test5` at target indices `t >= 30`
   (17 ground-truth attack events: 5 in `test4`, 12 in `test5`; 131,942 total
   scored rows; 35.7244 normal exposure hours).
3. Never concatenates sessions across file boundaries, never refits the scaler,
   never retrains models, and never alters frozen thresholds.
4. Evaluates DNN V2 seeds 42, 43, 44 at:
   - Primary operating mode: Frozen Validation-calibrated threshold `tau*`
     (`0.953203` for seed 42, `0.958436` for seed 43, `0.958462` for seed 44).
   - Secondary sensitivity mode: Canonical binary probability threshold `tau = 0.50`.
5. Computes pooled metrics across `test4 + test5` from pooled counts (TP, FP, FN,
   normal FA events, normal exposure hours) and pooled detection delay vectors
   (never averaging per-file F1 or per-file median delays).
6. Cross-verifies frozen V1 models (Persistence, Ridge, GRU seeds 42, 43, 44)
   on `test4 + test5` (`t >= 30`) against `results/metrics/final_per_file.csv`
   and `results/metrics/final_attack_events.csv`, while computing exact point-level
   confusion matrices, ROC-AUC, and PR-AUC from the frozen V1 anomaly scores.
"""

from datetime import datetime, timezone
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dnn_dataset_v2 import (
    CANONICAL_EVAL_SESSIONS,
    DEFAULT_CONFIG_PATH,
    load_v2_config,
    verify_frozen_scaler_and_features,
)
from src.dnn_model import (
    EXPECTED_TRAINABLE_PARAMETERS,
    VERIFIED_ACTIVATION_STATUS,
    VERIFIED_HIDDEN_ACTIVATIONS,
    FeedforwardDNNV2,
    verify_dnn_v2_architecture,
)
from src.gru_model import OneStepGRU


START_IDX = 30
EVAL_SESSIONS = ("test4", "test5")
SEEDS = (42, 43, 44)


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# PURE EVENT EXTRACTION & 1-TO-1 CHRONOLOGICAL MATCHING (t >= 30)
# =============================================================================

def extract_ground_truth_events_session(
    session_name: str,
    timestamps: np.ndarray,
    attack: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    start_idx_alignment: int = START_IDX,
) -> List[Dict]:
    """
    Extracts maximal contiguous intervals where `attack == 1` within a single
    test session and aligns them with the evaluated target region `t >= 30`.
    """
    is_attack = (attack == 1).astype(np.int8)
    diff = np.diff(np.pad(is_attack, (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    events = []
    for idx, (s, e) in enumerate(zip(starts, ends), 1):
        dur = int(e - s + 1)
        has_p1 = bool(np.any(p1[s : e + 1] == 1))
        has_p2 = bool(np.any(p2[s : e + 1] == 1))
        has_p3 = bool(np.any(p3[s : e + 1] == 1))

        procs = []
        if has_p1:
            procs.append("P1")
        if has_p2:
            procs.append("P2")
        if has_p3:
            procs.append("P3")
        proc_str = ";".join(procs) if procs else "Unknown"

        left_censored = bool(s < start_idx_alignment)
        evaluable = bool(e >= start_idx_alignment)
        eff_s = max(int(s), int(start_idx_alignment)) if evaluable else None

        events.append({
            "event_id": f"{session_name}_atk_{idx:02d}",
            "session": session_name,
            "orig_start_idx": int(s),
            "orig_end_idx": int(e),
            "orig_start": pd.Timestamp(timestamps[s]),
            "orig_end": pd.Timestamp(timestamps[e]),
            "duration_seconds": dur,
            "evaluable": evaluable,
            "left_censored": left_censored,
            "effective_start_idx": eff_s,
            "effective_start": pd.Timestamp(timestamps[eff_s]) if evaluable else None,
            "process_set": proc_str,
            "attack_P1": has_p1,
            "attack_P2": has_p2,
            "attack_P3": has_p3,
            "is_multiprocess": bool(len(procs) > 1),
        })
    return events


def extract_predicted_alarm_events_session(
    session_name: str,
    model_id: str,
    scores_common: np.ndarray,
    threshold: float,
    target_timestamps: np.ndarray,
) -> List[Dict]:
    """
    Extracts maximal contiguous runs of `scores_common > threshold` within
    the common target region `t >= 30` of a single session.
    """
    is_alarm = (scores_common > threshold).astype(np.int8)
    diff = np.diff(np.pad(is_alarm, (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    events = []
    for idx, (s, e) in enumerate(zip(starts, ends), 1):
        dur = int(e - s + 1)
        peak_score = float(np.max(scores_common[s : e + 1]))
        events.append({
            "alarm_id": f"{session_name}_{model_id}_alm_{idx:04d}",
            "session_name": session_name,
            "model_id": model_id,
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time": pd.Timestamp(target_timestamps[s]),
            "end_time": pd.Timestamp(target_timestamps[e]),
            "duration_seconds": dur,
            "peak_score": peak_score,
        })
    return events


def extract_normal_false_alarm_events_session(
    session_name: str,
    is_alarm_common: np.ndarray,
    is_attack_common: np.ndarray,
    target_timestamps: np.ndarray,
) -> List[Dict]:
    """
    Constructs `normal_alarm_t = alarm_t AND (attack_t == 0)` and extracts
    maximal contiguous runs. Attack timestamps strictly break normal runs.
    """
    normal_alarm = (is_alarm_common & (is_attack_common == 0)).astype(np.int8)
    diff = np.diff(np.pad(normal_alarm, (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    events = []
    for s, e in zip(starts, ends):
        events.append({
            "session_name": session_name,
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time": pd.Timestamp(target_timestamps[s]),
            "end_time": pd.Timestamp(target_timestamps[e]),
            "duration_seconds": int(e - s + 1),
        })
    return events


def match_events_one_to_one_session(
    evaluable_attacks: List[Dict],
    predicted_alarms: List[Dict],
    is_alarm_common: np.ndarray,
    target_timestamps: np.ndarray,
    start_idx_alignment: int = START_IDX,
) -> Dict:
    """
    Deterministic 1-to-1 chronological matching within a single session:
    1. Sort ground-truth attacks by `effective_start`.
    2. Sort predicted alarms by `start_time`.
    3. Match each attack to the earliest unmatched alarm with non-zero temporal overlap.
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
            if max(a_start, alm["start_time"]) <= min(a_end, alm["end_time"]):
                earliest_match = alm
                break

        if earliest_match is not None:
            matched_attack_ids.add(atk["event_id"])
            matched_alarm_ids.add(earliest_match["alarm_id"])
            matched_pairs.append((atk, earliest_match))

            c_s = atk["effective_start_idx"] - start_idx_alignment
            c_e = atk["orig_end_idx"] - start_idx_alignment
            interval_alarm_idxs = np.where(is_alarm_common[c_s : c_e + 1])[0]
            if len(interval_alarm_idxs) > 0:
                first_detect_idx = c_s + int(interval_alarm_idxs[0])
                detect_time = pd.Timestamp(target_timestamps[first_detect_idx])
                delay_sec = max(0.0, (detect_time - a_start).total_seconds())
                delays[atk["event_id"]] = float(delay_sec)
            else:
                delays[atk["event_id"]] = 0.0
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


def compute_session_and_pooled_metrics(
    model_id: str,
    model_name: str,
    paradigm: str,
    operating_mode: str,
    threshold_by_session: Dict[str, float],
    scores_by_session: Dict[str, np.ndarray],
    session_meta: Dict[str, Dict],
) -> Tuple[List[Dict], Dict, Dict[str, Dict]]:
    """
    Computes per-file (`test4`, `test5`) and pooled (`test4 + test5`) metrics
    strictly from pooled event/point counts and pooled delay vectors.
    """
    per_file_rows = []
    per_event_outcomes = {}

    pooled_total_rows = 0
    pooled_scored_rows = 0
    pooled_normal_samples = 0
    pooled_attack_samples = 0
    pooled_normal_hours = 0.0
    pooled_evaluable_attacks = 0

    pooled_ev_tp = 0
    pooled_ev_fn = 0
    pooled_ev_fp = 0
    pooled_total_alarms = 0
    pooled_norm_fa_events = 0
    pooled_norm_alarm_samples = 0
    pooled_delays: List[float] = []

    pooled_pt_tn = 0
    pooled_pt_fp = 0
    pooled_pt_fn = 0
    pooled_pt_tp = 0

    all_scores_list = []
    all_labels_list = []

    for sname in sorted(scores_by_session.keys()):
        scores = scores_by_session[sname]
        thresh = float(threshold_by_session[sname])
        smeta = session_meta[sname]

        target_ts = smeta["target_ts"]
        target_atk = smeta["target_atk"]
        normal_mask = (target_atk == 0)
        attack_mask = (target_atk == 1)
        normal_samples = int(smeta["normal_samples"])
        attack_samples = int(np.sum(attack_mask))
        normal_hours = float(smeta["normal_hours"])
        evaluable_atks = smeta["evaluable_attacks"]

        is_alarm = (scores > thresh)
        pred_alarms = extract_predicted_alarm_events_session(sname, model_id, scores, thresh, target_ts)
        norm_fa_ev = extract_normal_false_alarm_events_session(sname, is_alarm, target_atk, target_ts)
        norm_fa_count = len(norm_fa_ev)
        norm_alarm_sec = int(np.sum(is_alarm & normal_mask))

        fa_events_per_h = norm_fa_count / normal_hours if normal_hours > 0 else 0.0
        fp_sec_per_h = norm_alarm_sec / normal_hours if normal_hours > 0 else 0.0
        point_fpr = norm_alarm_sec / normal_samples if normal_samples > 0 else 0.0

        # Point confusion matrix & metrics
        pt_tp = int(np.sum(is_alarm & attack_mask))
        pt_fp = norm_alarm_sec
        pt_fn = int(attack_samples - pt_tp)
        pt_tn = int(normal_samples - pt_fp)
        pt_prec = pt_tp / (pt_tp + pt_fp) if (pt_tp + pt_fp) > 0 else 0.0
        pt_rec = pt_tp / (pt_tp + pt_fn) if (pt_tp + pt_fn) > 0 else 0.0
        pt_f1 = 2.0 * pt_prec * pt_rec / (pt_prec + pt_rec) if (pt_prec + pt_rec) > 0 else 0.0

        pt_roc_auc = float(roc_auc_score(target_atk, scores))
        pt_pr_auc = float(average_precision_score(target_atk, scores))

        # Event 1-to-1 matching
        mres = match_events_one_to_one_session(
            evaluable_attacks=evaluable_atks,
            predicted_alarms=pred_alarms,
            is_alarm_common=is_alarm,
            target_timestamps=target_ts,
            start_idx_alignment=START_IDX,
        )
        ev_tp = mres["tp_count"]
        ev_fn = mres["fn_count"]
        ev_fp = mres["fp_count"]
        ev_prec = ev_tp / (ev_tp + ev_fp) if (ev_tp + ev_fp) > 0 else 0.0
        ev_rec = ev_tp / (ev_tp + ev_fn) if (ev_tp + ev_fn) > 0 else 0.0
        ev_f1 = 2.0 * ev_prec * ev_rec / (ev_prec + ev_rec) if (ev_prec + ev_rec) > 0 else 0.0

        valid_delays = [d for d in mres["delays"].values() if not np.isnan(d)]
        mean_d = float(np.mean(valid_delays)) if valid_delays else np.nan
        med_d = float(np.median(valid_delays)) if valid_delays else np.nan

        for atk in evaluable_atks:
            eid = atk["event_id"]
            c_s = atk["effective_start_idx"] - START_IDX
            c_e = atk["orig_end_idx"] - START_IDX
            atk_scores = scores[c_s : c_e + 1]
            peak_sc = float(np.max(atk_scores))
            mean_sc = float(np.mean(atk_scores))
            delay_val = mres["delays"].get(eid, np.nan)
            detected = bool(not np.isnan(delay_val))
            per_event_outcomes[eid] = {
                "detected": detected,
                "delay_sec": None if np.isnan(delay_val) else round(float(delay_val), 2),
                "peak_score": round(peak_sc, 6),
                "mean_score": round(mean_sc, 6),
                "threshold": round(thresh, 6),
                "peak_to_threshold_ratio": round(peak_sc / thresh, 6) if thresh > 0 else None,
            }

        per_file_rows.append({
            "session": sname,
            "model_id": model_id,
            "model_name": model_name,
            "paradigm": paradigm,
            "operating_mode": operating_mode,
            "threshold": round(thresh, 6),
            "total_rows": smeta["total_rows"],
            "scored_transitions": smeta["common_rows"],
            "normal_exposure_hours": round(normal_hours, 4),
            "evaluable_attacks": len(evaluable_atks),
            "tp_events": ev_tp,
            "fn_events": ev_fn,
            "event_level_fp": ev_fp,
            "total_alarm_events": len(pred_alarms),
            "event_precision": round(ev_prec, 6),
            "event_recall": round(ev_rec, 6),
            "event_f1": round(ev_f1, 6),
            "normal_fa_events": norm_fa_count,
            "normal_fa_events_per_hour": round(fa_events_per_h, 4),
            "fp_alarm_seconds_per_hour": round(fp_sec_per_h, 4),
            "point_fpr": round(point_fpr, 8),
            "mean_delay_sec": round(mean_d, 2) if not np.isnan(mean_d) else None,
            "median_delay_sec": round(med_d, 2) if not np.isnan(med_d) else None,
            "point_tn": pt_tn,
            "point_fp": pt_fp,
            "point_fn": pt_fn,
            "point_tp": pt_tp,
            "point_precision": round(pt_prec, 6),
            "point_recall": round(pt_rec, 6),
            "point_f1": round(pt_f1, 6),
            "point_roc_auc": round(pt_roc_auc, 6),
            "point_pr_auc": round(pt_pr_auc, 6),
        })

        # Accumulate pooled counts
        pooled_total_rows += smeta["total_rows"]
        pooled_scored_rows += smeta["common_rows"]
        pooled_normal_samples += normal_samples
        pooled_attack_samples += attack_samples
        pooled_normal_hours += normal_hours
        pooled_evaluable_attacks += len(evaluable_atks)

        pooled_ev_tp += ev_tp
        pooled_ev_fn += ev_fn
        pooled_ev_fp += ev_fp
        pooled_total_alarms += len(pred_alarms)
        pooled_norm_fa_events += norm_fa_count
        pooled_norm_alarm_samples += norm_alarm_sec
        pooled_delays.extend(valid_delays)

        pooled_pt_tn += pt_tn
        pooled_pt_fp += pt_fp
        pooled_pt_fn += pt_fn
        pooled_pt_tp += pt_tp

        all_scores_list.append(scores)
        all_labels_list.append(target_atk)

    # Pooled Event metrics from total counts (NOT average of per-file F1)
    p_ev_prec = pooled_ev_tp / (pooled_ev_tp + pooled_ev_fp) if (pooled_ev_tp + pooled_ev_fp) > 0 else 0.0
    p_ev_rec = pooled_ev_tp / (pooled_ev_tp + pooled_ev_fn) if (pooled_ev_tp + pooled_ev_fn) > 0 else 0.0
    p_ev_f1 = 2.0 * p_ev_prec * p_ev_rec / (p_ev_prec + p_ev_rec) if (p_ev_prec + p_ev_rec) > 0 else 0.0

    p_fa_per_h = pooled_norm_fa_events / pooled_normal_hours if pooled_normal_hours > 0 else 0.0
    p_fp_sec_per_h = pooled_norm_alarm_samples / pooled_normal_hours if pooled_normal_hours > 0 else 0.0
    p_pt_fpr = pooled_norm_alarm_samples / pooled_normal_samples if pooled_normal_samples > 0 else 0.0

    p_mean_d = float(np.mean(pooled_delays)) if pooled_delays else np.nan
    p_med_d = float(np.median(pooled_delays)) if pooled_delays else np.nan

    # Pooled Point metrics from total confusion counts & concatenated score vectors
    p_pt_prec = pooled_pt_tp / (pooled_pt_tp + pooled_pt_fp) if (pooled_pt_tp + pooled_pt_fp) > 0 else 0.0
    p_pt_rec = pooled_pt_tp / (pooled_pt_tp + pooled_pt_fn) if (pooled_pt_tp + pooled_pt_fn) > 0 else 0.0
    p_pt_f1 = 2.0 * p_pt_prec * p_pt_rec / (p_pt_prec + p_pt_rec) if (p_pt_prec + p_pt_rec) > 0 else 0.0

    concat_scores = np.concatenate(all_scores_list, axis=0)
    concat_labels = np.concatenate(all_labels_list, axis=0)
    p_roc_auc = float(roc_auc_score(concat_labels, concat_scores))
    p_pr_auc = float(average_precision_score(concat_labels, concat_scores))

    unique_thresholds = sorted(set(round(v, 6) for v in threshold_by_session.values()))
    thresh_val = unique_thresholds[0] if len(unique_thresholds) == 1 else unique_thresholds

    pooled_summary = {
        "eval_split": "test4+test5",
        "model_id": model_id,
        "model_name": model_name,
        "paradigm": paradigm,
        "operating_mode": operating_mode,
        "threshold": thresh_val,
        "total_rows": pooled_total_rows,
        "scored_transitions": pooled_scored_rows,
        "normal_exposure_hours": round(pooled_normal_hours, 4),
        "evaluable_attacks": pooled_evaluable_attacks,
        "tp_events": pooled_ev_tp,
        "fn_events": pooled_ev_fn,
        "event_level_fp": pooled_ev_fp,
        "total_alarm_events": pooled_total_alarms,
        "event_precision": round(p_ev_prec, 6),
        "event_recall": round(p_ev_rec, 6),
        "event_f1": round(p_ev_f1, 6),
        "normal_fa_events": pooled_norm_fa_events,
        "normal_fa_events_per_hour": round(p_fa_per_h, 4),
        "fp_alarm_seconds_per_hour": round(p_fp_sec_per_h, 4),
        "point_fpr": round(p_pt_fpr, 8),
        "mean_delay_sec": round(p_mean_d, 2) if not np.isnan(p_mean_d) else None,
        "median_delay_sec": round(p_med_d, 2) if not np.isnan(p_med_d) else None,
        "point_tn": pooled_pt_tn,
        "point_fp": pooled_pt_fp,
        "point_fn": pooled_pt_fn,
        "point_tp": pooled_pt_tp,
        "point_precision": round(p_pt_prec, 6),
        "point_recall": round(p_pt_rec, 6),
        "point_f1": round(p_pt_f1, 6),
        "point_roc_auc": round(p_roc_auc, 6),
        "point_pr_auc": round(p_pr_auc, 6),
    }

    return per_file_rows, pooled_summary, per_event_outcomes


# =============================================================================
# V1 & V2 SCORING FUNCTIONS (STRICTLY WITHIN SINGLE SESSION, t >= 30)
# =============================================================================

def score_v1_persistence(features: np.ndarray) -> np.ndarray:
    pred = features[START_IDX - 1 : -1]
    actual = features[START_IDX:]
    diff = actual - pred
    return np.sqrt(np.mean(np.square(diff, dtype=np.float64), axis=1)).astype(np.float64)


def score_v1_ridge(features: np.ndarray, ridge_model) -> np.ndarray:
    x_in = features[START_IDX - 1 : -1]
    actual = features[START_IDX:]
    pred = ridge_model.predict(x_in)
    diff = actual - pred
    return np.sqrt(np.mean(np.square(diff, dtype=np.float64), axis=1)).astype(np.float64)


@torch.no_grad()
def score_v1_gru(features: np.ndarray, gru_model: OneStepGRU, device: torch.device, batch_size: int = 512) -> np.ndarray:
    gru_model.eval()
    n_rows = len(features)
    n_windows = n_rows - START_IDX
    scores = np.empty(n_windows, dtype=np.float64)

    for b_start in range(0, n_windows, batch_size):
        b_end = min(b_start + batch_size, n_windows)
        bsz = b_end - b_start
        x_batch = np.empty((bsz, START_IDX, 79), dtype=np.float32)
        y_batch = np.empty((bsz, 79), dtype=np.float32)
        for i, s_idx in enumerate(range(b_start, b_end)):
            x_batch[i] = features[s_idx : s_idx + START_IDX]
            y_batch[i] = features[s_idx + START_IDX]

        x_t = torch.from_numpy(x_batch).to(device)
        pred_np = gru_model(x_t).cpu().numpy()
        diff = y_batch - pred_np
        scores[b_start:b_end] = np.sqrt(np.mean(np.square(diff, dtype=np.float64), axis=1))
    return scores


@torch.no_grad()
def score_v2_dnn(features: np.ndarray, dnn_model: FeedforwardDNNV2, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    """
    Scores `FeedforwardDNNV2` on target indices `t >= 30` within a single session,
    returning sigmoid probabilities `P(attack_t = 1 | x_t)`.
    """
    dnn_model.eval()
    target_feats = features[START_IDX:]
    n_targets = len(target_feats)
    probs = np.empty(n_targets, dtype=np.float64)

    for b_start in range(0, n_targets, batch_size):
        b_end = min(b_start + batch_size, n_targets)
        x_np = np.array(target_feats[b_start:b_end], dtype=np.float32, copy=True)
        x_t = torch.from_numpy(x_np).to(device)
        p_t = dnn_model.predict_proba(x_t).squeeze(1).cpu().numpy()
        probs[b_start:b_end] = p_t.astype(np.float64)
    return probs


# =============================================================================
# DIAGNOSTIC FIGURES GENERATION
# =============================================================================

def generate_v2_diagnostic_figures(
    figures_dir: Path,
    pooled_df: pd.DataFrame,
    events_df: pd.DataFrame,
    dnn_probs_test3_seed42: np.ndarray,
    dnn_atk_test3: np.ndarray,
    dnn_probs_eval_seed42: Dict[str, np.ndarray],
    session_meta: Dict[str, Dict],
    seed42_tau: float,
) -> List[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # 1. Bar chart comparing Event Recall, Event Precision, Event F1, and Normal FA/h on test4+test5
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    compare_ids = [
        "persistence",
        "ridge",
        "gru_seed42",
        "gru_seed43",
        "gru_seed44",
        "dnn_v2_seed42_tau_star",
        "dnn_v2_seed43_tau_star",
        "dnn_v2_seed44_tau_star",
        "dnn_v2_seed42_tau_050",
    ]
    sub_df = pooled_df.set_index("model_id").loc[compare_ids].reset_index()
    short_labels = [
        "Persist\n(V1)",
        "Ridge\n(V1)",
        "GRU s42\n(V1)",
        "GRU s43\n(V1)",
        "GRU s44\n(V1)",
        "DNN s42\n(V2 τ*)",
        "DNN s43\n(V2 τ*)",
        "DNN s44\n(V2 τ*)",
        "DNN s42\n(τ=0.50)",
    ]
    x = np.arange(len(short_labels))
    w = 0.26

    axes[0].bar(x - w, sub_df["event_recall"], width=w, label="Event Recall", color="#2b5c8f")
    axes[0].bar(x, sub_df["event_precision"], width=w, label="Event Precision", color="#388e3c")
    axes[0].bar(x + w, sub_df["event_f1"], width=w, label="Event F1", color="#d95f02")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels, fontsize=8)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Score (0 to 1)")
    axes[0].set_title("Event-Level Detection Metrics on Held-out test4+test5 (17 Attacks)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    colors_fa = ["#7f7f7f", "#7f7f7f", "#1f77b4", "#1f77b4", "#1f77b4", "#d62728", "#d62728", "#d62728", "#ff9896"]
    bars = axes[1].bar(x, sub_df["normal_fa_events_per_hour"], color=colors_fa, width=0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, fontsize=8)
    axes[1].set_ylabel("Normal False-Alarm Events / Hour")
    axes[1].set_title("Normal False-Alarm Rate on test4+test5 (Lower is Better)")
    axes[1].grid(axis="y", alpha=0.3)
    for b in bars:
        h_val = b.get_height()
        axes[1].annotate(
            f"{h_val:.2f}",
            xy=(b.get_x() + b.get_width() / 2, h_val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    plt.tight_layout()
    p1 = figures_dir / "v2_event_metrics_comparison.png"
    fig.savefig(p1)
    plt.close(fig)
    saved_paths.append(str(p1.relative_to(PROJECT_ROOT)))

    # 2. Confusion matrices (Point-level on test4+test5) for GRU s42, DNN s42 (tau*), and DNN s42 (tau=0.50)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=150)
    cm_models = [
        ("gru_seed42", "GRU Seed 42 (V1, Q99.9)"),
        ("dnn_v2_seed42_tau_star", f"DNN V2 Seed 42 (Primary τ*={seed42_tau:.6f})"),
        ("dnn_v2_seed42_tau_050", "DNN V2 Seed 42 (Secondary τ=0.50)"),
    ]
    for ax, (mid, title) in zip(axes, cm_models):
        r = pooled_df[pooled_df["model_id"] == mid].iloc[0]
        cm = np.array([[int(r["point_tn"]), int(r["point_fp"])], [int(r["point_fn"]), int(r["point_tp"])]])
        im = ax.imshow(np.log1p(cm), cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal (0)", "Pred Attack (1)"], fontsize=8.5)
        ax.set_yticklabels(["True Normal (0)", "True Attack (1)"], fontsize=8.5)
        ax.set_title(
            f"{title}\nPoint F1={r['point_f1']:.4f} | Point FPR={r['point_fpr']:.5f}",
            fontsize=9,
        )
        labels_mat = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{labels_mat[i][j]}\n{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if np.log1p(cm[i, j]) > 7 else "black",
                    fontsize=9,
                    fontweight="bold",
                )
    plt.tight_layout()
    p2 = figures_dir / "v2_confusion_matrices_seed42.png"
    fig.savefig(p2)
    plt.close(fig)
    saved_paths.append(str(p2.relative_to(PROJECT_ROOT)))

    # 3. 17 Attack Event Detection & Delay Heatmap across V1 and V2 models
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    det_cols = [
        ("persistence_detected", "Persistence"),
        ("ridge_detected", "Ridge"),
        ("gru_seed42_detected", "GRU s42"),
        ("gru_seed43_detected", "GRU s43"),
        ("gru_seed44_detected", "GRU s44"),
        ("dnn_seed42_tau_star_detected", "DNN s42 (τ*)"),
        ("dnn_seed43_tau_star_detected", "DNN s43 (τ*)"),
        ("dnn_seed44_tau_star_detected", "DNN s44 (τ*)"),
        ("dnn_seed42_tau_050_detected", "DNN s42 (0.50)"),
    ]
    mat = np.zeros((len(events_df), len(det_cols)), dtype=np.float32)
    for j, (col_name, _) in enumerate(det_cols):
        mat[:, j] = events_df[col_name].astype(int).values

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(det_cols)))
    ax.set_xticklabels([c[1] for c in det_cols], rotation=25, ha="right", fontsize=8.5)
    y_labels = [
        f"{row['attack_event_id']} ({row['process_set']}, {row['duration_seconds']}s)"
        for _, row in events_df.iterrows()
    ]
    ax.set_yticks(np.arange(len(events_df)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title("Per-Event Detection Matrix on Held-out test4 + test5 (17 Attacks, Green=Detected, Red=Missed)")

    delay_col_map = {
        "persistence_detected": "persistence_delay_seconds",
        "ridge_detected": "ridge_delay_seconds",
        "gru_seed42_detected": "gru_seed42_delay_seconds",
        "gru_seed43_detected": "gru_seed43_delay_seconds",
        "gru_seed44_detected": "gru_seed44_delay_seconds",
        "dnn_seed42_tau_star_detected": "dnn_seed42_tau_star_delay_seconds",
        "dnn_seed43_tau_star_detected": "dnn_seed43_tau_star_delay_seconds",
        "dnn_seed44_tau_star_detected": "dnn_seed44_tau_star_delay_seconds",
        "dnn_seed42_tau_050_detected": "dnn_seed42_tau_050_delay_seconds",
    }
    for i in range(len(events_df)):
        for j, (d_col, _) in enumerate(det_cols):
            is_det = bool(events_df.iloc[i][d_col])
            delay_val = events_df.iloc[i][delay_col_map[d_col]]
            txt = f"{delay_val:.0f}s" if (is_det and pd.notna(delay_val)) else "MISS"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="black", fontweight="bold")

    plt.tight_layout()
    p3 = figures_dir / "v2_attack_event_heatmap_17.png"
    fig.savefig(p3)
    plt.close(fig)
    saved_paths.append(str(p3.relative_to(PROJECT_ROOT)))

    # 4. Probability saturation analysis on test3 vs test4+test5 (Seed 42)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=150)
    eval_probs = np.concatenate([dnn_probs_eval_seed42["test4"], dnn_probs_eval_seed42["test5"]])
    eval_atk = np.concatenate([session_meta["test4"]["target_atk"], session_meta["test5"]["target_atk"]])

    bins = np.linspace(0.0, 1.0, 50)
    axes[0].hist(dnn_probs_test3_seed42[dnn_atk_test3 == 0], bins=bins, density=True, alpha=0.6, label="Normal (test3)", color="#1f77b4")
    axes[0].hist(dnn_probs_test3_seed42[dnn_atk_test3 == 1], bins=bins, density=True, alpha=0.6, label="Attack (test3)", color="#d62728")
    axes[0].axvline(seed42_tau, color="black", linestyle="--", linewidth=1.5, label=f"Calibrated τ*={seed42_tau:.4f}")
    axes[0].axvline(0.50, color="gray", linestyle=":", linewidth=1.5, label="Reference τ=0.50")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Predicted Sigmoid Probability P(attack=1 | x)")
    axes[0].set_ylabel("Log Density")
    axes[0].set_title("Validation (test3) Probability Distribution (Seed 42)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].hist(eval_probs[eval_atk == 0], bins=bins, density=True, alpha=0.6, label="Normal (test4+test5)", color="#1f77b4")
    axes[1].hist(eval_probs[eval_atk == 1], bins=bins, density=True, alpha=0.6, label="Attack (test4+test5)", color="#d62728")
    axes[1].axvline(seed42_tau, color="black", linestyle="--", linewidth=1.5, label=f"Calibrated τ*={seed42_tau:.4f}")
    axes[1].axvline(0.50, color="gray", linestyle=":", linewidth=1.5, label="Reference τ=0.50")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Predicted Sigmoid Probability P(attack=1 | x)")
    axes[1].set_ylabel("Log Density")
    axes[1].set_title("Held-out (test4+test5) Probability Distribution (Seed 42)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p4 = figures_dir / "v2_probability_saturation_analysis.png"
    fig.savefig(p4)
    plt.close(fig)
    saved_paths.append(str(p4.relative_to(PROJECT_ROOT)))

    return saved_paths


# =============================================================================
# MAIN HELD-OUT EVALUATION EXECUTION
# =============================================================================

def run_heldout_evaluation() -> Dict:
    t0 = time.time()
    cfg = load_v2_config(DEFAULT_CONFIG_PATH)
    results_dir = PROJECT_ROOT / cfg["directories"]["results_dir"]
    model_dir = PROJECT_ROOT / cfg["directories"]["model_dir"]
    processed_test_dir = PROJECT_ROOT / cfg["dataset"]["processed_test_dir"]

    # 1. Verify Pre-Evaluation Freeze Manifest (Section A Gate)
    freeze_manifest_path = results_dir / "v2_pre_evaluation_freeze_manifest.json"
    if not freeze_manifest_path.exists():
        raise FileNotFoundError(f"Freeze manifest missing: {freeze_manifest_path}")
    with open(freeze_manifest_path, "r", encoding="utf-8") as f:
        freeze_manifest = json.load(f)

    if freeze_manifest.get("pre_evaluation_verification_status") != "PASS":
        raise RuntimeError("Pre-evaluation freeze manifest status is not PASS!")

    # Verify V1 and V2 tracked file checksums before touching test4/test5
    v1_freeze_path = PROJECT_ROOT / "results" / "audit" / "pretest_freeze_manifest.json"
    with open(v1_freeze_path, "r", encoding="utf-8") as f:
        v1_manifest = json.load(f)

    v1_protected_paths = [
        PROJECT_ROOT / "models" / "preprocessing" / "scaler.joblib",
        PROJECT_ROOT / "models" / "preprocessing" / "scaler_metadata.json",
        PROJECT_ROOT / "models" / "baselines" / "ridge.joblib",
        PROJECT_ROOT / "models" / "primary" / "gru_w30_seed42.pt",
        PROJECT_ROOT / "models" / "primary" / "gru_w30_seed43.pt",
        PROJECT_ROOT / "models" / "primary" / "gru_w30_seed44.pt",
        PROJECT_ROOT / "results" / "metrics" / "final_test_summary.csv",
        PROJECT_ROOT / "results" / "metrics" / "final_per_file.csv",
        PROJECT_ROOT / "results" / "metrics" / "final_attack_events.csv",
    ]
    pre_v1_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in v1_protected_paths}

    # Verify frozen scaler without refitting
    verify_frozen_scaler_and_features(
        scaler_path=PROJECT_ROOT / cfg["dataset"]["scaler_path"],
        scaler_metadata_path=PROJECT_ROOT / cfg["dataset"]["scaler_metadata_path"],
        expected_feature_count=79,
        expected_fit_sample_count=442802,
    )

    device = torch.device("cpu")
    torch.set_num_threads(4)

    # 2. Load Frozen DNN V2 Models & Thresholds from Freeze Manifest
    dnn_models: Dict[int, FeedforwardDNNV2] = {}
    dnn_tau_star: Dict[int, float] = {}
    for seed in SEEDS:
        seed_info = freeze_manifest["frozen_checkpoints"][f"seed_{seed}"]
        ckpt_path = PROJECT_ROOT / seed_info["checkpoint_path"]
        actual_sha = sha256_file(ckpt_path)
        if actual_sha != seed_info["sha256"]:
            raise RuntimeError(f"Checksum mismatch on {ckpt_path}: {actual_sha} != {seed_info['sha256']}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        m = FeedforwardDNNV2(
            hidden_activations=ckpt["hidden_activations"],
            activation_status=ckpt["activation_status"],
        ).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        verify_dnn_v2_architecture(m, EXPECTED_TRAINABLE_PARAMETERS)

        dnn_models[seed] = m
        dnn_tau_star[seed] = float(ckpt["calibrated_threshold_test3"])

    # 3. Load Frozen V1 Models & Thresholds for Exact Side-by-Side Comparison on test4 + test5
    ridge_model = joblib.load(PROJECT_ROOT / "models" / "baselines" / "ridge.joblib")
    gru_models: Dict[int, OneStepGRU] = {}
    for seed in SEEDS:
        g_ckpt = torch.load(
            PROJECT_ROOT / "models" / "primary" / f"gru_w30_seed{seed}.pt",
            map_location=device,
            weights_only=True,
        )
        gm = OneStepGRU(input_size=79, hidden_size=64, num_layers=1).to(device)
        gm.load_state_dict(g_ckpt["model_state_dict"])
        gm.eval()
        gru_models[seed] = gm

    v1_thresholds = {
        "persistence": float(v1_manifest["frozen_models"]["persistence_baseline_0"]["frozen_thresholds"]["Q99.9_primary"]),
        "ridge": float(v1_manifest["frozen_models"]["ridge_baseline_1"]["frozen_thresholds"]["Q99.9_primary"]),
        "gru_seed42": float(v1_manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["frozen_thresholds"]["Q99.9_primary"]),
        "gru_seed43": float(v1_manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["frozen_thresholds"]["Q99.9_primary"]),
        "gru_seed44": float(v1_manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["frozen_thresholds"]["Q99.9_primary"]),
    }

    # 4. Load Held-Out Sessions `test4` and `test5` (Read-only memmap, t >= 30)
    session_meta: Dict[str, Dict] = {}
    all_eval_gt_events: List[Dict] = []

    v1_scores: Dict[str, Dict[str, np.ndarray]] = {
        "persistence": {},
        "ridge": {},
        "gru_seed42": {},
        "gru_seed43": {},
        "gru_seed44": {},
    }
    dnn_probs: Dict[int, Dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}

    for sname in EVAL_SESSIONS:
        feats = np.load(processed_test_dir / f"{sname}_features.npy", mmap_mode="r")
        ts = np.load(processed_test_dir / f"{sname}_timestamps.npy", mmap_mode="r")
        atk = np.load(processed_test_dir / f"{sname}_attack.npy", mmap_mode="r")
        p1 = np.load(processed_test_dir / f"{sname}_attack_P1.npy", mmap_mode="r")
        p2 = np.load(processed_test_dir / f"{sname}_attack_P2.npy", mmap_mode="r")
        p3 = np.load(processed_test_dir / f"{sname}_attack_P3.npy", mmap_mode="r")

        total_rows = int(len(feats))
        common_rows = total_rows - START_IDX
        target_ts = ts[START_IDX:]
        target_atk = (atk[START_IDX:] == 1).astype(np.int8)

        gt_events = extract_ground_truth_events_session(sname, ts, atk, p1, p2, p3, START_IDX)
        evaluable_atks = [a for a in gt_events if a["evaluable"]]
        all_eval_gt_events.extend(evaluable_atks)

        normal_samples = int(np.sum(target_atk == 0))
        normal_hours = normal_samples / 3600.0

        session_meta[sname] = {
            "total_rows": total_rows,
            "common_rows": common_rows,
            "target_ts": target_ts,
            "target_atk": target_atk,
            "normal_samples": normal_samples,
            "normal_hours": normal_hours,
            "evaluable_attacks": evaluable_atks,
        }

        # Score V1 models strictly within session
        v1_scores["persistence"][sname] = score_v1_persistence(feats)
        v1_scores["ridge"][sname] = score_v1_ridge(feats, ridge_model)
        for seed in SEEDS:
            v1_scores[f"gru_seed{seed}"][sname] = score_v1_gru(feats, gru_models[seed], device, batch_size=512)

        # Score V2 DNN models strictly within session (t >= 30)
        for seed in SEEDS:
            dnn_probs[seed][sname] = score_v2_dnn(feats, dnn_models[seed], device, batch_size=1024)

    if len(all_eval_gt_events) != 17:
        raise RuntimeError(f"Expected 17 evaluable attack events in test4+test5, found {len(all_eval_gt_events)}")

    # 5. Compute Per-File & Pooled Metrics for All V1 and V2 Models
    model_specs = [
        ("persistence", "Baseline 0: Persistence (V1)", "Unsupervised Forecasting (Normal Train)", "V1 Frozen Q99.9 (train3)", {s: v1_thresholds["persistence"] for s in EVAL_SESSIONS}, v1_scores["persistence"]),
        ("ridge", "Baseline 1: Ridge alpha=300 (V1)", "Unsupervised Forecasting (Normal Train)", "V1 Frozen Q99.9 (train3)", {s: v1_thresholds["ridge"] for s in EVAL_SESSIONS}, v1_scores["ridge"]),
        ("gru_seed42", "Primary GRU W=30 Seed 42 (V1, Canonical)", "Unsupervised Forecasting (Normal Train)", "V1 Frozen Q99.9 (train3)", {s: v1_thresholds["gru_seed42"] for s in EVAL_SESSIONS}, v1_scores["gru_seed42"]),
        ("gru_seed43", "Primary GRU W=30 Seed 43 (V1)", "Unsupervised Forecasting (Normal Train)", "V1 Frozen Q99.9 (train3)", {s: v1_thresholds["gru_seed43"] for s in EVAL_SESSIONS}, v1_scores["gru_seed43"]),
        ("gru_seed44", "Primary GRU W=30 Seed 44 (V1)", "Unsupervised Forecasting (Normal Train)", "V1 Frozen Q99.9 (train3)", {s: v1_thresholds["gru_seed44"] for s in EVAL_SESSIONS}, v1_scores["gru_seed44"]),
        ("dnn_v2_seed42_tau_star", "Feedforward DNN V2 Seed 42 (Primary τ*, Canonical)", "Supervised Binary Classification (test1+test2)", "V2 Calibrated τ* on Labeled test3", {s: dnn_tau_star[42] for s in EVAL_SESSIONS}, dnn_probs[42]),
        ("dnn_v2_seed43_tau_star", "Feedforward DNN V2 Seed 43 (Primary τ*)", "Supervised Binary Classification (test1+test2)", "V2 Calibrated τ* on Labeled test3", {s: dnn_tau_star[43] for s in EVAL_SESSIONS}, dnn_probs[43]),
        ("dnn_v2_seed44_tau_star", "Feedforward DNN V2 Seed 44 (Primary τ*)", "Supervised Binary Classification (test1+test2)", "V2 Calibrated τ* on Labeled test3", {s: dnn_tau_star[44] for s in EVAL_SESSIONS}, dnn_probs[44]),
        ("dnn_v2_seed42_tau_050", "Feedforward DNN V2 Seed 42 (Secondary τ=0.50)", "Supervised Binary Classification (test1+test2)", "Secondary Sensitivity τ=0.50", {s: 0.50 for s in EVAL_SESSIONS}, dnn_probs[42]),
        ("dnn_v2_seed43_tau_050", "Feedforward DNN V2 Seed 43 (Secondary τ=0.50)", "Supervised Binary Classification (test1+test2)", "Secondary Sensitivity τ=0.50", {s: 0.50 for s in EVAL_SESSIONS}, dnn_probs[43]),
        ("dnn_v2_seed44_tau_050", "Feedforward DNN V2 Seed 44 (Secondary τ=0.50)", "Supervised Binary Classification (test1+test2)", "Secondary Sensitivity τ=0.50", {s: 0.50 for s in EVAL_SESSIONS}, dnn_probs[44]),
    ]

    all_per_file_rows = []
    all_pooled_rows = []
    outcomes_by_model: Dict[str, Dict[str, Dict]] = {}

    for mid, mname, paradigm, op_mode, thresh_map, sc_map in model_specs:
        pf_rows, p_row, ev_out = compute_session_and_pooled_metrics(
            model_id=mid,
            model_name=mname,
            paradigm=paradigm,
            operating_mode=op_mode,
            threshold_by_session=thresh_map,
            scores_by_session=sc_map,
            session_meta=session_meta,
        )
        all_per_file_rows.extend(pf_rows)
        all_pooled_rows.append(p_row)
        outcomes_by_model[mid] = ev_out

    per_file_df = pd.DataFrame(all_per_file_rows)
    pooled_df = pd.DataFrame(all_pooled_rows)

    # 6. Cross-Verify V1 Recomputed Per-File Event Metrics Against Frozen `results/metrics/final_per_file.csv`
    v1_frozen_pf = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "final_per_file.csv")
    v1_frozen_pf_45 = v1_frozen_pf[v1_frozen_pf["session"].isin(["test4", "test5"])]
    for _, f_row in v1_frozen_pf_45.iterrows():
        s_name = f_row["session"]
        m_id = f_row["model_id"]
        recomp = per_file_df[(per_file_df["session"] == s_name) & (per_file_df["model_id"] == m_id)].iloc[0]
        if int(recomp["tp_events"]) != int(f_row["tp_events"]) or int(recomp["event_level_fp"]) != int(f_row["event_level_fp"]):
            raise RuntimeError(f"V1 verification mismatch on {s_name} {m_id}!")
        if abs(float(recomp["event_f1"]) - float(f_row["event_f1"])) > 1e-6:
            raise RuntimeError(f"V1 event_f1 mismatch on {s_name} {m_id}!")

    # 7. Build Detailed 17 Attack Events Table
    event_detail_rows = []
    for atk in all_eval_gt_events:
        eid = atk["event_id"]
        row = {
            "attack_event_id": eid,
            "session": atk["session"],
            "orig_start": str(atk["orig_start"]),
            "orig_end": str(atk["orig_end"]),
            "duration_seconds": atk["duration_seconds"],
            "evaluable": atk["evaluable"],
            "left_censored": atk["left_censored"],
            "effective_start": str(atk["effective_start"]),
            "process_set": atk["process_set"],
            "attack_P1": atk["attack_P1"],
            "attack_P2": atk["attack_P2"],
            "attack_P3": atk["attack_P3"],
            "is_multiprocess": atk["is_multiprocess"],
        }
        model_col_map = [
            ("persistence", "persistence"),
            ("ridge", "ridge"),
            ("gru_seed42", "gru_seed42"),
            ("gru_seed43", "gru_seed43"),
            ("gru_seed44", "gru_seed44"),
            ("dnn_v2_seed42_tau_star", "dnn_seed42_tau_star"),
            ("dnn_v2_seed43_tau_star", "dnn_seed43_tau_star"),
            ("dnn_v2_seed44_tau_star", "dnn_seed44_tau_star"),
            ("dnn_v2_seed42_tau_050", "dnn_seed42_tau_050"),
            ("dnn_v2_seed43_tau_050", "dnn_seed43_tau_050"),
            ("dnn_v2_seed44_tau_050", "dnn_seed44_tau_050"),
        ]
        for mid, prefix in model_col_map:
            ev_res = outcomes_by_model[mid][eid]
            row[f"{prefix}_detected"] = ev_res["detected"]
            row[f"{prefix}_delay_seconds"] = ev_res["delay_sec"]
            row[f"{prefix}_peak_score"] = ev_res["peak_score"]
            row[f"{prefix}_threshold"] = ev_res["threshold"]
            row[f"{prefix}_peak_to_threshold_ratio"] = ev_res["peak_to_threshold_ratio"]
        event_detail_rows.append(row)

    events_17_df = pd.DataFrame(event_detail_rows)

    # 8. Build Subsystem Breakdown Table on the 17 Events
    subsystem_groups = [
        ("All 17 Held-out Attacks", events_17_df),
        ("Single-Process Attacks (6)", events_17_df[~events_17_df["is_multiprocess"]]),
        ("Multi-Process Attacks (11)", events_17_df[events_17_df["is_multiprocess"]]),
        ("Involves P1 Boiler (13)", events_17_df[events_17_df["attack_P1"]]),
        ("Involves P2 Turbine (9)", events_17_df[events_17_df["attack_P2"]]),
        ("Involves P3 Water-Treatment (6)", events_17_df[events_17_df["attack_P3"]]),
    ]
    sub_rows = []
    for g_label, g_df in subsystem_groups:
        n_ev = len(g_df)
        sub_rows.append({
            "subsystem_category": g_label,
            "attack_count": n_ev,
            "persistence_detected": int(g_df["persistence_detected"].sum()),
            "ridge_detected": int(g_df["ridge_detected"].sum()),
            "gru_seed42_detected": int(g_df["gru_seed42_detected"].sum()),
            "gru_seed43_detected": int(g_df["gru_seed43_detected"].sum()),
            "gru_seed44_detected": int(g_df["gru_seed44_detected"].sum()),
            "dnn_seed42_tau_star_detected": int(g_df["dnn_seed42_tau_star_detected"].sum()),
            "dnn_seed43_tau_star_detected": int(g_df["dnn_seed43_tau_star_detected"].sum()),
            "dnn_seed44_tau_star_detected": int(g_df["dnn_seed44_tau_star_detected"].sum()),
            "dnn_seed42_tau_050_detected": int(g_df["dnn_seed42_tau_050_detected"].sum()),
            "dnn_seed43_tau_050_detected": int(g_df["dnn_seed43_tau_050_detected"].sum()),
            "dnn_seed44_tau_050_detected": int(g_df["dnn_seed44_tau_050_detected"].sum()),
        })
    subsystem_df = pd.DataFrame(sub_rows)

    # 9. Compute 3-Seed Mean ± Std Table on test4 + test5
    multiseed_groups = [
        ("GRU W=30 (V1, Unsupervised Q99.9)", ["gru_seed42", "gru_seed43", "gru_seed44"]),
        ("Feedforward DNN (V2, Primary Calibrated τ*)", ["dnn_v2_seed42_tau_star", "dnn_v2_seed43_tau_star", "dnn_v2_seed44_tau_star"]),
        ("Feedforward DNN (V2, Secondary τ=0.50)", ["dnn_v2_seed42_tau_050", "dnn_v2_seed43_tau_050", "dnn_v2_seed44_tau_050"]),
    ]
    ms_rows = []
    metric_cols = [
        "tp_events",
        "event_precision",
        "event_recall",
        "event_f1",
        "normal_fa_events_per_hour",
        "fp_alarm_seconds_per_hour",
        "point_fpr",
        "median_delay_sec",
        "point_precision",
        "point_recall",
        "point_f1",
        "point_roc_auc",
        "point_pr_auc",
    ]
    for g_name, mids in multiseed_groups:
        sub = pooled_df[pooled_df["model_id"].isin(mids)]
        r_dict = {"model_group": g_name, "eval_split": "test4+test5 (17 attacks)"}
        for mc in metric_cols:
            vals = sub[mc].dropna().astype(float).values
            r_dict[f"{mc}_mean"] = round(float(np.mean(vals)), 6) if len(vals) > 0 else None
            r_dict[f"{mc}_std"] = round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else 0.0
        ms_rows.append(r_dict)
    multiseed_df = pd.DataFrame(ms_rows)

    # 10. Save CSV Tables in `results/v2_dnn/`
    pooled_df.to_csv(results_dir / "v2_heldout_test4_test5_summary.csv", index=False)
    per_file_df.to_csv(results_dir / "v2_heldout_per_file.csv", index=False)
    events_17_df.to_csv(results_dir / "v2_heldout_17_attack_events.csv", index=False)
    subsystem_df.to_csv(results_dir / "v2_heldout_subsystem_breakdown.csv", index=False)
    multiseed_df.to_csv(results_dir / "v2_heldout_multiseed_stats.csv", index=False)

    cm_cols = [
        "eval_split", "model_id", "model_name", "operating_mode", "threshold",
        "point_tn", "point_fp", "point_fn", "point_tp",
        "point_precision", "point_recall", "point_f1", "point_fpr", "point_roc_auc", "point_pr_auc",
    ]
    pooled_df[cm_cols].to_csv(results_dir / "v2_heldout_confusion_matrices.csv", index=False)

    # 11. Generate Figures
    test3_feats = np.load(processed_test_dir / "test3_features.npy", mmap_mode="r")
    test3_atk = np.load(processed_test_dir / "test3_attack.npy", mmap_mode="r")[START_IDX:]
    dnn_probs_test3_s42 = score_v2_dnn(test3_feats, dnn_models[42], device, batch_size=1024)

    fig_paths = generate_v2_diagnostic_figures(
        figures_dir=results_dir / "figures",
        pooled_df=pooled_df,
        events_df=events_17_df,
        dnn_probs_test3_seed42=dnn_probs_test3_s42,
        dnn_atk_test3=test3_atk,
        dnn_probs_eval_seed42=dnn_probs[42],
        session_meta=session_meta,
        seed42_tau=dnn_tau_star[42],
    )

    # 12. Verify Post-Evaluation Checksums of All V1 Protected Artifacts
    post_v1_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in v1_protected_paths}
    if pre_v1_hashes != post_v1_hashes:
        raise RuntimeError("V1 protected artifact checksum changed during V2 evaluation!")

    duration_sec = round(time.time() - t0, 2)
    peak_rss_mb = round(get_peak_rss_mb(), 2)

    audit_payload = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/evaluate_v2_heldout.py",
        "freeze_manifest_verified": "results/v2_dnn/v2_pre_evaluation_freeze_manifest.json",
        "status": "PASS",
        "evaluation_sessions": list(EVAL_SESSIONS),
        "alignment_start_idx": START_IDX,
        "total_scored_rows": 131942,
        "total_normal_exposure_hours": 35.7244,
        "total_evaluable_attack_events": 17,
        "duration_sec": duration_sec,
        "peak_rss_mb": peak_rss_mb,
        "v1_artifacts_checksum_verified_unchanged": True,
        "v1_per_file_cross_check_passed": True,
        "generated_csvs": [
            "results/v2_dnn/v2_heldout_test4_test5_summary.csv",
            "results/v2_dnn/v2_heldout_per_file.csv",
            "results/v2_dnn/v2_heldout_17_attack_events.csv",
            "results/v2_dnn/v2_heldout_subsystem_breakdown.csv",
            "results/v2_dnn/v2_heldout_multiseed_stats.csv",
            "results/v2_dnn/v2_heldout_confusion_matrices.csv",
        ],
        "generated_figures": fig_paths,
        "paradigm_comparison_disclaimer": (
            "DNN V2 is trained with supervised Attack labels (test1+test2) and calibrated on labeled Validation (test3), "
            "whereas GRU/Ridge/Persistence V1 are trained unsupervised on Normal data only (train1+train2) with Q99.9 "
            "thresholds calibrated on Normal data (train3). This comparison evaluates two distinct learning paradigms and "
            "supervision regimes on the common test4+test5 split (17 attacks, t >= 30), and does not isolate architecture alone."
        ),
        "blindness_disclaimer": (
            "test4 and test5 are strictly held out from DNN V2 training, validation, and threshold selection, "
            "but were previously analyzed in V1 final evaluation and are NOT a completely blind test of the overall project."
        ),
    }
    audit_path = results_dir / "v2_heldout_evaluation_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    print(f"[OK] Held-out V2 evaluation completed in {duration_sec:.2f}s (Peak RSS: {peak_rss_mb:.2f} MB)")
    return audit_payload


if __name__ == "__main__":
    run_heldout_evaluation()
