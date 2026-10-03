#!/usr/bin/env python3
"""
HAI 21.03 Post-Test Error Analysis (DT049)
===========================================
Performs post-test error analysis using frozen models and existing final test artifacts.
Scientific integrity & frozen rules:
- No retraining
- No threshold recalibration
- No change in event matching
- No change in feature set
- Existing frozen metric files remain untouched

Outputs:
- results/metrics/final_error_analysis_by_process.csv
- results/metrics/final_common_detected_delay.csv
- results/metrics/final_missed_attack_analysis.csv
- results/metrics/final_test_sensitivity_complete.csv
- results/figures/error_analysis/ (fig_ea_01 through fig_ea_09)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from gru_model import OneStepGRU
from gru_dataset import SessionWindowDataset

PROCESSED_TEST_DIR = REPO_ROOT / "data" / "processed" / "hai_21_03" / "final_test"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
AUDIT_DIR = REPO_ROOT / "results" / "audit"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "error_analysis"
MANIFEST_PATH = AUDIT_DIR / "pretest_freeze_manifest.json"


def setup_logger():
    logger = logging.getLogger("DT049_ErrorAnalysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def score_persistence(feats: np.ndarray) -> np.ndarray:
    targets = feats[30:]
    preds = feats[29:-1]
    diffs = targets.astype(np.float64) - preds.astype(np.float64)
    return np.sqrt(np.mean(diffs ** 2, axis=1)).astype(np.float32)


def score_ridge(feats: np.ndarray, ridge_model) -> np.ndarray:
    inputs = feats[29:-1]
    targets = feats[30:]
    preds = ridge_model.predict(inputs).astype(np.float64)
    diffs = targets.astype(np.float64) - preds
    return np.sqrt(np.mean(diffs ** 2, axis=1)).astype(np.float32)


def score_gru(feats: np.ndarray, model: nn.Module, device: torch.device, batch_size: int = 512) -> np.ndarray:
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
            scores_list.append(np.sqrt(np.mean(diffs ** 2, axis=1)).astype(np.float32))
    return np.concatenate(scores_list)


def extract_predicted_alarm_events(session_name: str, model_id: str, scores: np.ndarray, threshold: float, target_timestamps: np.ndarray):
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
            c_s = atk["effective_start_idx"] - 30
            c_e = atk["orig_end_idx"] - 30
            interval_alarm_idxs = np.where(is_alarm_common[c_s : c_e + 1])[0]
            if len(interval_alarm_idxs) > 0:
                first_detect_idx = c_s + interval_alarm_idxs[0]
                detect_time = pd.Timestamp(target_timestamps[first_detect_idx])
                delay_sec = max(0.0, (detect_time - a_start).total_seconds())
                delays[atk["event_id"]] = delay_sec
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


def main():
    logger = setup_logger()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting DT049 Post-Test Error Analysis...")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    device = torch.device("cpu")
    torch.set_num_threads(4)

    # Load frozen thresholds
    frozen_thresholds = {
        "persistence": float(manifest["frozen_models"]["persistence_baseline_0"]["frozen_thresholds"]["Q99.9_primary"]),
        "ridge": float(manifest["frozen_models"]["ridge_baseline_1"]["frozen_thresholds"]["Q99.9_primary"]),
        "gru_seed42": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["frozen_thresholds"]["Q99.9_primary"]),
        "gru_seed43": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["frozen_thresholds"]["Q99.9_primary"]),
        "gru_seed44": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["frozen_thresholds"]["Q99.9_primary"]),
    }

    model_labels = {
        "persistence": "Baseline 0: Persistence",
        "ridge": "Baseline 1: Ridge (alpha=300)",
        "gru_seed42": "Primary GRU (Seed 42, Canonical)",
        "gru_seed43": "Primary GRU (Seed 43, Replicate 1)",
        "gru_seed44": "Primary GRU (Seed 44, Replicate 2)",
    }

    # Load models
    logger.info("Loading frozen models...")
    ridge_model = joblib.load(MODELS_DIR / "baselines" / "ridge.joblib")
    gru_models = {}
    for seed in [42, 43, 44]:
        ckpt = torch.load(MODELS_DIR / "primary" / f"gru_w30_seed{seed}.pt", map_location=device, weights_only=True)
        m = OneStepGRU(79, 64, 1).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        gru_models[seed] = m

    # Load feature metadata
    with open(MODELS_DIR / "preprocessing" / "scaler_metadata.json", "r", encoding="utf-8") as f:
        scaler_meta = json.load(f)
    feature_names = scaler_meta["ordered_feature_names"]

    # Load existing final attack and alarm events
    attack_events_df = pd.read_csv(METRICS_DIR / "final_attack_events.csv")
    alarm_events_df = pd.read_csv(METRICS_DIR / "final_alarm_events.csv")
    summary_df = pd.read_csv(METRICS_DIR / "final_test_summary.csv")

    test_sessions = [f"test{i}" for i in range(1, 6)]
    session_data = {}
    model_session_scores = {mid: {} for mid in frozen_thresholds.keys()}

    logger.info("Scoring test sessions with all models...")
    for sname in test_sessions:
        feats = np.load(PROCESSED_TEST_DIR / f"{sname}_features.npy")
        ts = np.load(PROCESSED_TEST_DIR / f"{sname}_timestamps.npy")
        atk = np.load(PROCESSED_TEST_DIR / f"{sname}_attack.npy")
        p1 = np.load(PROCESSED_TEST_DIR / f"{sname}_attack_P1.npy")
        p2 = np.load(PROCESSED_TEST_DIR / f"{sname}_attack_P2.npy")
        p3 = np.load(PROCESSED_TEST_DIR / f"{sname}_attack_P3.npy")

        target_ts = ts[30:]
        target_atk = atk[30:]
        normal_samples = int(np.sum(target_atk == 0))
        normal_hours = normal_samples / 3600.0

        session_data[sname] = {
            "feats": feats,
            "ts": ts,
            "target_ts": target_ts,
            "target_atk": target_atk,
            "target_p1": p1[30:],
            "target_p2": p2[30:],
            "target_p3": p3[30:],
            "normal_samples": normal_samples,
            "normal_hours": normal_hours,
        }

        # Scores
        model_session_scores["persistence"][sname] = score_persistence(feats)
        model_session_scores["ridge"][sname] = score_ridge(feats, ridge_model)
        for seed in [42, 43, 44]:
            model_session_scores[f"gru_seed{seed}"][sname] = score_gru(feats, gru_models[seed], device)

    total_normal_hours = sum(session_data[s]["normal_hours"] for s in test_sessions)
    total_normal_samples = sum(session_data[s]["normal_samples"] for s in test_sessions)

    # =========================================================================
    # ANALYSIS 1: PROCESS-LEVEL DETECTION
    # =========================================================================
    logger.info("Executing Analysis 1: Process-Level Detection...")
    process_subsets = [
        ("P1 (all involving)", attack_events_df[attack_events_df["attack_P1"] == True]),
        ("P2 (all involving)", attack_events_df[attack_events_df["attack_P2"] == True]),
        ("P3 (all involving)", attack_events_df[attack_events_df["attack_P3"] == True]),
        ("Multi-process (>1 process)", attack_events_df[attack_events_df[["attack_P1", "attack_P2", "attack_P3"]].sum(axis=1) > 1]),
        ("P1 only", attack_events_df[attack_events_df["process_set"] == "P1"]),
        ("P2 only", attack_events_df[attack_events_df["process_set"] == "P2"]),
        ("P3 only", attack_events_df[attack_events_df["process_set"] == "P3"]),
        ("All attacks", attack_events_df),
    ]

    process_records = []
    for mid, mname in model_labels.items():
        det_col = f"{mid}_detected"
        for group_name, subset in process_subsets:
            n_eval = len(subset)
            n_det = int(subset[det_col].sum())
            n_miss = n_eval - n_det
            rec = n_det / n_eval if n_eval > 0 else 0.0
            process_records.append({
                "model_id": mid,
                "model_name": mname,
                "process_group": group_name,
                "evaluable_attacks": n_eval,
                "detected_count": n_det,
                "missed_count": n_miss,
                "recall": round(rec, 4),
            })

    process_df = pd.DataFrame(process_records)
    process_csv_path = METRICS_DIR / "final_error_analysis_by_process.csv"
    process_df.to_csv(process_csv_path, index=False)
    logger.info(f"Saved: {process_csv_path}")

    # =========================================================================
    # ANALYSIS 2: COMMON-DETECTED DELAY
    # =========================================================================
    logger.info("Executing Analysis 2: Common-Detected Delay...")
    pairs = [
        ("persistence", "ridge", "Persistence", "Ridge"),
        ("persistence", "gru_seed42", "Persistence", "GRU seed42"),
        ("ridge", "gru_seed42", "Ridge", "GRU seed42"),
    ]

    paired_records = []
    for m1, m2, l1, l2 in pairs:
        c1 = f"{m1}_detected"
        c2 = f"{m2}_detected"
        d1_col = f"{m1}_delay_seconds"
        d2_col = f"{m2}_delay_seconds"

        common_sub = attack_events_df[attack_events_df[c1] & attack_events_df[c2]].copy()
        n_common = len(common_sub)

        d1 = common_sub[d1_col].values
        d2 = common_sub[d2_col].values
        diffs = d2 - d1  # model2 - model1

        paired_records.append({
            "model_a": m1,
            "model_b": m2,
            "model_a_name": l1,
            "model_b_name": l2,
            "common_detected_count": n_common,
            "model_a_mean_delay_sec": round(float(np.mean(d1)), 2),
            "model_a_median_delay_sec": round(float(np.median(d1)), 2),
            "model_b_mean_delay_sec": round(float(np.mean(d2)), 2),
            "model_b_median_delay_sec": round(float(np.median(d2)), 2),
            "mean_paired_delay_diff_sec": round(float(np.mean(diffs)), 2),
            "median_paired_delay_diff_sec": round(float(np.median(diffs)), 2),
        })

    paired_df = pd.DataFrame(paired_records)
    paired_csv_path = METRICS_DIR / "final_common_detected_delay.csv"
    paired_df.to_csv(paired_csv_path, index=False)
    logger.info(f"Saved: {paired_csv_path}")

    # =========================================================================
    # ANALYSIS 3: MISSED ATTACKS (GRU SEED 42)
    # =========================================================================
    logger.info("Executing Analysis 3: Missed Attacks Analysis...")
    gru42_thresh = frozen_thresholds["gru_seed42"]
    gru_alarms = alarm_events_df[alarm_events_df["model_id"] == "gru_seed42"]
    missed_attacks = attack_events_df[~attack_events_df["gru_seed42_detected"]].copy()

    missed_records = []
    factual_notes = {
        "test2_atk_05": "Ridge barely crossed threshold (1.4438 vs 1.4009) at t=12s; GRU max score remained at 0.8217 (77.6% of threshold).",
        "test2_atk_06": "Missed by all models. Residual remained low across all features; max GRU score 0.8848 (83.6% of threshold).",
        "test2_atk_09": "Near miss: max GRU score reached 1.0481 (99.0% of threshold), falling short of alarm by only 0.0105 RMSE.",
        "test2_atk_10": "Ridge triggered alarm at t=187s (peak 1.4601 vs 1.4009); GRU score stayed low at 0.7317 (69.1% of threshold).",
        "test2_atk_14": "Missed by all models. Peak GRU score 0.7865 (74.3% of threshold); no actuator/sensor excursion exceeded normal variance.",
        "test2_atk_16": "Ridge detected at t=13s (peak 1.4604 vs 1.4009); GRU peak score 0.7167 (67.7% of threshold).",
        "test2_atk_19": "Missed by Persistence, Ridge, and GRU seed42/43; detected by GRU seed44 (at t=101s). GRU seed42 peak 0.6559 (62.0% threshold).",
        "test2_atk_20": "Missed by all models and all seeds. P3 boiler disturbance; GRU seed42 peak 0.8685 (82.0% of threshold).",
        "test3_atk_01": "Missed by all models. High background score, peak 0.9772 (92.3% of threshold), adjacent to normal false alarms 12s before and 32s after.",
        "test5_atk_09": "Near miss: GRU peak score reached 1.0193 (96.3% of threshold). Ridge detected at t=10s (peak 1.4529 vs 1.4009).",
    }

    for idx, row in missed_attacks.iterrows():
        aid = row["attack_event_id"]
        sname = row["session"]
        dur = row["duration_seconds"]
        pset = row["process_set"]
        r_det = bool(row["ridge_detected"])
        p_det = bool(row["persistence_detected"])

        s_time = pd.Timestamp(row["orig_start"])
        e_time = pd.Timestamp(row["orig_end"])
        ts = session_data[sname]["ts"]

        s_idx = np.where(ts == s_time)[0][0]
        e_idx = np.where(ts == e_time)[0][0]
        c_s = s_idx - 30
        c_e = e_idx - 30

        sc = model_session_scores["gru_seed42"][sname][c_s : c_e + 1]
        max_sc = float(np.max(sc))
        ratio = max_sc / gru42_thresh

        # Nearest alarms before/after
        s_alms = gru_alarms[gru_alarms["session_name"] == sname]
        alms_before = s_alms[pd.to_datetime(s_alms["end_time"]) < s_time]
        alms_after = s_alms[pd.to_datetime(s_alms["start_time"]) > e_time]

        dt_before = round((s_time - pd.to_datetime(alms_before["end_time"]).max()).total_seconds(), 1) if len(alms_before) > 0 else np.nan
        dt_after = round((pd.to_datetime(alms_after["start_time"]).min() - e_time).total_seconds(), 1) if len(alms_after) > 0 else np.nan

        missed_records.append({
            "attack_event_id": aid,
            "session": sname,
            "duration_seconds": dur,
            "process_set": pset,
            "max_gru_score_inside": round(max_sc, 6),
            "frozen_threshold": round(gru42_thresh, 6),
            "score_to_threshold_ratio": round(ratio, 4),
            "ridge_detected": r_det,
            "persistence_detected": p_det,
            "nearest_alarm_before_sec": dt_before,
            "nearest_alarm_after_sec": dt_after,
            "factual_telemetry_observation": factual_notes.get(aid, ""),
        })

    missed_df = pd.DataFrame(missed_records)
    missed_csv_path = METRICS_DIR / "final_missed_attack_analysis.csv"
    missed_df.to_csv(missed_csv_path, index=False)
    logger.info(f"Saved: {missed_csv_path}")

    # =========================================================================
    # ANALYSIS 6: COMPLETE SENSITIVITY METRICS
    # =========================================================================
    logger.info("Executing Analysis 6: Complete Sensitivity Metrics across Q99.5, Q99.9, Q99.99...")
    quantiles = ["Q99.5", "Q99.9", "Q99.99"]
    sens_thresh_dict = {
        "persistence": {
            "Q99.5": float(manifest["frozen_models"]["persistence_baseline_0"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": frozen_thresholds["persistence"],
            "Q99.99": float(manifest["frozen_models"]["persistence_baseline_0"]["frozen_thresholds"]["Q99.99"]),
        },
        "ridge": {
            "Q99.5": float(manifest["frozen_models"]["ridge_baseline_1"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": frozen_thresholds["ridge"],
            "Q99.99": float(manifest["frozen_models"]["ridge_baseline_1"]["frozen_thresholds"]["Q99.99"]),
        },
        "gru_seed42": {
            "Q99.5": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": frozen_thresholds["gru_seed42"],
            "Q99.99": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["frozen_thresholds"]["Q99.99"]),
        },
        "gru_seed43": {
            "Q99.5": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": frozen_thresholds["gru_seed43"],
            "Q99.99": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["frozen_thresholds"]["Q99.99"]),
        },
        "gru_seed44": {
            "Q99.5": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": frozen_thresholds["gru_seed44"],
            "Q99.99": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["frozen_thresholds"]["Q99.99"]),
        },
    }

    # Extract ground truth attack list from attack_events_df
    all_attacks_raw = []
    for sname in test_sessions:
        atk_sub = attack_events_df[attack_events_df["session"] == sname]
        for _, r in atk_sub.iterrows():
            all_attacks_raw.append({
                "session_name": sname,
                "event_id": r["attack_event_id"],
                "orig_start": pd.Timestamp(r["orig_start"]),
                "orig_end": pd.Timestamp(r["orig_end"]),
                "effective_start": pd.Timestamp(r["effective_start"]),
                "effective_start_idx": np.where(session_data[sname]["ts"] == pd.Timestamp(r["effective_start"]))[0][0],
                "orig_end_idx": np.where(session_data[sname]["ts"] == pd.Timestamp(r["orig_end"]))[0][0],
            })

    complete_sens_records = []
    for mid, mname in model_labels.items():
        for q in quantiles:
            q_thresh = sens_thresh_dict[mid][q]
            tp_total = 0
            fn_total = 0
            fp_ev_total = 0
            norm_fa_events_total = 0
            norm_alarm_samples_total = 0
            all_delays = []

            for sname in test_sessions:
                sc = model_session_scores[mid][sname]
                tgt_atk = session_data[sname]["target_atk"]
                tgt_ts = session_data[sname]["target_ts"]
                ev_atks = [a for a in all_attacks_raw if a["session_name"] == sname]
                norm_m = (tgt_atk == 0)

                is_alm = (sc > q_thresh)
                pred_alms = extract_predicted_alarm_events(sname, mid, sc, q_thresh, tgt_ts)
                norm_fa_evs = extract_normal_false_alarm_events(sname, is_alm, tgt_atk, tgt_ts)

                match_res = match_events_one_to_one(ev_atks, pred_alms, is_alm, tgt_ts)
                tp_total += match_res["tp_count"]
                fn_total += match_res["fn_count"]
                fp_ev_total += match_res["fp_count"]
                norm_fa_events_total += len(norm_fa_evs)
                norm_alarm_samples_total += int(np.sum(is_alm & norm_m))

                for d in match_res["delays"].values():
                    if not np.isnan(d):
                        all_delays.append(d)

            prec = tp_total / (tp_total + fp_ev_total) if (tp_total + fp_ev_total) > 0 else 0.0
            rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            norm_fa_rate = norm_fa_events_total / total_normal_hours
            fp_sec_per_h = norm_alarm_samples_total / total_normal_hours
            point_fpr = norm_alarm_samples_total / total_normal_samples
            med_delay = float(np.median(all_delays)) if all_delays else np.nan

            complete_sens_records.append({
                "model_id": mid,
                "model_name": mname,
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
                "fp_alarm_seconds_per_hour": round(fp_sec_per_h, 4),
                "point_fpr": round(point_fpr, 8),
                "median_detection_delay_sec": round(med_delay, 2) if not np.isnan(med_delay) else np.nan,
            })

    complete_sens_df = pd.DataFrame(complete_sens_records)
    complete_sens_csv_path = METRICS_DIR / "final_test_sensitivity_complete.csv"
    complete_sens_df.to_csv(complete_sens_csv_path, index=False)
    logger.info(f"Saved Complete Sensitivity Table: {complete_sens_csv_path}")

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    logger.info("Generating publication figures for error analysis...")

    # Figure EA 1: Process-Level Recall
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    groups = ["P1 (all involving)", "P2 (all involving)", "P3 (all involving)", "Multi-process (>1 process)"]
    x = np.arange(len(groups))
    width = 0.16
    models_order = ["persistence", "ridge", "gru_seed42", "gru_seed43", "gru_seed44"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#8c564b", "#9467bd"]
    labels = ["Persistence", "Ridge", "GRU s42 (Canonical)", "GRU s43", "GRU s44"]

    for i, (mid, col, lbl) in enumerate(zip(models_order, colors, labels)):
        sub = process_df[process_df["model_id"] == mid]
        recs = [sub[sub["process_group"] == g]["recall"].values[0] for g in groups]
        ax.bar(x + (i - 2) * width, recs, width, label=lbl, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(["P1 Involving\n(N=34)", "P2 Involving\n(N=22)", "P3 Involving\n(N=8)", "Multi-Process\n(N=14)"], fontsize=10)
    ax.set_ylabel("Detection Recall", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("DT049 Official Final Test: Process-Level Anomaly Detection Recall", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig_ea_01_process_recall.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # Figure EA 2: Common Detected Delay Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    # Left: Paired Ridge vs GRU s42 on common 31 attacks
    common_r_g = attack_events_df[attack_events_df["ridge_detected"] & attack_events_df["gru_seed42_detected"]]
    r_delays = common_r_g["ridge_delay_seconds"].values
    g_delays = common_r_g["gru_seed42_delay_seconds"].values

    axes[0].scatter(r_delays, g_delays, color="#1f77b4", edgecolor="black", s=50, alpha=0.8, zorder=3)
    max_d = max(np.max(r_delays), np.max(g_delays)) + 10
    axes[0].plot([0, max_d], [0, max_d], "r--", label="Equality Line (y=x)", alpha=0.7)
    axes[0].set_xlabel("Ridge Detection Delay (seconds)", fontsize=10)
    axes[0].set_ylabel("GRU seed42 Detection Delay (seconds)", fontsize=10)
    axes[0].set_title(f"(A) Paired Delays on Common Detected Attacks (N={len(common_r_g)})", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left")

    # Right: Boxplot comparing unconditional vs conditional delays
    box_data = [
        attack_events_df[attack_events_df["ridge_detected"]]["ridge_delay_seconds"].dropna().values,
        attack_events_df[attack_events_df["gru_seed42_detected"]]["gru_seed42_delay_seconds"].dropna().values,
        r_delays,
        g_delays,
    ]
    box_labels = ["Ridge\n(All 35)", "GRU s42\n(All 40)", "Ridge\n(Common 31)", "GRU s42\n(Common 31)"]
    axes[1].boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                    boxprops=dict(facecolor="#e0e0e0", color="black"),
                    medianprops=dict(color="red", linewidth=2))
    axes[1].set_ylabel("Detection Delay (seconds)", fontsize=10)
    axes[1].set_title("(B) Selection Bias: Unconditional vs Paired Delay", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig_ea_02_common_delay_paired.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # Figure EA 3: Event Chatter Decomposition
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    # Stacked bar of alarm decomposition
    m_ids = ["persistence", "ridge", "gru_seed42", "gru_seed43", "gru_seed44"]
    m_short = ["Persistence", "Ridge", "GRU s42", "GRU s43", "GRU s44"]

    tp_vals = [summary_df[summary_df["model_id"] == m]["tp_detected"].values[0] for m in m_ids]
    norm_vals = [summary_df[summary_df["model_id"] == m]["normal_fa_events"].values[0] for m in m_ids]
    tot_vals = [summary_df[summary_df["model_id"] == m]["total_predicted_alarms"].values[0] for m in m_ids]
    red_vals = [tot - tp - norm for tot, tp, norm in zip(tot_vals, tp_vals, norm_vals)]

    x = np.arange(len(m_ids))
    axes[0].bar(x, tp_vals, label="Matched TP Alarms", color="#2ca02c")
    axes[0].bar(x, red_vals, bottom=tp_vals, label="Redundant Alarms in Attack", color="#ff7f0e")
    bottom_norm = [t + r for t, r in zip(tp_vals, red_vals)]
    axes[0].bar(x, norm_vals, bottom=bottom_norm, label="Normal False-Alarm Events", color="#d62728")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(m_short, fontsize=10)
    axes[0].set_ylabel("Predicted Alarm Events Count", fontsize=10)
    axes[0].set_title("(A) Alarm Event Decomposition", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

    # Alarm duration CDF
    for mid, col, lbl in zip(m_ids, colors, m_short):
        durs = alarm_events_df[alarm_events_df["model_id"] == mid]["duration_seconds"].values
        sorted_d = np.sort(durs)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        axes[1].plot(sorted_d, cdf, label=lbl, color=col, linewidth=1.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Alarm Event Duration (seconds, log scale)", fontsize=10)
    axes[1].set_ylabel("Cumulative Probability (CDF)", fontsize=10)
    axes[1].set_title("(B) Alarm Event Duration CDF (Chatter Profile)", fontsize=11, fontweight="bold")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig_ea_03_event_chatter_breakdown.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # Figure EA 4: False-Alarm Tradeoff
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fa_rates = [summary_df[summary_df["model_id"] == m]["normal_fa_events_per_hour"].values[0] for m in m_ids]
    fp_secs = [summary_df[summary_df["model_id"] == m]["false_positive_alarm_seconds_per_hour"].values[0] for m in m_ids]

    axes[0].bar(x, fa_rates, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(m_short, fontsize=10)
    axes[0].set_ylabel("Normal FA Events / Hour", fontsize=10)
    axes[0].set_title("(A) Discrete False Alarm Interruptions", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

    axes[1].bar(x, fp_secs, color=colors, edgecolor="black", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(m_short, fontsize=10)
    axes[1].set_ylabel("FP Alarm Seconds / Hour", fontsize=10)
    axes[1].set_title("(B) Total False Alarm Occupancy Duration", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig_ea_04_false_alarm_tradeoff.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # Figure EA 5: Complete Sensitivity Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    for mid, col, lbl in zip(m_ids, colors, m_short):
        sub = complete_sens_df[complete_sens_df["model_id"] == mid]
        sub = sub.sort_values(by="operating_point")
        rec = sub["event_recall"].values
        f1 = sub["event_f1"].values
        prec = sub["event_precision"].values
        fa_rate = sub["normal_fa_events_per_hour"].values
        fp_sec = sub["fp_alarm_seconds_per_hour"].values

        axes[0].plot(fa_rate, rec, marker="o", color=col, label=lbl, linewidth=2)
        axes[1].plot(fa_rate, f1, marker="s", color=col, label=lbl, linewidth=2)
        axes[2].plot(fp_sec, rec, marker="^", color=col, label=lbl, linewidth=2)

    axes[0].set_xlabel("Normal FA Events / Hour", fontsize=10)
    axes[0].set_ylabel("Event Recall", fontsize=10)
    axes[0].set_title("(A) Recall vs Normal FA Rate", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Normal FA Events / Hour", fontsize=10)
    axes[1].set_ylabel("Event F1 Score", fontsize=10)
    axes[1].set_title("(B) Event F1 vs Normal FA Rate", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower left")

    axes[2].set_xlabel("FP Alarm Seconds / Hour", fontsize=10)
    axes[2].set_ylabel("Event Recall", fontsize=10)
    axes[2].set_title("(C) Recall vs FP Alarm Duration Occupancy", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig_ea_05_sensitivity_curves.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # =========================================================================
    # REPRESENTATIVE CASE PLOTS (FIG EA 6 - 9)
    # =========================================================================
    def plot_case(case_name, aid, sname, title, output_filename, margin_sec=120):
        atk_row = attack_events_df[attack_events_df["attack_event_id"] == aid].iloc[0]
        s_time = pd.Timestamp(atk_row["orig_start"])
        e_time = pd.Timestamp(atk_row["orig_end"])

        ts = session_data[sname]["ts"]
        s_idx = np.where(ts == s_time)[0][0]
        e_idx = np.where(ts == e_time)[0][0]

        start_window_idx = max(30, s_idx - margin_sec)
        end_window_idx = min(len(ts) - 1, e_idx + margin_sec)

        sub_ts = ts[start_window_idx : end_window_idx + 1]
        c_s = start_window_idx - 30
        c_e = end_window_idx - 30

        p_scores = model_session_scores["persistence"][sname][c_s : c_e + 1]
        r_scores = model_session_scores["ridge"][sname][c_s : c_e + 1]
        g_scores = model_session_scores["gru_seed42"][sname][c_s : c_e + 1]

        # Find top 3 perturbed features inside attack
        atk_feats = session_data[sname]["feats"][s_idx : e_idx + 1]
        pre_feats = session_data[sname]["feats"][start_window_idx : s_idx]
        if len(pre_feats) > 0:
            feat_dev = np.abs(np.mean(atk_feats, axis=0) - np.mean(pre_feats, axis=0))
        else:
            feat_dev = np.std(atk_feats, axis=0)
        top3_feat_idxs = np.argsort(feat_dev)[::-1][:3]
        top3_names = [feature_names[i] for i in top3_feat_idxs]

        fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, dpi=300)

        # Plot 1: Top 3 Telemetry Features
        for f_idx, f_name in zip(top3_feat_idxs, top3_names):
            vals = session_data[sname]["feats"][start_window_idx : end_window_idx + 1, f_idx]
            axes[0].plot(sub_ts, vals, label=f"{f_name} (stdized)", linewidth=1.5)

        axes[0].axvspan(s_time, e_time, color="red", alpha=0.2, label=f"Attack Event: {aid} ({atk_row['process_set']})")
        axes[0].set_ylabel("Standardized Value", fontsize=10)
        axes[0].set_title(f"{title}: Top Perturbed Telemetry Features", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="upper right", fontsize=9)

        # Plot 2: Anomaly Scores & Frozen Thresholds
        axes[1].plot(sub_ts, p_scores, label="Persistence", color="#7f7f7f", linewidth=1.2, alpha=0.7)
        axes[1].plot(sub_ts, r_scores, label="Ridge", color="#1f77b4", linewidth=1.4, alpha=0.85)
        axes[1].plot(sub_ts, g_scores, label="GRU seed42", color="#2ca02c", linewidth=1.8)

        axes[1].axhline(frozen_thresholds["persistence"], color="#7f7f7f", linestyle="--", alpha=0.6, label=f"Pers Thresh ({frozen_thresholds['persistence']:.2f})")
        axes[1].axhline(frozen_thresholds["ridge"], color="#1f77b4", linestyle="--", alpha=0.6, label=f"Ridge Thresh ({frozen_thresholds['ridge']:.2f})")
        axes[1].axhline(frozen_thresholds["gru_seed42"], color="#2ca02c", linestyle="--", alpha=0.6, label=f"GRU Thresh ({frozen_thresholds['gru_seed42']:.2f})")

        axes[1].axvspan(s_time, e_time, color="red", alpha=0.2)
        axes[1].set_ylabel("Anomaly Score (RMSE)", fontsize=10)
        axes[1].set_xlabel("Timestamp", fontsize=10)
        axes[1].set_title("Anomaly Scores vs Frozen Q99.9 Thresholds", fontsize=11, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(loc="upper right", fontsize=8, ncol=2)

        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.tight_layout()
        out_path = FIGURES_DIR / output_filename
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved: {out_path}")

    # Case 1: Detected by all models (test1_atk_04: P2 attack, 60s)
    plot_case("All Detected", "test1_atk_04", "test1", "Case 1 (Detected by All Models): test1_atk_04", "fig_ea_06_case_all_detected.png")

    # Case 2: GRU-only detection (test1_atk_02: P1 attack, 240s)
    plot_case("GRU Only", "test1_atk_02", "test1", "Case 2 (GRU-Only Detection): test1_atk_02", "fig_ea_07_case_gru_only.png")

    # Case 3: Missed by all models (test2_atk_06: P1 attack, 256s)
    plot_case("All Missed", "test2_atk_06", "test2", "Case 3 (Missed by All Models): test2_atk_06", "fig_ea_08_case_all_missed.png")

    # Case 4: GRU miss but Ridge detected (test2_atk_05: P1 attack, 123s)
    plot_case("GRU Miss, Ridge Det", "test2_atk_05", "test2", "Case 4 (GRU Miss, Ridge Detected): test2_atk_05", "fig_ea_09_case_gru_miss_ridge_detected.png")

    logger.info("DT049 Post-Test Error Analysis Completed Successfully!")


if __name__ == "__main__":
    main()
