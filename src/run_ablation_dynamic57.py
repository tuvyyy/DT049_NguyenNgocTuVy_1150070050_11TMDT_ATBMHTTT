#!/usr/bin/env python3
"""
HAI 21.03 Post-Hoc Exploratory Ablation: Dynamic 57 Features vs 79 Primary
=========================================================================
Canonical standalone implementation for DT049 Dynamic-57 Feature Ablation.
Evaluates the impact of restricting the RMSE anomaly score calculation to
the 57 dynamic features (features with variance > 0 during development training)
versus the primary 79-feature formulation.

Scientific constraints:
- Models remain completely unchanged and frozen (Persistence, Ridge, GRU seeds 42, 43, 44).
- Thresholds for the 57-feature score are calibrated on the normal validation dataset
  (train3_features.npy) at the predetermined quantiles (Q99.5, Q99.9, Q99.99).
- Evaluation on official test1-test5 uses the identical common timeline (t >= 30)
  and deterministic 1-to-1 chronological event matching.
- Frozen primary results remain untouched.

Outputs:
- results/metrics/ablation_dynamic57_summary.csv
- results/metrics/ablation_dynamic57_multiseed.csv
- results/metrics/ablation_dynamic57_changed_events.csv
- results/figures/ablation_dynamic57/
  * fig_ab_01_primary_vs_dynamic57.png
  * fig_ab_02_detection_changes.png
  * fig_ab_03_false_alarm_tradeoff.png
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from gru_model import OneStepGRU
from gru_dataset import SessionWindowDataset

PROCESSED_TEST_DIR = REPO_ROOT / "data" / "processed" / "hai_21_03" / "final_test"
VAL_FEATS_PATH = REPO_ROOT / "data" / "processed" / "hai_21_03" / "train3_features.npy"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "ablation_dynamic57"
MODELS_DIR = REPO_ROOT / "models"
MANIFEST_PATH = REPO_ROOT / "results" / "audit" / "pretest_freeze_manifest.json"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger():
    logger = logging.getLogger("DT049_AblationDynamic57")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


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
    logger.info("Executing Post-Hoc Exploratory Ablation: Dynamic 57 Features vs 79 Primary...")

    # 1. Feature partition
    with open(MODELS_DIR / "preprocessing" / "scaler_metadata.json", "r", encoding="utf-8") as f:
        scaler_meta = json.load(f)
    feature_names = scaler_meta["ordered_feature_names"]
    zero_cols = scaler_meta["zero_variance_training_features"]
    dyn_indices = [i for i, c in enumerate(feature_names) if c not in zero_cols]

    logger.info(f"Total features: {len(feature_names)} | Excluded zero-variance: {len(zero_cols)} | Retained dynamic: {len(dyn_indices)}")

    device = torch.device("cpu")
    torch.set_num_threads(4)

    # Load frozen models
    ridge_model = joblib.load(MODELS_DIR / "baselines" / "ridge.joblib")
    gru_models = {}
    for seed in [42, 43, 44]:
        ckpt = torch.load(MODELS_DIR / "primary" / f"gru_w30_seed{seed}.pt", map_location=device, weights_only=True)
        m = OneStepGRU(79, 64, 1).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        gru_models[seed] = m

    # 2. Validation threshold calibration on train3
    logger.info("Calibrating 57-feature thresholds on normal validation dataset (train3)...")
    val_feats = np.load(VAL_FEATS_PATH)
    quantiles = ["Q99.5", "Q99.9", "Q99.99"]
    q_vals = [0.995, 0.999, 0.9999]

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    thresholds_79 = {
        "persistence": {
            "Q99.5": float(manifest["frozen_models"]["persistence_baseline_0"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": float(manifest["frozen_models"]["persistence_baseline_0"]["frozen_thresholds"]["Q99.9_primary"]),
            "Q99.99": float(manifest["frozen_models"]["persistence_baseline_0"]["frozen_thresholds"]["Q99.99"]),
        },
        "ridge": {
            "Q99.5": float(manifest["frozen_models"]["ridge_baseline_1"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": float(manifest["frozen_models"]["ridge_baseline_1"]["frozen_thresholds"]["Q99.9_primary"]),
            "Q99.99": float(manifest["frozen_models"]["ridge_baseline_1"]["frozen_thresholds"]["Q99.99"]),
        },
        "gru_seed42": {
            "Q99.5": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["frozen_thresholds"]["Q99.9_primary"]),
            "Q99.99": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_42"]["frozen_thresholds"]["Q99.99"]),
        },
        "gru_seed43": {
            "Q99.5": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["frozen_thresholds"]["Q99.9_primary"]),
            "Q99.99": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_43"]["frozen_thresholds"]["Q99.99"]),
        },
        "gru_seed44": {
            "Q99.5": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["frozen_thresholds"]["Q99.5"]),
            "Q99.9": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["frozen_thresholds"]["Q99.9_primary"]),
            "Q99.99": float(manifest["frozen_models"]["gru_primary"]["checkpoints"]["seed_44"]["frozen_thresholds"]["Q99.99"]),
        },
    }

    thresholds_57 = {}
    # Persistence 57 val
    p_diff_val_57 = val_feats[30:, dyn_indices] - val_feats[29:-1, dyn_indices]
    p_sc_val_57 = np.sqrt(np.mean(p_diff_val_57 ** 2, axis=1)).astype(np.float32)
    thresholds_57["persistence"] = {q: float(np.quantile(p_sc_val_57, qv)) for q, qv in zip(quantiles, q_vals)}

    # Ridge 57 val
    r_preds_val = ridge_model.predict(val_feats[29:-1])
    r_diff_val_57 = val_feats[30:, dyn_indices] - r_preds_val[:, dyn_indices]
    r_sc_val_57 = np.sqrt(np.mean(r_diff_val_57 ** 2, axis=1)).astype(np.float32)
    thresholds_57["ridge"] = {q: float(np.quantile(r_sc_val_57, qv)) for q, qv in zip(quantiles, q_vals)}

    # GRU 57 val
    ds_val = SessionWindowDataset([val_feats], window_size=30, stride=1)
    loader_val = DataLoader(ds_val, batch_size=1024, shuffle=False)
    for seed in [42, 43, 44]:
        mid = f"gru_seed{seed}"
        m = gru_models[seed]
        sc_val_57 = []
        with torch.no_grad():
            for bx, by in loader_val:
                preds = m(bx.to(device)).cpu().numpy()
                diffs = by.numpy()[:, dyn_indices] - preds[:, dyn_indices]
                sc_val_57.append(np.sqrt(np.mean(diffs ** 2, axis=1)).astype(np.float32))
        sc_val_57 = np.concatenate(sc_val_57)
        thresholds_57[mid] = {q: float(np.quantile(sc_val_57, qv)) for q, qv in zip(quantiles, q_vals)}

    del val_feats, p_sc_val_57, r_sc_val_57, ds_val, loader_val

    # 3. Official test evaluation
    test_sessions = [f"test{i}" for i in range(1, 6)]
    attack_events_df = pd.read_csv(METRICS_DIR / "final_attack_events.csv")

    session_data = {}
    for sname in test_sessions:
        feats = np.load(PROCESSED_TEST_DIR / f"{sname}_features.npy")
        ts = np.load(PROCESSED_TEST_DIR / f"{sname}_timestamps.npy")
        atk = np.load(PROCESSED_TEST_DIR / f"{sname}_attack.npy")

        target_ts = ts[30:]
        target_atk = atk[30:]
        norm_mask = (target_atk == 0)
        norm_samples = int(np.sum(norm_mask))
        norm_hours = norm_samples / 3600.0

        session_data[sname] = {
            "feats": feats,
            "ts": ts,
            "target_ts": target_ts,
            "target_atk": target_atk,
            "norm_mask": norm_mask,
            "normal_samples": norm_samples,
            "normal_hours": norm_hours,
        }

    total_normal_hours = sum(session_data[s]["normal_hours"] for s in test_sessions)
    total_normal_samples = sum(session_data[s]["normal_samples"] for s in test_sessions)

    test_scores_79 = {mid: {} for mid in thresholds_79}
    test_scores_57 = {mid: {} for mid in thresholds_57}

    for sname in test_sessions:
        feats = session_data[sname]["feats"]
        # Persistence
        diff_all = feats[30:] - feats[29:-1]
        test_scores_79["persistence"][sname] = np.sqrt(np.mean(diff_all ** 2, axis=1)).astype(np.float32)
        test_scores_57["persistence"][sname] = np.sqrt(np.mean(diff_all[:, dyn_indices] ** 2, axis=1)).astype(np.float32)

        # Ridge
        r_preds = ridge_model.predict(feats[29:-1])
        r_diff = feats[30:] - r_preds
        test_scores_79["ridge"][sname] = np.sqrt(np.mean(r_diff ** 2, axis=1)).astype(np.float32)
        test_scores_57["ridge"][sname] = np.sqrt(np.mean(r_diff[:, dyn_indices] ** 2, axis=1)).astype(np.float32)

        # GRUs
        ds = SessionWindowDataset([feats], window_size=30, stride=1)
        loader = DataLoader(ds, batch_size=1024, shuffle=False)
        g_diffs = {seed: [] for seed in [42, 43, 44]}
        with torch.no_grad():
            for bx, by in loader:
                for seed in [42, 43, 44]:
                    preds = gru_models[seed](bx.to(device)).cpu().numpy()
                    diff = by.numpy() - preds
                    g_diffs[seed].append(diff)

        for seed in [42, 43, 44]:
            mid = f"gru_seed{seed}"
            cat_diff = np.concatenate(g_diffs[seed])
            test_scores_79[mid][sname] = np.sqrt(np.mean(cat_diff ** 2, axis=1)).astype(np.float32)
            test_scores_57[mid][sname] = np.sqrt(np.mean(cat_diff[:, dyn_indices] ** 2, axis=1)).astype(np.float32)

    model_labels = {
        "persistence": "Baseline 0: Persistence",
        "ridge": "Baseline 1: Ridge (alpha=300)",
        "gru_seed42": "Primary GRU (Seed 42, Canonical)",
        "gru_seed43": "Primary GRU (Seed 43, Replicate 1)",
        "gru_seed44": "Primary GRU (Seed 44, Replicate 2)",
    }

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

    comparison_records = []
    attack_detections = {atk["event_id"]: {"attack_event_id": atk["event_id"], "session": atk["session_name"]} for atk in all_attacks_raw}

    for mid, mname in model_labels.items():
        for config_name, feat_count, sc_dict, th_dict in [
            ("Primary (79 features)", 79, test_scores_79, thresholds_79),
            ("Ablation (57 dynamic features)", 57, test_scores_57, thresholds_57),
        ]:
            for q in quantiles:
                thresh = th_dict[mid][q]
                tp_total = 0
                fn_total = 0
                fp_ev_total = 0
                norm_fa_events_total = 0
                norm_alarm_samples_total = 0
                all_delays = []

                for sname in test_sessions:
                    sc = sc_dict[mid][sname]
                    tgt_atk = session_data[sname]["target_atk"]
                    tgt_ts = session_data[sname]["target_ts"]
                    norm_m = session_data[sname]["norm_mask"]
                    ev_atks = [a for a in all_attacks_raw if a["session_name"] == sname]

                    is_alm = (sc > thresh)
                    pred_alms = extract_predicted_alarm_events(sname, mid, sc, thresh, tgt_ts)
                    norm_fa_evs = extract_normal_false_alarm_events(sname, is_alm, tgt_atk, tgt_ts)

                    match_res = match_events_one_to_one(ev_atks, pred_alms, is_alm, tgt_ts)
                    tp_total += match_res["tp_count"]
                    fn_total += match_res["fn_count"]
                    fp_ev_total += match_res["fp_count"]
                    norm_fa_events_total += len(norm_fa_evs)
                    norm_alarm_samples_total += int(np.sum(is_alm & norm_m))

                    for aid, d in match_res["delays"].items():
                        if not np.isnan(d):
                            all_delays.append(d)
                        if q == "Q99.9":
                            prefix = "prim79" if feat_count == 79 else "ab57"
                            attack_detections[aid][f"{mid}_{prefix}_detected"] = (not np.isnan(d))
                            attack_detections[aid][f"{mid}_{prefix}_delay"] = round(d, 2) if not np.isnan(d) else np.nan

                prec = tp_total / (tp_total + fp_ev_total) if (tp_total + fp_ev_total) > 0 else 0.0
                rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

                norm_fa_rate = norm_fa_events_total / total_normal_hours
                fp_sec_per_h = norm_alarm_samples_total / total_normal_hours
                point_fpr = norm_alarm_samples_total / total_normal_samples
                mean_delay = float(np.mean(all_delays)) if all_delays else np.nan
                med_delay = float(np.median(all_delays)) if all_delays else np.nan

                comparison_records.append({
                    "model_id": mid,
                    "model_name": mname,
                    "feature_configuration": config_name,
                    "feature_count": feat_count,
                    "operating_point": q,
                    "threshold_value": round(thresh, 6),
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
                    "mean_detection_delay_sec": round(mean_delay, 2) if not np.isnan(mean_delay) else np.nan,
                    "median_detection_delay_sec": round(med_delay, 2) if not np.isnan(med_delay) else np.nan,
                })

    comparison_df = pd.DataFrame(comparison_records)
    summary_out = METRICS_DIR / "ablation_dynamic57_summary.csv"
    comparison_df.to_csv(summary_out, index=False)
    logger.info(f"Saved: {summary_out}")

    # Build multi-seed table
    multiseed_records = []
    for config_name, feat_count in [("Primary (79 features)", 79), ("Ablation (57 dynamic features)", 57)]:
        for q in quantiles:
            f1s, recs, precs, fa_rates, fp_secs, fprs, delays = [], [], [], [], [], [], []
            for seed in [42, 43, 44]:
                mid = f"gru_seed{seed}"
                row = comparison_df[(comparison_df["model_id"] == mid) &
                                    (comparison_df["feature_count"] == feat_count) &
                                    (comparison_df["operating_point"] == q)].iloc[0]
                multiseed_records.append({
                    "feature_configuration": config_name,
                    "feature_count": feat_count,
                    "operating_point": q,
                    "seed": str(seed),
                    "event_recall": row["event_recall"],
                    "event_precision": row["event_precision"],
                    "event_f1": row["event_f1"],
                    "normal_fa_events_per_hour": row["normal_fa_events_per_hour"],
                    "fp_alarm_seconds_per_hour": row["fp_alarm_seconds_per_hour"],
                    "point_fpr": row["point_fpr"],
                    "median_detection_delay_sec": row["median_detection_delay_sec"],
                })
                f1s.append(row["event_f1"])
                recs.append(row["event_recall"])
                precs.append(row["event_precision"])
                fa_rates.append(row["normal_fa_events_per_hour"])
                fp_secs.append(row["fp_alarm_seconds_per_hour"])
                fprs.append(row["point_fpr"])
                delays.append(row["median_detection_delay_sec"])

            multiseed_records.append({
                "feature_configuration": config_name,
                "feature_count": feat_count,
                "operating_point": q,
                "seed": "3-seed mean ± std",
                "event_recall": f"{np.mean(recs):.4f} ± {np.std(recs):.4f}",
                "event_precision": f"{np.mean(precs):.4f} ± {np.std(precs):.4f}",
                "event_f1": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
                "normal_fa_events_per_hour": f"{np.mean(fa_rates):.4f} ± {np.std(fa_rates):.4f}",
                "fp_alarm_seconds_per_hour": f"{np.mean(fp_secs):.4f} ± {np.std(fp_secs):.4f}",
                "point_fpr": f"{np.mean(fprs):.6f} ± {np.std(fprs):.6f}",
                "median_detection_delay_sec": f"{np.mean(delays):.2f} ± {np.std(delays):.2f}",
            })

    multiseed_df = pd.DataFrame(multiseed_records)
    multiseed_out = METRICS_DIR / "ablation_dynamic57_multiseed.csv"
    multiseed_df.to_csv(multiseed_out, index=False)
    logger.info(f"Saved: {multiseed_out}")

    # Build changed events table
    changed_records = []
    for idx, r in attack_events_df.iterrows():
        aid = r["attack_event_id"]
        ab_r = attack_detections[aid]

        p79 = bool(ab_r["persistence_prim79_detected"])
        p57 = bool(ab_r["persistence_ab57_detected"])
        r79 = bool(ab_r["ridge_prim79_detected"])
        r57 = bool(ab_r["ridge_ab57_detected"])
        g42_79 = bool(ab_r["gru_seed42_prim79_detected"])
        g42_57 = bool(ab_r["gru_seed42_ab57_detected"])
        g43_79 = bool(ab_r["gru_seed43_prim79_detected"])
        g43_57 = bool(ab_r["gru_seed43_ab57_detected"])
        g44_79 = bool(ab_r["gru_seed44_prim79_detected"])
        g44_57 = bool(ab_r["gru_seed44_ab57_detected"])

        changed = (p79 != p57) or (r79 != r57) or (g42_79 != g42_57) or (g43_79 != g43_57) or (g44_79 != g44_57)

        changed_records.append({
            "attack_event_id": aid,
            "session": r["session"],
            "duration_seconds": r["duration_seconds"],
            "process_set": r["process_set"],
            "persistence_status_79": "Detected" if p79 else "Missed",
            "persistence_status_57": "Detected" if p57 else "Missed",
            "ridge_status_79": "Detected" if r79 else "Missed",
            "ridge_status_57": "Detected" if r57 else "Missed",
            "gru_s42_status_79": "Detected" if g42_79 else "Missed",
            "gru_s42_status_57": "Detected" if g42_57 else "Missed",
            "gru_s43_status_79": "Detected" if g43_79 else "Missed",
            "gru_s43_status_57": "Detected" if g43_57 else "Missed",
            "gru_s44_status_79": "Detected" if g44_79 else "Missed",
            "gru_s44_status_57": "Detected" if g44_57 else "Missed",
            "changed_status_any_model": changed,
        })

    changed_df = pd.DataFrame(changed_records)
    changed_out = METRICS_DIR / "ablation_dynamic57_changed_events.csv"
    changed_df.to_csv(changed_out, index=False)
    logger.info(f"Saved: {changed_out}")

    # 4. Generate Figures
    logger.info("Generating publication figures for ablation_dynamic57...")
    m_order = ["persistence", "ridge", "gru_seed42", "gru_seed43", "gru_seed44"]
    m_labels = ["Persistence", "Ridge", "GRU s42\n(Canonical)", "GRU s43", "GRU s44"]

    # Figure 1: Primary vs Dynamic57 Metrics at Q99.9
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    q_sub = comparison_df[comparison_df["operating_point"] == "Q99.9"]
    sub_79 = q_sub[q_sub["feature_count"] == 79].set_index("model_id").loc[m_order]
    sub_57 = q_sub[q_sub["feature_count"] == 57].set_index("model_id").loc[m_order]

    x = np.arange(len(m_order))
    width = 0.35

    axes[0].bar(x - width/2, sub_79["event_recall"], width, label="Primary (79 feats)", color="#1f77b4")
    axes[0].bar(x + width/2, sub_57["event_recall"], width, label="Ablation (57 dynamic)", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(m_labels, fontsize=9)
    axes[0].set_ylabel("Event Recall", fontsize=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(A) Event Recall (Q99.9)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].legend(loc="lower right")

    axes[1].bar(x - width/2, sub_79["event_f1"], width, label="Primary (79 feats)", color="#1f77b4")
    axes[1].bar(x + width/2, sub_57["event_f1"], width, label="Ablation (57 dynamic)", color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(m_labels, fontsize=9)
    axes[1].set_ylabel("Event F1 Score", fontsize=10)
    axes[1].set_ylim(0, 0.30)
    axes[1].set_title("(B) Event F1 Score (Q99.9)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1].legend(loc="upper left")

    axes[2].bar(x - width/2, sub_79["normal_fa_events_per_hour"], width, label="Primary (79 feats)", color="#1f77b4")
    axes[2].bar(x + width/2, sub_57["normal_fa_events_per_hour"], width, label="Ablation (57 dynamic)", color="#ff7f0e")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(m_labels, fontsize=9)
    axes[2].set_ylabel("Normal FA Events / Hour", fontsize=10)
    axes[2].set_title("(C) Normal False Alarm Rate (Q99.9)", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "fig_ab_01_primary_vs_dynamic57.png"
    plt.savefig(fig1_path)
    plt.close()
    logger.info(f"Saved: {fig1_path}")

    # Figure 2: Binary Detection Heatmap across 50 attacks
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    col_labels = [
        "Pers (79)", "Pers (57)",
        "Ridge (79)", "Ridge (57)",
        "GRU s42 (79)", "GRU s42 (57)",
        "GRU s43 (79)", "GRU s43 (57)",
        "GRU s44 (79)", "GRU s44 (57)",
    ]
    matrix = []
    row_labels = []
    for idx, r in changed_df.iterrows():
        aid = r["attack_event_id"]
        pset = r["process_set"]
        row_labels.append(f"{aid} ({pset})")
        row_vals = [
            1 if r["persistence_status_79"] == "Detected" else 0,
            1 if r["persistence_status_57"] == "Detected" else 0,
            1 if r["ridge_status_79"] == "Detected" else 0,
            1 if r["ridge_status_57"] == "Detected" else 0,
            1 if r["gru_s42_status_79"] == "Detected" else 0,
            1 if r["gru_s42_status_57"] == "Detected" else 0,
            1 if r["gru_s43_status_79"] == "Detected" else 0,
            1 if r["gru_s43_status_57"] == "Detected" else 0,
            1 if r["gru_s44_status_79"] == "Detected" else 0,
            1 if r["gru_s44_status_57"] == "Detected" else 0,
        ]
        matrix.append(row_vals)

    matrix = np.array(matrix)
    cax = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("DT049 Attack Detection Comparison: Primary 79 vs Ablation 57 Dynamic (Exact Invariance)", fontsize=11, fontweight="bold")

    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)

    cbar = fig.colorbar(cax, ax=ax, ticks=[0, 1], fraction=0.02, pad=0.02)
    cbar.ax.set_yticklabels(["Missed (0)", "Detected (1)"])

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "fig_ab_02_detection_changes.png"
    plt.savefig(fig2_path)
    plt.close()
    logger.info(f"Saved: {fig2_path}")

    # Figure 3: False Alarm Tradeoff
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#8c564b", "#9467bd"]
    for mid, col, lbl in zip(m_order, colors, m_labels):
        r79 = comparison_df[(comparison_df["model_id"] == mid) & (comparison_df["feature_count"] == 79) & (comparison_df["operating_point"] == "Q99.9")].iloc[0]
        r57 = comparison_df[(comparison_df["model_id"] == mid) & (comparison_df["feature_count"] == 57) & (comparison_df["operating_point"] == "Q99.9")].iloc[0]
        ax.scatter(r79["normal_fa_events_per_hour"], r79["fp_alarm_seconds_per_hour"], color=col, marker="o", s=80, label=f"{lbl.split()[0]} (79 feats)")
        ax.scatter(r57["normal_fa_events_per_hour"], r57["fp_alarm_seconds_per_hour"], color=col, marker="x", s=80, linewidth=2, label=f"{lbl.split()[0]} (57 dyn)")

    ax.set_xlabel("Normal False-Alarm Events / Hour (Operator Interruption Frequency)", fontsize=10)
    ax.set_ylabel("False-Positive Alarm Seconds / Hour (Temporal Occupancy)", fontsize=10)
    ax.set_title("False-Alarm Operational Tradeoff: 79 vs 57 Features", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    fig3_path = FIGURES_DIR / "fig_ab_03_false_alarm_tradeoff.png"
    plt.savefig(fig3_path)
    plt.close()
    logger.info(f"Saved: {fig3_path}")

    logger.info("DT049 Dynamic 57 Post-Hoc Exploratory Ablation Execution Completed Successfully!")


if __name__ == "__main__":
    main()
