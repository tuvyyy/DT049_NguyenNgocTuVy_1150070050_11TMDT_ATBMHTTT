#!/usr/bin/env python3
"""
Baseline 0: One-Step Persistence Predictor for DT049 (HAI 21.03)
===============================================================
Evaluates non-parametric persistence baseline: X_hat[i+1] = X[i].
Primary Anomaly Score: RMSE across all 79 standardized features.

Methodological constraints:
1. Strict session boundaries: train1, train2, and train3 are processed independently;
   no predictions or windows span across session boundaries.
2. Threshold calibration is performed EXCLUSIVELY on Normal Validation (train3).
3. Zero test files (test1-test5) or attack labels are accessed or evaluated.
4. Primary threshold is Q99.9 empirical quantile of validation scores.
5. All 79 features are preserved in the baseline.
"""

import os
import sys
import gc
import json
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime, timezone
import psutil
import sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "baseline_persistence.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "hai_21_03"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "baseline_persistence"
AUDIT_DIR = REPO_ROOT / "results" / "audit"
MODELS_DIR = REPO_ROOT / "models" / "preprocessing"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("HAI2103_Persistence")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_persistence_scores(features_mmap: np.ndarray, chunk_size: int = 50_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes one-step persistence residuals and anomaly scores chunk-wise:
    X_hat[i+1] = X[i]  =>  residual = X[i+1] - X[i]
    Returns: (rmse_scores, mae_scores, max_abs_res, mean_squared_feature_errors)
    """
    total_rows, num_features = features_mmap.shape
    num_scores = total_rows - 1

    rmse_scores = np.empty(num_scores, dtype=np.float32)
    mae_scores = np.empty(num_scores, dtype=np.float32)
    max_abs_res = np.empty(num_scores, dtype=np.float32)
    feature_sq_err_sum = np.zeros(num_features, dtype=np.float64)

    for start_idx in range(0, num_scores, chunk_size):
        end_idx = min(start_idx + chunk_size, num_scores)
        # Slices: x_prev = X[start_idx : end_idx], x_curr = X[start_idx + 1 : end_idx + 1]
        x_prev = features_mmap[start_idx:end_idx]
        x_curr = features_mmap[start_idx + 1 : end_idx + 1]
        diff = x_curr - x_prev  # shape (chunk_len, 79)

        # Vectorized metrics
        diff_sq = diff ** 2
        feature_sq_err_sum += np.sum(diff_sq, axis=0)

        chunk_rmse = np.sqrt(np.mean(diff_sq, axis=1)).astype(np.float32)
        chunk_mae = np.mean(np.abs(diff), axis=1).astype(np.float32)
        chunk_max_res = np.max(np.abs(diff), axis=1).astype(np.float32)

        rmse_scores[start_idx:end_idx] = chunk_rmse
        mae_scores[start_idx:end_idx] = chunk_mae
        max_abs_res[start_idx:end_idx] = chunk_max_res

    mean_feature_rmse = np.sqrt(feature_sq_err_sum / num_scores).astype(np.float32)
    return rmse_scores, mae_scores, max_abs_res, mean_feature_rmse


def extract_alarm_events(is_alarm: np.ndarray) -> List[dict]:
    """
    Extracts contiguous runs where is_alarm == True with zero filtering or merging.
    """
    events = []
    in_alarm = False
    start_idx = 0

    for i in range(len(is_alarm)):
        if is_alarm[i]:
            if not in_alarm:
                in_alarm = True
                start_idx = i
        else:
            if in_alarm:
                in_alarm = False
                events.append({"start_idx": start_idx, "end_idx": i - 1, "duration_seconds": i - start_idx})

    if in_alarm:
        events.append({"start_idx": start_idx, "end_idx": len(is_alarm) - 1, "duration_seconds": len(is_alarm) - start_idx})

    return events


def compute_distribution_summary(scores: np.ndarray) -> dict:
    return {
        "count": int(len(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "median": float(np.median(scores)),
        "q25": float(np.quantile(scores, 0.25, method="linear")),
        "q75": float(np.quantile(scores, 0.75, method="linear")),
        "q95": float(np.quantile(scores, 0.95, method="linear")),
        "q99": float(np.quantile(scores, 0.99, method="linear")),
        "q99_5": float(np.quantile(scores, 0.995, method="linear")),
        "q99_9": float(np.quantile(scores, 0.999, method="linear")),
        "q99_99": float(np.quantile(scores, 0.9999, method="linear")),
    }


def main():
    logger = setup_logger()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    peak_mem = get_memory_mb()

    logger.info("=================================================================")
    logger.info("Starting Baseline 0: One-Step Persistence Predictor for DT049")
    logger.info(f"Initial Memory Usage: {peak_mem:.2f} MB")
    logger.info("=================================================================")

    config = load_config()

    # Load scaler metadata for ordered feature names
    scaler_meta_path = MODELS_DIR / "scaler_metadata.json"
    with open(scaler_meta_path, "r") as f:
        scaler_meta = json.load(f)
    feature_names = scaler_meta["ordered_feature_names"]
    num_features = len(feature_names)
    logger.info(f"Loaded feature metadata: {num_features} features.")

    # 1. Compute Persistence Scores on Development Training (train1 and train2 separately)
    logger.info("Computing persistence scores on Development Training sessions...")
    t1_f = np.load(PROCESSED_DIR / "train1_features.npy", mmap_mode="r")
    t2_f = np.load(PROCESSED_DIR / "train2_features.npy", mmap_mode="r")

    t1_rmse, t1_mae, t1_max_res, t1_feat_rmse = compute_persistence_scores(t1_f)
    t2_rmse, t2_mae, t2_max_res, t2_feat_rmse = compute_persistence_scores(t2_f)

    train_rmse_scores = np.concatenate([t1_rmse, t2_rmse])
    train_exposure_seconds = len(train_rmse_scores)
    train_exposure_hours = train_exposure_seconds / 3600.0
    del t1_f, t2_f, t1_mae, t1_max_res, t2_mae, t2_max_res
    gc.collect()

    peak_mem = max(peak_mem, get_memory_mb())
    logger.info(f"Dev Training Scores: {train_exposure_seconds:,} samples ({train_exposure_hours:.2f} hours; train1: 216,000; train2: 226,800).")
    logger.info(f"Training RMSE Score: Mean={train_rmse_scores.mean():.4f}, Median={np.median(train_rmse_scores):.4f}, Max={train_rmse_scores.max():.4f}")

    # 2. Compute Persistence Scores on Normal Validation (train3)
    logger.info("Computing persistence scores on Normal Validation session (train3)...")
    t3_f = np.load(PROCESSED_DIR / "train3_features.npy", mmap_mode="r")
    t3_ts = np.load(PROCESSED_DIR / "train3_timestamps.npy", mmap_mode="r")

    val_rmse_scores, val_mae_scores, val_max_res, val_feat_rmse = compute_persistence_scores(t3_f)
    val_scored_timestamps = t3_ts[1:]  # shape (478800,)
    del t3_f, t3_ts
    gc.collect()

    val_exposure_seconds = len(val_rmse_scores)
    val_exposure_hours = val_exposure_seconds / 3600.0

    peak_mem = max(peak_mem, get_memory_mb())
    logger.info(f"Normal Validation Scores: {len(val_rmse_scores):,} samples ({val_exposure_hours:.2f} hours).")
    logger.info(f"Validation RMSE Score: Mean={val_rmse_scores.mean():.4f}, Median={np.median(val_rmse_scores):.4f}, Max={val_rmse_scores.max():.4f}")

    # 3. Distribution Summaries
    train_dist = compute_distribution_summary(train_rmse_scores)
    val_dist = compute_distribution_summary(val_rmse_scores)

    # 4. Threshold Calibration on Normal Validation (train3)
    logger.info("=================================================================")
    logger.info("Calibrating Operating Thresholds on Normal Validation (train3)...")
    logger.info(f"Quantile method: {config['threshold_calibration']['numpy_quantile_method']}")
    logger.info(f"Scored Validation Exposure: {val_exposure_seconds:,} seconds = {val_exposure_hours:.2f} hours.")
    logger.info(f"Scored Train Exposure: {train_exposure_seconds:,} seconds = {train_exposure_hours:.2f} hours.")

    operating_targets = config["threshold_calibration"]["operating_targets"]
    validation_results = []
    calibrated_thresholds = {}

    for target_cfg in operating_targets:
        q_name = target_cfg["name"]
        q_val = float(target_cfg["quantile"])
        role = target_cfg["role"]

        # Exact empirical quantile on normal validation (train3)
        threshold_val = float(np.quantile(val_rmse_scores, q_val, method="linear"))

        # Validation exceedance and events (strict inequality: score > threshold)
        is_alarm_val = (val_rmse_scores > threshold_val)
        val_point_exceedances = int(np.sum(is_alarm_val))
        val_point_exceedance_rate = val_point_exceedances / val_exposure_seconds
        val_alarm_seconds = val_point_exceedances
        val_alarm_sec_per_hour = val_alarm_seconds / val_exposure_hours

        val_events = extract_alarm_events(is_alarm_val)
        val_event_count = len(val_events)
        val_events_per_hour = val_event_count / val_exposure_hours
        val_durations = [e["duration_seconds"] for e in val_events]
        val_mean_dur = float(np.mean(val_durations)) if val_durations else 0.0
        val_med_dur = float(np.median(val_durations)) if val_durations else 0.0
        val_max_dur = int(np.max(val_durations)) if val_durations else 0

        # Development Training evaluation on same threshold (without crossing session boundaries)
        is_alarm_t1 = (t1_rmse > threshold_val)
        is_alarm_t2 = (t2_rmse > threshold_val)
        train_point_exceedances = int(np.sum(is_alarm_t1) + np.sum(is_alarm_t2))
        train_point_exceedance_rate = train_point_exceedances / train_exposure_seconds
        train_alarm_seconds = train_point_exceedances
        train_alarm_sec_per_hour = train_alarm_seconds / train_exposure_hours

        t1_events = extract_alarm_events(is_alarm_t1)
        t2_events = extract_alarm_events(is_alarm_t2)
        train_event_count = len(t1_events) + len(t2_events)
        train_events_per_hour = train_event_count / train_exposure_hours
        train_durations = [e["duration_seconds"] for e in t1_events] + [e["duration_seconds"] for e in t2_events]
        train_mean_dur = float(np.mean(train_durations)) if train_durations else 0.0
        train_med_dur = float(np.median(train_durations)) if train_durations else 0.0
        train_max_dur = int(np.max(train_durations)) if train_durations else 0

        calibrated_thresholds[q_name] = {
            "role": role,
            "quantile": q_val,
            "threshold": threshold_val,
            # Validation metrics
            "point_exceedance_count": val_point_exceedances,
            "point_exceedance_rate": val_point_exceedance_rate,
            "alarm_seconds": val_alarm_seconds,
            "alarm_seconds_per_hour": round(val_alarm_sec_per_hour, 4),
            "false_alarm_events_count": val_event_count,
            "false_alarm_events_per_hour": round(val_events_per_hour, 4),
            "mean_event_duration_seconds": round(val_mean_dur, 2),
            "median_event_duration_seconds": round(val_med_dur, 2),
            "max_event_duration_seconds": val_max_dur,
            # Sub-dictionaries for explicit partitioning
            "validation": {
                "point_exceedance_count": val_point_exceedances,
                "point_exceedance_rate": val_point_exceedance_rate,
                "alarm_seconds_per_hour": round(val_alarm_sec_per_hour, 4),
                "false_alarm_events_count": val_event_count,
                "false_alarm_events_per_hour": round(val_events_per_hour, 4),
                "mean_duration_seconds": round(val_mean_dur, 2),
                "median_duration_seconds": round(val_med_dur, 2),
                "max_duration_seconds": val_max_dur,
            },
            "development_training": {
                "point_exceedance_count": train_point_exceedances,
                "point_exceedance_rate": train_point_exceedance_rate,
                "alarm_seconds_per_hour": round(train_alarm_sec_per_hour, 4),
                "false_alarm_events_count": train_event_count,
                "false_alarm_events_per_hour": round(train_events_per_hour, 4),
                "mean_duration_seconds": round(train_mean_dur, 2),
                "median_duration_seconds": round(train_med_dur, 2),
                "max_duration_seconds": train_max_dur,
            }
        }

        validation_results.append({
            "target_name": q_name,
            "role": role,
            "quantile": q_val,
            "threshold": round(threshold_val, 6),
            # Validation columns
            "val_point_exceedance_count": val_point_exceedances,
            "val_point_exceedance_rate": round(val_point_exceedance_rate, 8),
            "val_alarm_seconds": val_alarm_seconds,
            "val_alarm_seconds_per_hour": round(val_alarm_sec_per_hour, 4),
            "val_false_alarm_events_count": val_event_count,
            "val_false_alarm_events_per_hour": round(val_events_per_hour, 4),
            "val_mean_duration_seconds": round(val_mean_dur, 2),
            "val_median_duration_seconds": round(val_med_dur, 2),
            "val_max_duration_seconds": val_max_dur,
            # Training columns
            "train_point_exceedance_count": train_point_exceedances,
            "train_point_exceedance_rate": round(train_point_exceedance_rate, 8),
            "train_alarm_seconds": train_alarm_seconds,
            "train_alarm_seconds_per_hour": round(train_alarm_sec_per_hour, 4),
            "train_false_alarm_events_count": train_event_count,
            "train_false_alarm_events_per_hour": round(train_events_per_hour, 4),
            "train_mean_duration_seconds": round(train_mean_dur, 2),
            "train_median_duration_seconds": round(train_med_dur, 2),
            "train_max_duration_seconds": train_max_dur,
        })

        logger.info(
            f"[{q_name} | {role.upper()}] Thresh = {threshold_val:.6f} | "
            f"Val: {val_point_exceedances:,} pts ({val_point_exceedance_rate*100:.4f}%), {val_event_count} ev ({val_events_per_hour:.3f} ev/h, {val_alarm_sec_per_hour:.2f} s/h, max {val_max_dur}s) | "
            f"Train: {train_point_exceedances:,} pts ({train_point_exceedance_rate*100:.4f}%), {train_event_count} ev ({train_events_per_hour:.3f} ev/h, {train_alarm_sec_per_hour:.2f} s/h, max {train_max_dur}s)"
        )

    # Save validation CSV table
    val_metrics_df = pd.DataFrame(validation_results)
    val_metrics_csv_path = METRICS_DIR / "baseline_persistence_validation.csv"
    val_metrics_df.to_csv(val_metrics_csv_path, index=False)
    logger.info(f"Saved validation metrics table: {val_metrics_csv_path}")

    # Primary operating point
    primary_thresh_info = calibrated_thresholds["Q99.9"]
    primary_threshold = primary_thresh_info["threshold"]

    # 5. Save Comprehensive Reproducibility JSON
    json_payload = {
        "metadata": {
            "baseline_model": "one_step_persistence",
            "version": config["version"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "preprocessing_manifest": "results/audit/hai_21_03_preprocessing_manifest.csv",
            "scaler_metadata": "models/preprocessing/scaler_metadata.json",
            "primary_operating_threshold": "Q99.9",
            "primary_threshold_value": primary_threshold,
            "feature_count": num_features,
            "zero_variance_features_count": len(scaler_meta["zero_variance_training_features"]),
            "zero_variance_features_scale_handling": "scaler scale=1.0 (raw deviation preserved if dynamic in test)",
        },
        "scored_exposure": {
            "development_training_samples": len(train_rmse_scores),
            "development_training_exposure_seconds": train_exposure_seconds,
            "development_training_exposure_hours": train_exposure_hours,
            "normal_validation_samples": val_exposure_seconds,
            "normal_validation_exposure_seconds": val_exposure_seconds,
            "normal_validation_exposure_hours": val_exposure_hours,
            "validation_first_scored_timestamp": str(val_scored_timestamps[0]),
            "validation_last_scored_timestamp": str(val_scored_timestamps[-1]),
        },
        "score_distributions": {
            "development_training_rmse": train_dist,
            "normal_validation_rmse": val_dist,
        },
        "threshold_calibration": calibrated_thresholds,
    }

    json_path = METRICS_DIR / "baseline_persistence_thresholds.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    logger.info(f"Saved threshold JSON: {json_path}")

    # 6. Render Baseline Figures
    logger.info("Rendering Baseline Figures...")

    # Figure 1: Train vs Validation Score Distribution Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Baseline 0 (Persistence): Anomaly Score Distributions\n[Development Training vs. Normal Validation]", fontsize=12, fontweight="bold")

    ax1.hist(train_rmse_scores, bins=100, density=True, color="#2b5c8f", alpha=0.7, label="Dev Train (train1+2)")
    ax1.hist(val_rmse_scores, bins=100, density=True, color="#d35400", alpha=0.6, label="Normal Val (train3)")
    ax1.axvline(primary_threshold, color="black", linestyle="--", lw=1.5, label=f"Primary Thresh Q99.9 ({primary_threshold:.4f})")
    ax1.set_xlabel("RMSE Anomaly Score", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.set_title("Full Distribution Density", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    # Log-scale tail view
    ax2.hist(train_rmse_scores, bins=100, density=True, color="#2b5c8f", alpha=0.7, label="Dev Train (train1+2)", log=True)
    ax2.hist(val_rmse_scores, bins=100, density=True, color="#d35400", alpha=0.6, label="Normal Val (train3)", log=True)
    ax2.axvline(calibrated_thresholds["Q99.5"]["threshold"], color="blue", linestyle=":", lw=1.2, label=f"Q99.5 ({calibrated_thresholds['Q99.5']['threshold']:.4f})")
    ax2.axvline(primary_threshold, color="black", linestyle="--", lw=1.5, label=f"Q99.9 ({primary_threshold:.4f})")
    ax2.axvline(calibrated_thresholds["Q99.99"]["threshold"], color="red", linestyle=":", lw=1.2, label=f"Q99.99 ({calibrated_thresholds['Q99.99']['threshold']:.4f})")
    ax2.set_xlabel("RMSE Anomaly Score", fontsize=10)
    ax2.set_ylabel("Log Density", fontsize=10)
    ax2.set_title("Extreme Tail Log-Density & Operating Thresholds", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "fig_01_score_distribution_comparison.png"
    plt.savefig(fig1_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig1_path}")

    # Figure 2: Validation Score Timeline with Primary Threshold
    fig, ax = plt.subplots(figsize=(14, 5))
    downsample_factor = 20
    val_ts_down = pd.to_datetime(val_scored_timestamps[::downsample_factor])
    val_scores_down = val_rmse_scores[::downsample_factor]

    ax.plot(val_ts_down, val_scores_down, color="#2c3e50", lw=0.8, alpha=0.85, label="Persistence RMSE (train3 normal val)")
    ax.axhline(primary_threshold, color="#c0392b", linestyle="--", lw=1.8, label=f"Primary Threshold Q99.9 = {primary_threshold:.4f} ({primary_thresh_info['false_alarm_events_per_hour']} ev/h)")

    # Highlight false alarms on timeline
    val_alarm_mask = (val_rmse_scores > primary_threshold)
    alarm_ts = pd.to_datetime(val_scored_timestamps[val_alarm_mask])
    alarm_scores = val_rmse_scores[val_alarm_mask]
    ax.scatter(alarm_ts, alarm_scores, color="#e74c3c", s=18, zorder=5, label=f"False Alarm Points ({len(alarm_ts)} pts)")

    ax.set_xlabel("Timeline (August 04, 2020 - August 10, 2020)", fontsize=10)
    ax.set_ylabel("RMSE Score", fontsize=10)
    ax.set_title(
        f"Normal Validation (train3) Anomaly Score Timeline & Primary Q99.9 Alarm Events\n"
        f"[Total Scored Exposure: {val_exposure_hours:.1f} Hours | {primary_thresh_info['false_alarm_events_count']} False Alarm Events ({primary_thresh_info['false_alarm_events_per_hour']} ev/h)]",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "fig_02_validation_score_timeline.png"
    plt.savefig(fig2_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig2_path}")

    # Figure 3: Validation False-Alarm Event Duration Distribution
    primary_events = extract_alarm_events(val_alarm_mask)
    primary_durations = [e["duration_seconds"] for e in primary_events]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Validation False-Alarm Event Durations (Primary Threshold Q99.9 = {primary_threshold:.4f})\n"
        f"[Total Events = {len(primary_events)} | Mean = {primary_thresh_info['mean_event_duration_seconds']}s | Median = {primary_thresh_info['median_event_duration_seconds']}s | Max = {primary_thresh_info['max_event_duration_seconds']}s]",
        fontsize=11,
        fontweight="bold",
    )

    ax1.hist(primary_durations, bins=20, color="#d35400", edgecolor="black", alpha=0.75)
    ax1.set_xlabel("Event Duration (Seconds)", fontsize=10)
    ax1.set_ylabel("Event Count", fontsize=10)
    ax1.set_title("Histogram of False Alarm Event Durations", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.boxplot(primary_durations, patch_artist=True, boxprops=dict(facecolor="#edbb99"), medianprops=dict(color="black", lw=1.5))
    ax2.set_ylabel("Duration (Seconds)", fontsize=10)
    ax2.set_xticks([1])
    ax2.set_xticklabels([f"Q99.9 Events (N={len(primary_events)})"])
    ax2.set_title(f"Boxplot of Event Durations (Range: {min(primary_durations) if primary_durations else 0}s - {max(primary_durations) if primary_durations else 0}s)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig3_path = FIGURES_DIR / "fig_03_false_alarm_event_durations.png"
    plt.savefig(fig3_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig3_path}")

    # Figure 4: Top Residual Feature Contributors during Normal Validation
    fig, ax = plt.subplots(figsize=(13, 7))
    feat_series = pd.Series(val_feat_rmse, index=feature_names).sort_values(ascending=True)
    top_contributors = feat_series.tail(25)

    subsystem_colors = {"P1": "#1f77b4", "P2": "#ff7f0e", "P3": "#2ca02c", "P4": "#d62728"}
    bar_colors = [subsystem_colors.get(c.split("_")[0], "#7f7f7f") for c in top_contributors.index]

    ax.barh(top_contributors.index, top_contributors.values, color=bar_colors, edgecolor="none")
    ax.set_xlabel("Mean Root Mean Squared Step Difference (Standardized Units)", fontsize=10)
    ax.set_title("Top 25 Feature Contributors to One-Step Persistence Residuals (train3 Validation)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    # Legend for subsystems
    handles = [plt.Rectangle((0,0),1,1, color=subsystem_colors[k]) for k in ["P1", "P2", "P3", "P4"]]
    ax.legend(handles, ["P1 (Boiler)", "P2 (Turbine)", "P3 (Water)", "P4 (HIL)"], loc="lower right", fontsize=9)

    plt.tight_layout()
    fig4_path = FIGURES_DIR / "fig_04_top_residual_contributors.png"
    plt.savefig(fig4_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig4_path}")

    peak_mem = max(peak_mem, get_memory_mb())
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    logger.info("=================================================================")
    logger.info("Baseline 0 Execution Completed Successfully")
    logger.info(f"Primary Operating Point: Q99.9 Threshold = {primary_threshold:.6f}")
    logger.info(f"Validation False Alarm Event Rate: {primary_thresh_info['false_alarm_events_per_hour']} events/hour")
    logger.info(f"Elapsed Time: {elapsed:.2f} seconds")
    logger.info(f"Peak Process Memory: {peak_mem:.2f} MB")
    logger.info("=================================================================")

    return {
        "primary_threshold": primary_threshold,
        "primary_events_per_hour": primary_thresh_info["false_alarm_events_per_hour"],
        "peak_mem_mb": round(peak_mem, 2),
        "elapsed_sec": round(elapsed, 2),
    }


if __name__ == "__main__":
    main()
