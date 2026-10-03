#!/usr/bin/env python3
"""
Baseline 1: One-Step Multi-Output Ridge Predictor for DT049 (HAI 21.03)
======================================================================
Evaluates linear transition model: X_hat[t+1] = W * X[t] + b using Ridge regression.
Primary Anomaly Score: RMSE across all 79 standardized feature residuals.

Methodological rules:
1. Strict session boundaries: train1, train2, and train3 transitions are created
   independently; never bridge from the end of train1 to the start of train2.
2. Alpha candidate selection is conducted EXCLUSIVELY on Normal Validation (train3).
3. Tie-breaking rule: if candidate validation mean RMSEs are numerically tied within
   documented tolerance (np.isclose, atol=1e-6, rtol=1e-5), choose the smaller alpha.
4. Threshold calibration is performed on train3 validation scores using predetermined
   quantiles (Q99.5, Q99.9 primary, Q99.99) with strict inequality (score > threshold).
5. All 79 features are preserved; 22 train-zero-variance features receive W=0 weights.
6. Zero test files (test1-test5) or attack labels are accessed or evaluated.
"""

import os
import sys
import gc
import json
import time
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime, timezone
import psutil
import joblib
import sklearn
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "baseline_ridge.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "hai_21_03"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "baseline_ridge"
MODELS_BASELINES_DIR = REPO_ROOT / "models" / "baselines"
MODELS_PREPROC_DIR = REPO_ROOT / "models" / "preprocessing"
HAI_COMMIT_HASH = "2a814cebc9a66b06c9e5cd545e2d72e65d383737"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("HAI2103_Ridge")
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
        "mean": float(np.mean(scores.astype(np.float64))),
        "std": float(np.std(scores.astype(np.float64))),
        "median": float(np.median(scores)),
        "q25": float(np.quantile(scores, 0.25, method="linear")),
        "q75": float(np.quantile(scores, 0.75, method="linear")),
        "q95": float(np.quantile(scores, 0.95, method="linear")),
        "q99": float(np.quantile(scores, 0.99, method="linear")),
        "q99_5": float(np.quantile(scores, 0.995, method="linear")),
        "q99_9": float(np.quantile(scores, 0.999, method="linear")),
        "q99_99": float(np.quantile(scores, 0.9999, method="linear")),
    }


def evaluate_ridge_scores(
    model: Ridge,
    features_mmap: np.ndarray,
    chunk_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates Ridge predictions in chunks to conserve memory:
    X_hat[t+1] = X[t] @ W.T + b.T
    residual = X[t+1] - X_hat[t+1]
    Uses float64 accumulation for squared errors.
    Returns: (rmse_scores, mae_scores, max_abs_res, mean_feature_rmse)
    """
    total_rows, num_features = features_mmap.shape
    num_scores = total_rows - 1

    rmse_scores = np.empty(num_scores, dtype=np.float32)
    mae_scores = np.empty(num_scores, dtype=np.float32)
    max_abs_res = np.empty(num_scores, dtype=np.float32)
    feature_sq_err_sum = np.zeros(num_features, dtype=np.float64)

    # Pre-extract weights for fast matrix multiplication
    W_T = model.coef_.T.astype(np.float32)  # shape (79, 79)
    intercept = model.intercept_.astype(np.float32)  # shape (79,)

    for start_idx in range(0, num_scores, chunk_size):
        end_idx = min(start_idx + chunk_size, num_scores)
        x_prev = features_mmap[start_idx:end_idx]          # shape (chunk_len, 79)
        x_curr = features_mmap[start_idx + 1:end_idx + 1]  # shape (chunk_len, 79)

        # One-step prediction
        x_pred = np.matmul(x_prev, W_T) + intercept
        residual = x_curr - x_pred  # shape (chunk_len, 79)

        # float64 accumulation for squared errors
        res_64 = residual.astype(np.float64)
        diff_sq = res_64 ** 2
        feature_sq_err_sum += np.sum(diff_sq, axis=0)

        chunk_rmse = np.sqrt(np.mean(diff_sq, axis=1)).astype(np.float32)
        chunk_mae = np.mean(np.abs(res_64), axis=1).astype(np.float32)
        chunk_max_res = np.max(np.abs(res_64), axis=1).astype(np.float32)

        rmse_scores[start_idx:end_idx] = chunk_rmse
        mae_scores[start_idx:end_idx] = chunk_mae
        max_abs_res[start_idx:end_idx] = chunk_max_res

    mean_feature_rmse = np.sqrt(feature_sq_err_sum / num_scores).astype(np.float32)
    return rmse_scores, mae_scores, max_abs_res, mean_feature_rmse


def main():
    logger = setup_logger()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    start_rss = get_memory_mb()
    peak_rss = start_rss

    logger.info("=================================================================")
    logger.info("Starting Baseline 1: One-Step Multi-Output Ridge Predictor")
    logger.info(f"Initial Memory Usage: {start_rss:.2f} MB")
    logger.info("=================================================================")

    config = load_config()

    # Load scaler metadata for ordered feature names
    scaler_meta_path = MODELS_PREPROC_DIR / "scaler_metadata.json"
    with open(scaler_meta_path, "r", encoding="utf-8") as f:
        scaler_meta = json.load(f)
    feature_names = scaler_meta["ordered_feature_names"]
    num_features = len(feature_names)
    zero_var_features = scaler_meta["zero_variance_training_features"]
    zero_var_indices = [feature_names.index(f) for f in zero_var_features]
    logger.info(f"Loaded feature metadata: {num_features} features ({len(zero_var_features)} zero-variance).")

    # 1. Build Training Transition Matrices without crossing session boundaries
    logger.info("Constructing Development Training transition matrices from train1 and train2...")
    t1_f = np.load(PROCESSED_DIR / "train1_features.npy", mmap_mode="r")
    t2_f = np.load(PROCESSED_DIR / "train2_features.npy", mmap_mode="r")

    n1_trans = len(t1_f) - 1  # 216,000
    n2_trans = len(t2_f) - 1  # 226,800
    total_train_trans = n1_trans + n2_trans  # 442,800
    train_exposure_hours = total_train_trans / 3600.0

    X_train = np.empty((total_train_trans, num_features), dtype=np.float32)
    Y_train = np.empty((total_train_trans, num_features), dtype=np.float32)

    # Slice train1 transitions
    X_train[:n1_trans] = t1_f[:-1]
    Y_train[:n1_trans] = t1_f[1:]

    # Slice train2 transitions
    X_train[n1_trans:] = t2_f[:-1]
    Y_train[n1_trans:] = t2_f[1:]

    del t1_f, t2_f
    gc.collect()

    peak_rss = max(peak_rss, get_memory_mb())
    logger.info(f"Assembled X_train: shape={X_train.shape}, dtype={X_train.dtype} ({X_train.nbytes / 1e6:.2f} MB)")
    logger.info(f"Assembled Y_train: shape={Y_train.shape}, dtype={Y_train.dtype} ({Y_train.nbytes / 1e6:.2f} MB)")
    logger.info(f"Total training transitions: {total_train_trans:,} ({train_exposure_hours:.2f} hours; train1: {n1_trans:,}, train2: {n2_trans:,}).")
    logger.info(f"Current RSS: {get_memory_mb():.2f} MB")

    # Normal Validation session (train3)
    t3_f = np.load(PROCESSED_DIR / "train3_features.npy", mmap_mode="r")
    t3_ts = np.load(PROCESSED_DIR / "train3_timestamps.npy", mmap_mode="r")
    val_trans_count = len(t3_f) - 1  # 478,800
    val_exposure_hours = val_trans_count / 3600.0
    val_scored_timestamps = t3_ts[1:]
    logger.info(f"Loaded Normal Validation session (train3): {val_trans_count:,} transitions ({val_exposure_hours:.2f} hours).")

    # 2. Model Selection over Predetermined Alpha Candidates
    alpha_candidates = config["model"]["alpha_candidates"]
    tie_atol = float(config["model"]["tie_tolerance"]["atol"])
    tie_rtol = float(config["model"]["tie_tolerance"]["rtol"])
    logger.info("=================================================================")
    logger.info(f"Evaluating Candidate Alphas on Normal Validation: {alpha_candidates}")
    logger.info(f"Tie-break rule: np.isclose(cand, best, atol={tie_atol}, rtol={tie_rtol}) -> choose smaller alpha")

    selection_records = []
    candidate_models: Dict[float, Ridge] = {}
    candidate_val_rmse_scores: Dict[float, np.ndarray] = {}

    for alpha in alpha_candidates:
        fit_t0 = time.time()
        model = Ridge(alpha=alpha, solver="cholesky", fit_intercept=True)
        model.fit(X_train, Y_train)
        fit_duration = time.time() - fit_t0

        eval_t0 = time.time()
        val_rmse, val_mae, val_max_res, _ = evaluate_ridge_scores(model, t3_f)
        eval_duration = time.time() - eval_t0

        # High-precision float64 accumulation for mean RMSE
        val_mean_rmse = float(np.mean(val_rmse.astype(np.float64)))
        val_median_rmse = float(np.median(val_rmse))
        val_q99_rmse = float(np.quantile(val_rmse, 0.99, method="linear"))
        val_max_rmse = float(np.max(val_rmse))

        # Check weights for 22 zero-variance features
        zero_col_coef_max = float(np.max(np.abs(model.coef_[:, zero_var_indices])))

        logger.info(
            f"Alpha = {alpha:>6} | Val Mean RMSE = {val_mean_rmse:.8f} | "
            f"Median = {val_median_rmse:.6f} | Q99 = {val_q99_rmse:.6f} | "
            f"Fit Time = {fit_duration:.3f}s | Zero-var weight max = {zero_col_coef_max:.1e}"
        )

        selection_records.append({
            "alpha": alpha,
            "val_mean_rmse": val_mean_rmse,
            "val_median_rmse": val_median_rmse,
            "val_q99_rmse": val_q99_rmse,
            "val_max_rmse": val_max_rmse,
            "zero_variance_coef_max": zero_col_coef_max,
            "fit_duration_seconds": round(fit_duration, 4),
            "eval_duration_seconds": round(eval_duration, 4),
        })

        candidate_models[alpha] = model
        candidate_val_rmse_scores[alpha] = val_rmse
        peak_rss = max(peak_rss, get_memory_mb())

    # Save model selection CSV
    selection_df = pd.DataFrame(selection_records)
    selection_csv_path = METRICS_DIR / "baseline_ridge_model_selection.csv"
    selection_df.to_csv(selection_csv_path, index=False)
    logger.info(f"Saved candidate selection table: {selection_csv_path}")

    # Determine best alpha with exact tie-breaking logic
    # Find absolute minimum mean RMSE
    best_record = min(selection_records, key=lambda r: r["val_mean_rmse"])
    min_mean_rmse = best_record["val_mean_rmse"]

    # Check for numerical ties within tolerance
    tied_candidates = []
    for r in selection_records:
        cand_alpha = r["alpha"]
        cand_rmse = r["val_mean_rmse"]
        if np.isclose(cand_rmse, min_mean_rmse, atol=tie_atol, rtol=tie_rtol):
            tied_candidates.append(cand_alpha)

    # Tie-break rule: choose smaller alpha among tied candidates
    selected_alpha = min(tied_candidates)
    best_model = candidate_models[selected_alpha]
    selected_val_rmse = candidate_val_rmse_scores[selected_alpha]

    logger.info("=================================================================")
    logger.info(f"MODEL SELECTION RESULT:")
    logger.info(f"Min Validation Mean RMSE: {min_mean_rmse:.8f}")
    logger.info(f"Tied candidates within tolerance (atol={tie_atol}): {tied_candidates}")
    logger.info(f"SELECTED ALPHA: {selected_alpha} (Tie-break rule applied: smaller alpha)")
    logger.info(f"Selected Model Coefficient Shape: {best_model.coef_.shape}")
    logger.info(f"Selected Model Intercept Shape: {best_model.intercept_.shape}")
    logger.info("=================================================================")

    # Free non-selected candidate scores to conserve memory
    for alpha in alpha_candidates:
        if alpha != selected_alpha:
            del candidate_val_rmse_scores[alpha]
            del candidate_models[alpha]
    gc.collect()

    # 3. Save Selected Ridge Model and Metadata
    model_save_path = MODELS_BASELINES_DIR / "ridge.joblib"
    joblib.dump(best_model, model_save_path)
    logger.info(f"Saved frozen Ridge model: {model_save_path}")

    meta_save_path = MODELS_BASELINES_DIR / "ridge_metadata.json"
    ridge_metadata = {
        "model_name": "one_step_ridge",
        "version": config["version"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "hai_commit_reference": HAI_COMMIT_HASH,
        "preprocessing_metadata_reference": "models/preprocessing/scaler_metadata.json",
        "preprocessing_manifest": "results/audit/hai_21_03_preprocessing_manifest.csv",
        "solver": "cholesky",
        "fit_intercept": True,
        "selected_alpha": selected_alpha,
        "candidate_alphas": alpha_candidates,
        "selection_criterion": "lowest_validation_mean_rmse",
        "tie_tolerance": {"atol": tie_atol, "rtol": tie_rtol},
        "tied_candidates": tied_candidates,
        "selected_alpha_val_mean_rmse": min_mean_rmse,
        "feature_count": num_features,
        "ordered_feature_names": feature_names,
        "zero_variance_features_count": len(zero_var_features),
        "zero_variance_features": zero_var_features,
        "zero_variance_handling": "22 train-zero-variance columns are identically zero after training scaling and receive zero learned input contribution (W=0) under primary Ridge fit",
        "source_training_sessions": ["train1", "train2"],
        "training_transition_count": total_train_trans,
        "normal_validation_transition_count": val_trans_count,
        "model_coefficient_shape": list(best_model.coef_.shape),
        "model_intercept_shape": list(best_model.intercept_.shape),
    }
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(ridge_metadata, f, indent=2)
    logger.info(f"Saved Ridge metadata: {meta_save_path}")

    # 4. Compute Comprehensive Ridge Anomaly Scores on Training & Validation
    logger.info("Computing Ridge anomaly scores on Development Training sessions...")
    t1_f_raw = np.load(PROCESSED_DIR / "train1_features.npy", mmap_mode="r")
    t2_f_raw = np.load(PROCESSED_DIR / "train2_features.npy", mmap_mode="r")

    t1_rmse, t1_mae, t1_max_res, t1_feat_rmse = evaluate_ridge_scores(best_model, t1_f_raw)
    t2_rmse, t2_mae, t2_max_res, t2_feat_rmse = evaluate_ridge_scores(best_model, t2_f_raw)
    del t1_f_raw, t2_f_raw
    gc.collect()

    train_rmse_scores = np.concatenate([t1_rmse, t2_rmse])
    del X_train, Y_train
    gc.collect()

    # Full evaluation on Validation
    val_rmse_scores, val_mae_scores, val_max_res, val_feat_rmse = evaluate_ridge_scores(best_model, t3_f)

    train_dist = compute_distribution_summary(train_rmse_scores)
    val_dist = compute_distribution_summary(val_rmse_scores)

    logger.info(f"Ridge Training Scores: {len(train_rmse_scores):,} samples | Mean={train_dist['mean']:.4f}, Median={train_dist['median']:.4f}, Max={train_dist['max']:.4f}")
    logger.info(f"Ridge Validation Scores: {len(val_rmse_scores):,} samples | Mean={val_dist['mean']:.4f}, Median={val_dist['median']:.4f}, Max={val_dist['max']:.4f}")

    # 5. Threshold Calibration on Normal Validation (train3)
    logger.info("=================================================================")
    logger.info("Calibrating Operating Thresholds on Normal Validation (train3)...")
    operating_targets = config["threshold_calibration"]["operating_targets"]
    validation_results = []
    calibrated_thresholds = {}

    for target_cfg in operating_targets:
        q_name = target_cfg["name"]
        q_val = float(target_cfg["quantile"])
        role = target_cfg["role"]

        # Empirical quantile on normal validation (train3)
        threshold_val = float(np.quantile(val_rmse_scores, q_val, method="linear"))

        # Validation alarm events (strict inequality: score > threshold)
        is_alarm_val = (val_rmse_scores > threshold_val)
        val_point_exceedances = int(np.sum(is_alarm_val))
        val_point_exceedance_rate = val_point_exceedances / val_trans_count
        val_alarm_seconds = val_point_exceedances
        val_alarm_sec_per_hour = val_alarm_seconds / val_exposure_hours

        val_events = extract_alarm_events(is_alarm_val)
        val_event_count = len(val_events)
        val_events_per_hour = val_event_count / val_exposure_hours
        val_durations = [e["duration_seconds"] for e in val_events]
        val_mean_dur = float(np.mean(val_durations)) if val_durations else 0.0
        val_med_dur = float(np.median(val_durations)) if val_durations else 0.0
        val_max_dur = int(np.max(val_durations)) if val_durations else 0

        # Development Training evaluation on same threshold (without session bridging)
        is_alarm_t1 = (t1_rmse > threshold_val)
        is_alarm_t2 = (t2_rmse > threshold_val)
        train_point_exceedances = int(np.sum(is_alarm_t1) + np.sum(is_alarm_t2))
        train_point_exceedance_rate = train_point_exceedances / total_train_trans
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
            "point_exceedance_count": val_point_exceedances,
            "point_exceedance_rate": val_point_exceedance_rate,
            "alarm_seconds": val_alarm_seconds,
            "alarm_seconds_per_hour": round(val_alarm_sec_per_hour, 4),
            "false_alarm_events_count": val_event_count,
            "false_alarm_events_per_hour": round(val_events_per_hour, 4),
            "mean_event_duration_seconds": round(val_mean_dur, 2),
            "median_event_duration_seconds": round(val_med_dur, 2),
            "max_event_duration_seconds": val_max_dur,
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
            "val_point_exceedance_count": val_point_exceedances,
            "val_point_exceedance_rate": round(val_point_exceedance_rate, 8),
            "val_alarm_seconds": val_alarm_seconds,
            "val_alarm_seconds_per_hour": round(val_alarm_sec_per_hour, 4),
            "val_false_alarm_events_count": val_event_count,
            "val_false_alarm_events_per_hour": round(val_events_per_hour, 4),
            "val_mean_duration_seconds": round(val_mean_dur, 2),
            "val_median_duration_seconds": round(val_med_dur, 2),
            "val_max_duration_seconds": val_max_dur,
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
    val_metrics_csv_path = METRICS_DIR / "baseline_ridge_validation.csv"
    val_metrics_df.to_csv(val_metrics_csv_path, index=False)
    logger.info(f"Saved validation metrics table: {val_metrics_csv_path}")

    # Primary operating point
    primary_thresh_info = calibrated_thresholds["Q99.9"]
    primary_threshold = primary_thresh_info["threshold"]

    # Save Comprehensive Reproducibility JSON
    json_payload = {
        "metadata": {
            "baseline_model": "one_step_ridge",
            "version": config["version"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "hai_commit_reference": HAI_COMMIT_HASH,
            "preprocessing_manifest": "results/audit/hai_21_03_preprocessing_manifest.csv",
            "scaler_metadata": "models/preprocessing/scaler_metadata.json",
            "ridge_metadata": "models/baselines/ridge_metadata.json",
            "primary_operating_threshold": "Q99.9",
            "primary_threshold_value": primary_threshold,
            "selected_alpha": selected_alpha,
            "feature_count": num_features,
            "zero_variance_features_count": len(zero_var_features),
            "zero_variance_handling": "scaler scale=1.0, learned Ridge weights identically zero",
        },
        "scored_exposure": {
            "development_training_samples": total_train_trans,
            "development_training_exposure_seconds": total_train_trans,
            "development_training_exposure_hours": train_exposure_hours,
            "normal_validation_samples": val_trans_count,
            "normal_validation_exposure_seconds": val_trans_count,
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

    json_path = METRICS_DIR / "baseline_ridge_thresholds.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    logger.info(f"Saved threshold JSON: {json_path}")

    # 6. Direct Side-by-Side Comparison with Baseline 0 (Persistence)
    logger.info("Building direct comparison table: Persistence vs. Ridge...")
    persistence_json_path = METRICS_DIR / "baseline_persistence_thresholds.json"
    with open(persistence_json_path, "r", encoding="utf-8") as f:
        persist_data = json.load(f)

    p_val_dist = persist_data["score_distributions"]["normal_validation_rmse"]
    p_q999 = persist_data["threshold_calibration"]["Q99.9"]
    r_val_dist = val_dist
    r_q999 = primary_thresh_info

    comparison_rows = [
        {"metric": "Validation Scored Exposure (Hours)", "persistence_baseline_0": 133.0, "ridge_baseline_1": 133.0, "delta_ridge_vs_persistence": "0.0"},
        {"metric": "Validation Scored Transitions", "persistence_baseline_0": 478800, "ridge_baseline_1": 478800, "delta_ridge_vs_persistence": "0"},
        {"metric": "Validation Mean RMSE", "persistence_baseline_0": round(p_val_dist["mean"], 6), "ridge_baseline_1": round(r_val_dist["mean"], 6), "delta_ridge_vs_persistence": f"{round(r_val_dist['mean'] - p_val_dist['mean'], 6):+}"},
        {"metric": "Validation Median RMSE", "persistence_baseline_0": round(p_val_dist["median"], 6), "ridge_baseline_1": round(r_val_dist["median"], 6), "delta_ridge_vs_persistence": f"{round(r_val_dist['median'] - p_val_dist['median'], 6):+}"},
        {"metric": "Validation Std Dev RMSE", "persistence_baseline_0": round(p_val_dist["std"], 6), "ridge_baseline_1": round(r_val_dist["std"], 6), "delta_ridge_vs_persistence": f"{round(r_val_dist['std'] - p_val_dist['std'], 6):+}"},
        {"metric": "Validation Q99 RMSE", "persistence_baseline_0": round(p_val_dist["q99"], 6), "ridge_baseline_1": round(r_val_dist["q99"], 6), "delta_ridge_vs_persistence": f"{round(r_val_dist['q99'] - p_val_dist['q99'], 6):+}"},
        {"metric": "Validation Q99.9 RMSE (Primary Threshold)", "persistence_baseline_0": round(p_q999["threshold"], 6), "ridge_baseline_1": round(r_q999["threshold"], 6), "delta_ridge_vs_persistence": f"{round(r_q999['threshold'] - p_q999['threshold'], 6):+}"},
        {"metric": "Validation Q99.9 Point Exceedances (pts)", "persistence_baseline_0": p_q999["point_exceedance_count"], "ridge_baseline_1": r_q999["point_exceedance_count"], "delta_ridge_vs_persistence": f"{r_q999['point_exceedance_count'] - p_q999['point_exceedance_count']:+}"},
        {"metric": "Validation Q99.9 Alarm Seconds / Hour", "persistence_baseline_0": p_q999["alarm_seconds_per_hour"], "ridge_baseline_1": r_q999["alarm_seconds_per_hour"], "delta_ridge_vs_persistence": f"{round(r_q999['alarm_seconds_per_hour'] - p_q999['alarm_seconds_per_hour'], 4):+}"},
        {"metric": "Validation Q99.9 False Alarm Events (count)", "persistence_baseline_0": p_q999["false_alarm_events_count"], "ridge_baseline_1": r_q999["false_alarm_events_count"], "delta_ridge_vs_persistence": f"{r_q999['false_alarm_events_count'] - p_q999['false_alarm_events_count']:+}"},
        {"metric": "Validation Q99.9 False Alarm Rate (events/hour)", "persistence_baseline_0": p_q999["false_alarm_events_per_hour"], "ridge_baseline_1": r_q999["false_alarm_events_per_hour"], "delta_ridge_vs_persistence": f"{round(r_q999['false_alarm_events_per_hour'] - p_q999['false_alarm_events_per_hour'], 4):+}"},
        {"metric": "Validation Q99.9 Mean Event Duration (s)", "persistence_baseline_0": p_q999["mean_event_duration_seconds"], "ridge_baseline_1": r_q999["mean_event_duration_seconds"], "delta_ridge_vs_persistence": f"{round(r_q999['mean_event_duration_seconds'] - p_q999['mean_event_duration_seconds'], 2):+}"},
        {"metric": "Validation Q99.9 Median Event Duration (s)", "persistence_baseline_0": p_q999["median_event_duration_seconds"], "ridge_baseline_1": r_q999["median_event_duration_seconds"], "delta_ridge_vs_persistence": f"{round(r_q999['median_event_duration_seconds'] - p_q999['median_event_duration_seconds'], 2):+}"},
        {"metric": "Validation Q99.9 Max Event Duration (s)", "persistence_baseline_0": p_q999["max_event_duration_seconds"], "ridge_baseline_1": r_q999["max_event_duration_seconds"], "delta_ridge_vs_persistence": f"{r_q999['max_event_duration_seconds'] - p_q999['max_event_duration_seconds']:+}"},
    ]
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = METRICS_DIR / "baseline_comparison_validation.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    logger.info(f"Saved direct comparison table: {comp_csv_path}")

    # 7. Render Diagnostic Figures
    logger.info("Rendering Baseline 1 Diagnostic Figures...")

    # Figure 1: Validation Mean RMSE vs. Alpha
    fig, ax = plt.subplots(figsize=(8, 4.5))
    alphas = [r["alpha"] for r in selection_records]
    val_rmses = [r["val_mean_rmse"] for r in selection_records]
    ax.plot(alphas, val_rmses, marker="o", color="#2980b9", lw=2, markersize=8, label="Validation Mean RMSE")
    ax.scatter([selected_alpha], [min_mean_rmse], color="#e74c3c", s=140, zorder=5, label=f"Selected alpha={selected_alpha} ({min_mean_rmse:.6f})")
    ax.set_xscale("log")
    ax.set_xlabel("Ridge Regularization Alpha (log scale)", fontsize=10)
    ax.set_ylabel("Validation Mean RMSE", fontsize=10)
    ax.set_title("Baseline 1: Alpha Selection on Normal Validation (train3)\n[Criterion: Lowest Mean RMSE across 478,800 transitions]", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "fig_01_alpha_selection_rmse.png"
    plt.savefig(fig1_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig1_path}")

    # Figure 2: Persistence vs. Ridge Score Distribution Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Baseline Comparison: Score Distributions on Normal Validation (train3)\n[Persistence vs. Ridge (alpha=selected)]", fontsize=12, fontweight="bold")

    # Load persistence val scores for plot
    p_val_f = np.load(PROCESSED_DIR / "train3_features.npy", mmap_mode="r")
    p_diff = np.diff(p_val_f, axis=0)
    p_rmse_scores = np.sqrt(np.mean(p_diff ** 2, axis=1)).astype(np.float32)
    del p_diff, p_val_f
    gc.collect()

    ax1.hist(p_rmse_scores, bins=100, density=True, color="#2b5c8f", alpha=0.6, label=f"Persistence (Mean={p_val_dist['mean']:.4f})")
    ax1.hist(val_rmse_scores, bins=100, density=True, color="#27ae60", alpha=0.6, label=f"Ridge (Mean={r_val_dist['mean']:.4f})")
    ax1.axvline(primary_threshold, color="#e74c3c", linestyle="--", lw=1.5, label=f"Ridge Q99.9 ({primary_threshold:.4f})")
    ax1.axvline(p_q999["threshold"], color="#2b5c8f", linestyle=":", lw=1.5, label=f"Persist Q99.9 ({p_q999['threshold']:.4f})")
    ax1.set_xlabel("RMSE Anomaly Score", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.set_title("Full Distribution Density", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Tail log-density
    ax2.hist(p_rmse_scores, bins=100, density=True, color="#2b5c8f", alpha=0.6, label="Persistence", log=True)
    ax2.hist(val_rmse_scores, bins=100, density=True, color="#27ae60", alpha=0.6, label="Ridge", log=True)
    ax2.axvline(calibrated_thresholds["Q99.5"]["threshold"], color="blue", linestyle=":", lw=1.2, label=f"Ridge Q99.5 ({calibrated_thresholds['Q99.5']['threshold']:.4f})")
    ax2.axvline(primary_threshold, color="#e74c3c", linestyle="--", lw=1.5, label=f"Ridge Q99.9 ({primary_threshold:.4f})")
    ax2.axvline(calibrated_thresholds["Q99.99"]["threshold"], color="darkred", linestyle=":", lw=1.2, label=f"Ridge Q99.99 ({calibrated_thresholds['Q99.99']['threshold']:.4f})")
    ax2.set_xlabel("RMSE Anomaly Score", fontsize=10)
    ax2.set_ylabel("Log Density", fontsize=10)
    ax2.set_title("Extreme Tail Log-Density & Ridge Operating Thresholds", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "fig_02_persistence_vs_ridge_distributions.png"
    plt.savefig(fig2_path, dpi=180)
    plt.close()
    del p_rmse_scores
    gc.collect()
    logger.info(f"Saved: {fig2_path}")

    # Figure 3: Ridge Validation Timeline with Q99.9 Threshold
    fig, ax = plt.subplots(figsize=(14, 5))
    downsample_factor = 20
    val_ts_down = pd.to_datetime(val_scored_timestamps[::downsample_factor])
    val_scores_down = val_rmse_scores[::downsample_factor]

    ax.plot(val_ts_down, val_scores_down, color="#1e8449", lw=0.8, alpha=0.85, label="Ridge RMSE (train3 normal val)")
    ax.axhline(primary_threshold, color="#c0392b", linestyle="--", lw=1.8, label=f"Primary Threshold Q99.9 = {primary_threshold:.4f} ({primary_thresh_info['false_alarm_events_per_hour']} ev/h)")

    # Scatter false alarm points
    val_alarm_mask = (val_rmse_scores > primary_threshold)
    alarm_ts = pd.to_datetime(val_scored_timestamps[val_alarm_mask])
    alarm_scores = val_rmse_scores[val_alarm_mask]
    ax.scatter(alarm_ts, alarm_scores, color="#e74c3c", s=18, zorder=5, label=f"Ridge False Alarm Points ({len(alarm_ts)} pts)")

    ax.set_xlabel("Timeline (August 04, 2020 - August 10, 2020)", fontsize=10)
    ax.set_ylabel("RMSE Score", fontsize=10)
    ax.set_title(
        f"Ridge Validation Anomaly Score Timeline & Primary Q99.9 Alarm Events\n"
        f"[Total Scored Exposure: {val_exposure_hours:.1f} Hours | {primary_thresh_info['false_alarm_events_count']} False Alarm Events ({primary_thresh_info['false_alarm_events_per_hour']} ev/h)]",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "fig_03_ridge_validation_timeline.png"
    plt.savefig(fig3_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig3_path}")

    # Figure 4: Ridge False-Alarm Event Duration Distribution
    primary_events = extract_alarm_events(val_alarm_mask)
    primary_durations = [e["duration_seconds"] for e in primary_events]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Ridge Validation False-Alarm Event Durations (Primary Threshold Q99.9 = {primary_threshold:.4f})\n"
        f"[Total Events = {len(primary_events)} | Mean = {primary_thresh_info['mean_event_duration_seconds']}s | Median = {primary_thresh_info['median_event_duration_seconds']}s | Max = {primary_thresh_info['max_event_duration_seconds']}s]",
        fontsize=11,
        fontweight="bold",
    )

    ax1.hist(primary_durations, bins=20, color="#27ae60", edgecolor="black", alpha=0.75)
    ax1.set_xlabel("Event Duration (Seconds)", fontsize=10)
    ax1.set_ylabel("Event Count", fontsize=10)
    ax1.set_title("Histogram of False Alarm Event Durations", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.boxplot(primary_durations, patch_artist=True, boxprops=dict(facecolor="#a9dfbf"), medianprops=dict(color="black", lw=1.5))
    ax2.set_ylabel("Duration (Seconds)", fontsize=10)
    ax2.set_xticks([1])
    ax2.set_xticklabels([f"Ridge Q99.9 Events (N={len(primary_events)})"])
    ax2.set_title(f"Boxplot of Event Durations (Range: {min(primary_durations) if primary_durations else 0}s - {max(primary_durations) if primary_durations else 0}s)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig4_path = FIGURES_DIR / "fig_04_ridge_event_durations.png"
    plt.savefig(fig4_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig4_path}")

    # Figure 5: Top Residual Contributors on Normal Validation
    fig, ax = plt.subplots(figsize=(13, 7))
    feat_series = pd.Series(val_feat_rmse, index=feature_names).sort_values(ascending=True)
    top_contributors = feat_series.tail(25)

    subsystem_colors = {"P1": "#1f77b4", "P2": "#ff7f0e", "P3": "#2ca02c", "P4": "#d62728"}
    bar_colors = [subsystem_colors.get(c.split("_")[0], "#7f7f7f") for c in top_contributors.index]

    ax.barh(top_contributors.index, top_contributors.values, color=bar_colors, edgecolor="none")
    ax.set_xlabel("Mean Root Mean Squared Step Residual (Standardized Units)", fontsize=10)
    ax.set_title("Top 25 Feature Contributors to Ridge Prediction Residuals (train3 Validation)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    handles = [plt.Rectangle((0,0),1,1, color=subsystem_colors[k]) for k in ["P1", "P2", "P3", "P4"]]
    ax.legend(handles, ["P1 (Boiler)", "P2 (Turbine)", "P3 (Water)", "P4 (HIL)"], loc="lower right", fontsize=9)

    plt.tight_layout()
    fig5_path = FIGURES_DIR / "fig_05_top_residual_contributors.png"
    plt.savefig(fig5_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig5_path}")

    # Final execution accounting
    peak_rss = max(peak_rss, get_memory_mb())
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    logger.info("=================================================================")
    logger.info("Baseline 1 (Ridge) Execution Completed Successfully")
    logger.info(f"Selected Alpha: {selected_alpha}")
    logger.info(f"Primary Operating Point: Q99.9 Threshold = {primary_threshold:.6f}")
    logger.info(f"Validation False Alarm Event Rate: {primary_thresh_info['false_alarm_events_per_hour']} events/hour")
    logger.info(f"Elapsed Time: {elapsed:.2f} seconds")
    logger.info(f"Measured Peak Process RSS: {peak_rss:.2f} MB")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
