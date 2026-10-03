#!/usr/bin/env python3
"""
HAI 21.03 Exploratory Data Analysis & Leakage-Safe Split Script for DT049
=======================================================================
This script performs rigorous, leakage-safe feature characterization,
distribution drift analysis, split validation, and visual EDA for the
HAI 21.03 dataset.

CRITICAL METHODOLOGICAL RULES:
1. No models are trained, no scalers are fit, no thresholds are tuned.
2. Only Development Training (train1, train2) and Normal Validation (train3)
   feature data are read. Official test files (test1-test5) are NEVER accessed
   for feature values, correlations, or distributions.
3. All 79 candidate telemetry features are preserved (no feature pruning).
4. Timestamp continuity is preserved strictly within independent recording
   sessions; file boundaries are never concatenated.
5. Correlation heatmap uses a reproducible deterministic sample from
   TRAINING DATA ONLY with fixed random seed (seed=42).
6. Descriptive attack-duration plots use previously audited label manifests
   and are strictly labeled DESCRIPTIVE ONLY.
"""

import os
import sys
import gc
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import psutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "source" / "hai_repo" / "hai-21.03"
RESULTS_AUDIT_DIR = REPO_ROOT / "results" / "audit"
RESULTS_FIGURES_DIR = REPO_ROOT / "results" / "figures" / "eda"
DOCS_DIR = REPO_ROOT / "docs"

# Split Definitions
TRAIN_FILES = ["train1.csv.gz", "train2.csv.gz"]
VAL_FILES = ["train3.csv.gz"]
TEST_FILES = [f"test{i}.csv.gz" for i in range(1, 6)]

RANDOM_SEED = 42
TARGET_CORRELATION_SAMPLES = 45_000  # 22,500 from train1, 22,500 from train2


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("HAI2103_EDA")
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


def main():
    logger = setup_logger()
    RESULTS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    peak_mem = get_memory_mb()

    logger.info("=================================================================")
    logger.info("Starting HAI 21.03 EDA & Leakage-Safe Split Analysis for DT049")
    logger.info(f"Initial Memory Usage: {peak_mem:.2f} MB")
    logger.info("=================================================================")

    # -------------------------------------------------------------------------
    # 1. Schema & Candidate Features Identification
    # -------------------------------------------------------------------------
    sample_df = pd.read_csv(DATA_DIR / TRAIN_FILES[0], compression="gzip", nrows=5)
    all_cols = list(sample_df.columns)
    ground_truth_cols = ["time", "attack", "attack_P1", "attack_P2", "attack_P3"]
    candidate_features = [c for c in all_cols if c not in ground_truth_cols]

    logger.info(f"Total Columns: {len(all_cols)}")
    logger.info(f"Ground Truth / Metadata Columns: {ground_truth_cols}")
    logger.info(f"Candidate Telemetry Features: {len(candidate_features)}")

    # -------------------------------------------------------------------------
    # 2. Load Development Training Data (train1 + train2)
    # -------------------------------------------------------------------------
    logger.info("Loading Development Training Data (train1.csv.gz & train2.csv.gz)...")
    train1_df = pd.read_csv(DATA_DIR / "train1.csv.gz", compression="gzip", usecols=["time"] + candidate_features)
    train2_df = pd.read_csv(DATA_DIR / "train2.csv.gz", compression="gzip", usecols=["time"] + candidate_features)

    train1_features = train1_df[candidate_features]
    train2_features = train2_df[candidate_features]
    train_features = pd.concat([train1_features, train2_features], axis=0, ignore_index=True)

    peak_mem = max(peak_mem, get_memory_mb())
    logger.info(f"Training Data Loaded: train1 ({len(train1_df):,} rows), train2 ({len(train2_df):,} rows) -> Total ({len(train_features):,} rows)")
    logger.info(f"Current Process Memory: {get_memory_mb():.2f} MB")

    # -------------------------------------------------------------------------
    # 3. Load Normal Validation Data (train3)
    # -------------------------------------------------------------------------
    logger.info("Loading Normal Validation Data (train3.csv.gz)...")
    val_df = pd.read_csv(DATA_DIR / "train3.csv.gz", compression="gzip", usecols=["time"] + candidate_features)
    val_features = val_df[candidate_features]

    peak_mem = max(peak_mem, get_memory_mb())
    logger.info(f"Validation Data Loaded: train3 ({len(val_features):,} rows)")
    logger.info(f"Current Process Memory: {get_memory_mb():.2f} MB")

    # -------------------------------------------------------------------------
    # 4. Feature Characterization: Train Only & Train vs. Validation
    # -------------------------------------------------------------------------
    logger.info("Computing exact statistical characterizations on training data...")

    # Load descriptive audit for global constant reference
    audit_col_summary_path = RESULTS_AUDIT_DIR / "hai_21_03_column_summary.csv"
    globally_constant_audit_ref = set()
    if audit_col_summary_path.exists():
        aud_df = pd.read_csv(audit_col_summary_path)
        globally_constant_audit_ref = set(aud_df[aud_df["is_globally_constant"] == True]["column_name"].unique())

    stats_records = []
    retention_records = []

    # Vectorized / column-wise computation
    for col in candidate_features:
        s_t1 = train1_features[col]
        s_t2 = train2_features[col]
        s_train = train_features[col]
        s_val = val_features[col]

        subsystem = col.split("_")[0] if "_" in col else "OTHER"

        # Constancy per split
        const_t1 = bool(s_t1.min() == s_t1.max())
        const_t2 = bool(s_t2.min() == s_t2.max())
        const_train = bool(s_train.min() == s_train.max())
        const_val = bool(s_val.min() == s_val.max())
        const_train_val = const_train and const_val and (s_train.iloc[0] == s_val.iloc[0])

        # Train statistics
        t_min = float(s_train.min())
        t_max = float(s_train.max())
        t_mean = float(s_train.mean())
        t_std = float(s_train.std())
        t_q01 = float(s_train.quantile(0.01))
        t_q25 = float(s_train.quantile(0.25))
        t_med = float(s_train.median())
        t_q75 = float(s_train.quantile(0.75))
        t_q99 = float(s_train.quantile(0.99))
        t_iqr = t_q75 - t_q25
        t_dyn_range = t_q99 - t_q01
        t_unique = int(s_train.nunique())
        t_unique_ratio = t_unique / len(s_train)

        # Validation statistics
        v_min = float(s_val.min())
        v_max = float(s_val.max())
        v_mean = float(s_val.mean())
        v_std = float(s_val.std())
        v_med = float(s_val.median())
        v_iqr = float(s_val.quantile(0.75) - s_val.quantile(0.25))

        # Drift metrics (Train vs. Validation)
        mean_diff = abs(v_mean - t_mean)
        std_ratio = v_std / (t_std + 1e-12) if t_std > 0 else (1.0 if v_std == 0 else np.nan)
        normalized_shift = mean_diff / (t_std + 1e-12) if t_std > 0 else (0.0 if mean_diff == 0 else np.nan)

        # Descriptive Near-Zero Variance indicators
        nzv_flag = bool((t_std == 0.0) or (t_iqr == 0.0) or (t_unique_ratio < 0.0001))

        stats_records.append({
            "feature_name": col,
            "subsystem": subsystem,
            "train_min": t_min,
            "train_max": t_max,
            "train_mean": round(t_mean, 6),
            "train_std": round(t_std, 6),
            "train_q01": round(t_q01, 6),
            "train_q25": round(t_q25, 6),
            "train_median": round(t_med, 6),
            "train_q75": round(t_q75, 6),
            "train_q99": round(t_q99, 6),
            "train_iqr": round(t_iqr, 6),
            "train_range_99_01": round(t_dyn_range, 6),
            "train_unique_count": t_unique,
            "train_unique_ratio": round(t_unique_ratio, 6),
            "is_constant_train": const_train,
            "val_min": v_min,
            "val_max": v_max,
            "val_mean": round(v_mean, 6),
            "val_std": round(v_std, 6),
            "val_median": round(v_med, 6),
            "val_iqr": round(v_iqr, 6),
            "is_constant_val": const_val,
            "drift_mean_diff": round(mean_diff, 6),
            "drift_normalized_shift": round(normalized_shift, 4),
            "drift_std_ratio": round(std_ratio, 4),
        })

        retention_records.append({
            "feature_name": col,
            "subsystem": subsystem,
            "constant_in_train1": const_t1,
            "constant_in_train2": const_t2,
            "constant_in_training_set": const_train,
            "constant_in_validation": const_val,
            "constant_in_train_plus_val": const_train_val,
            "globally_constant_descriptive_audit": (col in globally_constant_audit_ref),
            "train_std": round(t_std, 6),
            "train_iqr": round(t_iqr, 6),
            "train_dynamic_range_99_01": round(t_dyn_range, 6),
            "train_unique_ratio": round(t_unique_ratio, 6),
            "near_zero_variance_descriptive": nzv_flag,
            "retention_status": "RETAINED (Preserved for modeling)",
        })

    train_stats_df = pd.DataFrame(stats_records)
    train_stats_df.to_csv(RESULTS_AUDIT_DIR / "hai_21_03_train_feature_stats.csv", index=False)
    logger.info(f"Saved: {RESULTS_AUDIT_DIR / 'hai_21_03_train_feature_stats.csv'}")

    retention_df = pd.DataFrame(retention_records)
    retention_df.to_csv(RESULTS_AUDIT_DIR / "hai_21_03_feature_retention_analysis.csv", index=False)
    logger.info(f"Saved: {RESULTS_AUDIT_DIR / 'hai_21_03_feature_retention_analysis.csv'}")

    # Log constancy findings
    train_constants = retention_df[retention_df["constant_in_training_set"] == True]["feature_name"].tolist()
    val_constants = retention_df[retention_df["constant_in_validation"] == True]["feature_name"].tolist()
    logger.info(f"Constant features in TRAINING SET ({len(train_constants)}): {train_constants}")
    logger.info(f"Constant features in VALIDATION SET ({len(val_constants)}): {val_constants}")

    # -------------------------------------------------------------------------
    # 5. Split Integrity & Calendar Overlap Analysis
    # -------------------------------------------------------------------------
    logger.info("Building Leakage-Safe Split Summary and Calendar Relationships...")
    audit_file_summary_path = RESULTS_AUDIT_DIR / "hai_21_03_file_summary.csv"
    audit_files_info = {}
    if audit_file_summary_path.exists():
        af_df = pd.read_csv(audit_file_summary_path)
        for _, row in af_df.iterrows():
            audit_files_info[row["filename"]] = {
                "first_timestamp": row["first_timestamp"],
                "last_timestamp": row["last_timestamp"],
                "duration_hours": row["duration_hours"],
                "total_rows": row["total_rows"],
                "attack_events": row["observed_attacks"],
                "attack_seconds": row["attack_seconds"],
            }

    split_records = [
        {
            "filename": "train1.csv.gz",
            "split_role": "Development training",
            "calendar_start": "2020-07-11 00:00:00",
            "calendar_end": "2020-07-13 12:00:00",
            "duration_hours": 60.0,
            "total_rows": 216001,
            "attack_events": 0,
            "attack_seconds": 0,
            "session_relationship": "Independent normal recording session 1; calendar-overlaps 12h with test3 on 2020-07-13.",
            "leakage_guard": "Strict file boundary; fit scalers on train only; no window crossing file bounds.",
        },
        {
            "filename": "train2.csv.gz",
            "split_role": "Development training",
            "calendar_start": "2020-07-31 22:00:00",
            "calendar_end": "2020-08-03 13:00:00",
            "duration_hours": 63.0,
            "total_rows": 226801,
            "attack_events": 0,
            "attack_seconds": 0,
            "session_relationship": "Independent normal recording session 2; recorded 18 days after train1 and 9.5 hours after test5.",
            "leakage_guard": "Strict file boundary; combined with train1 for training parameters only.",
        },
        {
            "filename": "train3.csv.gz",
            "split_role": "Normal validation",
            "calendar_start": "2020-08-04 22:00:00",
            "calendar_end": "2020-08-10 11:00:00",
            "duration_hours": 133.0,
            "total_rows": 478801,
            "attack_events": 0,
            "attack_seconds": 0,
            "session_relationship": "Independent normal validation session; recorded 1.38 days after train2; strictly normal (0 attacks).",
            "leakage_guard": "Normal validation loss and drift checks only; NEVER used for attack recall or threshold tuning.",
        },
        {
            "filename": "test1.csv.gz",
            "split_role": "Official final attack test",
            "calendar_start": "2020-07-07 15:00:00",
            "calendar_end": "2020-07-08 03:00:00",
            "duration_hours": 12.0,
            "total_rows": 43201,
            "attack_events": 5,
            "attack_seconds": 629,
            "session_relationship": "Independent test recording 1; calendar-precedes train1 by 2.88 days.",
            "leakage_guard": "Completely sealed; zero feature or label usage in model development.",
        },
        {
            "filename": "test2.csv.gz",
            "split_role": "Official final attack test",
            "calendar_start": "2020-07-09 15:00:00",
            "calendar_end": "2020-07-11 00:00:00",
            "duration_hours": 33.0,
            "total_rows": 118801,
            "attack_events": 20,
            "attack_seconds": 3449,
            "session_relationship": "Independent test recording 2; calendar-precedes train1 by 0 seconds (ends exactly as train1 begins).",
            "leakage_guard": "Completely sealed; zero feature or label usage in model development.",
        },
        {
            "filename": "test3.csv.gz",
            "split_role": "Official final attack test",
            "calendar_start": "2020-07-13 00:00:00",
            "calendar_end": "2020-07-14 06:00:00",
            "duration_hours": 30.0,
            "total_rows": 108001,
            "attack_events": 8,
            "attack_seconds": 1535,
            "session_relationship": "Independent test recording 3; calendar-overlaps 12h with train1 (different experimental runs on testbed).",
            "leakage_guard": "Completely sealed; zero feature or label usage in model development.",
        },
        {
            "filename": "test4.csv.gz",
            "split_role": "Official final attack test",
            "calendar_start": "2020-07-28 12:00:00",
            "calendar_end": "2020-07-28 23:00:00",
            "duration_hours": 11.0,
            "total_rows": 39601,
            "attack_events": 5,
            "attack_seconds": 1157,
            "session_relationship": "Independent test recording 4; recorded between train1 and train2.",
            "leakage_guard": "Completely sealed; zero feature or label usage in model development.",
        },
        {
            "filename": "test5.csv.gz",
            "split_role": "Official final attack test",
            "calendar_start": "2020-07-30 10:50:00",
            "calendar_end": "2020-07-31 12:30:00",
            "duration_hours": 25.67,
            "total_rows": 92401,
            "attack_events": 12,
            "attack_seconds": 2177,
            "session_relationship": "Independent test recording 5; ends 9.5 hours before train2 begins.",
            "leakage_guard": "Completely sealed; zero feature or label usage in model development.",
        },
    ]

    split_summary_df = pd.DataFrame(split_records)
    split_summary_df.to_csv(RESULTS_AUDIT_DIR / "hai_21_03_split_summary.csv", index=False)
    logger.info(f"Saved: {RESULTS_AUDIT_DIR / 'hai_21_03_split_summary.csv'}")

    # -------------------------------------------------------------------------
    # 6. Select Non-Constant Representative Signals (TRAINING ONLY)
    # -------------------------------------------------------------------------
    logger.info("Selecting non-constant representative signals from TRAINING DATA ONLY...")
    # Select from training data based on non-constancy and dynamic range
    subsystem_candidates = {
        "P1": [c for c in candidate_features if c.startswith("P1_") and c not in train_constants],
        "P2": [c for c in candidate_features if c.startswith("P2_") and c not in train_constants],
        "P3": [c for c in candidate_features if c.startswith("P3_") and c not in train_constants],
        "P4": [c for c in candidate_features if c.startswith("P4_") and c not in train_constants],
    }

    # Selected signals with documented rationale:
    # P1: P1_LIT01 (Boiler level transmitter - continuous physical process with rich variance)
    # P2: P2_CO_rpm (Turbine compressor RPM - dynamic rotating machinery signal; P2_VTR01 is constant and rejected!)
    # P3: P3_LIT01 (Water treatment tank level - continuous level indicator; P3_LH and P3_LL are constant and rejected!)
    # P4: P4_ST_PT01 (Steam turbine pressure transmitter - prominent thermodynamic sensor)
    representative_signals = {
        "P1": "P1_LIT01",
        "P2": "P2_CO_rpm",
        "P3": "P3_LIT01",
        "P4": "P4_ST_PT01",
    }

    for sub, sig in representative_signals.items():
        sig_std = train_features[sig].std()
        sig_range = train_features[sig].quantile(0.99) - train_features[sig].quantile(0.01)
        logger.info(f"Selected [{sub}] Representative Signal: {sig} (Train Std: {sig_std:.4f}, Dynamic Range Q99-Q01: {sig_range:.4f})")

    # -------------------------------------------------------------------------
    # 7. Correlation Heatmap on Deterministic Training Sample (TRAIN ONLY)
    # -------------------------------------------------------------------------
    logger.info(f"Generating deterministic correlation sample (Seed: {RANDOM_SEED}, Target: {TARGET_CORRELATION_SAMPLES})...")
    # Sample 22,500 rows from train1 and 22,500 rows from train2
    n_per_file = TARGET_CORRELATION_SAMPLES // 2
    s1 = train1_features.sample(n=n_per_file, random_state=RANDOM_SEED)
    s2 = train2_features.sample(n=n_per_file, random_state=RANDOM_SEED + 1)
    corr_sample_df = pd.concat([s1, s2], axis=0, ignore_index=True)
    logger.info(f"Sampled exactly {len(corr_sample_df):,} training rows for correlation calculation.")

    corr_matrix = corr_sample_df.corr(method="pearson")
    # For constant features with std=0, corr will be NaN: fill diagonal with 1.0, off-diagonal with 0.0 for plotting
    corr_vals = corr_matrix.to_numpy(copy=True)
    np.fill_diagonal(corr_vals, 1.0)
    corr_matrix_filled = pd.DataFrame(corr_vals, index=corr_matrix.index, columns=corr_matrix.columns).fillna(0.0)

    # Free large training & validation feature dataframes to minimize peak RAM
    rep_sig_names = list(representative_signals.values())
    rep_t1 = train1_features[rep_sig_names].copy()
    rep_t2 = train2_features[rep_sig_names].copy()
    rep_val = val_features[rep_sig_names].copy()
    del train1_features, train2_features, train_features, val_features, corr_sample_df
    gc.collect()
    logger.info(f"Memory after cleanup before plotting: {get_memory_mb():.2f} MB")

    # -------------------------------------------------------------------------
    # 8. Visual EDA Figure Generation (TRAIN & VALIDATION ONLY)
    # -------------------------------------------------------------------------
    logger.info("Rendering EDA Figures...")

    # Plot 1: Telemetry Overview across Development Training Sessions
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("HAI 21.03 Development Training Telemetry Overview (train1 & train2)\n[Non-constant Representative Sensors Selected on Training Data Only]", fontsize=13, fontweight="bold")

    plot_colors = {"P1": "#1f77b4", "P2": "#ff7f0e", "P3": "#2ca02c", "P4": "#d62728"}
    t1_time = pd.to_datetime(train1_df["time"]).iloc[::20]
    t2_time = pd.to_datetime(train2_df["time"]).iloc[::20]
    t1_rep_down = rep_t1.iloc[::20]
    t2_rep_down = rep_t2.iloc[::20]
    del train1_df, train2_df
    gc.collect()

    for idx, (sub, sig) in enumerate(representative_signals.items()):
        ax = axes[idx]
        ax.plot(t1_time, t1_rep_down[sig], label=f"train1 ({sig})", color=plot_colors[sub], alpha=0.85, lw=1.0)
        ax.plot(t2_time, t2_rep_down[sig], label=f"train2 ({sig})", color=plot_colors[sub], alpha=0.85, lw=1.0, linestyle="--")
        ax.set_ylabel(f"{sub}\n{sig}", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Timestamp (Independent Sessions: train1 = July 11-13; train2 = July 31 - Aug 03)", fontsize=10)
    plt.tight_layout()
    fig1_path = RESULTS_FIGURES_DIR / "eda_01_telemetry_overview.png"
    plt.savefig(fig1_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig1_path}")

    # Plot 2: Feature Dynamic Ranges & Variances on Training Data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("HAI 21.03 Candidate Features Dynamic Range & Variance (Training Data: train1 + train2)", fontsize=13, fontweight="bold")

    sorted_by_range = retention_df.sort_values(by="train_dynamic_range_99_01", ascending=True)
    top_dyn = sorted_by_range.tail(35)
    ax1.barh(top_dyn["feature_name"], top_dyn["train_dynamic_range_99_01"], color="#2b5c8f", edgecolor="none")
    ax1.set_xlabel(r"Dynamic Range ($Q_{99} - Q_{01}$)", fontsize=10)
    ax1.set_title(r"Top 35 Features by Dynamic Range ($Q_{99} - Q_{01}$)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax1.tick_params(axis="y", labelsize=8)

    sorted_by_std = retention_df.sort_values(by="train_std", ascending=True)
    top_std = sorted_by_std.tail(35)
    ax2.barh(top_std["feature_name"], top_std["train_std"], color="#c0392b", edgecolor="none")
    ax2.set_xlabel(r"Standard Deviation ($\sigma$)", fontsize=10)
    ax2.set_title(r"Top 35 Features by Standard Deviation ($\sigma$)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax2.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    fig2_path = RESULTS_FIGURES_DIR / "eda_02_feature_variance_and_ranges.png"
    plt.savefig(fig2_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig2_path}")

    # Plot 3: Distribution Comparison Across Normal Runs (train1 vs. train2 vs. train3)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Normal Telemetry Distribution & Drift Analysis\n[Development Training: train1, train2 vs. Normal Validation: train3]", fontsize=13, fontweight="bold")

    axes_flat = axes.flatten()
    for idx, (sub, sig) in enumerate(representative_signals.items()):
        ax = axes_flat[idx]
        data_to_plot = [
            rep_t1[sig].dropna().values,
            rep_t2[sig].dropna().values,
            rep_val[sig].dropna().values,
        ]
        bplot = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", lw=1.5),
        )
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["train1 (Dev)", "train2 (Dev)", "train3 (Val)"])
        colors = ["#7fb3d5", "#a9cce3", "#f9e79f"]
        for patch, c in zip(bplot["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_title(f"Subsystem {sub}: {sig}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Sensor Value", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig3_path = RESULTS_FIGURES_DIR / "eda_03_distribution_drift_comparison.png"
    plt.savefig(fig3_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig3_path}")

    # Plot 4: Correlation Heatmap (TRAINING DATA ONLY, 45,000 samples)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_matrix_filled.values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation Coefficient ($r$)", fontsize=10)

    # Subsystem tick boundaries
    subsystem_labels = [c.split("_")[0] for c in candidate_features]
    ax.set_xticks(range(len(candidate_features)))
    ax.set_yticks(range(len(candidate_features)))
    ax.set_xticklabels(candidate_features, rotation=90, fontsize=6)
    ax.set_yticklabels(candidate_features, fontsize=6)
    ax.set_title(
        f"HAI 21.03 Candidate Features Pearson Correlation Heatmap\n"
        f"[TRAINING DATA ONLY: train1 + train2 | Sample Size: {TARGET_CORRELATION_SAMPLES:,} | Seed: {RANDOM_SEED}]",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    fig4_path = RESULTS_FIGURES_DIR / "eda_04_train_feature_correlation_heatmap.png"
    plt.savefig(fig4_path, dpi=180)
    plt.close()
    logger.info(f"Saved: {fig4_path}")

    # Plot 5: Descriptive Attack Duration Distribution (From Audited Labels Manifest Only)
    attack_events_path = RESULTS_AUDIT_DIR / "hai_21_03_attack_events.csv"
    if attack_events_path.exists():
        events_df = pd.read_csv(attack_events_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            "HAI 21.03 Attack Duration Distribution\n"
            "[DESCRIPTIVE DATASET AUDIT DOCUMENTATION ONLY — NOT FOR WINDOW OR MODEL SELECTION]",
            fontsize=12,
            fontweight="bold",
            color="#922b21",
        )

        durations_sec = events_df["duration_seconds"].values
        durations_min = durations_sec / 60.0

        ax1.hist(durations_sec, bins=15, color="#c0392b", edgecolor="black", alpha=0.75)
        ax1.set_xlabel("Attack Event Duration (Seconds)", fontsize=10)
        ax1.set_ylabel("Count of Attack Events (Total = 50)", fontsize=10)
        ax1.set_title(f"Histogram of Attack Durations (Mean: {durations_sec.mean():.1f}s, Median: {np.median(durations_sec):.1f}s)", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.boxplot(durations_min, patch_artist=True, boxprops=dict(facecolor="#f5b7b1"), medianprops=dict(color="black", lw=1.5))
        ax2.set_ylabel("Duration (Minutes)", fontsize=10)
        ax2.set_xticks([1])
        ax2.set_xticklabels(["50 Attack Events"])
        ax2.set_title(f"Boxplot of Attack Durations (Range: {durations_min.min():.2f}m - {durations_min.max():.2f}m)", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig5_path = RESULTS_FIGURES_DIR / "eda_05_attack_duration_distribution.png"
        plt.savefig(fig5_path, dpi=180)
        plt.close()
        logger.info(f"Saved: {fig5_path}")

    # Peak memory update
    peak_mem = max(peak_mem, get_memory_mb())
    elapsed_sec = (datetime.now(timezone.utc) - start_time).total_seconds()

    logger.info("=================================================================")
    logger.info("EDA and Split Analysis Execution Completed Successfully")
    logger.info(f"Elapsed Time: {elapsed_sec:.2f} seconds")
    logger.info(f"Peak Process Memory: {peak_mem:.2f} MB")
    logger.info("=================================================================")

    return {
        "candidate_features_count": len(candidate_features),
        "train_rows": 442802,
        "val_rows": 478801,
        "train_constants_count": len(train_constants),
        "val_constants_count": len(val_constants),
        "peak_mem_mb": round(peak_mem, 2),
        "elapsed_sec": round(elapsed_sec, 2),
    }


if __name__ == "__main__":
    main()
