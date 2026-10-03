#!/usr/bin/env python3
"""
HAI 21.03 Preprocessing Pipeline for DT049
==========================================
Implements memory-safe streaming preprocessing, standardization, and array serialization.

Key implementation controls:
1. Streaming scaler fitting: StandardScaler.partial_fit() is called strictly on
   chunks from Development Training data (train1.csv.gz + train2.csv.gz = 442,802 rows).
2. Normal Validation (train3.csv.gz) and Official Test files (test1-test5) NEVER enter fitting.
3. Processed arrays are written incrementally via numpy.lib.format.open_memmap
   to data/processed/hai_21_03/.
4. All 79 candidate features are retained with float32 precision.
5. Timestamps are stored as datetime64[s].
6. Complete metadata is saved to models/preprocessing/scaler_metadata.json and scaler.joblib.
7. Verification manifest is saved to results/audit/hai_21_03_preprocessing_manifest.csv.
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
import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.lib.format import open_memmap

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "preprocessing.yaml"
SOURCE_DIR = REPO_ROOT / "data" / "source" / "hai_repo" / "hai-21.03"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "hai_21_03"
MODELS_DIR = REPO_ROOT / "models" / "preprocessing"
RESULTS_DIR = REPO_ROOT / "results" / "audit"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("HAI2103_Preprocess")
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


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    logger = setup_logger()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    peak_mem = get_memory_mb()

    logger.info("=================================================================")
    logger.info("Starting Streaming Preprocessing Pipeline for DT049 (HAI 21.03)")
    logger.info(f"Initial Memory Usage: {peak_mem:.2f} MB")
    logger.info("=================================================================")

    config = load_config()
    chunk_size = config["scaling"]["fit_chunk_size"]
    excluded_cols = set(config["features"]["excluded_columns"])

    # 1. Determine exact 79 candidate features in deterministic order
    sample_df = pd.read_csv(SOURCE_DIR / "train1.csv.gz", compression="gzip", nrows=2)
    candidate_features = [c for c in sample_df.columns if c not in excluded_cols]
    num_features = len(candidate_features)
    logger.info(f"Identified {num_features} candidate telemetry features.")

    if num_features != config["features"]["total_candidate_count"]:
        raise ValueError(f"Expected {config['features']['total_candidate_count']} features, got {num_features}")

    # 2. Fit StandardScaler strictly on train1 + train2 using streaming partial_fit
    logger.info("Fitting StandardScaler via streaming partial_fit on train1.csv.gz and train2.csv.gz...")
    scaler = StandardScaler()
    training_files = config["splits"]["development_training"]["files"]

    total_fit_rows = 0
    file_row_counts = {}

    for fname in training_files:
        fpath = SOURCE_DIR / fname
        logger.info(f"Streaming partial_fit on {fname}...")
        file_rows = 0
        for chunk in pd.read_csv(fpath, compression="gzip", chunksize=chunk_size, usecols=candidate_features):
            chunk_vals = chunk[candidate_features].to_numpy(dtype=np.float64)
            scaler.partial_fit(chunk_vals)
            file_rows += len(chunk_vals)
            total_fit_rows += len(chunk_vals)
            peak_mem = max(peak_mem, get_memory_mb())
        file_row_counts[fname] = file_rows
        logger.info(f"Finished partial_fit on {fname} ({file_rows:,} rows). Cumulative seen: {total_fit_rows:,}")

    expected_train_rows = config["splits"]["development_training"]["expected_rows"]
    if total_fit_rows != expected_train_rows:
        raise ValueError(f"Expected {expected_train_rows} training rows, but scaler saw {total_fit_rows}")

    # Identify zero-variance features
    # In streaming Welford's algorithm, constant columns accumulate floating-point residuals (< 1e-12)
    zero_variance_mask = (scaler.var_ < 1e-12)
    scaler.var_[zero_variance_mask] = 0.0
    scaler.scale_[zero_variance_mask] = 1.0
    zero_variance_features = [candidate_features[i] for i in range(num_features) if zero_variance_mask[i]]
    dynamic_features = [candidate_features[i] for i in range(num_features) if not zero_variance_mask[i]]
    logger.info(f"Identified {len(zero_variance_features)} zero-variance features in training data (variance < 1e-12).")
    logger.info(f"Zero-variance training features: {zero_variance_features}")

    # 3. Save scaler artifacts
    scaler_joblib_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_joblib_path)
    logger.info(f"Saved serialized scaler: {scaler_joblib_path}")

    # Read upstream commit hash from audit log if available
    upstream_commit = "2a814cebc9a66b06c9e5cd545e2d72e65d383737"
    audit_file = RESULTS_DIR / "dataset_download_audit.txt"
    if audit_file.exists():
        with open(audit_file, "r") as f:
            for line in f:
                if "Commit Hash:" in line:
                    upstream_commit = line.split("Commit Hash:")[-1].strip()

    scaler_metadata = {
        "preprocessing_config_version": config["version"],
        "dataset_name": config["dataset"]["name"],
        "dataset_version": config["dataset"]["version"],
        "hai_repository_commit": upstream_commit,
        "fit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fit_source_files": training_files,
        "fit_sample_count": total_fit_rows,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "feature_count": num_features,
        "ordered_feature_names": candidate_features,
        "zero_variance_training_features_count": len(zero_variance_features),
        "zero_variance_training_features": zero_variance_features,
        "dynamic_training_features_count": len(dynamic_features),
        "zero_variance_scale_handling": "scale set to 1.0 (transforms identically to 0.0 for normal constant input)",
        "timestamp_representation": "datetime64[s]",
        "output_dtype": "float32",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_var": scaler.var_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    scaler_metadata_path = MODELS_DIR / "scaler_metadata.json"
    with open(scaler_metadata_path, "w", encoding="utf-8") as f:
        json.dump(scaler_metadata, f, indent=2)
    logger.info(f"Saved scaler metadata JSON: {scaler_metadata_path}")

    # 4. Transform and write processed arrays via open_memmap
    # Files to process: train1, train2, train3
    files_to_process = [
        ("train1.csv.gz", file_row_counts["train1.csv.gz"], "train1"),
        ("train2.csv.gz", file_row_counts["train2.csv.gz"], "train2"),
        ("train3.csv.gz", 478801, "train3"),
    ]

    manifest_records = []

    for fname, total_rows, base_label in files_to_process:
        fpath = SOURCE_DIR / fname
        feat_npy_path = PROCESSED_DIR / f"{base_label}_features.npy"
        ts_npy_path = PROCESSED_DIR / f"{base_label}_timestamps.npy"

        logger.info(f"Transforming {fname} ({total_rows:,} rows) -> {feat_npy_path.name} & {ts_npy_path.name}...")

        # Initialize memmaps
        feat_mmap = open_memmap(feat_npy_path, mode="w+", dtype="float32", shape=(total_rows, num_features))
        ts_mmap = open_memmap(ts_npy_path, mode="w+", dtype="datetime64[s]", shape=(total_rows,))

        row_offset = 0
        for chunk in pd.read_csv(fpath, compression="gzip", chunksize=chunk_size, usecols=["time"] + candidate_features):
            chunk_len = len(chunk)
            chunk_feats = chunk[candidate_features].to_numpy(dtype=np.float64)
            # Transform using frozen training scaler
            transformed = scaler.transform(chunk_feats).astype(np.float32)

            feat_mmap[row_offset : row_offset + chunk_len] = transformed
            ts_mmap[row_offset : row_offset + chunk_len] = pd.to_datetime(chunk["time"]).values.astype("datetime64[s]")

            row_offset += chunk_len
            peak_mem = max(peak_mem, get_memory_mb())

        # Flush to disk and close
        del feat_mmap, ts_mmap
        gc.collect()

        if row_offset != total_rows:
            raise ValueError(f"Row count mismatch for {fname}: expected {total_rows}, wrote {row_offset}")

        logger.info(f"Finished writing {feat_npy_path.name} and {ts_npy_path.name}")

        # Compute file manifest records
        for p, dt, shp in [
            (feat_npy_path, "float32", f"({total_rows}, {num_features})"),
            (ts_npy_path, "datetime64[s]", f"({total_rows},)"),
        ]:
            f_size = p.stat().st_size
            f_sha = compute_sha256(p)
            manifest_records.append({
                "processed_file": str(p.relative_to(REPO_ROOT)),
                "source_file": fname,
                "shape": shp,
                "dtype": dt,
                "byte_size": f_size,
                "sha256": f_sha,
                "scaler_metadata_ref": str(scaler_metadata_path.relative_to(REPO_ROOT)),
            })

    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = RESULTS_DIR / "hai_21_03_preprocessing_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Saved preprocessing manifest: {manifest_path}")

    # 5. Run Verification Tests on Processed Outputs
    logger.info("=================================================================")
    logger.info("Running Preprocessing Integrity Verification Tests...")

    # Load arrays via mmap_mode="r"
    t1_f = np.load(PROCESSED_DIR / "train1_features.npy", mmap_mode="r")
    t2_f = np.load(PROCESSED_DIR / "train2_features.npy", mmap_mode="r")
    t3_f = np.load(PROCESSED_DIR / "train3_features.npy", mmap_mode="r")
    t1_t = np.load(PROCESSED_DIR / "train1_timestamps.npy", mmap_mode="r")
    t2_t = np.load(PROCESSED_DIR / "train2_timestamps.npy", mmap_mode="r")
    t3_t = np.load(PROCESSED_DIR / "train3_timestamps.npy", mmap_mode="r")

    # Shape & Dtype checks
    assert t1_f.shape == (216001, 79), f"train1 shape mismatch: {t1_f.shape}"
    assert t2_f.shape == (226801, 79), f"train2 shape mismatch: {t2_f.shape}"
    assert t3_f.shape == (478801, 79), f"train3 shape mismatch: {t3_f.shape}"
    assert t1_f.dtype == np.float32, f"train1 dtype mismatch: {t1_f.dtype}"
    assert t2_f.dtype == np.float32, f"train2 dtype mismatch: {t2_f.dtype}"
    assert t3_f.dtype == np.float32, f"train3 dtype mismatch: {t3_f.dtype}"
    assert t1_t.dtype == "datetime64[s]", f"train1 timestamp dtype mismatch: {t1_t.dtype}"
    assert len(t1_t) == len(t1_f), "train1 timestamp alignment mismatch"
    assert len(t2_t) == len(t2_f), "train2 timestamp alignment mismatch"
    assert len(t3_t) == len(t3_f), "train3 timestamp alignment mismatch"
    logger.info("PASSED: Shapes, dtypes, and timestamp row-for-row alignment verified.")

    # Check for NaNs and Infs
    for name, arr in [("train1", t1_f), ("train2", t2_f), ("train3", t3_f)]:
        assert not np.isnan(arr).any(), f"NaN detected in {name}!"
        assert not np.isinf(arr).any(), f"Inf detected in {name}!"
    logger.info("PASSED: Zero NaN and zero Inf values across all processed arrays.")

    # Verify zero-variance feature behavior
    zv_indices = [i for i, c in enumerate(candidate_features) if c in zero_variance_features]
    for zv_idx in zv_indices:
        assert (t1_f[:, zv_idx] == 0.0).all(), f"Non-zero in train1 for zero-variance feature {candidate_features[zv_idx]}"
        assert (t2_f[:, zv_idx] == 0.0).all(), f"Non-zero in train2 for zero-variance feature {candidate_features[zv_idx]}"
        assert (t3_f[:, zv_idx] == 0.0).all(), f"Non-zero in train3 for zero-variance feature {candidate_features[zv_idx]}"
    logger.info("PASSED: 22 train-zero-variance features transform to 0.0 identically across all normal sessions.")

    # Verify dynamic training features approximately zero-mean and unit-variance
    dyn_indices = [i for i, c in enumerate(candidate_features) if c not in zero_variance_features]
    # Sample 50,000 points from t1 and 50,000 from t2 to check means and variances
    sample_t1 = t1_f[::4, dyn_indices]
    sample_t2 = t2_f[::4, dyn_indices]
    sample_train = np.concatenate([sample_t1, sample_t2], axis=0)
    sample_means = np.mean(sample_train, axis=0)
    sample_stds = np.std(sample_train, axis=0)
    assert np.all(np.abs(sample_means) < 0.05), f"Sample means not close to 0: max diff = {np.max(np.abs(sample_means))}"
    assert np.all(np.abs(sample_stds - 1.0) < 0.05), f"Sample stds not close to 1: max diff = {np.max(np.abs(sample_stds - 1.0))}"
    logger.info("PASSED: Dynamic features standardize to approximately zero-mean and unit-variance.")

    peak_mem = max(peak_mem, get_memory_mb())
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    logger.info("=================================================================")
    logger.info("Preprocessing Completed Successfully")
    logger.info(f"Elapsed Time: {elapsed:.2f} seconds")
    logger.info(f"Peak Process Memory: {peak_mem:.2f} MB")
    logger.info("=================================================================")

    return {
        "total_train_rows": total_fit_rows,
        "num_features": num_features,
        "zero_variance_count": len(zero_variance_features),
        "peak_mem_mb": round(peak_mem, 2),
        "elapsed_sec": round(elapsed, 2),
    }


if __name__ == "__main__":
    main()
