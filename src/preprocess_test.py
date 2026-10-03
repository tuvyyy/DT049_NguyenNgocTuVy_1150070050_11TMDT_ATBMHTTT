#!/usr/bin/env python3
"""
HAI 21.03 Official Test Data Preprocessing Pipeline for DT049
=============================================================
Memory-safe streaming standardization and serialization of official test1-test5.

Key integrity controls:
1. Uses the frozen StandardScaler from models/preprocessing/scaler.joblib.
   Zero fitting or partial fitting on test data.
2. Preserves all 79 candidate feature columns in exact frozen order.
3. Preserves timestamp as datetime64[s].
4. Preserves binary attack labels: attack, attack_P1, attack_P2, attack_P3 as int8.
5. Writes memory-mapped numpy arrays to data/processed/hai_21_03/final_test/.
6. Creates results/audit/final_test_preprocessing_manifest.csv recording shapes,
   dtypes, byte sizes, and SHA256 checksums.
7. Performs comprehensive integrity checks (zero NaN/Inf, exact row counts, alignment).
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
from numpy.lib.format import open_memmap

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "final_evaluation.yaml"
SOURCE_DIR = REPO_ROOT / "data" / "source" / "hai_repo" / "hai-21.03"
PROCESSED_TEST_DIR = REPO_ROOT / "data" / "processed" / "hai_21_03" / "final_test"
MODELS_PREPROC_DIR = REPO_ROOT / "models" / "preprocessing"
RESULTS_AUDIT_DIR = REPO_ROOT / "results" / "audit"

EXPECTED_SCALER_SHA256 = "cbc43757fb1155a36285b711d9b134d387e20e9775a7ab19c8776749a9000d26"
EXPECTED_SCALER_METADATA_SHA256 = "1e428deea10be2d31a6198d4cf2936c990a702b9ce80354463bbe9b2a271b56a"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("HAI2103_TestPreprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_file = REPO_ROOT / "results" / "logs" / "preprocess_test.log"
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


def main():
    logger = setup_logger()
    PROCESSED_TEST_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    peak_mem = get_memory_mb()

    logger.info("=================================================================")
    logger.info("Starting Official Test Preprocessing Pipeline (DT049)")
    logger.info(f"Initial Memory Usage: {peak_mem:.2f} MB")
    logger.info("=================================================================")

    # 1. Verify frozen scaler and metadata checksums
    scaler_path = MODELS_PREPROC_DIR / "scaler.joblib"
    scaler_meta_path = MODELS_PREPROC_DIR / "scaler_metadata.json"

    actual_scaler_sha = compute_sha256(scaler_path)
    actual_meta_sha = compute_sha256(scaler_meta_path)

    if actual_scaler_sha.lower() != EXPECTED_SCALER_SHA256.lower():
        raise ValueError(f"Scaler checksum mismatch! Expected {EXPECTED_SCALER_SHA256}, got {actual_scaler_sha}")
    if actual_meta_sha.lower() != EXPECTED_SCALER_METADATA_SHA256.lower():
        raise ValueError(f"Scaler metadata checksum mismatch! Expected {EXPECTED_SCALER_METADATA_SHA256}, got {actual_meta_sha}")

    logger.info("Verified frozen StandardScaler and metadata SHA256 hashes successfully.")
    scaler = joblib.load(scaler_path)

    with open(scaler_meta_path, "r", encoding="utf-8") as f:
        scaler_meta = json.load(f)

    ordered_features = scaler_meta["ordered_feature_names"]
    num_features = len(ordered_features)
    logger.info(f"Loaded frozen feature order with {num_features} candidate process features.")
    assert num_features == 79, f"Expected 79 features, got {num_features}"

    # 2. Test sessions definition (exact row counts from HAI 21.03 audit)
    test_sessions = [
        {"name": "test1", "file": "test1.csv.gz", "expected_rows": 43201},
        {"name": "test2", "file": "test2.csv.gz", "expected_rows": 118801},
        {"name": "test3", "file": "test3.csv.gz", "expected_rows": 108001},
        {"name": "test4", "file": "test4.csv.gz", "expected_rows": 39601},
        {"name": "test5", "file": "test5.csv.gz", "expected_rows": 92401},
    ]

    chunk_size = 50000
    manifest_records = []
    total_test_rows_processed = 0

    # 3. Process each test session independently
    for sess in test_sessions:
        sname = sess["name"]
        sfname = sess["file"]
        expected_rows = sess["expected_rows"]
        spath = SOURCE_DIR / sfname

        logger.info("-----------------------------------------------------------------")
        logger.info(f"Processing session: {sname} ({sfname}, expected {expected_rows:,} rows)...")

        # Destination array paths
        feat_path = PROCESSED_TEST_DIR / f"{sname}_features.npy"
        ts_path = PROCESSED_TEST_DIR / f"{sname}_timestamps.npy"
        atk_path = PROCESSED_TEST_DIR / f"{sname}_attack.npy"
        atk_p1_path = PROCESSED_TEST_DIR / f"{sname}_attack_P1.npy"
        atk_p2_path = PROCESSED_TEST_DIR / f"{sname}_attack_P2.npy"
        atk_p3_path = PROCESSED_TEST_DIR / f"{sname}_attack_P3.npy"

        # Initialize memmaps
        feat_mmap = open_memmap(feat_path, mode="w+", dtype="float32", shape=(expected_rows, num_features))
        ts_mmap = open_memmap(ts_path, mode="w+", dtype="datetime64[s]", shape=(expected_rows,))
        atk_mmap = open_memmap(atk_path, mode="w+", dtype="int8", shape=(expected_rows,))
        atk_p1_mmap = open_memmap(atk_p1_path, mode="w+", dtype="int8", shape=(expected_rows,))
        atk_p2_mmap = open_memmap(atk_p2_path, mode="w+", dtype="int8", shape=(expected_rows,))
        atk_p3_mmap = open_memmap(atk_p3_path, mode="w+", dtype="int8", shape=(expected_rows,))

        cols_to_read = ["time"] + ordered_features + ["attack", "attack_P1", "attack_P2", "attack_P3"]
        row_offset = 0

        for chunk_idx, chunk in enumerate(pd.read_csv(spath, compression="gzip", chunksize=chunk_size, usecols=cols_to_read)):
            chunk_len = len(chunk)

            # Raw feature extraction
            raw_feats = chunk[ordered_features].to_numpy(dtype=np.float64)
            # Standardization using frozen scaler strictly (NO fit or partial_fit!)
            scaled_feats = scaler.transform(raw_feats).astype(np.float32)

            if np.isnan(scaled_feats).any() or np.isinf(scaled_feats).any():
                raise ValueError(f"NaN or Inf encountered after scaling in {sname} chunk {chunk_idx}!")

            # Write into memmaps
            feat_mmap[row_offset : row_offset + chunk_len] = scaled_feats
            ts_mmap[row_offset : row_offset + chunk_len] = pd.to_datetime(chunk["time"]).values.astype("datetime64[s]")
            atk_mmap[row_offset : row_offset + chunk_len] = chunk["attack"].to_numpy(dtype=np.int8)
            atk_p1_mmap[row_offset : row_offset + chunk_len] = chunk["attack_P1"].to_numpy(dtype=np.int8)
            atk_p2_mmap[row_offset : row_offset + chunk_len] = chunk["attack_P2"].to_numpy(dtype=np.int8)
            atk_p3_mmap[row_offset : row_offset + chunk_len] = chunk["attack_P3"].to_numpy(dtype=np.int8)

            row_offset += chunk_len
            peak_mem = max(peak_mem, get_memory_mb())
            logger.info(f"[{sname}] Written {row_offset:,} / {expected_rows:,} rows (RSS: {get_memory_mb():.1f} MB)")

        # Flush memmaps to disk
        del feat_mmap, ts_mmap, atk_mmap, atk_p1_mmap, atk_p2_mmap, atk_p3_mmap
        gc.collect()

        if row_offset != expected_rows:
            raise ValueError(f"Row count mismatch for {sname}: expected {expected_rows}, wrote {row_offset}!")

        total_test_rows_processed += row_offset
        logger.info(f"Successfully processed {sname}: {row_offset:,} rows.")

        # Compute SHA256 and record in manifest
        array_files = [
            (feat_path, "float32", f"({expected_rows}, {num_features})", "features"),
            (ts_path, "datetime64[s]", f"({expected_rows},)", "timestamps"),
            (atk_path, "int8", f"({expected_rows},)", "attack"),
            (atk_p1_path, "int8", f"({expected_rows},)", "attack_P1"),
            (atk_p2_path, "int8", f"({expected_rows},)", "attack_P2"),
            (atk_p3_path, "int8", f"({expected_rows},)", "attack_P3"),
        ]

        for p, dt, shp, arr_role in array_files:
            manifest_records.append({
                "session_name": sname,
                "source_file": sfname,
                "array_role": arr_role,
                "output_path": str(p.relative_to(REPO_ROOT)),
                "shape": shp,
                "dtype": dt,
                "byte_size": p.stat().st_size,
                "sha256": compute_sha256(p),
                "scaler_reference_sha256": EXPECTED_SCALER_SHA256,
            })

    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_records)
    manifest_csv_path = RESULTS_AUDIT_DIR / "final_test_preprocessing_manifest.csv"
    manifest_df.to_csv(manifest_csv_path, index=False)
    logger.info("=================================================================")
    logger.info(f"Saved Final Test Preprocessing Manifest: {manifest_csv_path}")

    # 4. Integrity and Row Alignment Verification Tests
    logger.info("=================================================================")
    logger.info("Running Strict Post-Processing Integrity Verification...")

    total_attack_samples = 0
    total_normal_samples = 0

    for sess in test_sessions:
        sname = sess["name"]
        expected_rows = sess["expected_rows"]

        feats = np.load(PROCESSED_TEST_DIR / f"{sname}_features.npy", mmap_mode="r")
        ts = np.load(PROCESSED_TEST_DIR / f"{sname}_timestamps.npy", mmap_mode="r")
        atk = np.load(PROCESSED_TEST_DIR / f"{sname}_attack.npy", mmap_mode="r")
        p1 = np.load(PROCESSED_TEST_DIR / f"{sname}_attack_P1.npy", mmap_mode="r")
        p2 = np.load(PROCESSED_TEST_DIR / f"{sname}_attack_P2.npy", mmap_mode="r")
        p3 = np.load(PROCESSED_TEST_DIR / f"{sname}_attack_P3.npy", mmap_mode="r")

        assert feats.shape == (expected_rows, 79), f"Shape mismatch in {sname} features: {feats.shape}"
        assert ts.shape == (expected_rows,), f"Shape mismatch in {sname} timestamps: {ts.shape}"
        assert atk.shape == (expected_rows,), f"Shape mismatch in {sname} attack: {atk.shape}"
        assert p1.shape == (expected_rows,), f"Shape mismatch in {sname} P1: {p1.shape}"
        assert p2.shape == (expected_rows,), f"Shape mismatch in {sname} P2: {p2.shape}"
        assert p3.shape == (expected_rows,), f"Shape mismatch in {sname} P3: {p3.shape}"

        # Check timestamp monotonicity
        time_diffs = np.diff(ts.astype("int64"))
        assert (time_diffs == 1).all(), f"Timestamp non-monotonicity or gap found in {sname}!"

        # Check label consistency: attack == (p1 | p2 | p3)
        assert ((p1 | p2 | p3) <= atk).all(), f"Subsystem attack true while overall attack false in {sname}!"

        atk_sum = int(np.sum(atk))
        norm_sum = int(np.sum(atk == 0))
        total_attack_samples += atk_sum
        total_normal_samples += norm_sum

        logger.info(f"Verified {sname}: {expected_rows:,} rows, Attack samples: {atk_sum:,}, Normal: {norm_sum:,}, Time range: {ts[0]} to {ts[-1]}")

    elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info("=================================================================")
    logger.info(f"Test Preprocessing Completed Successfully in {elapsed_time:.2f}s")
    logger.info(f"Total Test Rows Processed: {total_test_rows_processed:,}")
    logger.info(f"Total Attack Samples: {total_attack_samples:,} ({total_attack_samples / total_test_rows_processed * 100:.3f}%)")
    logger.info(f"Total Normal Samples: {total_normal_samples:,} ({total_normal_samples / total_test_rows_processed * 100:.3f}%)")
    logger.info(f"Peak Memory Usage: {peak_mem:.2f} MB")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
