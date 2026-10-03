#!/usr/bin/env python3
"""
HAI 21.03 Dataset Audit Script for DT049
========================================
This script performs a comprehensive, reproducible, memory-safe audit of the
official HAI 21.03 dataset files located at data/source/hai_repo/hai-21.03/.

Important rules followed:
- Original .csv.gz files remain compressed and unmodified.
- Chunked reading is used to keep memory consumption well within 4 GB RAM.
- Features are identified objectively: all columns except timestamp and label columns.
- Discrepancies between documented expectations and empirical observations are
  explicitly reported.
- Third-party package psutil is used solely for process memory telemetry.
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import psutil  # Third-party library installed in .venv, used for memory profiling
import numpy as np
import pandas as pd

# Define paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "source" / "hai_repo" / "hai-21.03"
RESULTS_DIR = REPO_ROOT / "results" / "audit"
DOCS_DIR = REPO_ROOT / "docs"

# Expected files in HAI 21.03
DATASET_FILES = [
    ("train1.csv.gz", "train", 60, 0),
    ("train2.csv.gz", "train", 63, 0),
    ("train3.csv.gz", "train", 229, 0),
    ("test1.csv.gz", "test", 12, 5),
    ("test2.csv.gz", "test", 33, 20),
    ("test3.csv.gz", "test", 30, 8),
    ("test4.csv.gz", "test", 11, 5),
    ("test5.csv.gz", "test", 26, 12),
]

CHUNK_SIZE = 100_000


def setup_logger(log_file: Path) -> logging.Logger:
    """Configures dual logging to console and audit log file."""
    logger = logging.getLogger("HAI2103_Audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def compute_sha256(filepath: Path) -> str:
    """Computes SHA256 hash of a file via 64KB chunk streaming."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_memory_usage_mb() -> float:
    """Returns current process RSS memory in Megabytes via psutil."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def audit_hai_dataset():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = RESULTS_DIR / "hai_21_03_audit.log"
    logger = setup_logger(log_file)

    logger.info("=================================================================")
    logger.info("Starting HAI 21.03 Dataset Audit for DT049")
    logger.info(f"Target dataset directory: {DATA_DIR}")
    logger.info(f"Initial Memory Usage: {get_memory_usage_mb():.2f} MB")
    logger.info("=================================================================")

    if not DATA_DIR.exists():
        logger.error(f"Dataset directory does not exist: {DATA_DIR}")
        sys.exit(1)

    peak_memory_mb = get_memory_usage_mb()

    # Data structures for results
    file_summaries = []
    column_summaries = []
    all_attack_events = []
    global_column_stats = {}  # col -> {'min': float, 'max': float, 'nulls': int, 'infs': int, 'roles': set}

    # Verify schema consistency across files
    reference_columns = None
    timestamp_col_name = "time"
    known_label_col_names = ["attack", "attack_P1", "attack_P2", "attack_P3"]

    for filename, role, exp_hours, exp_attacks in DATASET_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            logger.error(f"Required file not found: {filepath}")
            continue

        file_size_bytes = filepath.stat().st_size
        sha256_hash = compute_sha256(filepath)
        logger.info(f"Auditing file: {filename} ({role.upper()}) | Size: {file_size_bytes:,} bytes | SHA256: {sha256_hash[:12]}...")

        # Per-file accumulators
        total_rows = 0
        first_timestamp = None
        last_timestamp = None
        prev_timestamp = None
        is_chronological = True
        duplicate_timestamp_count = 0
        timestamp_gaps = []  # list of (t_prev, t_curr, gap_sec)
        
        file_col_stats = {}  # col -> {null_count, inf_count, min_val, max_val, dtype}
        label_distributions = {lbl: {} for lbl in known_label_col_names}

        # Attack event tracking across chunks
        # Event structure: dict(event_id, start_ts, end_ts, sample_count, active_p1, active_p2, active_p3, gaps)
        current_event = None
        file_events = []
        event_counter = 0

        # Chunked reading
        reader = pd.read_csv(filepath, compression="gzip", chunksize=CHUNK_SIZE)

        for chunk_idx, chunk in enumerate(reader):
            chunk_rows = len(chunk)
            total_rows += chunk_rows

            cols = list(chunk.columns)
            if reference_columns is None:
                reference_columns = cols
            elif cols != reference_columns:
                logger.warning(f"Schema mismatch in {filename} chunk {chunk_idx}! Columns differ from reference.")

            # Identify timestamp column
            if timestamp_col_name not in chunk.columns:
                raise ValueError(f"Expected timestamp column '{timestamp_col_name}' missing in {filename}")

            # Parse timestamps
            ts_series = pd.to_datetime(chunk[timestamp_col_name])
            if first_timestamp is None:
                first_timestamp = ts_series.iloc[0]

            # Check timestamp ordering and continuity
            if prev_timestamp is not None:
                # Check boundary between chunks
                boundary_delta = (ts_series.iloc[0] - prev_timestamp).total_seconds()
                if boundary_delta < 0:
                    is_chronological = False
                    logger.warning(f"Non-monotonic timestamp at chunk boundary in {filename}: {prev_timestamp} -> {ts_series.iloc[0]}")
                elif boundary_delta == 0:
                    duplicate_timestamp_count += 1
                elif boundary_delta > 1:
                    timestamp_gaps.append((str(prev_timestamp), str(ts_series.iloc[0]), int(boundary_delta)))

            # Intra-chunk timestamp deltas
            deltas = ts_series.diff().iloc[1:].dt.total_seconds()
            if (deltas < 0).any():
                is_chronological = False
                logger.warning(f"Non-monotonic timestamps detected inside chunk {chunk_idx} of {filename}")
            
            dup_intra = (deltas == 0).sum()
            duplicate_timestamp_count += int(dup_intra)

            intra_gaps = deltas[deltas > 1]
            if len(intra_gaps) > 0:
                for idx in intra_gaps.index:
                    t_before = ts_series.loc[idx - 1]
                    t_after = ts_series.loc[idx]
                    gap_sec = int(deltas.loc[idx])
                    timestamp_gaps.append((str(t_before), str(t_after), gap_sec))

            prev_timestamp = ts_series.iloc[-1]
            last_timestamp = ts_series.iloc[-1]

            # Label distribution tracking
            for lbl in known_label_col_names:
                if lbl in chunk.columns:
                    val_counts = chunk[lbl].value_counts(dropna=False).to_dict()
                    for val, count in val_counts.items():
                        # Use string representation of key to be JSON safe
                        s_val = str(val)
                        label_distributions[lbl][s_val] = label_distributions[lbl].get(s_val, 0) + int(count)

            # Attack event extraction for test files (or train if attacks present)
            if "attack" in chunk.columns:
                attack_arr = chunk["attack"].to_numpy()
                p1_arr = chunk["attack_P1"].to_numpy() if "attack_P1" in chunk.columns else np.zeros(chunk_rows)
                p2_arr = chunk["attack_P2"].to_numpy() if "attack_P2" in chunk.columns else np.zeros(chunk_rows)
                p3_arr = chunk["attack_P3"].to_numpy() if "attack_P3" in chunk.columns else np.zeros(chunk_rows)
                ts_arr = ts_series.to_numpy()

                for i in range(chunk_rows):
                    val = attack_arr[i]
                    is_pos = (val != 0 and not np.isnan(val))
                    curr_ts = pd.Timestamp(ts_arr[i])

                    if is_pos:
                        if current_event is None:
                            # Start of a new event
                            event_counter += 1
                            current_event = {
                                "file_name": filename,
                                "event_id": f"{filename.split('.')[0]}_event_{event_counter:02d}",
                                "event_index_in_file": event_counter,
                                "start_time": curr_ts,
                                "end_time": curr_ts,
                                "prev_time": curr_ts,
                                "sample_count": 1,
                                "p1_samples": 1 if (p1_arr[i] != 0 and not np.isnan(p1_arr[i])) else 0,
                                "p2_samples": 1 if (p2_arr[i] != 0 and not np.isnan(p2_arr[i])) else 0,
                                "p3_samples": 1 if (p3_arr[i] != 0 and not np.isnan(p3_arr[i])) else 0,
                                "internal_gaps": [],
                            }
                        else:
                            # Continuation of existing event
                            gap = (curr_ts - current_event["prev_time"]).total_seconds()
                            if gap > 1:
                                current_event["internal_gaps"].append((str(current_event["prev_time"]), str(curr_ts), int(gap)))
                            current_event["sample_count"] += 1
                            current_event["end_time"] = curr_ts
                            current_event["prev_time"] = curr_ts
                            if p1_arr[i] != 0 and not np.isnan(p1_arr[i]):
                                current_event["p1_samples"] += 1
                            if p2_arr[i] != 0 and not np.isnan(p2_arr[i]):
                                current_event["p2_samples"] += 1
                            if p3_arr[i] != 0 and not np.isnan(p3_arr[i]):
                                current_event["p3_samples"] += 1
                    else:
                        if current_event is not None:
                            # End of current event
                            file_events.append(current_event)
                            current_event = None

            # Track column statistics
            for col in cols:
                series = chunk[col]
                dtype_str = str(series.dtype)
                null_cnt = int(series.isna().sum())

                inf_cnt = 0
                is_num = pd.api.types.is_numeric_dtype(series.dtype)
                if is_num:
                    inf_cnt = int(np.isinf(series).sum())

                # Min/Max for constant detection
                valid_series = series.dropna()
                if len(valid_series) > 0 and is_num:
                    c_min = float(valid_series.min())
                    c_max = float(valid_series.max())
                else:
                    c_min, c_max = None, None

                if col not in file_col_stats:
                    file_col_stats[col] = {
                        "dtype": dtype_str,
                        "null_count": null_cnt,
                        "inf_count": inf_cnt,
                        "min_val": c_min,
                        "max_val": c_max,
                    }
                else:
                    st = file_col_stats[col]
                    st["null_count"] += null_cnt
                    st["inf_count"] += inf_cnt
                    if c_min is not None:
                        st["min_val"] = c_min if st["min_val"] is None else min(st["min_val"], c_min)
                    if c_max is not None:
                        st["max_val"] = c_max if st["max_val"] is None else max(st["max_val"], c_max)

            # Telemetry check
            current_mem = get_memory_usage_mb()
            if current_mem > peak_memory_mb:
                peak_memory_mb = current_mem

        # Close open event if file ended while in attack
        if current_event is not None:
            file_events.append(current_event)
            current_event = None

        # Convert file events to final records
        for ev in file_events:
            start_ts = ev["start_time"]
            end_ts = ev["end_time"]
            sample_cnt = ev["sample_count"]
            span_seconds = int((end_ts - start_ts).total_seconds()) + 1
            has_gap = (len(ev["internal_gaps"]) > 0) or (span_seconds != sample_cnt)
            duration_sec = span_seconds if not has_gap else sample_cnt

            subsystems = []
            if ev["p1_samples"] > 0:
                subsystems.append("P1")
            if ev["p2_samples"] > 0:
                subsystems.append("P2")
            if ev["p3_samples"] > 0:
                subsystems.append("P3")

            all_attack_events.append({
                "file_name": ev["file_name"],
                "event_id": ev["event_id"],
                "event_index_in_file": ev["event_index_in_file"],
                "start_time": str(start_ts),
                "end_time": str(end_ts),
                "sample_count": sample_cnt,
                "span_seconds": span_seconds,
                "duration_seconds": duration_sec,
                "has_internal_gap": has_gap,
                "internal_gaps_count": len(ev["internal_gaps"]),
                "internal_gaps_detail": str(ev["internal_gaps"]) if has_gap else "",
                "subsystems": "+".join(subsystems) if subsystems else "None",
                "p1_attack_samples": ev["p1_samples"],
                "p2_attack_samples": ev["p2_samples"],
                "p3_attack_samples": ev["p3_samples"],
            })

        # Calculate file duration
        duration_seconds = int((last_timestamp - first_timestamp).total_seconds()) + 1
        duration_hours = round(duration_seconds / 3600.0, 2)
        duration_match = (duration_hours == exp_hours)

        observed_attacks = len(file_events)
        attack_count_match = (observed_attacks == exp_attacks)

        # File constant columns count
        constant_cols_file = [
            c for c, st in file_col_stats.items()
            if st["min_val"] is not None and st["min_val"] == st["max_val"]
        ]

        total_file_nulls = sum(st["null_count"] for st in file_col_stats.values())
        total_file_infs = sum(st["inf_count"] for st in file_col_stats.values())
        attack_seconds = sum(v for k, v in label_distributions.get("attack", {}).items() if k != "0")

        file_summaries.append({
            "filename": filename,
            "role": role,
            "compressed_size_bytes": file_size_bytes,
            "sha256": sha256_hash,
            "total_rows": total_rows,
            "total_columns": len(reference_columns),
            "first_timestamp": str(first_timestamp),
            "last_timestamp": str(last_timestamp),
            "duration_seconds": duration_seconds,
            "duration_hours": duration_hours,
            "expected_duration_hours": exp_hours,
            "duration_match": duration_match,
            "is_chronological": is_chronological,
            "duplicate_timestamps": duplicate_timestamp_count,
            "gaps_count": len(timestamp_gaps),
            "gaps_detail": str(timestamp_gaps) if timestamp_gaps else "",
            "attack_seconds": attack_seconds,
            "observed_attacks": observed_attacks,
            "expected_attacks": exp_attacks,
            "attack_count_match": attack_count_match,
            "null_values_total": total_file_nulls,
            "inf_values_total": total_file_infs,
            "constant_columns_count": len(constant_cols_file),
            "constant_columns": ";".join(constant_cols_file),
            "label_distribution_attack": json.dumps(label_distributions.get("attack", {})),
            "label_distribution_attack_P1": json.dumps(label_distributions.get("attack_P1", {})),
            "label_distribution_attack_P2": json.dumps(label_distributions.get("attack_P2", {})),
            "label_distribution_attack_P3": json.dumps(label_distributions.get("attack_P3", {})),
        })

        # Update global column stats
        for col, st in file_col_stats.items():
            if col not in global_column_stats:
                global_column_stats[col] = {
                    "dtype": st["dtype"],
                    "null_count": st["null_count"],
                    "inf_count": st["inf_count"],
                    "global_min": st["min_val"],
                    "global_max": st["max_val"],
                    "constant_in_all_files": (st["min_val"] is not None and st["min_val"] == st["max_val"]),
                    "file_min_max": {filename: (st["min_val"], st["max_val"])},
                }
            else:
                gst = global_column_stats[col]
                gst["null_count"] += st["null_count"]
                gst["inf_count"] += st["inf_count"]
                if st["min_val"] is not None:
                    gst["global_min"] = st["min_val"] if gst["global_min"] is None else min(gst["global_min"], st["min_val"])
                if st["max_val"] is not None:
                    gst["global_max"] = st["max_val"] if gst["global_max"] is None else max(gst["global_max"], st["max_val"])
                is_file_const = (st["min_val"] is not None and st["min_val"] == st["max_val"])
                gst["constant_in_all_files"] = gst["constant_in_all_files"] and is_file_const
                gst["file_min_max"][filename] = (st["min_val"], st["max_val"])

            # Save detailed column summary row per file
            col_cat = "timestamp" if col == timestamp_col_name else ("label" if col in known_label_col_names else "candidate_feature")
            column_summaries.append({
                "filename": filename,
                "column_name": col,
                "column_category": col_cat,
                "dtype": st["dtype"],
                "null_count": st["null_count"],
                "inf_count": st["inf_count"],
                "min_val": st["min_val"],
                "max_val": st["max_val"],
                "is_constant_in_file": (st["min_val"] is not None and st["min_val"] == st["max_val"]),
            })

    # Global column constant check
    globally_constant_columns = []
    for col, gst in global_column_stats.items():
        if gst["global_min"] is not None and gst["global_min"] == gst["global_max"]:
            globally_constant_columns.append(col)

    # Classify candidate features
    all_columns = reference_columns if reference_columns else []
    candidate_features = [c for c in all_columns if c != timestamp_col_name and c not in known_label_col_names]
    subsystem_breakdown = {}
    for feat in candidate_features:
        prefix = feat.split("_")[0] if "_" in feat else "OTHER"
        subsystem_breakdown[prefix] = subsystem_breakdown.get(prefix, 0) + 1

    # Discrepancy analysis
    total_observed_attacks = sum(f["observed_attacks"] for f in file_summaries)
    total_expected_attacks = 50
    total_observed_duration_hours = sum(f["duration_hours"] for f in file_summaries)
    total_expected_duration_hours = 464

    # Save output CSVs
    file_summary_df = pd.DataFrame(file_summaries)
    file_summary_df.to_csv(RESULTS_DIR / "hai_21_03_file_summary.csv", index=False)
    logger.info(f"Saved file summary: {RESULTS_DIR / 'hai_21_03_file_summary.csv'}")

    column_summary_df = pd.DataFrame(column_summaries)
    # Add globally constant column flag
    column_summary_df["is_globally_constant"] = column_summary_df["column_name"].isin(globally_constant_columns)
    column_summary_df.to_csv(RESULTS_DIR / "hai_21_03_column_summary.csv", index=False)
    logger.info(f"Saved column summary: {RESULTS_DIR / 'hai_21_03_column_summary.csv'}")

    events_df = pd.DataFrame(all_attack_events)
    events_df.to_csv(RESULTS_DIR / "hai_21_03_attack_events.csv", index=False)
    logger.info(f"Saved attack events: {RESULTS_DIR / 'hai_21_03_attack_events.csv'}")

    # Build consolidated JSON
    audit_json = {
        "audit_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": "src/audit_dataset.py",
            "dataset_path": str(DATA_DIR),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "python_interpreter": sys.executable,
        },
        "schema_analysis": {
            "total_columns": len(all_columns),
            "documented_data_points": 78,
            "observed_candidate_features": len(candidate_features),
            "schema_discrepancy": {
                "documented_points": 78,
                "observed_candidate_features": len(candidate_features),
                "difference": len(candidate_features) - 78,
                "explanation": f"Observed 84 total CSV columns = 1 timestamp ('{timestamp_col_name}') + 4 ground-truth labels ({known_label_col_names}) + {len(candidate_features)} candidate telemetry features (README stated 78 data points, discrepancy = +{len(candidate_features) - 78})."
            },
            "timestamp_column": timestamp_col_name,
            "label_columns": known_label_col_names,
            "candidate_features_count": len(candidate_features),
            "candidate_features_subsystem_breakdown": subsystem_breakdown,
            "candidate_features_list": candidate_features,
            "globally_constant_columns": globally_constant_columns,
        },
        "files_summary": file_summaries,
        "attack_events_summary": {
            "total_observed_attacks": total_observed_attacks,
            "total_expected_attacks": total_expected_attacks,
            "attacks_match": (total_observed_attacks == total_expected_attacks),
            "events_per_file": {f["filename"]: f["observed_attacks"] for f in file_summaries if f["role"] == "test"},
            "events_with_internal_gaps": int(events_df["has_internal_gap"].sum()) if not events_df.empty else 0,
            "events": all_attack_events,
        },
        "dataset_duration_summary": {
            "total_observed_hours": total_observed_duration_hours,
            "total_expected_hours": total_expected_duration_hours,
            "duration_match": (total_observed_duration_hours == total_expected_duration_hours),
            "hours_per_file": {f["filename"]: f["duration_hours"] for f in file_summaries},
        }
    }

    with open(RESULTS_DIR / "hai_21_03_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_json, f, indent=2)
    logger.info(f"Saved audit JSON: {RESULTS_DIR / 'hai_21_03_audit.json'}")

    logger.info("=================================================================")
    logger.info("Audit Execution Finished Successfully")
    logger.info(f"Total Columns: {len(all_columns)} (1 timestamp, 4 labels, {len(candidate_features)} candidate features)")
    logger.info(f"Globally Constant Columns: {globally_constant_columns}")
    logger.info(f"Observed Attack Events: {total_observed_attacks} (Expected: {total_expected_attacks})")
    logger.info(f"Observed Duration: {total_observed_duration_hours} hours (Expected: {total_expected_duration_hours} hours)")
    logger.info(f"Peak Process Memory: {peak_memory_mb:.2f} MB")
    logger.info("=================================================================")

    return audit_json


if __name__ == "__main__":
    audit_hai_dataset()
