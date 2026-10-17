# HAI 21.03 Dataset Audit Report: DT049

## 1. Executive Summary

This report documents the empirical audit of the official **HAI Security Dataset, version 21.03**, downloaded via sparse checkout from `https://github.com/icsdataset/hai` (commit `2a814cebc9a66b06c9e5cd545e2d72e65d383737`).

The audit was executed using `src/audit_dataset.py` with Python 3.14.7 within the project virtual environment (`.venv`). All 8 dataset files were inspected directly in their compressed state (`.csv.gz`) without decompression, modification, or memory exhaustion.

### High-Level Comparison: Documented Expectations vs. Empirical Observations

| Dimension | Documented Expectation (Official README) | Locally Observed Value | Status / Discrepancy |
| :--- | :--- | :--- | :--- |
| **Total Files** | 8 files (3 train, 5 test) | 8 files (`.csv.gz`) | **Matched** |
| **Total Data Points** | 78 points/sec | 79 candidate process features (+1 timestamp, +4 labels = 84 total columns) | **Discrepancy: +1 feature** |
| **Attack Events** | 50 attacks (5, 20, 8, 5, 12) | 50 attacks (5, 20, 8, 5, 12) | **Matched** |
| **Train Duration** | 352 hours (60h, 63h, 229h) | 256.0 hours (60h, 63h, 133h) | **Discrepancy: -96h in `train3`** |
| **Test Duration** | 112 hours (12h, 33h, 30h, 11h, 26h) | 111.67 hours (12h, 33h, 30h, 11h, 25.67h) | **Matched (with rounding)** |
| **Total Duration** | 464 hours | 367.67 hours (1,323,608 seconds) | **Discrepancy: -96.33h** |
| **Training Labels** | Normal operating data (unsupervised) | 100% attack-free (`attack=0` across all train rows) | **Matched** |
| **Sampling Interval**| 1-second intervals (1 Hz) | Strictly 1-second monotonic intervals ($\Delta t = 1\text{ s}$) | **Matched (0 gaps, 0 duplicates)**|
| **Data Cleanliness** | Unspecified | 0 null values, 0 infinite values across all 1,323,608 rows | **Excellent** |

---

## 2. Schema and Feature Taxonomy

The CSV headers across all 8 files are 100% identical in column names, order, and count (84 columns total).

### Column Classification
1. **Timestamp Column (1 column)**:
   - `time`: Timestamp formatted as `YYYY-MM-DD HH:MM:SS` (parsed as UTC/local timestamp).
2. **Ground-Truth Label Columns (4 columns)**:
   - `attack`: Primary binary indicator (`0` = normal, `1` = cyber attack active).
   - `attack_P1`: Attack indicator specifically targeting Subsystem 1 (Boiler).
   - `attack_P2`: Attack indicator specifically targeting Subsystem 2 (Turbine).
   - `attack_P3`: Attack indicator specifically targeting Subsystem 3 (Water Treatment).
   *(These 4 columns must be strictly excluded from model feature inputs).*
3. **Candidate Model Features (79 columns)**:
   - Defined objectively as all non-timestamp, non-label columns.
   - Categorized by physical subsystem prefix:
     - **Subsystem 1 (Boiler / Feedwater)**: 38 columns (`P1_*`)
     - **Subsystem 2 (Turbine / Generator)**: 22 columns (`P2_*`)
     - **Subsystem 3 (Water Treatment)**: 7 columns (`P3_*`)
     - **Subsystem 4 (HIL Simulator / Balance of Plant)**: 12 columns (`P4_*`)

> [!IMPORTANT]
> **Discrepancy Note**: The official README states "Data Points (points/sec): 78". The actual CSV contains 79 process telemetry columns between `time` and `attack`.

---

## 3. Data Quality & Constant Columns

Across all 1,323,608 rows and 84 columns:
- **Missing / Null Values**: `0` (0.00%)
- **Infinite Numeric Values**: `0` (0.00%)
- **Duplicate Timestamps**: `0`
- **Duplicate Full Rows**: Because timestamps are strictly unique within each file, no full-row duplicates exist.

### Constant Columns Analysis
Of the 79 candidate model features, **19 columns are globally constant** across all 8 files (zero variance across 1,323,608 seconds):

| Subsystem | Globally Constant Features (Zero Variance) | Count |
| :--- | :--- | :--- |
| **P1** | `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP` | 7 |
| **P2** | `P2_ASD`, `P2_AutoGO`, `P2_MSD`, `P2_ManualGO`, `P2_RTR`, `P2_TripEx`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04` | 10 |
| **P3** | `P3_LH`, `P3_LL` | 2 |
| **P4** | *(None)* | 0 |
| **Total** | | **19** |

**Effective Non-Constant Candidate Features**:
$$79 - 19 = 60 \text{ informative candidate features}$$

In addition, individual files contain intermittent constant features:
- `P1_PCV02D`, `P2_Emerg`, and `P2_OnOff` are constant in `train1`, `train2`, `train3`, and `test1`, but vary in subsequent test splits.
- `attack_P3` is constant 0 in `test1` and `test3` (no attacks target Subsystem 3 in those two files).

---

## 4. File-by-File Summary and Duration Discrepancy

| File | Role | Compressed Size | Rows (Seconds) | First Timestamp | Last Timestamp | Observed Duration | Documented Duration | Attack Seconds | Attack Events |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train1.csv.gz` | Train | 30.65 MB | 216,001 | 2020-07-11 00:00:00 | 2020-07-13 12:00:00 | 60.00 h | 60 h | 0 | 0 |
| `train2.csv.gz` | Train | 31.43 MB | 226,801 | 2020-07-31 22:00:00 | 2020-08-03 13:00:00 | 63.00 h | 63 h | 0 | 0 |
| `train3.csv.gz` | Train | 67.96 MB | 478,801 | 2020-08-04 22:00:00 | 2020-08-10 11:00:00 | 133.00 h | 229 h | 0 | 0 |
| **Train Total** | | **130.04 MB** | **921,603** | | | **256.00 h** | **352 h** | **0** | **0** |
| `test1.csv.gz` | Test | 6.15 MB | 43,201 | 2020-07-07 15:00:00 | 2020-07-08 03:00:00 | 12.00 h | 12 h | 629 | 5 |
| `test2.csv.gz` | Test | 16.77 MB | 118,801 | 2020-07-09 15:00:00 | 2020-07-11 00:00:00 | 33.00 h | 33 h | 3,449 | 20 |
| `test3.csv.gz` | Test | 15.48 MB | 108,001 | 2020-07-13 00:00:00 | 2020-07-14 06:00:00 | 30.00 h | 30 h | 1,535 | 8 |
| `test4.csv.gz` | Test | 5.60 MB | 39,601 | 2020-07-28 12:00:00 | 2020-07-28 23:00:00 | 11.00 h | 11 h | 1,157 | 5 |
| `test5.csv.gz` | Test | 13.20 MB | 92,401 | 2020-07-30 10:50:00 | 2020-07-31 12:30:00 | 25.67 h | 26 h | 2,177 | 12 |
| **Test Total** | | **57.20 MB** | **402,005** | | | **111.67 h** | **112 h** | **8,947** | **50** |
| **Overall Total** | | **187.24 MB** | **1,323,608** | | | **367.67 h** | **464 h** | **8,947** | **50** |

> [!WARNING]
> **Major Discrepancy in `train3.csv.gz`**:
> The official documentation reports `train3` as having 229 hours of normal data (~824,400 rows). However, the actual downloaded `train3.csv.gz` contains exactly 478,801 rows spanning 2020-08-04 22:00:00 to 2020-08-10 11:00:00 (133.0 hours), resulting in a 96-hour deficit from the README claim.

### Upstream Integrity Verification & Discrepancy Classification
An upstream Git object verification was performed on `data/source/hai_repo` to determine whether this difference was due to local download truncation or upstream repository state:

1. **Working Tree Cleanliness**: `git -C data/source/hai_repo status` confirmed the working tree is completely clean (`nothing to commit, working tree clean`).
2. **Git Tree Blob Identity**: `git -C data/source/hai_repo ls-tree -l HEAD hai-21.03/` verified that at commit `2a814cebc9a66b06c9e5cd545e2d72e65d383737`, the upstream tree defines `train3.csv.gz` as blob `207e4b9f4c157505d5cb4a66e8ebe84d03575dab` with size **67,961,579 bytes**.
3. **Local Blob Matching**: Running `git -C data/source/hai_repo hash-object hai-21.03/train3.csv.gz` on the local file produces `207e4b9f4c157505d5cb4a66e8ebe84d03575dab` and byte count `67,961,579` (100% exact match). All 8 files in `hai-21.03` match their upstream Git tree blobs byte-for-byte.
4. **Classification**:
   - **Official README Claim**: 229 hours (documented as 246 MB uncompressed).
   - **Verified Git Repository State**: Blob `207e4b9f4c157505d5cb4a66e8ebe84d03575dab` (67,961,579 bytes compressed).
   - **Local Observed Data**: Exactly 478,801 contiguous 1 Hz rows spanning 133.00 hours.
   - **Conclusion**: The difference between 229h and 133h is classified as an **upstream documentation/data discrepancy** present in the official ETRI GitHub repository, **not local corruption or incomplete download**.

---

## 5. Attack Event Audit & Verification

An attack event was algorithmically identified as a contiguous positive run (`attack == 1`) bounded by rising edges ($0 \rightarrow 1$) and falling edges ($1 \rightarrow 0$), treating file boundaries independently.

### Attack Count Verification
- **Total Expected Attack Events**: 50
- **Total Observed Attack Events**: 50 (100% match)
- **Breakdown by Split**:
  - `test1`: 5 events (expected 5)
  - `test2`: 20 events (expected 20)
  - `test3`: 8 events (expected 8)
  - `test4`: 5 events (expected 5)
  - `test5`: 12 events (expected 12)

### Event Continuity & Sampling Validation
- **Internal Gaps**: **0 events** had internal timestamp discontinuities.
- For all 50 events:
  $$\text{duration\_seconds} = (t_{\text{end}} - t_{\text{start}}) + 1\text{ s} = \text{sample\_count}$$
- Event durations range from **17 seconds** (`test2_event_03`) to **422 seconds** (~7.0 minutes, `test2_event_02`).

### Target Subsystems Breakdown
- **Single-subsystem attacks**:
  - P1 only: 26 attacks
  - P2 only: 11 attacks
  - P3 only: 2 attacks
- **Multi-subsystem coordinated attacks**:
  - P1 + P2: 6 attacks
  - P1 + P3: 3 attacks
  - P2 + P3: 2 attacks

---

## 6. Audit Execution Telemetry

- **Script**: `src/audit_dataset.py`
- **Python Version**: 3.14.7 (`/home/kali/DT049/.venv/bin/python`)
- **Execution Strategy**: Sequential chunked streaming (`chunksize=100,000`)
- **Peak Process Memory**: **270.62 MB** (measured via `psutil`, well within the 4 GB RAM VM budget)
- **Execution Time**: ~9.5 seconds

### Audit Artifacts Generated
1. [`results/audit/hai_21_03_file_summary.csv`](file:///home/kali/DT049/results/audit/hai_21_03_file_summary.csv): Full metadata and integrity metrics per file.
2. [`results/audit/hai_21_03_column_summary.csv`](file:///home/kali/DT049/results/audit/hai_21_03_column_summary.csv): Detailed per-column, per-file statistics.
3. [`results/audit/hai_21_03_attack_events.csv`](file:///home/kali/DT049/results/audit/hai_21_03_attack_events.csv): Complete list of all 50 attack events with start/end timestamps, sample counts, and affected subsystems.
4. [`results/audit/hai_21_03_audit.json`](file:///home/kali/DT049/results/audit/hai_21_03_audit.json): Structured JSON payload for automated verification pipelines.
5. [`results/audit/hai_21_03_audit.log`](file:///home/kali/DT049/results/audit/hai_21_03_audit.log): Execution log detailing progress and diagnostics.
