# Dataset Selection: HAI 21.03

## 1. Selected Version
- **Dataset**: HAI Security Dataset
- **Version**: HAI 21.03

## 2. Official Source & Download Metadata
- **Repository URL**: https://github.com/icsdataset/hai
- **Default Branch**: `master`
- **Git Commit Hash**: `2a814cebc9a66b06c9e5cd545e2d72e65d383737`
- **Download Timestamp (UTC)**: `2026-10-02T11:42:07Z`
- **Local Checkout Path**: `data/source/hai_repo/hai-21.03/`
- **Actual Local Checkout Size**: `179M` (as verified via `du -sh`)

## 3. Rationale for Selection for DT049
- **Benchmark Stability**: HAI 21.03 served as the official evaluation dataset for the HAICon 2020 anomaly detection competition, providing published baselines for performance comparison.
- **Realistic Cyber-Physical Environment**: The dataset was collected on a Hardware-In-the-Loop (HIL) augmented ICS testbed integrating three physical subsystems (GE turbine, Emerson boiler, and FESTO water treatment) linked via a dSPACE HIL simulator.
- **Process Telemetry**: Provides 78 data points sampled at 1-second intervals. *(Note: whether all 78 points represent usable model features will be verified during the formal dataset audit).*
- **Rich Attack Scenarios**: Contains 50 distinct labeled attack scenarios distributed across 5 test splits with ground-truth anomaly intervals.
- **Documentation Sizing vs. Compressed Artifacts**: The official README table reports 471 MB of normal data across 352 hours and 205 MB of attack data across 112 hours (representing uncompressed CSV metrics). The actual sparse-checked-out compressed repository payload is 179 MB.

## 4. License and Documentation Inconsistency
An inconsistency exists within the official repository documentation regarding the license terms:
- **README License Section**: Explicitly states **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.
- **Dataset Metadata Section (Schema.org table)**: Lists the license as **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

This discrepancy is explicitly documented here for legal clarity. For project DT049, usage complies with the more restrictive ShareAlike terms (CC BY-SA 4.0) unless upstream clarifies otherwise.

## 5. Expected vs. Actual Downloaded Files
### Expected File Counts & Attack Distribution
- **Training Files (Normal Data)**: 3 files (train1, train2, train3) representing 352 total hours.
- **Testing Files (Attack Data)**: 5 files (test1 to test5) containing 50 attacks across 112 total hours:
  - `test1`: 5 attacks, 12 hours
  - `test2`: 20 attacks, 33 hours
  - `test3`: 8 attacks, 30 hours
  - `test4`: 5 attacks, 11 hours
  - `test5`: 12 attacks, 26 hours
- **Total Expected Files**: 8 files.
- **Total Expected Attacks**: 50 attacks.

### Actual Verified Files in Checkout
The sparse checkout confirmed exactly 8 files with `.csv.gz` extension:
1. `train1.csv.gz` (30,651,421 bytes)
2. `train2.csv.gz` (31,430,645 bytes)
3. `train3.csv.gz` (67,961,579 bytes)
4. `test1.csv.gz` (6,149,549 bytes)
5. `test2.csv.gz` (16,765,328 bytes)
6. `test3.csv.gz` (15,477,098 bytes)
7. `test4.csv.gz` (5,602,841 bytes)
8. `test5.csv.gz` (13,199,829 bytes)

*Note: All original dataset files remain compressed (`.csv.gz`) and unmodified in their source directory `data/source/hai_repo/hai-21.03/`.*

Integrity hashes for these files are recorded in `results/audit/hai_21_03_sha256.txt` and `results/audit/hai_21_03_file_manifest.txt`.

