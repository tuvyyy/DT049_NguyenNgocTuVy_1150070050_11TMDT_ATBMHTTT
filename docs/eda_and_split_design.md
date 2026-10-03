# Exploratory Data Analysis (EDA) and Leakage-Safe Split Design: DT049

## 1. Executive Summary & Split Terminology

This document establishes the exploratory analysis and methodological split framework for project DT049 using the **HAI 21.03 Security Dataset**.

### Methodological Rules Applied
1. **Zero Model Training & Scaling**: No models were trained, no scalers fit, and no thresholds tuned.
2. **Strict Test Data Isolation**: Test feature values (`test1` through `test5`) were **never accessed or read** for computing feature statistics, distributions, correlations, or variance checks.
3. **No Premature Feature Pruning**: All **79 candidate telemetry features** are preserved in the modeling candidate pool. Constancy metrics are recorded as descriptive facts, not selection decisions.
4. **Independent Recording Sessions**: Each `.csv.gz` file represents an independent recording session. Temporal order is preserved strictly **within each file**, and samples are **never concatenated across session boundaries**.

### Formal Split Terminology & Composition

| Split Role | Files Included | Calendar Range | Total Samples (1 Hz) | Duration (Hours) | Attacks | Role & Leakage Guard |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Development Training** | `train1.csv.gz`<br>`train2.csv.gz` | 2020-07-11 to 2020-07-13<br>2020-07-31 to 2020-08-03 | 442,802 | 123.00 h | 0 | Model parameter training and baseline feature statistics only. |
| **Normal Validation** | `train3.csv.gz` | 2020-08-04 to 2020-08-10 | 478,801 | 133.00 h | 0 | Validation loss, early stopping, operational drift evaluation, and false-alarm rate calibration. |
| **Official Final Attack Test** | `test1.csv.gz` to `test5.csv.gz` | 2020-07-07 to 2020-07-31 *(interleaved)* | 402,005 | 111.67 h | 50 (8,947 s) | **Completely sealed.** Evaluated once after freezing model, scaler, and threshold. |

---

## 2. Calendar Relationships & Non-Chronological Interleaving

The official ETRI dataset is **not globally chronological in calendar time**. The physical testbed was operated across July and August 2020 with interleaved testing and training campaigns:

```text
2020-07-07 15:00 ─── [test1: 12h, 5 attacks]
2020-07-09 15:00 ─── [test2: 33h, 20 attacks]
2020-07-11 00:00 ─── [train1: 60h, Normal Dev]  ◄── Starts immediately as test2 ends
2020-07-13 00:00 ─── [test3: 30h, 8 attacks]    ◄── 12-hour calendar overlap with train1!
2020-07-28 12:00 ─── [test4: 11h, 5 attacks]
2020-07-30 10:50 ─── [test5: 25.67h, 12 attacks]
2020-07-31 22:00 ─── [train2: 63h, Normal Dev]  ◄── Starts 9.5 hours after test5 ends
2020-08-04 22:00 ─── [train3: 133h, Normal Val] ◄── Starts 1.38 days after train2 ends
```

### Key Architectural Implications
1. **Overlap Clarification**: `train1` (ends 2020-07-13 12:00:00) and `test3` (starts 2020-07-13 00:00:00) share a 12-hour calendar period. On the physical testbed, these represent independent operational runs. Because `test3` is strictly isolated and never accessed during training, model-development leakage is prevented by the defined controls.
2. **No Cross-Session Windows**: Sliding temporal windows must never span across file boundaries. Each session begins at $t=0$ relative to its own recording.
3. **Validation Semantics**: `train3` contains **zero cyber attacks**. Therefore:
   - `train3` **can** evaluate reconstruction loss, prediction error, operational drift, and normal false-positive rates (FPR).
   - `train3` **cannot** be used to optimize attack recall, detection delay, or Event-based F1, as there are no positive attack labels in validation data.

---

## 3. Train-Only Feature Characterization

Using **Development Training data only (`train1` + `train2` = 442,802 samples)**, all 79 candidate telemetry features were characterized:
- Full metrics recorded in [`results/audit/hai_21_03_train_feature_stats.csv`](file:///home/kali/DT049/results/audit/hai_21_03_train_feature_stats.csv).
- Subsystem distribution:
  - **P1 (Boiler/Feedwater)**: 38 candidate features
  - **P2 (Turbine/Governor)**: 22 candidate features
  - **P3 (Water Treatment)**: 7 candidate features
  - **P4 (HIL Simulator/BOP)**: 12 candidate features

### Feature Constancy & Near-Zero Variance (NZV)
The analysis distinguishes between **descriptive facts** and **feature-selection decisions**:

- **Constant across Training Set (`train1` + `train2`)**: **22 features** exhibit zero variance ($\sigma = 0$, $\min = \max$):
  - `P1 (8)`: `P1_PCV02D`, `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`
  - `P2 (12)`: `P2_ASD`, `P2_AutoGO`, `P2_Emerg`, `P2_MSD`, `P2_ManualGO`, `P2_OnOff`, `P2_RTR`, `P2_TripEx`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`
  - `P3 (2)`: `P3_LH`, `P3_LL`
- **Constant in Validation (`train3`)**: The exact same **22 features** are constant in `train3`.
- **Intermittent Features**: `P1_PCV02D`, `P2_Emerg`, and `P2_OnOff` are constant in training and validation, but become dynamic in official test recordings.
- **Methodological Guard**: Any prior knowledge that particular training-constant features later become active in attack/test data belongs to descriptive dataset audit only and must not influence feature selection, preprocessing, window size, architecture, or threshold decisions.
- **Retention Decision**: **All 79 candidate features are retained** in [`results/audit/hai_21_03_feature_retention_analysis.csv`](file:///home/kali/DT049/results/audit/hai_21_03_feature_retention_analysis.csv). No features are removed at this stage.

---

## 4. Normal-to-Validation Operational Drift

Comparing the Development Training baseline (`train1` + `train2`, July 2020) against the Normal Validation set (`train3`, August 2020) reveals measurable operational drift across normal sessions:

| Feature | Subsystem | Train Mean ($\mu_{\text{tr}}$) | Val Mean ($\mu_{\text{val}}$) | Train Std ($\sigma_{\text{tr}}$) | Normalized Shift ($\frac{\Delta\mu}{\sigma_{\text{tr}}}$) | Physical Context |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `P1_PCV02Z` | P1 | 0.4485 | 0.3370 | 0.1142 | **0.9763 $\sigma$** | Pressure control valve position |
| `P2_SIT02` | P2 | 794.61 | 799.30 | 5.597 | **0.8379 $\sigma$** | Turbine speed indicator 2 |
| `P2_SIT01` | P2 | 794.60 | 799.28 | 5.594 | **0.8364 $\sigma$** | Turbine speed indicator 1 |
| `P1_B4002` | P1 | 32.06 | 32.49 | 0.7206 | **0.6024 $\sigma$** | Deaerator pressure |
| `P1_PCV01D` | P1 | 74.20 | 61.82 | 22.56 | **0.5489 $\sigma$** | Steam pressure control demand |
| `P4_ST_PS` | P4 | 4.887 | 3.659 | 2.331 | **0.5268 $\sigma$** | Steam turbine pressure setpoint |

> [!NOTE]
> This drift is physically normal in continuous process plants due to ambient environmental shifts, setpoint recalibrations, and loop retuning between distinct operational runs. Models that overfit to specific July setpoints will exhibit elevated reconstruction error on August validation data.

---

## 5. Representative Signal Selection (Training Data Only)

To visualize multi-loop dynamics without hard-coding or using constant signals, one informative, non-constant sensor was selected from each subsystem based strictly on Development Training data:

1. **P1 (Boiler)**: **`P1_LIT01`** (Boiler level transmitter)
   - Dynamic Range ($Q_{99} - Q_{01}$): 72.28 | $\sigma_{\text{train}} = 13.59$
   - Continuous closed-loop level tracking around feedwater setpoints.
2. **P2 (Turbine)**: **`P2_CO_rpm`** (Compressor speed)
   - Dynamic Range ($Q_{99} - Q_{01}$): 106.00 | $\sigma_{\text{train}} = 22.32$
   - Continuously varying rotating machinery speed. *(Note: `P2_VTR01` is constant and was rejected).*
3. **P3 (Water Treatment)**: **`P3_LIT01`** (Treated water level transmitter)
   - Dynamic Range ($Q_{99} - Q_{01}$): 14,232.00 | $\sigma_{\text{train}} = 4,287.34$
   - High-amplitude cycle oscillations. *(Note: `P3_LH` and `P3_LL` are constant switch states and were rejected).*
4. **P4 (HIL / Balance of Plant)**: **`P4_ST_PT01`** (Steam turbine pressure transmitter)
   - Dynamic Range ($Q_{99} - Q_{01}$): 112.00 | $\sigma_{\text{train}} = 19.38$
   - High-fidelity physical coupling signal from the dSPACE simulator.

---

## 6. Correlation Sampling Methodology

- **Source**: Development Training files only (`train1.csv.gz` and `train2.csv.gz`).
- **Sampling Scheme**: Deterministic stratified random sampling of **45,000 samples** (22,500 rows from `train1`, 22,500 rows from `train2`) using fixed random seed (`seed=42`).
- **Memory Optimization**: Correlation computed on $45,000 \times 79$ dataframe (~28 MB RAM).
- **Constant Features Handling**: Pearson correlation is mathematically undefined for zero-variance features ($\sigma = 0$). Any diagonal=1.0 and off-diagonal=0.0 values used for constant features in the correlation heatmap visualization are display conventions only, not measured correlations. Validation and test feature values were strictly excluded from the correlation calculation.

---

## 7. Class Imbalance Documentation (Descriptive Only)

Derived strictly from the label audit manifest [`results/audit/hai_21_03_attack_events.csv`](file:///home/kali/DT049/results/audit/hai_21_03_attack_events.csv) for documentation purposes only:
- **Total Test Samples**: 402,005 seconds (111.67 hours)
- **Normal Test Samples**: 393,058 seconds (97.77%)
- **Attack-Labeled Samples**: 8,947 seconds (**2.23% positive class ratio**)
- **Total Attack Events**: 50 distinct attacks across 5 test splits
- **Attack Duration Distribution**:
  - Minimum: **17 seconds** (0.28 minutes)
  - 25th Percentile: **98 seconds** (1.63 minutes)
  - Median: **178 seconds** (2.97 minutes)
  - Mean: **178.9 seconds** (2.98 minutes)
  - 75th Percentile: **256 seconds** (4.27 minutes)
  - Maximum: **422 seconds** (7.03 minutes)

> [!WARNING]
> **Methodological Restriction**: Attack durations must **not** be used to select model sliding-window lengths (e.g. tuning window size to 178s would be test-label leakage). Window length must be selected based purely on physical process time constants and model receptive fields.

---

## 8. Visual EDA Artifacts

All figures are saved under [`results/figures/eda/`](file:///home/kali/DT049/results/figures/eda/):

1. **[`eda_01_telemetry_overview.png`](file:///home/kali/DT049/results/figures/eda/eda_01_telemetry_overview.png)**: Time series of the 4 representative dynamic signals across `train1` and `train2`.
2. **[`eda_02_feature_variance_and_ranges.png`](file:///home/kali/DT049/results/figures/eda/eda_02_feature_variance_and_ranges.png)**: Top dynamic features ranked by $Q_{99} - Q_{01}$ range and standard deviation on training data.
3. **[`eda_03_distribution_drift_comparison.png`](file:///home/kali/DT049/results/figures/eda/eda_03_distribution_drift_comparison.png)**: Boxplot comparisons of normal distributions between `train1` (Dev), `train2` (Dev), and `train3` (Val).
4. **[`eda_04_train_feature_correlation_heatmap.png`](file:///home/kali/DT049/results/figures/eda/eda_04_train_feature_correlation_heatmap.png)**: $79 \times 79$ Pearson correlation heatmap generated from the 45,000-sample training subset (seed=42).
5. **[`eda_05_attack_duration_distribution.png`](file:///home/kali/DT049/results/figures/eda/eda_05_attack_duration_distribution.png)**: Histogram and boxplot of attack event durations (labeled strictly as *Descriptive Only*).

---

## 9. Proposed Future Preprocessing Design (Unexecuted Specification)

The following pipeline is planned for the subsequent preprocessing stage:

1. **Scaler Fitting**:
   - Fit scalers (`StandardScaler` or `RobustScaler`) **exclusively on Development Training data** (`train1` + `train2`).
   - Freeze the fitted scaler parameters $(\mu_{\text{train}}, \sigma_{\text{train}})$ or $(\text{median}_{\text{train}}, \text{IQR}_{\text{train}})$.
   - Apply `.transform()` to `train3` and `test1`–`test5` without re-fitting.
2. **Sliding Window Construction**:
   - Sliding windows of length $W$ with stride $S$ will be generated **independently within each file**.
   - No window may straddle the boundary between `train1` and `train2`, nor between train and validation.
   - For a file with $N$ rows, the number of generated windows is strictly $\lfloor (N - W) / S \rfloor + 1$.
3. **Threshold Calibration**:
   - Anomaly detection decision thresholds will be calibrated on `train3` validation reconstruction/prediction errors targeting a predefined normal False Alarm Rate (e.g. 1.0% or 0.5% FAR) without inspecting test data.
4. **Final Evaluation Protocol**:
   - Test files `test1` through `test5` will be ingested through the frozen pipeline only after all hyperparameters and thresholds are committed.
