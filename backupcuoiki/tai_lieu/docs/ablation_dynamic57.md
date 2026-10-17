# DT049 Post-Hoc Exploratory Ablation: Train-Dynamic Feature Score

**Document Version:** 1.0  
**Ablation Name:** Train-Dynamic Feature Score Ablation (79-feature primary score vs. 57-feature dynamic-only score)  
**Status:** Post-Hoc Exploratory Variant (Completed)  
**Scientific Constraint:** *This exploratory analysis was conducted after official final-test evaluation. It provides diagnostic physical insight and does NOT modify or replace the frozen primary final-test experiment (`dt049-final-test-v1`).*  
**Repository State:** Frozen at Git Commit [`331cc07cb72b8b3f30a64bbe7af52248bd0cd9ba`](file:///home/kali/DT049)  
**Execution Timestamp:** 2026-10-03  

---

## 1. Motivation & Scientific Objective

The DT049 primary anomaly detector aggregates prediction errors across all 79 standardized telemetry channels:
$$\text{RMSE}_{79}(t) = \sqrt{\frac{1}{79} \sum_{i=1}^{79} \left(y_{t, i} - \hat{y}_{t, i}\right)^2}$$

During initial feature analysis of normal development training data (`train1.csv.gz` and `train2.csv.gz`), exactly **22 channels exhibited zero empirical variance** ($\sigma = 0$), representing static actuator command bits or unactuated subsystems during baseline plant operation. The remaining **57 channels exhibited active continuous dynamics** ($\sigma > 0$).

This ablation addresses the core scientific question:
> **"How much detection information is contributed by channels that were constant during normal development training?"**

Specifically, if the anomaly score is computed strictly over the 57 dynamic channels:
$$\text{RMSE}_{57}(t) = \sqrt{\frac{1}{57} \sum_{i \in \mathcal{D}_{57}} \left(y_{t, i} - \hat{y}_{t, i}\right)^2}$$
does detection sensitivity or false-alarm rate improve, degrade, or remain invariant?

---

## 2. Feature Partition & Training-Only Provenance

### 2.1 Confirmation of Training-Only Partition
The 57/22 feature split was determined **strictly and exclusively from normal development training data** (`train1.csv.gz` and `train2.csv.gz`, 442,802 training samples) as documented in the frozen [`models/preprocessing/scaler_metadata.json`](file:///home/kali/DT049/models/preprocessing/scaler_metadata.json) (commit `6ce11fb8bcb682c10e125219d0b964f394b8a5d2`). Official test data (`test1`–`test5`) was completely sealed and never inspected during feature selection.

### 2.2 The 22 Excluded Training-Constant Channels
The 22 excluded channels whose development training variance was zero are:
1. `P1_PCV02D`
2. `P1_PP01AD`
3. `P1_PP01AR`
4. `P1_PP01BD`
5. `P1_PP01BR`
6. `P1_PP02D`
7. `P1_PP02R`
8. `P1_STSP`
9. `P2_ASD`
10. `P2_AutoGO`
11. `P2_Emerg`
12. `P2_MSD`
13. `P2_ManualGO`
14. `P2_OnOff`
15. `P2_RTR`
16. `P2_TripEx`
17. `P2_VTR01`
18. `P2_VTR02`
19. `P2_VTR03`
20. `P2_VTR04`
21. `P3_LH`
22. `P3_LL`

### 2.3 The 57 Retained Dynamic Channels (Exact Ordered List)
The 57 retained dynamic channels (in column order) are:
`P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_B4005`, `P1_B400B`, `P1_B4022`, `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02D`, `P1_FCV02Z`, `P1_FCV03D`, `P1_FCV03Z`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_FT03Z`, `P1_LCV01D`, `P1_LCV01Z`, `P1_LIT01`, `P1_PCV01D`, `P1_PCV01Z`, `P1_PCV02Z`, `P1_PIT01`, `P1_PIT02`, `P1_TIT01`, `P1_TIT02`, `P2_24Vdc`, `P2_CO_rpm`, `P2_HILout`, `P2_SIT01`, `P2_SIT02`, `P2_VT01`, `P2_VXT02`, `P2_VXT03`, `P2_VYT02`, `P2_VYT03`, `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_LIT01`, `P3_PIT01`, `P4_HT_FD`, `P4_HT_LD`, `P4_HT_PO`, `P4_HT_PS`, `P4_LD`, `P4_ST_FD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`, `P4_ST_PS`, `P4_ST_PT01`, `P4_ST_TT01`.

---

## 3. Strict Frozen Execution Rules

In strict adherence to post-test protocol:
1. **No Retraining:** The exact frozen weights of Baseline 0 (Persistence), Baseline 1 (Ridge $\alpha=300$), and Primary GRU ($W=30, H=64$, seeds 42, 43, 44) were used without retraining.
2. **Threshold Calibration on Validation Only:** For the 57-feature score, thresholds were recalibrated **only** on the 478,771 scored stride-1 transitions of normal validation dataset `train3_features.npy` at $Q_{99.5}$, $Q_{99.9}$, and $Q_{99.99}$. Test data and test labels were never accessed during threshold calibration.
3. **Preserved Test Evaluation Protocol:** Evaluation on official `test1`–`test5` used the identical common timeline ($t \ge 30$), deterministic 1-to-1 event matching, detection delay calculation, normal false-alarm definition, and session independence.

---

## 4. Mathematical Scale Rescaling on Normal Validation Data

During normal validation operation (`train3_features.npy`), the 22 training-constant channels have identically zero empirical variance and zero residual ($e_{t, i} = 0$ for $i \notin \mathcal{D}_{57}$).

Where the 22 excluded residuals are zero:
$$\sum_{i=1}^{79} e_{t, i}^2 = \sum_{i \in \mathcal{D}_{57}} e_{t, i}^2 \implies \text{RMSE}_{57}(t) = \sqrt{\frac{79}{57}} \cdot \text{RMSE}_{79}(t) = \kappa \cdot \text{RMSE}_{79}(t)$$

The exact constant scale factor is:
$$\kappa = \sqrt{\frac{79}{57}} \approx 1.177270$$
*(Note: $\kappa = \sqrt{79/57} \approx 1.177270$, whereas the dimension ratio itself is $79/57 \approx 1.385965$.)*

Because quantile thresholds are strictly monotonic order statistics calibrated on normal validation data where these 22 residuals are zero, the validation thresholds scale exactly by $\kappa$:
$$\theta_{57}^* = \kappa \cdot \theta_{79}^*$$

### Validation Threshold Calibration Results:

| Model ID | Operating Point | Primary Thresh $\theta_{79}^*$ | Ablation Thresh $\theta_{57}^*$ | Empirical Ratio ($\theta_{57}^* / \theta_{79}^*$) | Theoretical $\kappa = \sqrt{79/57}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `persistence` | $Q_{99.5}$ | 1.293395 | 1.522700 | **1.177290** | 1.177270 |
| `persistence` | **$Q_{99.9}$** | **1.680523** | **1.978435** | **1.177273** | 1.177270 |
| `persistence` | $Q_{99.99}$ | 1.866531 | 2.197414 | **1.177271** | 1.177270 |
| `ridge` | $Q_{99.5}$ | 0.851713 | 1.002705 | **1.177274** | 1.177270 |
| `ridge` | **$Q_{99.9}$** | **1.400928** | **1.649274** | **1.177272** | 1.177270 |
| `ridge` | $Q_{99.99}$ | 1.484236 | 1.747349 | **1.177273** | 1.177270 |
| `gru_seed42` | $Q_{99.5}$ | 0.695715 | 0.819044 | **1.177270** | 1.177270 |
| `gru_seed42` | **$Q_{99.9}$** | **1.058582** | **1.246237** | **1.177270** | 1.177270 |
| `gru_seed42` | $Q_{99.99}$ | 1.376288 | 1.620263 | **1.177270** | 1.177270 |
| `gru_seed43` | **$Q_{99.9}$** | **1.195869** | **1.407860** | **1.177270** | 1.177270 |
| `gru_seed44` | **$Q_{99.9}$** | **1.076832** | **1.267722** | **1.177270** | 1.177270 |

---

## 5. Comparative Evaluation Results ($Q_{99.9}$ Operating Point)

From [`results/metrics/ablation_dynamic57_summary.csv`](file:///home/kali/DT049/results/metrics/ablation_dynamic57_summary.csv):

| Model ID | Configuration | Feats ($D$) | Thresh $\theta^*$ | $TP$ | $FN$ | $FP_{ev}$ | Event Prec | Event Rec | Event F1 | Normal FA (ev/h) | FP Time (s/h) | Point FPR | Med. Delay |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `persistence` | Primary | 79 | 1.680523 | 22 | 28 | 430 | 0.0487 | 0.4400 | 0.0876 | 3.7016 | 3.9215 | 0.001089 | 40.5 s |
| `persistence` | **Ablation** | **57** | **1.978435** | **22** | **28** | **430** | **0.0487** | **0.4400** | **0.0876** | **3.7016** | **3.9215** | **0.001089** | **40.5 s** |
| `ridge` | Primary | 79 | 1.400928 | 35 | 15 | 515 | 0.0636 | 0.7000 | 0.1167 | 4.4804 | 6.5237 | 0.001812 | 13.0 s |
| `ridge` | **Ablation** | **57** | **1.649274** | **35** | **15** | **515** | **0.0636** | **0.7000** | **0.1167** | **4.4804** | **6.5237** | **0.001812** | **13.0 s** |
| `gru_seed42` | Primary | 79 | 1.058582 | 40 | 10 | 335 | 0.1067 | 0.8000 | 0.1882 | 2.8220 | 5.8915 | 0.001637 | 22.0 s |
| `gru_seed42` | **Ablation** | **57** | **1.246237** | **40** | **10** | **335** | **0.1067** | **0.8000** | **0.1882** | **2.8220** | **5.8915** | **0.001637** | **22.0 s** |
| `gru_seed43` | Primary | 79 | 1.195869 | 39 | 11 | 317 | 0.1096 | 0.7800 | 0.1921 | 2.7212 | 5.5433 | 0.001540 | 21.0 s |
| `gru_seed43` | **Ablation** | **57** | **1.407860** | **39** | **11** | **317** | **0.1096** | **0.7800** | **0.1921** | **2.7212** | **5.5433** | **0.001540** | **21.0 s** |
| `gru_seed44` | Primary | 79 | 1.076832 | 41 | 9 | 275 | 0.1297 | 0.8200 | 0.2240 | 2.2906 | 5.3142 | 0.001476 | 24.0 s |
| `gru_seed44` | **Ablation** | **57** | **1.267722** | **41** | **9** | **275** | **0.1297** | **0.8200** | **0.2240** | **2.2906** | **5.3142** | **0.001476** | **24.0 s** |

---

## 6. GRU Multi-Seed Aggregate (Seeds 42, 43, 44)

From [`results/metrics/ablation_dynamic57_multiseed.csv`](file:///home/kali/DT049/results/metrics/ablation_dynamic57_multiseed.csv):

| Configuration | Features | Event Recall | Event Precision | Event F1 | Normal FA (ev/h) | FP Time (s/h) | Point FPR | Median Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary** | 79 | $0.8000 \pm 0.0163$ | $0.1153 \pm 0.0103$ | $0.2015 \pm 0.0160$ | $2.6113 \pm 0.2304$ | $5.5830 \pm 0.2373$ | $0.001551 \pm 0.000066$ | $22.33 \pm 1.25\text{ s}$ |
| **Ablation** | 57 | $0.8000 \pm 0.0163$ | $0.1153 \pm 0.0103$ | $0.2015 \pm 0.0160$ | $2.6113 \pm 0.2304$ | $5.5830 \pm 0.2373$ | $0.001551 \pm 0.000066$ | $22.33 \pm 1.25\text{ s}$ |

The 3-seed aggregate performance is identical to 4 decimal places across all primary evaluation metrics.

---

## 7. Event-Level Change Analysis

From [`results/metrics/ablation_dynamic57_changed_events.csv`](file:///home/kali/DT049/results/metrics/ablation_dynamic57_changed_events.csv):

- **79 Detected $\to$ 57 Missed:** **0 attacks**
- **79 Missed $\to$ 57 Detected:** **0 attacks**
- **Total Changed Attacks Across All Models:** **0 out of 50 ($0.0\%$)**
- **Maximum Absolute Delay Difference Across All Detected Attacks:** **$0.0000\text{ seconds}$**

Every single attack detected under the 79-feature score was detected under the 57-feature score with identical onset timestamps and identical detection delays.

---

## 8. Scientific Interpretation

**"How much detection information is contributed by channels that were constant during normal development training?"**

Exact constant rescaling applies on normal validation because the 22 excluded channels have zero residual there:
$$\text{RMSE}_{57}(t) = \kappa \cdot \text{RMSE}_{79}(t), \quad \kappa = \sqrt{\frac{79}{57}} \approx 1.177270$$

On the official test set, some excluded channels become active (notably `P1_PCV02D`, `P2_Emerg`, `P2_OnOff`), so exact score scaling need not hold. Empirically, their contribution was not large enough to change any Q99.9 alarm decision, attack detection status, or detection delay in this experiment.

**Conclusion:** The 22 training-constant channels did not alter detector decisions under this post-hoc Q99.9 ablation on HAI 21.03. The primary 79-feature score remains the frozen primary benchmark.

---

## 9. Generated Artifacts

- Documentation: [`docs/ablation_dynamic57.md`](file:///home/kali/DT049/docs/ablation_dynamic57.md)
- Metric Tables:
  * [`results/metrics/ablation_dynamic57_summary.csv`](file:///home/kali/DT049/results/metrics/ablation_dynamic57_summary.csv)
  * [`results/metrics/ablation_dynamic57_multiseed.csv`](file:///home/kali/DT049/results/metrics/ablation_dynamic57_multiseed.csv)
  * [`results/metrics/ablation_dynamic57_changed_events.csv`](file:///home/kali/DT049/results/metrics/ablation_dynamic57_changed_events.csv)
- Figures in [`results/figures/ablation_dynamic57/`](file:///home/kali/DT049/results/figures/ablation_dynamic57/):
  * `fig_ab_01_primary_vs_dynamic57.png`
  * `fig_ab_02_detection_changes.png`
  * `fig_ab_03_false_alarm_tradeoff.png`
- Execution Script: [`src/run_ablation_dynamic57.py`](file:///home/kali/DT049/src/run_ablation_dynamic57.py)
