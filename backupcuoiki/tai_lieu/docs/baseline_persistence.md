# Baseline 0: One-Step Persistence Predictor

## 1. Architectural & Theoretical Formulation

The **One-Step Persistence Predictor** serves as the initial non-parametric benchmark (Baseline 0) for anomaly detection on the HAI 21.03 testbed dataset. In physical industrial control systems (ICS) operating in steady state, physical process variables (temperatures, pressures, flow rates, rotational speeds) exhibit significant temporal continuity at high sampling frequencies (1 Hz). The persistence predictor models the process state as a martingale under normal operating conditions:

$$\hat{\mathbf{x}}_t = \mathbf{x}_{t-1}$$

where $\mathbf{x}_t \in \mathbb{R}^D$ denotes the standardized process state vector at target timestamp $t$, with $D = 79$ candidate process telemetry variables.

### Anomaly Residual & Score Function
The step residual vector $\mathbf{r}_t \in \mathbb{R}^D$ measures the one-second departure from the preceding observed state:

$$\mathbf{r}_t = \mathbf{x}_t - \hat{\mathbf{x}}_t = \mathbf{x}_t - \mathbf{x}_{t-1}$$

The primary scalar anomaly score $s_t$ is defined as the Root Mean Squared Error (RMSE) across all $D = 79$ standardized features:

$$s_t = \text{RMSE}(\mathbf{r}_t) = \sqrt{\frac{1}{D} \sum_{j=1}^D (x_{t,j} - x_{t-1, j})^2}$$

Descriptive auxiliary metrics computed chunk-wise for structural comparison include:
- **Mean Absolute Error (MAE)**: $\text{MAE}(\mathbf{r}_t) = \frac{1}{D} \sum_{j=1}^D |x_{t,j} - x_{t-1, j}|$
- **Maximum Absolute Residual**: $\text{MaxAbs}(\mathbf{r}_t) = \max_{j \in \{1,\dots,D\}} |x_{t,j} - x_{t-1, j}|$

As specified by project methodology, model selection, threshold calibration, and alarm generation strictly use the primary **RMSE** score.

---

## 2. Data Processing & Preserved Feature Space

### 2.1 Training-Only Standardization
All telemetry variables are transformed using the frozen `StandardScaler` fitted streaming via Welford accumulation (`partial_fit`) across the 442,802 samples of Development Training (`train1.csv.gz` and `train2.csv.gz`). 

### 2.2 Strict Feature Preservation
In accordance with methodological constraints:
- **No feature selection or filtering is applied.** All 79 candidate process features are retained.
- The 22 features identified during audit as zero-variance in the training sets (`scale = 1.0`, `mean = c`) are preserved. Under normal training and validation conditions, their persistence differences are identically zero ($0.0 - 0.0 = 0.0$). Should any of these variables deviate during cyber-physical attacks, their raw standardized departures $(x - \mu_{\text{train}})$ will enter the score function without artificial suppression.

### 2.3 Strict Session Boundary Isolation
Residual differencing is executed independently within each recording session:
- **`train1`**: $N_1 = 216,001$ samples $\implies 216,000$ scored transitions ($60.00$ hours).
- **`train2`**: $N_2 = 226,801$ samples $\implies 226,800$ scored transitions ($63.00$ hours).
- **`train3`** (Normal Validation): $N_3 = 478,801$ samples $\implies 478,800$ scored transitions ($133.00$ hours).

**No difference, window, or prediction spans across file boundaries.** The boundary transition between `train1` and `train2` is never scored. Official attack test sessions (`test1`–`test5`) were not accessed, inspected, or transformed.

---

## 3. Threshold Calibration Protocol

### 3.1 Non-Monotonicity and Event-Rate Calibration Constraints
A critical methodological finding in continuous-time alarm generation is that the number of contiguous alarm events $E(\theta)$ is **non-monotonic** with respect to threshold $\theta$:
1. Raising $\theta$ can cause a single long alarm event to fragment into multiple shorter disjoint events, temporarily *increasing* the reported event count.
2. Lowering $\theta$ can cause distinct disjoint events to bridge across transient dips into a single persistent alarm, *decreasing* the reported event count.
3. In the pathological extreme ($\theta \to 0$), a continuous false alarm spanning the entire multi-day validation session yields exactly $E = 1$ event (an apparent rate of $\approx 0.0075$ events/hour) despite an unusable alarm occupancy of 100%.

Consequently, binary-searching event rates as an objective function is invalid.

### 3.2 Predetermined Empirical Quantile Protocol
Threshold calibration is conducted exclusively using predetermined empirical quantiles of the **Normal Validation (`train3`)** RMSE scores ($N = 478,800$ samples, $133.0$ hours of continuous normal operation):

$$\theta_q = \text{Quantile}_{\text{linear}}(S_{\text{val}}, q)$$

Three operating regimes are established:
1. **Primary Operating Point ($q = 0.999$, Q99.9)**: Designed for low false-alarm frequency in industrial settings.
2. **Sensitivity Lower Bound ($q = 0.995$, Q99.5)**: Evaluates high-sensitivity alarm behavior.
3. **Sensitivity Upper Bound ($q = 0.9999$, Q99.99)**: Evaluates extreme-tail alarm behavior.

### 3.3 Alarm and Contiguous Event Definition
- **Point Alarm Condition**: Strict inequality $s_t > \theta$. (Boundary equality $s_t = \theta$ is classified as normal).
- **Contiguous Alarm Event**: A maximal contiguous sequence of seconds $\{t_{\text{start}}, \dots, t_{\text{end}}\}$ where $s_t > \theta$ at every second. No post-processing smoothing, hysteresis deadbands, debounce timers, or duration filtering are applied in this raw baseline.

---

## 4. Empirical Score Distributions

Both Development Training and Normal Validation exhibit tight, right-skewed score distributions with strong statistical alignment, confirming the temporal stability of the testbed under normal operations.

| Statistic | Development Training (`train1` + `train2`) | Normal Validation (`train3`) |
| :--- | :---: | :---: |
| **Total Scored Samples** | 442,800 | 478,800 |
| **Exposure Duration** | 123.00 hours | 133.00 hours |
| **Minimum Score** | 0.007724 | 0.008639 |
| **25th Percentile (Q1)** | 0.232811 | 0.246679 |
| **Median (50th Percentile)** | 0.289934 | 0.304545 |
| **Mean** | 0.307635 | 0.322667 |
| **Standard Deviation** | 0.140514 | 0.143530 |
| **75th Percentile (Q3)** | 0.355022 | 0.370498 |
| **95th Percentile (Q95)** | 0.479565 | 0.499054 |
| **99th Percentile (Q99)** | 0.848787 | 0.882807 |
| **99.5th Percentile (Q99.5)** | 1.266769 | 1.293395 |
| **99.9th Percentile (Q99.9)** | 1.675494 | **1.680523** |
| **99.99th Percentile (Q99.99)**| 1.863256 | 1.866531 |
| **Maximum Score** | 3.230422 | 4.273721 |

The validation Q99.9 score ($1.680523$) differs from the training Q99.9 score ($1.675494$) by only **0.30%**, demonstrating that normal operating dynamics remained highly stationary between the July and August recording campaigns.

![Score Distributions](../results/figures/baseline_persistence/fig_01_score_distribution_comparison.png)

---

## 5. Threshold Calibration & Evaluation Results

Calibration on `train3` normal validation yields the following operating performance. The calibrated thresholds are also applied to Development Training as an independent normal consistency check (without cross-file bridging).

| Quantile | Target Role | Calibrated Threshold $\theta$ | Val Exceedances (pts / %) | Val Alarm Sec / Hour | Val False Alarm Events | Val False Alarm Rate (ev/h) | Train False Alarm Rate (ev/h) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q99.5** | Sensitivity Lower | `1.293395` | 2,394 (0.5000%) | 18.00 s/h | 1,454 | 10.932 ev/h | 10.398 ev/h |
| **Q99.9** | **Primary Operating Point** | **`1.680523`** | **479 (0.1000%)** | **3.60 s/h** | **471** | **3.541 ev/h** | **3.455 ev/h** |
| **Q99.99**| Sensitivity Upper | `1.866531` | 48 (0.0100%) | 0.36 s/h | 41 | 0.308 ev/h | 0.309 ev/h |

### Key Observations:
1. **Primary Operating Point (Q99.9)**:
   - Sets threshold $\theta^* = 1.680523$.
   - Generates **3.541 false-alarm events per hour** (471 events over 133 hours).
   - Generates **3.60 alarm seconds per hour** ($0.100\%$ total point occupancy).
   - Independent verification on Development Training yields **3.455 events per hour** and **3.50 alarm seconds per hour** ($0.097\%$ point occupancy), confirming excellent consistency across distinct sessions.
2. **Sensitivity Lower Bound (Q99.5)**:
   - At $\theta = 1.293395$, the false-alarm rate triples to **10.932 events/hour** ($18.00$ alarm seconds/hour), representing excessive operator fatigue.
3. **Sensitivity Upper Bound (Q99.99)**:
   - At $\theta = 1.866531$, the false-alarm rate drops to **0.308 events/hour** (roughly one false alarm every 3.25 hours, or $0.36$ alarm seconds/hour).

![Validation Timeline](../results/figures/baseline_persistence/fig_02_validation_score_timeline.png)

---

## 6. Alarm Morphology & Event Duration Profiling

Examining contiguous alarm runs reveals the temporal structure of normal baseline false alarms:

| Operating Threshold | Scored Events | Mean Duration (s) | Median Duration (s) | Max Duration (s) |
| :---: | :---: | :---: | :---: | :---: |
| **Q99.5** ($\theta = 1.293395$) | 1,454 | 1.65 s | 2.0 s | 3 s |
| **Q99.9** ($\theta = 1.680523$) | **471** | **1.02 s** | **1.0 s** | **3 s** |
| **Q99.99** ($\theta = 1.866531$) | 41 | 1.17 s | 1.0 s | 3 s |

### Duration Insights:
- At the **primary Q99.9 operating point**, the mean duration is **1.02 seconds** and median is **1.0 second**.
- Out of 471 alarm events on validation data, **463 events (98.3%) are exactly 1 second long** (isolated single-sample step spikes). Only 8 events persist for 2 seconds, and zero events exceed 3 seconds.
- This demonstrates that persistence false alarms are fundamentally transient step-discontinuity spikes (jitter/noise) rather than sustained anomalous trajectories. Future temporal smoothing or persistence persistence filters (e.g. minimum event duration $\ge 3$s) could suppress nearly 100% of these normal false alarms, though no such filtering is applied to this baseline.

![Alarm Durations](../results/figures/baseline_persistence/fig_03_false_alarm_event_durations.png)

---

## 7. Process Variable Contribution Analysis

Evaluating root mean squared residuals per feature across `train3` identifies the process sensors that undergo the largest high-frequency changes from second to second:

| Rank | Feature Name | Subsystem | Description / Role | Mean RMS Step Difference |
| :---: | :---: | :---: | :---: | :---: |
| 1 | `P2_24Vdc` | P2 (Turbine) | 24V DC power rail monitoring | 1.3905 |
| 2 | `P4_HT_FD` | P4 (HIL) | Hardware-in-the-loop feed variable | 1.2277 |
| 3 | `P2_HILout` | P2 (Turbine) | HIL interface signal | 1.2104 |
| 4 | `P4_ST_FD` | P4 (HIL) | Hardware-in-the-loop steam turbine variable | 1.1750 |
| 5 | `P2_CO_rpm` | P2 (Turbine) | Turbine rotational speed (rpm) | 0.9175 |
| 6 | `P2_VT01` | P2 (Turbine) | Turbine vibration transmitter | 0.6852 |
| 7 | `P2_VYT02` | P2 (Turbine) | Turbine Y-axis vibration transmitter | 0.6292 |
| 8 | `P2_VXT02` | P2 (Turbine) | Turbine X-axis vibration transmitter | 0.6208 |
| 9 | `P2_SIT01` | P2 (Turbine) | Speed indicating transmitter 1 | 0.6050 |
| 10 | `P2_SIT02` | P2 (Turbine) | Speed indicating transmitter 2 | 0.6009 |

Residual dynamics are dominated by:
1. Electrical sensor signals (`P2_24Vdc`) exhibiting high-frequency quantization or electrical noise.
2. Hardware-in-the-loop coupling signals (`P4_HT_FD`, `P2_HILout`, `P4_ST_FD`).
3. High-speed turbine rotational and vibration measurements (`P2_CO_rpm`, `P2_VT01`, `P2_VYT02`, `P2_VXT02`, `P2_SIT01`, `P2_SIT02`).

In contrast, slower thermal and hydraulic processes in P1 (Boiler) and P3 (Water Treatment) have substantially smaller one-step residuals under normal operation.

![Top Residual Contributors](../results/figures/baseline_persistence/fig_04_top_residual_contributors.png)

---

## 8. Computational Performance & Resource Profile

All computations were executed on the local VM using pure NumPy streaming over memory-mapped arrays:

- **Total Execution Time**: **2.33 seconds** (scoring 921,600 transitions across 79 features and generating 4 figures).
- **Peak Process Memory (RSS)**: **270.27 MB** (well within the Kali VM's 4 GB memory ceiling).
- **Zero GPU or Deep Learning Dependencies**: Fully framework-independent.

---

## 9. Verification of Leakage Safeguards

| Safeguard | Status | Verification Evidence |
| :--- | :---: | :--- |
| **Zero Test Set Exposure** | **VERIFIED** | `test1.csv.gz` through `test5.csv.gz` were not read, decompressed, preprocessed, or evaluated. |
| **Zero Attack Label Ingestion** | **VERIFIED** | No attack labels (`attack`, `attack_P1`, etc.) were loaded or utilized. |
| **Threshold Calibration Source** | **VERIFIED** | Thresholds were calibrated strictly on `train3` normal validation data. |
| **Strict Session Isolation** | **VERIFIED** | No differencing or predictions bridged `train1`, `train2`, and `train3`. |
| **All Features Preserved** | **VERIFIED** | All 79 features (including 22 zero-variance features) are retained in primary score. |

---

## 10. Summary Artifacts

The execution produced the following reproducible artifacts:
- **Metrics Table**: `results/metrics/baseline_persistence_validation.csv`
- **Thresholds & Metadata JSON**: `results/metrics/baseline_persistence_thresholds.json`
- **Configuration**: `configs/baseline_persistence.yaml`
- **Figures**:
  - `results/figures/baseline_persistence/fig_01_score_distribution_comparison.png`
  - `results/figures/baseline_persistence/fig_02_validation_score_timeline.png`
  - `results/figures/baseline_persistence/fig_03_false_alarm_event_durations.png`
  - `results/figures/baseline_persistence/fig_04_top_residual_contributors.png`
