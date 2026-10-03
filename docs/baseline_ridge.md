# Baseline 1: One-Step Multi-Output Ridge Predictor

## 1. Architectural & Theoretical Formulation

Baseline 1 transitions from the non-parametric persistence assumption ($\hat{\mathbf{x}}_{t+1} = \mathbf{x}_t$) to a learned linear state transition model:

$$\hat{\mathbf{x}}_{t+1} = \mathbf{x}_t W^T + \mathbf{b}^T$$

where $\mathbf{x}_t \in \mathbb{R}^D$ denotes the standardized process state vector at timestamp $t$ across all $D = 79$ candidate telemetry features, $W \in \mathbb{R}^{D \times D}$ represents the transition coefficient weight matrix, and $\mathbf{b} \in \mathbb{R}^D$ is the intercept vector.

### Objective Function & Cholesky Optimization
Ridge regression optimizes the regularized multi-output Frobenius-norm objective:

$$\min_{W, \mathbf{b}} \frac{1}{2} \|Y - (X W^T + \mathbf{1} \mathbf{b}^T)\|_F^2 + \frac{\alpha}{2} \|W\|_F^2$$

With centered data, the weights solve the regularized normal equations:

$$(X^T X + \alpha I) W^T = X^T Y$$

Because the number of training transitions ($N = 442,800$) significantly exceeds the feature dimensionality ($D = 79$) with ratio $N / D \approx 5,605$, solving via the compact $79 \times 79$ Gram matrix $(X^T X + \alpha I)$ using the **Cholesky solver** (`solver="cholesky"`, `fit_intercept=True`) provides an exact, non-iterative, deterministic closed-form solution in $O(N D^2 + D^3)$ operations.

### Anomaly Residual & Score Function
For target timestamp $t+1$, the residual vector is:

$$\mathbf{r}_{t+1} = \mathbf{x}_{t+1} - \hat{\mathbf{x}}_{t+1}$$

The primary anomaly score $s_{t+1}$ is the Root Mean Squared Error (RMSE) across all $D = 79$ standardized feature residuals:

$$s_{t+1} = \text{RMSE}(\mathbf{r}_{t+1}) = \sqrt{\frac{1}{D} \sum_{j=1}^D (x_{t+1, j} - \hat{x}_{t+1, j})^2}$$

All squared-error sums and validation aggregate statistics are accumulated in `float64` precision to avoid numerical precision loss.

---

## 2. Zero-Variance Feature Preservation

All 79 candidate telemetry features are retained in the model:
- The 22 features identified during audit as zero-variance across the training data (`scale = 1.0`, `mean = c`) are standardized to identically zero ($X_{:, j} = 0$).
- In the regularized Gram matrix $(X^T X + \alpha I)$, the diagonal entries for these columns equal $0 + \alpha = \alpha > 0$. For any positive regularization parameter ($\alpha \ge 0.01$), the system is strictly positive definite.
- Furthermore, because $X_{:, j} = 0$, $X^T Y$ is identically zero for these rows, resulting in exact zero learned input weights:

$$W_{j, :} = \mathbf{0}$$

- The model assigns zero weight to these features during normal operations, but if an attack induces anomalous variations in test data, their standardized departures $(x - \mu_{\text{train}})$ will enter the score function directly without artificial suppression.

---

## 3. Session Isolation & Pair Construction

Transition pairs are constructed strictly within recording sessions:
- **`train1`**: $N_1 = 216,001$ rows $\implies X_1 = \text{train1}[0:216000]$, $Y_1 = \text{train1}[1:216001]$ ($216,000$ transitions, $60.00$ hours).
- **`train2`**: $N_2 = 226,801$ rows $\implies X_2 = \text{train2}[0:226800]$, $Y_2 = \text{train2}[1:226801]$ ($226,800$ transitions, $63.00$ hours).
- **Development Training**: Vertically stacked $X_{\text{train}} = [X_1; X_2]$, $Y_{\text{train}} = [Y_1; Y_2]$ ($442,800$ transitions, $123.00$ hours). **No transition pair connects the end of `train1` to the start of `train2`.**
- **Normal Validation (`train3`)**: $N_3 = 478,801$ rows $\implies X_3 = \text{train3}[0:478800]$, $Y_3 = \text{train3}[1:478801]$ ($478,800$ transitions, $133.00$ hours), evaluated in complete isolation.
- Official attack test sessions (`test1`–`test5`) and attack labels were not accessed, transformed, or evaluated.

---

## 4. Model Selection & Boundary-Extension Check

In the initial grid (`[0.01, 0.1, 1, 10, 100]`), $\alpha = 100$ was selected at the boundary. To ensure the optimal hyperparameter was not truncated, a single predetermined boundary-extension check was conducted across the fixed 13-point grid spanning eight orders of magnitude:

$$\alpha \in [0.01, 0.1, 1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]$$

| Candidate Alpha $\alpha$ | Validation Mean RMSE | Validation Median RMSE | Validation Q99 RMSE | Validation Max RMSE | Fit Duration (s) | Eval Duration (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.01` | 0.24613451 | 0.232832 | 0.614476 | 3.639762 | 4.333s | 0.475s |
| `0.1` | 0.24613757 | 0.232837 | 0.614563 | 3.639692 | 0.284s | 0.329s |
| `1.0` | 0.24611376 | 0.232815 | 0.614560 | 3.639551 | 0.215s | 0.375s |
| `10.0` | 0.24598840 | 0.232667 | 0.614949 | 3.638827 | 0.239s | 0.341s |
| `100.0` | 0.24571059 | 0.232407 | 0.618588 | 3.637705 | 0.223s | 0.347s |
| **`300.0`** | **0.24568216** | **0.232257** | **0.621933** | **3.637515** | **0.220s** | **0.360s** |
| `1000.0` | 0.24588542 | 0.232268 | 0.626580 | 3.637498 | 0.437s | 0.376s |
| `3000.0` | 0.24648787 | 0.232737 | 0.630208 | 3.637747 | 0.243s | 0.368s |
| `10000.0` | 0.24896339 | 0.235093 | 0.633853 | 3.638697 | 0.228s | 0.367s |
| `30000.0` | 0.25735935 | 0.243128 | 0.640199 | 3.640922 | 0.230s | 0.368s |
| `100000.0` | 0.28361815 | 0.269222 | 0.652503 | 3.645566 | 0.224s | 0.368s |
| `300000.0` | 0.34149909 | 0.328369 | 0.685769 | 3.651744 | 0.231s | 0.421s |
| `1000000.0`| 0.46319497 | 0.454311 | 0.790897 | 3.662146 | 0.249s | 0.559s |

### Optimization Findings:
1. **Grid Optimization Result**: The best regularization setting within the predetermined extended grid was alpha = 300, and the minimum occurred away from the grid boundaries ($\mu_{\text{val}} = 0.24568216$).
2. **Convex U-Curve**: As regularization increases beyond $\alpha = 300$, error steadily climbs ($0.245885$ at $\alpha=1000$, $0.283618$ at $\alpha=10^5$, and $0.463195$ at $\alpha=10^6$).
3. **Tie-Break Check**: No candidates tied within numerical tolerance (`np.isclose`, $\text{atol} = 10^{-6}, \text{rtol} = 10^{-5}$). $\alpha = 300.0$ was uniquely optimal.
4. **Permanent Freezing**: Having verified an interior optimum, Baseline 1 is permanently frozen at **$\alpha^* = 300.0$**.

![Alpha Selection](../results/figures/baseline_ridge/fig_01_alpha_selection_rmse.png)

---

## 5. Anomaly Score Distributions

Evaluating the frozen Ridge model ($\alpha^* = 300.0$) yields the following residual score distributions:

| Statistic | Development Training (`train1` + `train2`) | Normal Validation (`train3`) |
| :--- | :---: | :---: |
| **Scored Transitions** | 442,800 | 478,800 |
| **Scored Duration** | 123.00 hours | 133.00 hours |
| **Minimum** | 0.036795 | 0.036851 |
| **25th Percentile (Q1)** | 0.173823 | 0.187221 |
| **Median (50th Percentile)** | 0.217441 | 0.232257 |
| **Mean** | 0.231354 | 0.245682 |
| **Standard Deviation** | 0.104970 | 0.106999 |
| **75th Percentile (Q3)** | 0.267635 | 0.283000 |
| **95th Percentile (Q95)** | 0.363569 | 0.377716 |
| **99th Percentile (Q99)** | 0.598074 | 0.621933 |
| **99.5th Percentile (Q99.5)** | 0.810549 | 0.851713 |
| **99.9th Percentile (Q99.9)** | 1.403409 | **1.400928** |
| **99.99th Percentile (Q99.99)**| 1.488352 | 1.484236 |
| **Maximum** | 2.697148 | 3.637515 |

The validation Q99.9 score ($1.400928$) and training Q99.9 score ($1.403409$) differ by only **$0.18\%$**. Rather than claiming global process stationarity across disparate operating regimes, we observe specifically that **the upper-tail residual-score distributions are similar across development training and normal validation**.

![Score Distributions](../results/figures/baseline_ridge/fig_02_persistence_vs_ridge_distributions.png)

---

## 6. Threshold Calibration & Alarm Performance

Threshold calibration follows the predetermined empirical quantile protocol on Normal Validation (`train3`):

| Quantile | Target Role | Calibrated Threshold $\theta$ | Val Exceedances (pts / %) | Val Alarm Sec / Hour | Val False Alarm Events | Val False Alarm Rate (ev/h) | Train False Alarm Rate (ev/h) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q99.5** | Sensitivity Lower | `0.851713` | 2,394 (0.5000%) | 18.00 s/h | 2,379 | 17.887 ev/h | 16.260 ev/h |
| **Q99.9** | **Primary Operating Point** | **`1.400928`** | **479 (0.1000%)** | **3.60 s/h** | **475** | **3.571 ev/h** | **3.724 ev/h** |
| **Q99.99**| Sensitivity Upper | `1.484236` | 48 (0.0100%) | 0.36 s/h | 45 | 0.338 ev/h | 0.382 ev/h |

### Duration Morphology:
- At the **primary Q99.9 operating point** ($\theta^* = 1.400928$):
  - Mean duration: **1.01 seconds**
  - Median duration: **1.0 second**
  - Maximum duration: **3 seconds**
  - **471 out of 475 validation events (99.16%) are exactly 1 second long** (isolated single-sample step spikes). Only 4 events persist for 2 seconds, and zero events exceed 3 seconds.

![Validation Timeline](../results/figures/baseline_ridge/fig_03_ridge_validation_timeline.png)
![Event Durations](../results/figures/baseline_ridge/fig_04_ridge_event_durations.png)

---

## 7. Comparative Analysis: Persistence (Baseline 0) vs. Ridge (Baseline 1)

The following table provides a direct side-by-side comparison on identical Normal Validation exposure ($478,800$ transitions, $133.00$ hours):

| Evaluation Metric | Baseline 0: Persistence | Baseline 1: Ridge ($\alpha=300.0$) | Difference ($\Delta$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Scored Exposure** | 133.00 hours | 133.00 hours | 0.00 h | 0.0% |
| **Validation Scored Transitions** | 478,800 | 478,800 | 0 | 0.0% |
| **Validation Mean RMSE** | 0.322667 | **0.245682** | **-0.076985** | **-23.86%** |
| **Validation Median RMSE** | 0.304545 | **0.232257** | **-0.072288** | **-23.74%** |
| **Validation Std Dev RMSE** | 0.143530 | **0.106999** | **-0.036531** | **-25.45%** |
| **Validation Q99 RMSE** | 0.882807 | **0.621933** | **-0.260874** | **-29.55%** |
| **Primary Threshold $\theta^*$ (Q99.9)**| 1.680523 | **1.400928** | **-0.279595** | **-16.64%** |
| **Validation Alarm Seconds / Hour** | 3.6015 s/h | 3.6015 s/h | 0.0000 s/h | 0.0% (by construction) |
| **Validation False Alarm Event Rate** | 3.5414 ev/h | 3.5714 ev/h | +0.0300 ev/h | +0.85% |
| **Mean Event Duration** | 1.02 s | 1.01 s | -0.01 s | -0.98% |
| **Median Event Duration** | 1.0 s | 1.0 s | 0.0 s | 0.0% |
| **Max Event Duration** | 3 s | 3 s | 0 s | 0.0% |
| **Execution Runtime** | 2.33 s | 16.06 s | +13.73 s | — |
| **Measured Peak Process RSS** | 270.27 MB | 669.70 MB | +399.43 MB | (Safe for 4 GB RAM) |

### Key Methodological Insights:
1. **Model Accuracy vs. Quantile Occupancy**:
   - **Q99.9 validation alarm occupancy is approximately 0.1% by construction** because each model receives its own empirical quantile threshold calibrated on the validation set.
   - Therefore, **equal alarm-seconds/hour ($3.60\text{ s/h}$) at Q99.9 is not evidence that the detectors perform equally**.
2. **Substantial Prediction Error Reduction**:
   - Ridge regression reduces mean one-step prediction residuals by **$23.86\%$** ($0.3227 \to 0.2457$) and median residuals by **$23.74\%$** ($0.3045 \to 0.2323$).
   - The upper tail is significantly tightened: Q99 error falls by **$29.55\%$** ($0.8828 \to 0.6219$), and the primary Q99.9 threshold lowers by **$16.64\%$** ($1.6805 \to 1.4009$).
3. **Invariant Alarm Morphology**:
   - Despite the reduction in residuals, the morphology of false alarms remains essentially identical: false-alarm event rates are virtually identical ($3.54\text{ ev/h}$ vs. $3.57\text{ ev/h}$), and $>99\%$ of false alarms consist of 1-second transient spikes.
4. **Sealed Test Integrity**:
   - Direct comparison is conducted strictly on normal validation data. **No claim of attack-detection superiority is made** because the official attack test data remains sealed.

---

## 8. Process Variable Contribution Analysis

Evaluating root mean squared residuals per feature under Ridge predictions ($\alpha^* = 300.0$) indicates which process sensors exhibit the largest unexplained variance:

| Rank | Feature Name | Subsystem | Description / Role | Mean RMS Residual | Persistence Mean RMS |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `P2_24Vdc` | P2 (Turbine) | 24V DC power rail monitoring | 1.3895 | 1.3905 |
| 2 | `P4_HT_FD` | P4 (HIL) | Hardware-in-the-loop feed variable | 1.2271 | 1.2277 |
| 3 | `P2_HILout` | P2 (Turbine) | HIL interface signal | 1.2100 | 1.2104 |
| 4 | `P4_ST_FD` | P4 (HIL) | HIL steam turbine variable | 1.1741 | 1.1750 |
| 5 | `P2_CO_rpm` | P2 (Turbine) | Turbine rotational speed (rpm) | 0.9165 | 0.9175 |
| 6 | `P2_VT01` | P2 (Turbine) | Turbine vibration transmitter | 0.6844 | 0.6852 |
| 7 | `P2_VYT02` | P2 (Turbine) | Turbine Y-axis vibration transmitter | 0.6284 | 0.6292 |
| 8 | `P2_VXT02` | P2 (Turbine) | Turbine X-axis vibration transmitter | 0.6201 | 0.6208 |
| 9 | `P2_SIT01` | P2 (Turbine) | Speed indicating transmitter 1 | 0.6044 | 0.6050 |
| 10 | `P2_SIT02` | P2 (Turbine) | Speed indicating transmitter 2 | 0.6004 | 0.6009 |

Residual dynamics remain concentrated in high-frequency electrical signals, HIL coupling variables, and rotational/vibrational sensors, where measurement noise cannot be linearly predicted.

![Top Residual Contributors](../results/figures/baseline_ridge/fig_05_top_residual_contributors.png)

---

## 9. Reproducibility & Permanent Freezing

Baseline 1 is permanently frozen with the following saved artifacts:
- **Trained Model Weights**: [`models/baselines/ridge.joblib`](file:///home/kali/DT049/models/baselines/ridge.joblib)
- **Model Metadata**: [`models/baselines/ridge_metadata.json`](file:///home/kali/DT049/models/baselines/ridge_metadata.json)
- **Model Selection Table**: [`results/metrics/baseline_ridge_model_selection.csv`](file:///home/kali/DT049/results/metrics/baseline_ridge_model_selection.csv)
- **Validation Metrics Table**: [`results/metrics/baseline_ridge_validation.csv`](file:///home/kali/DT049/results/metrics/baseline_ridge_validation.csv)
- **Calibrated Thresholds JSON**: [`results/metrics/baseline_ridge_thresholds.json`](file:///home/kali/DT049/results/metrics/baseline_ridge_thresholds.json)
- **Comparison Table**: [`results/metrics/baseline_comparison_validation.csv`](file:///home/kali/DT049/results/metrics/baseline_comparison_validation.csv)
