# Primary Model: One-Step Multivariate GRU Predictor

## 1. Architectural & Theoretical Formulation

The primary deep-learning detector for DT049 is a compact, one-step multivariate Gated Recurrent Unit (GRU) predictor. While Baseline 0 assumes static persistence ($\hat{\mathbf{x}}_{t+1} = \mathbf{x}_t$) and Baseline 1 learns instantaneous static linear transitions ($\hat{\mathbf{x}}_{t+1} = \mathbf{x}_t W^T + \mathbf{b}^T$), the GRU incorporates sequential temporal dynamics over a historical sliding window of $W$ previous seconds:

$$\mathbf{h}_t = \text{GRU}(\mathbf{x}_{t-W+1 : t+1}, \mathbf{h}_0)$$
$$\hat{\mathbf{x}}_{t+1} = \mathbf{W}_{\text{out}} \mathbf{h}_t + \mathbf{b}_{\text{out}}$$

where $\mathbf{x} \in \mathbb{R}^{B \times W \times 79}$ represents the batch of standardized telemetry features over $W$ previous seconds, $\mathbf{h}_t \in \mathbb{R}^{B \times 64}$ is the final hidden state vector, and $\hat{\mathbf{x}}_{t+1} \in \mathbb{R}^{B \times 79}$ is the predicted process state vector at the next second.

### Fixed Architecture & Trainable Parameters
- **Input Dimension**: $D = 79$ (all candidate process features retained, including the 22 training zero-variance features)
- **Hidden Size**: $H = 64$
- **Recurrent Layers**: $1$
- **Recurrent Dropout**: $0.0$ (single-layer GRU does not apply recurrent dropout between layers)
- **Output Layer**: $\text{Linear}(64, 79)$
- **Trainable Parameter Count**:
  - GRU Input-to-Hidden: $3 \times (64 \times 79 + 64) = 15,360$
  - GRU Hidden-to-Hidden: $3 \times (64 \times 64 + 64) = 12,480$
  - Linear Output: $79 \times 64 + 79 = 5,135$
  - **Total Trainable Parameters**: **$32,975$** ($\approx \mathbf{131.9\text{ KB}}$ in `float32`)

### Training Hyperparameters & Thread Control
- **Loss**: Mean Squared Error (MSE) over all 79 process features
- **Optimizer**: Adam ($\text{lr} = 10^{-3}, \text{weight\_decay} = 10^{-5}$)
- **Gradient Clipping**: `max_norm = 1.0`
- **Batch Size**: $256$
- **DataLoader Workers**: $0$ (single-process CPU streaming from memory-mapped `.npy` arrays)
- **Early Stopping**: `patience = 5` epochs, `min_delta = 1e-5`, monitored on validation mean RMSE
- **CPU Thread Allocation**: `torch.set_num_threads(4)`, `torch.set_num_interop_threads(1)`

---

## 2. Computational Strategy & Stride Controls

To ensure computationally tractable training and validation on a 4 GB RAM Kali Linux VM with 4 vCPUs without sacrificing temporal resolution:

1. **Training Stride ($S = 5\text{ seconds}$)**:
   - Slices sliding windows every 5 seconds strictly within recording boundaries (`train1` and `train2`).
   - Slashes training workload by $80\%$ ($\approx 88,550$ windows instead of $\approx 442,800$), while preserving physical steady-state diversity.
2. **Model Selection Validation Stride ($S = 5\text{ seconds}$)**:
   - Slices validation windows every 5 seconds on `train3` ($95,755$ windows for $W=30$) during epoch training and early-stopping evaluations.
   - Avoids evaluating $\approx 478\text{k}$ windows after every single training epoch, significantly speeding up model selection.
3. **Full-Resolution Validation Stride ($S = 1\text{ second}$)**:
   - Evaluated **only** on the final frozen window size across seeds 42, 43, and 44.
   - Evaluates all $478,771$ 1 Hz windows ($132.99\text{ hours}$) for final threshold calibration, alarm rate calculation, and duration profiling.

---

## 3. Stage A: Window-Length Selection

Three candidate window lengths were evaluated on Development Training with `seed = 42` using validation stride $S=5$ for early stopping:

| Candidate Window $W$ | Training Windows ($S=5$) | Validation Windows ($S=5$) | Selected Epoch | Stop Epoch | Stop Reason | Validation Mean RMSE ($S=5$) | Run Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$W = 10\\text{s}$** | $88,558$ | $95,759$ | 25 | 30 | `early_stopping` | $0.22637874$ | $243.79\\text{s}$ |
| **$W = 30\\text{s}$** | $88,550$ | $95,755$ | **27** | **32** | `early_stopping` | **$0.22597168$** | **$533.40\\text{s}$** |
| **$W = 60\\text{s}$** | $88,538$ | $95,749$ | 13 | 18 | `early_stopping` | $0.22942426$ | $884.75\\text{s}$ |

### Window Selection Decision:
- **Selected Window**: **$W^* = 30\text{ seconds}$** achieves the lowest validation mean RMSE ($0.22597168$).
- **Selection Margins**:
  - $W=10 - W=30$: $+0.00040705$
  - $W=60 - W=30$: $+0.00345258$
- In accordance with the protocol, $W^* = 30$ was permanently frozen and no further window searches were performed.

![Window Selection](../results/figures/gru/fig_01_window_selection_rmse.png)
![Training Curves](../results/figures/gru/fig_02_training_loss_curves.png)

---

## 4. Stage B: Three-Seed Stability Evaluation

For the selected window $W^* = 30$, the Stage A seed 42 checkpoint was preserved, and two additional runs were trained with seeds 43 and 44. All three models were subsequently evaluated at full 1 Hz resolution ($S = 1$, $478,771$ windows, $132.9919\text{ hours}$):

| Run / Seed | Selected Epoch | Stop Epoch | Stop Reason | Full Val Mean RMSE ($S=1$) | Val Median RMSE | Val Q99 RMSE | Q99.9 Threshold $\\theta^*$ | Q99.9 False Alarms (ev/h) | Q99.9 Alarm Sec/h | Max Duration | Run Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 27 | 32 | `early_stopping` | $0.22561317$ | $0.214421$ | $0.524117$ | $1.058582$ | $3.5641\\text{ ev/h}$ | $3.6017\\text{ s/h}$ | 2s | $533.40\\text{s}$ |
| **Seed 43** | 17 | 22 | `early_stopping` | $0.22945071$ | $0.217115$ | $0.567546$ | $1.195869$ | $3.5566\\text{ ev/h}$ | $3.6017\\text{ s/h}$ | 4s | $583.69\\text{s}$ |
| **Seed 44** | 29 | 34 | `early_stopping` | $0.22713761$ | $0.215848$ | $0.537123$ | $1.076832$ | $3.5491\\text{ ev/h}$ | $3.6017\\text{ s/h}$ | 4s | $781.02\\text{s}$ |
| **Mean $\\pm$ Std** | — | — | — | **$0.227400 \\pm 0.001578$** | **$0.215795 \\pm 0.001100$** | **$0.542929 \\pm 0.018204$** | **$1.110428 \\pm 0.060874$** | **$3.5566 \\pm 0.0061\\text{ ev/h}$** | **$3.6017\\text{ s/h}$** | 4s | **$632.70\\text{s}$** |

### Stability Insights:
1. **RMSE Consistency**: Validation mean RMSE varies by only $\sigma = 0.001578$ across seeds (relative standard deviation $\sigma / \mu = 0.69\%$), demonstrating robust convergence.
2. **Threshold Stability**: The primary Q99.9 threshold averages $1.110428 \pm 0.060874$.
3. **Alarm Rate Invariance**: Contiguous false-alarm event rates at Q99.9 are remarkably stable: $3.557 \pm 0.006\text{ events/hour}$ across seeds.
4. **All Three Runs Preserved**: In accordance with project instructions, no single seed was chosen as a "winner"; all three runs constitute the multi-seed stability baseline.

![Multi-Seed Stability](../results/figures/gru/fig_05_multiseed_threshold_stability.png)

---

## 5. Tri-Model Benchmark Comparison: Persistence vs. Ridge vs. GRU

The following table presents the direct side-by-side comparison across all three detector stages evaluated on identical Normal Validation exposure (`train3`, $\approx 133\text{ hours}$):

| Evaluation Metric | Baseline 0: Persistence | Baseline 1: Ridge ($\alpha=300$) | Primary GRU (Seed 42) | Primary GRU (3-Seed Mean) | $\Delta$ (GRU Mean vs. Persist) | $\Delta$ (GRU Mean vs. Ridge) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scored Exposure** | $133.00\text{ h}$ | $133.00\text{ h}$ | $132.99\text{ h}$ | $132.99\text{ h}$ | $-0.01\text{ h}$ | $-0.01\text{ h}$ |
| **Scored Transitions** | $478,800$ | $478,800$ | $478,771$ | $478,771$ | $-29$ | $-29$ |
| **Validation Mean RMSE** | $0.322667$ | $0.245682$ | **$0.225613$** | **$0.227400$** | **$-29.53\%$** | **$-7.44\%$** |
| **Validation Median RMSE** | $0.304545$ | $0.232257$ | **$0.214421$** | **$0.215795$** | **$-29.14\%$** | **$-7.09\%$** |
| **Validation Std Dev RMSE** | $0.143530$ | $0.106999$ | **$0.090822$** | **$0.093389$** | **$-34.93\%$** | **$-12.72\%$** |
| **Validation Q99 RMSE** | $0.882807$ | $0.621933$ | **$0.524117$** | **$0.542929$** | **$-38.50\%$** | **$-12.70\%$** |
| **Primary Threshold $\theta^*$ (Q99.9)**| $1.680523$ | $1.400928$ | **$1.058582$** | **$1.110428$** | **$-33.92\%$** | **$-20.74\%$** |
| **Validation Alarm Seconds / Hour** | $3.6015\text{ s/h}$ | $3.6015\text{ s/h}$ | $3.6017\text{ s/h}$ | $3.6017\text{ s/h}$ | $0.0\%$ (by construction) | $0.0\%$ (by construction) |
| **Validation False Alarm Event Rate** | $3.5414\text{ ev/h}$ | $3.5714\text{ ev/h}$ | $3.5641\text{ ev/h}$ | $3.5566\text{ ev/h}$ | $+0.43\%$ | $-0.41\%$ |
| **Mean Event Duration** | $1.02\text{ s}$ | $1.01\text{ s}$ | $1.01\text{ s}$ | $1.01\text{ s}$ | $-0.98\%$ | $0.0\%$ |
| **Median Event Duration** | $1.0\text{ s}$ | $1.0\text{ s}$ | $1.0\text{ s}$ | $1.0\text{ s}$ | $0.0\%$ | $0.0\%$ |
| **Max Event Duration** | $3\text{ s}$ | $3\text{ s}$ | $2\text{ s}$ | $4\text{ s}$ | $+1\text{ s}$ | $+1\text{ s}$ |

### Critical Methodological Insights:
1. **Quantile Calibration vs. Detector Quality**:
   - **Q99.9 validation alarm occupancy is approximately 0.1% by construction** because each model receives its own empirical quantile threshold calibrated on normal validation data.
   - Therefore, **equal alarm-seconds/hour ($3.60\text{ s/h}$) at Q99.9 is an artifact of quantile calibration, not evidence that the detectors perform equally**.
2. **Substantial Reduction in Prediction Residuals**:
   - The primary GRU lowers mean prediction error by **$29.53\%$** relative to Persistence and **$7.44\%$** relative to Ridge.
   - Upper-tail residual errors drop even more dramatically: Q99 error falls by **$38.50\%$** vs. Persistence and **$12.70\%$** vs. Ridge, driving a **$33.92\%$** reduction in the primary threshold ($1.6805 \to 1.1104$).
3. **Alarm Morphology Invariance**:
   - Despite major reductions in score magnitude, the temporal structure of normal alarms is virtually unchanged: $\approx 3.56\text{ events/hour}$ with a median duration of $1.0\text{ second}$ and $>99\%$ of false alarms being single-sample spikes.
4. **Sealed Test Integrity**:
   - Comparison is based strictly on normal validation residuals and alarm morphology. **No claim of attack-detection superiority is made** because the official attack test data remains fully sealed.

![Score Distributions](../results/figures/gru/fig_03_score_distribution_comparison.png)
![Validation Timeline](../results/figures/gru/fig_04_gru_validation_timeline.png)

---

## 6. Methodological Limitation: Validation Reuse

`train3` is reused for:
1. Early stopping during model training,
2. Window-size selection ($W \in \{10, 30, 60\}$), and
3. Final normal threshold calibration (Q99.5, Q99.9, Q99.99).

While this does **not** constitute test data leakage (since official attack test files `test1`–`test5` remain completely untouched and sealed), repeated use of a single normal-validation recording campaign can introduce a degree of **validation-selection bias**. This constraint is documented as an inherent methodological limitation of the dataset split structure.

---

## 7. Computational Profile & Resource Footprint

- **Hardware / Device**: 4 vCPUs, CPU-only execution (`torch.set_num_threads(4)`)
- **Total Workflow Runtime**: **$3,108.00\text{ seconds}$** ($\approx 51.80\text{ minutes}$)
  - Stage A ($W=10, 30, 60$, seed 42): $1,163.29\text{s}$ ($\approx 19.4\text{ min}$)
  - Stage B (seeds 43 and 44): $673.49\text{s}$ ($\approx 11.2\text{ min}$)
  - Stage C (full stride=1 evaluation across 3 seeds): $54.72\text{s}$ ($\approx 0.9\text{ min}$)
  - Figures and tables generation: $7.39\text{s}$
- **Measured Peak RSS**: **$865.34\text{ MB}$** (comfortably within the 4 GB RAM limit of the Kali VM).

---

## 8. Saved Checkpoints & SHA256 Manifest

All model checkpoints and metadata are permanently saved and tracked:

| Checkpoint File | Size | SHA256 Hash |
| :--- | :---: | :--- |
| `models/primary/gru_w10_seed42.pt` | $401\text{ KB}$ | `507c5f33575c094a39e6fa4dd1c758fe5314f52b0caaa7d42493ca6d6601b501` |
| `models/primary/gru_w30_seed42.pt` | $401\text{ KB}$ | `48013cdb36a7490330ba615874399ff21f0a2c3f9b5f7f6edc756168233e2c3d` |
| `models/primary/gru_w60_seed42.pt` | $401\text{ KB}$ | `464ced4dd45d15e6dc3c294efcbce44cd2ded3d906d5da6929b7b66127fa2b79` |
| `models/primary/gru_w30_seed43.pt` | $401\text{ KB}$ | `ac4bfb705c5dd28e114d32b7a9bc654dc2acc7875180f86f06c7661780d7c75c` |
| `models/primary/gru_w30_seed44.pt` | $401\text{ KB}$ | `34443577a9a3a91f2dbeddbb35481c9e1610f3e8d0f40fb360a67af960e3c660` |

Metadata file: [`models/primary/gru_metadata.json`](file:///home/kali/DT049/models/primary/gru_metadata.json)
Thresholds file: [`results/metrics/gru_thresholds.json`](file:///home/kali/DT049/results/metrics/gru_thresholds.json)
