# DT049 — Multivariate Time-Series Anomaly Detection on HAI 21.03

Multivariate time-series anomaly detection on the Hardware-in-the-Loop Security (HAI) 21.03 benchmark using leakage-aware predictive baselines (Persistence, Ridge) and a compact Gated Recurrent Unit (GRU) model under a frozen official test protocol.

---

## 1. Project Overview

DT049 is an experimental course research project conducting a selective replication study inspired by the methodology of the assigned 2025 *Computers & Security* paper, adapted to the public HAI 21.03 industrial control systems (ICS) dataset. 

The project establishes a leakage-controlled experimental framework for multivariate anomaly detection across 79 sensor and actuator telemetry channels. It evaluates three detector families:
1. **Baseline 0 (Persistence):** Naive one-step identity persistence baseline.
2. **Baseline 1 (Ridge Regression):** Linear regularized multi-output predictor ($\alpha = 300.0$).
3. **Primary Model (GRU):** Compact 1-layer Gated Recurrent Unit neural network ($W=30$, Hidden=64).

Evaluation focuses on operational industrial metrics: contiguous attack event-level Precision, Recall, and F1, operational false-alarm rates during prolonged normal operation, and empirical detection latency (delay). 

> [!NOTE]
> This study does **not** claim direct numerical replication of the original paper's published metrics. The original paper evaluates a custom physical testbed with different process dynamics, variable sets, and attack topologies; DT049 investigates whether the architectural insights hold when systematically adapted to the HAI 21.03 benchmark under strict leakage controls.

---

## 2. Main Research Question

> **Can a compact temporal predictor such as GRU improve attack-event detection and detection latency over simple Persistence and Ridge baselines on HAI 21.03, under a frozen leakage-aware evaluation protocol?**

As a secondary post-hoc question, DT049 investigates through the **Dynamic57 Ablation** whether zero-variance channels identified during training contribute meaningfully to anomaly scoring and operational alarm decisions on official test data.

---

## 3. Dataset

The project uses the **HAI (Hardware-in-the-loop Augmented ICS) Security Dataset 21.03**, released by the Electronics and Telecommunications Research Institute (ETRI).
- **Upstream Repository**: [https://github.com/icsdataset/hai](https://github.com/icsdataset/hai)
- **Upstream Git Commit**: `2a814cebc9a66b06c9e5cd545e2d72e65d383737`
- **Total Files**: 8 compressed CSV files (`train1`–`train3`, `test1`–`test5`)
- **Telemetry Channels**: 79 candidate continuous process variables across four interconnected physical subsystems (P1: Boiler, P2: Turbine, P3: Water Treatment, P4: HIL Simulator)
- **Official Test Attacks**: 50 contiguous multi-sensor physical attack events

| Split Name | File Name | Role | Rows (Seconds) | Duration | Attack Seconds | Attack Events |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Development Train 1 | `train1.csv.gz` | Normal Model Fitting | 216,001 | 60.00 h | 0 | 0 |
| Development Train 2 | `train2.csv.gz` | Normal Model Fitting | 226,801 | 63.00 h | 0 | 0 |
| Validation Normal | `train3.csv.gz` | Model Selection & Threshold Calibration | 478,801 | 133.00 h | 0 | 0 |
| Official Test 1 | `test1.csv.gz` | Final Unseen Test Session | 43,201 | 12.00 h | 629 | 5 |
| Official Test 2 | `test2.csv.gz` | Final Unseen Test Session | 118,801 | 33.00 h | 3,449 | 20 |
| Official Test 3 | `test3.csv.gz` | Final Unseen Test Session | 108,001 | 30.00 h | 1,535 | 8 |
| Official Test 4 | `test4.csv.gz` | Final Unseen Test Session | 39,601 | 11.00 h | 1,157 | 5 |
| Official Test 5 | `test5.csv.gz` | Final Unseen Test Session | 92,401 | 25.67 h | 2,177 | 12 |
| **Total Test** | `test1`–`test5` | **Frozen Official Final Test** | **402,005** | **111.67 h** | **8,947** | **50** |

*Note: Raw dataset archives are gitignored and not duplicated in the repository. SHA-256 integrity hashes for all raw files are permanently cataloged in `results/audit/`.*

### Documentation Discrepancy in `train3.csv.gz`
The upstream ETRI dataset README documents `train3.csv.gz` as containing 229 hours of normal data. However, the actual upstream Git repository blob at commit `2a814cebc9a66b06c9e5cd545e2d72e65d383737` (Git blob `207e4b9f4c157505d5cb4a66e8ebe84d03575dab`, size 67,961,579 bytes compressed) contains exactly 478,801 contiguous 1 Hz records spanning 133.00 hours (a 96-hour documentation deficit). SHA-256 and Git tree verification confirm this is an upstream ETRI documentation error, not local download corruption.

---

## 4. Leakage-Controlled Experimental Design

The project enforces strict chronological and partition boundaries to prevent data leakage:
- **Development Training (`train1` + `train2`):** 442,802 normal seconds (123.0 hours). Exclusively used to fit the `StandardScaler`, determine channel variance, train Ridge regression, and optimize GRU model weights.
- **Normal Validation (`train3`):** 478,801 normal seconds (133.0 hours). Exclusively used for early stopping checkpoint selection and a priori calibration of operating anomaly thresholds.
- **Official Final Test (`test1`–`test5`):** 402,005 seconds (111.67 hours) containing 50 physical attacks. Fully held out until after all models, window lengths, hyperparameters, and thresholds were permanently frozen.

> [!IMPORTANT]
> Earlier descriptive dataset audit used official test data and labels for integrity and event-structure characterization. No test-derived information was used for model fitting, feature selection, hyperparameter/window selection, model selection, or threshold calibration.

---

## 5. Models

All models predict the next-second standardized telemetry vector $\hat{\mathbf{x}}_t \in \mathbb{R}^{79}$ given preceding observations:

1. **Baseline 0 — Persistence:**
   $$\hat{\mathbf{x}}_t = \mathbf{x}_{t-1}$$
   Parameter-free identity baseline capturing naive temporal autocorrelation.
2. **Baseline 1 — Ridge Regression:**
   $$\hat{\mathbf{x}}_t = \mathbf{x}_{t-1} \mathbf{W} + \mathbf{b}$$
   Multi-output linear predictor with L2 regularization ($\alpha = 300.0$) fit closed-form on standardized training transitions.
3. **Primary Model — GRU (Gated Recurrent Unit):**
   - Input dimension: 79 channels
   - Window size: $W = 30$ seconds (lookback window $[\mathbf{x}_{t-W}, \dots, \mathbf{x}_{t-1}]$)
   - Hidden units: 64
   - Recurrent layers: 1 (compact architecture with 32,975 trainable parameters)
   - Output dimension: 79 channels
   - Seeds evaluated: 42, 43, 44 (Seed 42 designated as canonical primary; Seeds 43 and 44 serve as stability replicates)
   - Loss function: Mean Squared Error (MSE) optimized with Adam ($lr = 10^{-3}$) and cosine annealing
   - Execution: Strict **CPU-only execution** (no GPU dependencies).

---

## 6. Anomaly Score & Threshold Calibration

For all detectors, the anomaly score at second $t$ is the Root Mean Squared Error (normalized L2 prediction residual) across all 79 standardized channels:
$$\text{score}(t) = \sqrt{\frac{1}{79} \sum_{j=1}^{79} \left( x_{t, j} - \hat{x}_{t, j} \right)^2}$$

### Threshold Calibration
Operating thresholds were permanently calibrated on the normal validation split (`train3`) prior to evaluating test data:
- **Primary Operating Point ($Q_{99.9}$):** 99.9th empirical quantile of normal validation residuals ($1 \text{ alarm per } 1,000 \text{ normal seconds}$ target).
  - Persistence: $\tau = 1.680523$
  - Ridge: $\tau = 1.400928$
  - GRU Seed 42: $\tau = 1.058582$ (Seed 43: $\tau = 1.195869$, Seed 44: $\tau = 1.076832$)
- **Sensitivity Quantiles:** $Q_{99.5}$ (lower bound) and $Q_{99.99}$ (upper bound) evaluated strictly as sensitivity bounds.

---

## 7. Frozen Final-Test Protocol

Final test evaluation was conducted under a frozen evaluation protocol ([docs/final_test_protocol.md](docs/final_test_protocol.md)):
- **Common Comparison Timeline:** Evaluated on all transitions $t \ge 30$ to ensure exact 1-to-1 paired comparability across $W=1$ baselines and $W=30$ GRU. This yields exactly 401,855 scored transitions and preserves 100% (50/50) of ground-truth test attack events.
- **Session Independence:** Each test file (`test1`–`test5`) is processed independently; no window spans a file boundary.
- **Event Definition:** 
  - Ground-truth attack event: Maximal contiguous run where $\text{attack} == 1$.
  - Predicted alarm event: Maximal contiguous run where $\text{score}(t) > \tau$.
- **Strict Chronological Matching:** A predicted alarm event is matched to an attack event if they overlap in time. Each attack can be detected at most once (first overlapping alarm matches; subsequent in-attack alarms are treated as redundant and counted as event-level FP). Unmatched alarms during normal periods constitute normal false alarms.
- **Evaluation Metrics:**
  - $\text{Event Recall} = \text{TP} / \text{Evaluable Attacks}$
  - $\text{Event Precision} = \text{TP} / \text{Total Predicted Alarm Events}$
  - $\text{Event F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
  - $\text{Detection Delay} = t_{\text{first alarm}} - t_{\text{attack start}}$ (seconds)
  - $\text{Normal False Alarm Rate} = \frac{\text{Unmatched Alarms in Normal Time}}{\text{Normal Exposure Hours}}$ ($\text{events/hour}$)
  - $\text{False Positive Alarm Time} = \text{Total False Alarm Seconds / Normal Exposure Hours}$ ($\text{seconds/hour}$)
  - $\text{Point FPR} = \text{False Alarm Seconds} / \text{Normal Seconds}$

> [!NOTE]
> The protocol strictly distinguishes **event-level unmatched alarms** (which include redundant re-triggers inside attacks) from **operational normal false alarms** (spurious interruptions during true attack-free periods).

---

## 8. Main Frozen Results

Official frozen test results across all 5 test sessions (401,855 scored transitions, 109.1411 normal exposure hours) at the primary $Q_{99.9}$ threshold:

| Model | Detected (TP) | Missed (FN) | Event Recall | Event Precision | Event F1 | Normal FA (ev/h) | FP Time (s/h) | Median Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0 (Persistence)** | 22 / 50 | 28 | 0.4400 | 0.0487 | 0.0876 | 3.7016 | **3.92 s/h** | 40.5 s |
| **Baseline 1 (Ridge $\alpha=300$)** | 35 / 50 | 15 | 0.7000 | 0.0636 | 0.1167 | 4.4804 | 6.52 s/h | **13.0 s** |
| **Primary GRU (Seed 42, Canonical)** | **40 / 50** | **10** | **0.8000** | **0.1067** | **0.1882** | **2.8220** | 5.89 s/h | **22.0 s** |
| *Primary GRU (Seed 43, Replicate 1)* | 39 / 50 | 11 | 0.7800 | 0.1096 | 0.1921 | 2.7212 | 5.54 s/h | 21.0 s |
| *Primary GRU (Seed 44, Replicate 2)* | 41 / 50 | 9 | 0.8200 | 0.1297 | 0.2240 | 2.2906 | 5.31 s/h | 24.0 s |
| **GRU 3-Seed Aggregate** | **40.0 ± 0.8** | **10.0 ± 0.8** | **0.8000 ± 0.0163** | **0.1153 ± 0.0103** | **0.2015 ± 0.0160** | **2.6113 ± 0.2305** | 5.58 ± 0.24 s/h | **22.33 ± 1.25 s** |

### Key Scientific Findings & Tradeoffs
1. **Detection Sensitivity:** GRU significantly outperforms baselines, capturing $80.0\%$ of ground-truth attacks ($40/50$), compared to $44.0\%$ for Persistence and $70.0\%$ for Ridge.
2. **False-Alarm Mitigation:** GRU suppresses discrete operator false-alarm interruptions to $2.61\text{ events/hour}$ across 3 seeds ($29.5\%$ reduction vs Persistence, $41.7\%$ reduction vs Ridge).
3. **The Alarm Time Tradeoff:** While GRU produces the fewest discrete interruption events, Persistence achieves the lowest total alarmed duration ($3.92\text{ s/hour}$ vs $5.89\text{ s/hour}$ for GRU seed 42) because Persistence errors recover immediately in 1–2 seconds, whereas GRU prediction residuals ring for several seconds after transient process spikes.
4. **Context for Event Precision:** Low event precision (~$0.11$) is an inherent property of evaluating over 109.14 hours ($392,908$ normal seconds) of continuous telemetry under strict 1-to-1 event matching without artificial hysteresis or heuristic alarm merging. It must **not** be misrepresented as poor detector quality.

---

## 9. Sensitivity Analysis

A sensitivity sweep across predefined validation quantiles was evaluated strictly for robustness analysis:

| Operating Point | Model | Recall | Precision | Event F1 | Normal FA Rate | Median Delay | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$Q_{99.5}$** | GRU Seed 42 | 0.9800 | 0.0269 | 0.0524 | 14.8707 ev/h | 15.0 s | Sensitivity Lower Bound |
| **$Q_{99.9}$** | **GRU Seed 42** | **0.8000** | **0.1067** | **0.1882** | **2.8220 ev/h** | **22.0 s** | **FROZEN PRIMARY OPERATING POINT** |
| **$Q_{99.99}$** | GRU Seed 42 | 0.7200 | 0.4000 | 0.5143 | 0.3390 ev/h | 24.5 s | Sensitivity Upper Bound |

> [!WARNING]
> While $Q_{99.99}$ achieves higher test Event F1 ($0.5143$), **$Q_{99.9}$ remains the permanent primary operating point**. In strict leakage-controlled science, a threshold calibrated on validation data cannot be replaced post-hoc by a test-optimal threshold.

---

## 10. Dynamic57 Feature Ablation

During development training, 22 of the 79 telemetry channels exhibited zero variance in normal operations, while 57 exhibited non-zero dynamic variance. The post-hoc exploratory **Dynamic57 Ablation** tested whether restricting the anomaly score to the 57 dynamic channels altered detector performance.

### Theoretical Derivation & Scaling Factor
On normal validation data (`train3`), the excluded 22 channels have zero residual ($e_j(t) = 0$). Under this condition, the 57-feature score is an exact constant rescaling of the 79-feature score:
$$\kappa = \sqrt{\frac{79}{57}} \approx 1.177270$$
$$\text{RMSE}_{57}(t) = \sqrt{\frac{79}{57}} \cdot \text{RMSE}_{79}(t) = \kappa \cdot \text{RMSE}_{79}(t)$$
*(Note: The channel dimension ratio is $\frac{79}{57} \approx 1.385965$, so the root-mean-square scale factor is $\kappa = \sqrt{\frac{79}{57}} \approx 1.177270$.)*

Because both the score and the quantile threshold scale by $\kappa$, alarm decisions are mathematically invariant on normal validation data.

### Official Test Conclusion
On the official test set, some excluded channels become active, so exact score scaling need not hold. Empirically, their contribution was not large enough to change any $Q_{99.9}$ alarm decision:
> **"The 22 training-constant channels did not alter detector decisions under this post-hoc Q99.9 ablation on HAI 21.03."**

Across all 50 attacks, the binary detection status, false-alarm counts, and detection delays were identical between the 79-feature and 57-feature configurations.

---

## 11. Repository Structure

```text
DT049/
├── README.md                            # Project documentation and reproduction guide
├── requirements-lock.txt                # Exact pip dependency lockfile
├── configs/                             # Configuration files
│   ├── final_evaluation.yaml            # Frozen test protocol parameters
│   └── primary_gru.yaml                 # Canonical GRU training specification
├── data/                                # Data storage (raw data gitignored)
│   ├── processed/hai_21_03/             # Memory-mapped processed arrays
│   └── source/                          # Upstream repository submodule / download destination
├── demo/                                # Interactive Streamlit dashboard
│   ├── app.py                           # Multi-page dashboard application
│   └── ui_utils.py                      # UI caching and plot rendering helpers
├── docs/                                # Detailed technical documentation
│   ├── final_test_protocol.md           # Formal evaluation protocol
│   ├── final_error_analysis.md          # Post-test error and delay analysis
│   ├── ablation_dynamic57.md            # Post-hoc Dynamic57 ablation study
│   └── environment.md                   # Computational specifications & RAM audit
├── models/                              # Serialized model checkpoints
│   ├── preprocessing/scaler.joblib      # Frozen StandardScaler fit on train1+train2
│   ├── baselines/ridge.joblib           # Frozen Ridge regression model
│   └── primary/gru_w30_seed*.pt         # Frozen PyTorch checkpoints (seeds 42, 43, 44)
├── results/                             # Evaluation outputs and audit trails
│   ├── audit/                           # SHA-256 manifests and execution audit logs
│   ├── metrics/                         # Official test summary and event-level CSVs
│   └── figures/                         # High-resolution diagnostic visualizations
└── src/                                 # Executable Python pipeline modules
    ├── preprocess.py                    # Training data scaler fit and normalization
    ├── preprocess_test.py               # Test data transformation
    ├── baseline_persistence.py          # Persistence baseline pipeline
    ├── baseline_ridge.py                # Ridge baseline pipeline
    ├── train_gru.py                     # Primary GRU multi-seed training
    ├── evaluate_final_test.py           # Frozen official test evaluation engine
    ├── analyze_final_errors.py          # Post-test error diagnostics
    └── run_ablation_dynamic57.py        # Post-hoc Dynamic57 ablation engine
```

---

## 12. Reproducibility & Audit Trail

The complete scientific development and evaluation history is preserved in Git:

| Checkpoint Name | Commit Hash / Tag | Description |
| :--- | :--- | :--- |
| **Upstream HAI Commit** | `2a814cebc9a66b06c9e5cd545e2d72e65d383737` | Verified ETRI repository source state |
| **Development Freeze** | `6ce11fb8bcb682c10e125219d0b964f394b8a5d2` | Permanent freeze of code, models, and thresholds |
| **Evaluation Protocol Freeze** | `6a0613b87a0a9d89c89b6ae848350b44c1aa1740` | Locked evaluation protocol document (`dt049-pretest-freeze`) |
| **Official Final-Test Tag** | `dt049-final-test-v1` | Permanent Git tag for official final test evaluation |
| **Final-Test Execution** | `331cc07cb72b8b3f30a64bbe7af52248bd0cd9ba` | Recorded official final test metric outputs |
| **Post-Test Analysis** | `4104576d1c4ad69edd87cc97ea6463974becf460` | Added error analysis and Dynamic57 ablation |
| **Dashboard Interface** | `a9f03c940b066bd3bea46007a21f372915f45db9` | Committed Streamlit interactive demo |

All preprocessing arrays, model weights, and result CSVs are verified by SHA-256 manifests in `results/audit/`.

---

## 13. Environment & Software Versions

The environment was executed entirely on a single Kali Linux virtual machine:
- **Operating System:** Kali GNU/Linux Rolling 2026.2 (Linux 6.19.14+kali-amd64)
- **Python Version:** `3.14.7`
- **Execution Target:** CPU-only (PyTorch CPU build, strictly zero GPU dependencies)
- **Virtual Machine RAM:** 4.0 GB physical RAM

### Verified Package Versions
```text
numpy==2.5.3
pandas==3.0.6
scipy==1.18.1
scikit-learn==1.9.1
matplotlib==3.11.2
PyYAML==6.0.3
torch==2.14.0+cpu
streamlit==1.65.0
joblib==1.6.0
tqdm==4.70.1
psutil==7.2.2
```

---

## 14. Running the Interactive Dashboard

An interactive dashboard allows full visual inspection of model comparisons, attack timelines, error diagnostics, sensitivity sweeps, and ablation metrics:

```bash
# 1. Navigate to project root inside the Kali VM
cd /home/kali/DT049

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Launch Streamlit server
streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501
```

Once running, access the application locally at:
**[http://127.0.0.1:8501](http://127.0.0.1:8501)**

> [!NOTE]
> The dashboard operates in read-only mode using existing frozen disk artifacts; it performs zero model inference, test re-evaluation, or dataset alteration during runtime.

---

## 15. Reproducing Development Experiments

All stages of the experimental pipeline are fully scriptable using executable Python modules in `src/`:

### Step 1: Preprocessing
```bash
# Fit StandardScaler on train1+train2 and transform train1, train2, train3
python src/preprocess.py

# Transform official test1..test5 using the frozen scaler (no refitting)
python src/preprocess_test.py
```

### Step 2: Baseline Models
```bash
# Evaluate Baseline 0 (Persistence) and calibrate validation threshold
python src/baseline_persistence.py

# Fit Baseline 1 (Ridge alpha=300) and calibrate validation threshold
python src/baseline_ridge.py
```

### Step 3: Primary GRU Training
```bash
# Train GRU across seeds 42, 43, 44 with early stopping on train3
python src/train_gru.py
```

### Step 4: Official Final-Test Evaluation
```bash
# Evaluate frozen models on test1..test5 at Q99.9 and sensitivity quantiles
python src/evaluate_final_test.py
```

### Step 5: Post-Test Analysis & Ablation
```bash
# Run process-level error breakdown and paired common-delay analysis
python src/analyze_final_errors.py

# Run post-hoc Dynamic57 feature score ablation
python src/run_ablation_dynamic57.py
```

---

## 16. Limitations

1. **Testbed Specificity:** Results are established on the HAI 21.03 testbed and may not directly generalize to other industrial benchmarks (e.g., SWaT, WADI) that feature different physical topologies or network attack surfaces.
2. **Normal-Only Calibration:** Threshold calibration relies on the normal validation split (`train3`). While free of attacks, long-term seasonal sensor drift in real industrial plants may require adaptive recalibration.
3. **Validation Data Re-Use:** `train3` was utilized both for GRU early stopping and threshold quantile selection, which is methodologically sound for unsupervised anomaly detection but couples model checkpoint selection with the operating point.
4. **Modest Event Precision:** Event precision remains modest (~$0.11$) because of strict 1-to-1 event matching across 109.14 hours of continuous operation without artificial hysteresis or alarm merging.
5. **Metric Duality:** Alarm event rate and temporal alarm occupancy reflect different operational realities: GRU minimizes operator interruption frequency ($2.61\text{ ev/h}$), whereas Persistence minimizes total alarmed seconds ($3.92\text{ s/h}$).
6. **Architectural Scope:** Only one compact GRU architecture ($W=30$, Hidden=64, 1 layer) was evaluated; deeper networks or transformer-based sequence predictors were outside the project scope.
7. **Exploratory Status of Dynamic57:** The Dynamic57 ablation was formulated and conducted post-hoc after final test evaluation and serves as exploratory evidence, not as a replacement for the frozen primary evaluation.
8. **Disparity from Original Reference Paper:** The original 2025 Computers & Security paper uses a different custom dataset; quantitative metrics cannot be directly compared across different experimental physical domains.

---

## 17. Dashboard

Run the local Streamlit dashboard for interactive exploration:
```text
http://127.0.0.1:8501
```
*(Screenshot placeholder: Launch the dashboard in Kali Linux to view interactive figures, attack timelines, and comparative diagnostics.)*
