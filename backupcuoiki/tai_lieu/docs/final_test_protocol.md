# Final Test Evaluation Protocol (DT049)

**Project**: University DT049 — Industrial Control System Anomaly Detection  
**Target Dataset**: Official HAI 21.03 Testbed (`test1.csv.gz` through `test5.csv.gz`)  
**Protocol Version**: 1.0.0  
**Development Freeze Commit**: `6ce11fb8bcb682c10e125219d0b964f394b8a5d2`  
**Freeze Manifest**: [`results/audit/pretest_freeze_manifest.json`](file:///home/kali/DT049/results/audit/pretest_freeze_manifest.json)  
**Configuration**: [`configs/final_evaluation.yaml`](file:///home/kali/DT049/configs/final_evaluation.yaml)  

---

## 1. Scientific Integrity & Methodological Demarcation

### Core Integrity Statement
> [!IMPORTANT]
> **Earlier dataset audit used official test data and labels for integrity and descriptive characterization, including attack-event structure. No test-derived information was used for model fitting, feature selection, hyperparameter or window selection, model selection, or threshold calibration.**

### Demarcation Between Dataset Audit and Model Development
The non-inferential dataset audit previously conducted on HAI 21.03 characterized file integrity, column schemas, physical tag descriptions, and contiguous attack-event boundaries:
1. No predictive models or linear baselines were fitted to test data.
2. No hyperparameters (such as Ridge regularizer $\alpha$, GRU window size $W$, hidden dimensions, learning rates, or early stopping patience) were informed by test data.
3. No feature selection or variance masking was guided by test distributions.
4. No anomaly score distributions or attack labels from `test1`–`test5` were examined to tune, calibrate, or select detection thresholds.

Model development is permanently complete, and all model weights, feature transformers, and decision thresholds are permanently frozen in development freeze commit `6ce11fb8bcb682c10e125219d0b964f394b8a5d2` prior to test evaluation.

---

## 2. Frozen Artifacts & Checksum Manifest

All models, preprocessing objects, and decision thresholds are locked as specified in the Pre-Test Freeze Manifest:

| Artifact Role | File Path | Frozen Checksum (SHA256) | Key Hyperparameters / Settings | Frozen Operating Threshold (Q99.9) |
| :--- | :--- | :--- | :--- | :---: |
| **StandardScaler** | `models/preprocessing/scaler.joblib` | `cbc43757fb1155a36285b711d9b134d387e20e9775a7ab19c8776749a9000d26` | Fitted strictly on `train1` + `train2` (442,802 rows, 79 features) | — |
| **Scaler Metadata** | `models/preprocessing/scaler_metadata.json` | `1e428deea10be2d31a6198d4cf2936c990a702b9ce80354463bbe9b2a271b56a` | 22 zero-variance features retained (`scale=1.0`) | — |
| **Baseline 0 (Persistence)** | *Parameter-free algorithmic baseline* | *Formula:* $\hat{\mathbf{x}}_{t+1} = \mathbf{x}_t$ | 79 features, zero parameters | $\theta^* = 1.680523$ |
| **Baseline 1 (Ridge)** | `models/baselines/ridge.joblib` | `8985e6b46fad2b659a20cfd9bc1a3fc8f1d57be44ae3c6fec81bbd5cb3b7caff` | $\alpha = 300.0$, solver `cholesky`, `fit_intercept=True` | $\theta^* = 1.400928$ |
| **Ridge Metadata** | `models/baselines/ridge_metadata.json` | `b3ea2895db085c1143aeccf20049a0c7087390e74a103fdaa4d08b6647ca258d` | Best $\alpha$ from predetermined extended grid | — |
| **Primary GRU (Seed 42)** | `models/primary/gru_w30_seed42.pt` | `48013cdb36a7490330ba615874399ff21f0a2c3f9b5f7f6edc756168233e2c3d` | $W=30$, $H=64$, 1 layer, 32,975 params (**Canonical Primary**) | $\theta^* = 1.058582$ |
| **Primary GRU (Seed 43)** | `models/primary/gru_w30_seed43.pt` | `ac4bfb705c5dd28e114d32b7a9bc654dc2acc7875180f86f06c7661780d7c75c` | $W=30$, $H=64$, 1 layer, 32,975 params (Stability Replicate 1) | $\theta^* = 1.195869$ |
| **Primary GRU (Seed 44)** | `models/primary/gru_w30_seed44.pt` | `34443577a9a3a91f2dbeddbb35481c9e1610f3e8d0f40fb360a67af960e3c660` | $W=30$, $H=64$, 1 layer, 32,975 params (Stability Replicate 2) | $\theta^* = 1.076832$ |
| **GRU Metadata** | `models/primary/gru_metadata.json` | `453783636af47c6b42d22f53c9caf1d593cb4ec308f675e5f95398d3b20d21b1` | Documented 50-epoch convergence, early stopping | — |
| **GRU Thresholds** | `results/metrics/gru_thresholds.json` | `3b9f80c719e690be5c9da18b6ba50af5c03be88420f7e38f0e4398733145eccf` | Calibrated on `train3` full 1 Hz validation | — |

---

## 3. Score Formulation & Terminology

The primary anomaly score for all models is defined as:
$$\text{RMSE (normalized } L_2 \text{ residual) across the 79 standardized features}$$

$$s_t = \sqrt{\frac{1}{79} \sum_{j=1}^{79} (x_{t, j} - \hat{x}_{t, j})^2}$$

All squared differences are accumulated in `float64` precision before taking the mean and square root.

---

## 4. Common Final Evaluation Timeline

To ensure rigorous, scientifically fair, side-by-side comparison across Baseline 0 (Persistence), Baseline 1 (Ridge), and the Primary GRU, every model must be evaluated on an identical scored timeline within each official test recording:

### The Historical Window Constraint
- The Primary GRU requires a historical sliding window of $W = 30$ seconds (process observations at indices $t-29, t-28, \dots, t$) to predict the state vector at target second $t+1$.
- Consequently, for a recording session starting at row index $0$, the earliest target prediction the GRU can formulate is at **target row index $30$** (using rows $0$ through $29$ as historical context).

### The Common Evaluation Rule
1. **Target Row Index $\ge 30$**:
   - In each of the five official test recordings (`test1` through `test5`), the common evaluable timeline consists strictly of rows with **`target row index >= 30`** (0-indexed).
2. **Exclusion of Early Baseline Scores**:
   - While Persistence and Ridge can technically compute one-step predictions starting from target row index $1$ (using row index $0$), their first $29$ target predictions (row indices $1$ to $29$) **must be excluded from the primary cross-model benchmark metrics**.
   - This guarantees that Persistence, Ridge, and all GRU seeds are scored over the exact same seconds and identical physical process intervals.
3. **Session Boundary Independence**:
   - Sliding windows and one-step predictions **must never cross recording session boundaries**.
   - Each test file (`test1` through `test5`) is processed as an isolated, continuous recording.

---

## 5. Ground-Truth Attack Event Formulation & Process Labels

Attack labels in the HAI dataset are provided at 1-second granularity in the binary `attack` column:

1. **Maximal Contiguous Runs**:
   - A ground-truth attack event is defined as a maximal contiguous run where:
     $$\text{attack}_t == 1$$
     within an individual test file.
   - Attack events are never merged across file boundaries.
2. **Recorded Event Metadata**:
   - `file`: Recording name (`test1`–`test5`).
   - `attack_event_id`: Unique identifier (e.g. `test1_attack_01`).
   - `original_start`: Timestamp of first sample where $\text{attack} == 1$.
   - `original_end`: Timestamp of last sample where $\text{attack} == 1$.
   - `duration_seconds`: Total continuous elapsed seconds ($\text{end} - \text{start} + 1$).
3. **Process-Label Event Summary**:
   For each ground-truth attack event, process labels are summarized deterministically:
   - `attack_P1_event = any(attack_P1 == 1 within the event)`
   - `attack_P2_event = any(attack_P2 == 1 within the event)`
   - `attack_P3_event = any(attack_P3 == 1 within the event)`
   - `process_set`: List of active process tags during the event (e.g. `["P1", "P3"]`).
4. **Pre-Region and Left-Censored Attacks**:
   - **Not Evaluable Attacks**: Any attack event whose entire duration falls strictly before row index $30$ cannot be predicted by the GRU under the common timeline. It must be marked as `"not evaluable"` and reported explicitly in the audit catalog.
   - **Left-Censored Attacks**: If an attack begins prior to row index $30$ but continues into the evaluable region (row index $\ge 30$), it is marked as `"left-censored"`. For detection-delay calculation, its **effective attack start** is defined as the first evaluable timestamp (row index $30$). The original physical start timestamp is preserved in the event table.

---

## 6. Predicted Alarm Events & Filter-Free Policy

For any frozen model and decision threshold $\theta$:

1. **Point Decision Rule**:
   $$\text{alarm}_t = (\text{score}_t > \theta)$$
   *(Note: $\text{score}_t == \theta$ is classified as normal).*
2. **Alarm Event Definition**:
   - A predicted alarm event is defined as a maximal contiguous run of:
     $$\text{alarm}_t == \text{True}$$
3. **Strictly Filter-Free Evaluation**:
   The primary evaluation enforces pure model residual behavior without post-processing artifacts:
   - **No Temporal Smoothing**: Scores are not smoothed using rolling averages, medians, or exponential filters.
   - **No Hysteresis**: Detection does not use separate assertion and de-assertion thresholds.
   - **No Alarm Merging**: Alarm events separated by even a single normal sample ($1\text{ second}$) are treated as distinct events.
   - **No Minimum-Duration Filter**: A single-second exceedance is treated as an active alarm event.

---

## 7. Deterministic One-to-One Chronological Event Matching

To prevent double-counting and eliminate ambiguity in event-level scoring, alarms and attacks are matched using a strictly deterministic, chronological one-to-one algorithm:

```
Algorithm 1: Deterministic Chronological Event Matching
Input: Ground-Truth Attack Events {A_1, ..., A_m} in test file F
       Predicted Alarm Events {E_1, ..., E_n} in test file F
Output: Matched Pairs (True Positives), Unmatched Attacks (False Negatives),
        Unmatched Alarms (Event-Level False Positives)

1. Sort {A_i} in ascending order of start timestamp.
2. Sort {E_j} in ascending order of start timestamp.
3. Initialize MatchedAttacks = {}, MatchedAlarms = {}.
4. For each attack A_i in {A_1, ..., A_m}:
     Find the earliest alarm E_j in {E_1, ..., E_n} such that:
       a) E_j not in MatchedAlarms
       b) Temporal overlap(A_i, E_j) > 0 seconds:
          max(start(A_i), start(E_j)) <= min(end(A_i), end(E_j))
     If such E_j exists:
       Match A_i with E_j
       Add A_i to MatchedAttacks
       Add E_j to MatchedAlarms
5. Return:
     True Positives (TP)                  = MatchedPairs (count = |MatchedAttacks|)
     False Negatives (FN)                 = Evaluable Attacks not in MatchedAttacks
     Event-Level False Positives (FP_ev)  = Predicted Alarm Events not in MatchedAlarms
```

### Constraints & Event-Level FP Definition:
- Each predicted alarm event can match at most **one** ground-truth attack event.
- Each ground-truth attack event can match at most **one** predicted alarm event.
- **Event-Level False Positives ($\text{FP}_{\text{ev}}$)**: All predicted alarm events that do not match an attack event. This includes alarms during completely normal periods **and** redundant alarm chatter occurring during an attack window (after the first matching alarm has been paired). This strictly penalizes excessive chatter in Event Precision and Event F1.

---

## 8. Primary Event Metrics

Using the counts obtained from Algorithm 1:

1. **Event Precision**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}_{\text{ev}}}$$
   *(If $\text{TP} + \text{FP}_{\text{ev}} == 0$, Precision is defined as $0.0$).*
2. **Event Recall**:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   *(If $\text{TP} + \text{FN} == 0$, Recall is defined as $0.0$).*
3. **Event F1 Score**:
   $$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   *(If $\text{Precision} + \text{Recall} == 0.0$, F1 is defined as $0.0$).*

---

## 9. Detection Delay Formulation

For each detected attack event $A_i$ (where a valid match exists):

1. **Detection Timestamp**:
   The timestamp of the **first alarm-positive point** inside the evaluable attack window:
   $$t_{\text{detect}} = \min \{ t \mid t \in [\text{effective\_start}(A_i), \text{end}(A_i)] \land \text{alarm}_t == \text{True} \}$$
2. **Detection Delay**:
   $$\Delta t_{\text{delay}} = t_{\text{detect}} - \text{effective\_start}(A_i) \quad (\text{seconds})$$
3. **Summary Statistics**:
   The distribution of detection delays across all detected attacks is reported with:
   - Mean delay
   - Median delay
   - Standard deviation
   - Minimum delay
   - Maximum delay
   - 95th percentile delay
4. **Handling of Missed Attacks**:
   For false negative attacks (missed attacks), detection delay is recorded as **`NA` (missing / null)**. Missed attacks are **never assigned an artificial infinite delay** ($\infty$), which would distort numerical moments.

---

## 10. Separation of Event-Level FP vs. Normal False Alarms

> [!CAUTION]
> **Do not conflate Event-Level FP with Normal False-Alarm Rate.**
> - **Event-Level FP** counts *all unmatched predicted alarm events* (including redundant chatter during attacks) and is used strictly for **Event Precision and Event F1**.
> - **Normal False Alarms** measure the *operational false-alarm burden strictly during normal operating conditions* ($\text{attack} == 0$), where both numerator and denominator refer to the exact same normal exposure.

### Normal False-Alarm Definitions:
1. **Normal Scored Exposure**:
   $$\text{Normal Scored Samples} = \sum_{t \ge 30} \mathbb{I}(\text{attack}_t == 0)$$
   $$\text{Normal Scored Exposure (Hours)} = \frac{\text{Normal Scored Samples}}{3600}$$
2. **Normal False-Alarm Sample**:
   $$\text{normal\_alarm}_t = (\text{alarm}_t == \text{True}) \land (\text{attack}_t == 0)$$
3. **Normal False-Alarm Event**:
   A maximal contiguous run of $\text{normal\_alarm}_t == \text{True}$ within each test file. Attack timestamps strictly break normal false-alarm runs.
4. **Normal Operational Metrics**:
   - **Normal False-Alarm Events / Hour**:
     $$\text{Normal FA Events / Hour} = \frac{\text{Count of Normal False-Alarm Events}}{\text{Normal Scored Exposure (Hours)}}$$
   - **False-Positive Alarm Seconds / Hour**:
     $$\text{FP Alarm Seconds / Hour} = \frac{\text{Count of Normal Scored Samples with } \text{alarm}_t == \text{True}}{\text{Normal Scored Exposure (Hours)}}$$
   - **Point False-Positive Rate (FPR)**:
     $$\text{Point FPR} = \frac{\text{Count of Normal Scored Samples with } \text{alarm}_t == \text{True}}{\text{Total Normal Scored Samples}}$$

---

## 11. Multi-Seed Stability Policy for Primary GRU

To guarantee scientific rigor and avoid post-hoc cherry-picking:

1. **Three-Seed Evaluation**:
   - All three frozen GRU models (seeds 42, 43, 44) will be executed independently across `test1`–`test5` under identical conditions using their respective frozen validation thresholds.
2. **Reported Aggregate**:
   - The primary GRU performance will be reported as:
     $$\text{Mean} \pm \text{Standard Deviation across Seeds 42, 43, 44}$$
3. **Canonical Headline Model**:
   - **Seed 42** is designated *a priori* as the canonical representative model for headline timeline figures, system demos, and individual attack plots.
   - Selecting a different seed based on superior test performance is strictly prohibited.

---

## 12. Final Test Execution Plan & Output Manifest

Upon authorized execution of the test stage, the evaluation script will generate the following structured outputs:

### Metric Tables (`results/metrics/`):
- `final_test_summary.csv`: Direct side-by-side benchmark (Persistence vs. Ridge vs. GRU Seed 42 vs. GRU 3-Seed Mean).
- `final_test_multiseed.csv`: Detailed per-seed evaluation for seeds 42, 43, and 44 across all five test sessions.
- `final_attack_events.csv`: Ground-truth attack catalog with duration, process labels (`attack_P1`, `attack_P2`, `attack_P3`, `process_set`), and detection status per model.
- `final_alarm_events.csv`: Complete listing of all predicted alarm events, durations, and matching outcomes.
- `final_detection_delays.csv`: Per-attack detection delay breakdown for each model.
- `final_per_file.csv`: Granular performance breakdown broken down by individual test recording (`test1` through `test5`).

### Publication Figures (`results/figures/final_test/`):
- `fig_01_model_comparison_f1_precision_recall.png`: Bar chart of Event F1, Precision, and Recall.
- `fig_02_attack_timeline_overview.png`: Complete timeline of official test sessions with ground-truth attacks and canonical GRU alarms.
- `fig_03_detection_delay_distribution.png`: Box plot and CDF of detection delays.
- `fig_04_attack_detection_matrix.png`: Heatmap indicating detection vs. miss status for each individual attack.
- `fig_05_false_alarm_analysis.png`: Distribution of false-alarm events, durations, and rates per recording.
- `fig_06_representative_attacks_detail.png`: Detailed telemetry and residual traces for representative detected and missed attacks.

---

## 13. Verification of Current State

- **Development Freeze Commit**: `6ce11fb8bcb682c10e125219d0b964f394b8a5d2`
- **HAI Upstream Commit**: `2a814cebc9a66b06c9e5cd545e2d72e65d383737`
- **Official Test Data Status**: **STRICTLY SEALED** (`test1.csv.gz` through `test5.csv.gz` unmodified since download).
- **Execution State**: **STOPPED**. No test data read, preprocessed, or evaluated in this step.
