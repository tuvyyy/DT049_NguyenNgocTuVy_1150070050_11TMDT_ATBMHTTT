# DT049 Post-Test Error Analysis & Scientific Verification

**Document Version:** 1.0  
**Status:** Completed & Frozen Supplementary Post-Test Analysis  
**Repository State:** Frozen at Git Tag [`dt049-final-test-v1`](file:///home/kali/DT049) (`331cc07cb72b8b3f30a64bbe7af52248bd0cd9ba`)  
**Upstream Evaluation Protocol:** [`docs/final_test_protocol.md`](file:///home/kali/DT049/docs/final_test_protocol.md)  
**Execution Timestamp:** 2026-10-03  

---

## Executive Summary & Methodological Constraints

This document provides post-test diagnostic error analysis for the official HAI 21.03 final test evaluation (`test1.csv.gz` through `test5.csv.gz`). In strict accordance with the pre-test freeze protocol:
1. **No Retraining or Parameter Adjustment:** All models (Baseline 0 Persistence, Baseline 1 Ridge $\alpha=300$, and Primary GRU $W=30, H=64$ across seeds 42, 43, 44) remain permanently frozen.
2. **No Recalibration of Thresholds:** The operating thresholds established exclusively on normal validation data (`train3_features.npy`) at quantile $Q_{99.9}$ remain the canonical primary evaluation points.
3. **No Modification of Evaluation Rules:** The common timeline ($t \ge 30$), deterministic 1-to-1 chronological event matching, and strict separation between event-level false positives and normal false alarms are strictly preserved.
4. **Frozen Primary Metrics Intact:** The original primary metric tables remain unmodified. The analyses presented below provide supplementary physical, statistical, and operational context to explain detector behavior.

---

## 1. Process-Level Detection Analysis

### 1.1 Evaluability and Subsystem Attribution
Across the 5 official test sessions, exactly **50 ground-truth attack events** occurred ($100\%$ evaluable on the common timeline $t \ge 30$). The official HAI 21.03 dataset labels denote the physical subsystems directly targeted:
- **Process 1 (P1 - Water Treatment):** 34 attacks involved P1 (23 targeting P1 exclusively; 11 multi-process).
- **Process 2 (P2 - Storage & Pumping):** 22 attacks involved P2 (11 targeting P2 exclusively; 11 multi-process).
- **Process 3 (P3 - Boiler):** 8 attacks involved P3 (2 targeting P3 exclusively; 6 multi-process).
- **Multi-Process Attacks:** 14 attacks spanned multiple subsystems (8 P1;P2, 3 P1;P3, 3 P2;P3).

> [!NOTE]
> Physical mechanisms are not inferred solely from process labels. Rather, subsystem labels designate the target boundary within the plant telemetry.

### 1.2 Subsystem Detection Breakdown
The detection counts, misses, and recall rates across all detectors are summarized below (persisted in [`results/metrics/final_error_analysis_by_process.csv`](file:///home/kali/DT049/results/metrics/final_error_analysis_by_process.csv)):

| Model ID | Model Name | Subsystem / Process Group | Evaluable Attacks ($N$) | Detected ($TP$) | Missed ($FN$) | Event Recall |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `persistence` | Baseline 0: Persistence | P1 (all involving) | 34 | 11 | 23 | 0.3235 |
| `persistence` | Baseline 0: Persistence | P2 (all involving) | 22 | 16 | 6 | 0.7273 |
| `persistence` | Baseline 0: Persistence | P3 (all involving) | 8 | 5 | 3 | 0.6250 |
| `persistence` | Baseline 0: Persistence | Multi-process (>1 process) | 14 | 10 | 4 | 0.7143 |
| `persistence` | Baseline 0: Persistence | P1 only | 23 | 4 | 19 | 0.1739 |
| `persistence` | Baseline 0: Persistence | P2 only | 11 | 8 | 3 | 0.7273 |
| `persistence` | Baseline 0: Persistence | P3 only | 2 | 0 | 2 | 0.0000 |
| `persistence` | Baseline 0: Persistence | **All Attacks** | **50** | **22** | **28** | **0.4400** |
| `ridge` | Baseline 1: Ridge ($\alpha=300$) | P1 (all involving) | 34 | 21 | 13 | 0.6176 |
| `ridge` | Baseline 1: Ridge ($\alpha=300$) | P2 (all involving) | 22 | 22 | 0 | **1.0000** |
| `ridge` | Baseline 1: Ridge ($\alpha=300$) | P3 (all involving) | 8 | 5 | 3 | 0.6250 |
| `ridge` | Baseline 1: Ridge ($\alpha=300$) | Multi-process (>1 process) | 14 | 13 | 1 | 0.9286 |
| `ridge` | Baseline 1: Ridge ($\alpha=300$) | P1 only | 23 | 11 | 12 | 0.4783 |
| `ridge` | Baseline 1: Ridge ($\alpha=300$) | P2 only | 11 | 11 | 0 | **1.0000** |
| `ridge` | Baseline 1: Ridge ($\alpha=300$) | P3 only | 2 | 0 | 2 | 0.0000 |
| `ridge` | Baseline 1: Ridge ($\alpha=300$) | **All Attacks** | **50** | **35** | **15** | **0.7000** |
| `gru_seed42` | Primary GRU (Canonical) | P1 (all involving) | 34 | 26 | 8 | **0.7647** |
| `gru_seed42` | Primary GRU (Canonical) | P2 (all involving) | 22 | 22 | 0 | **1.0000** |
| `gru_seed42` | Primary GRU (Canonical) | P3 (all involving) | 8 | 6 | 2 | **0.7500** |
| `gru_seed42` | Primary GRU (Canonical) | Multi-process (>1 process) | 14 | 14 | 0 | **1.0000** |
| `gru_seed42` | Primary GRU (Canonical) | P1 only | 23 | 15 | 8 | **0.6522** |
| `gru_seed42` | Primary GRU (Canonical) | P2 only | 11 | 11 | 0 | **1.0000** |
| `gru_seed42` | Primary GRU (Canonical) | P3 only | 2 | 0 | 2 | 0.0000 |
| `gru_seed42` | Primary GRU (Canonical) | **All Attacks** | **50** | **40** | **10** | **0.8000** |
| `gru_seed43` | Primary GRU (Replicate 1) | P1 (all involving) | 34 | 25 | 9 | 0.7353 |
| `gru_seed43` | Primary GRU (Replicate 1) | P2 (all involving) | 22 | 22 | 0 | **1.0000** |
| `gru_seed43` | Primary GRU (Replicate 1) | P3 (all involving) | 8 | 6 | 2 | **0.7500** |
| `gru_seed43` | Primary GRU (Replicate 1) | Multi-process (>1 process) | 14 | 14 | 0 | **1.0000** |
| `gru_seed43` | Primary GRU (Replicate 1) | **All Attacks** | **50** | **39** | **11** | **0.7800** |
| `gru_seed44` | Primary GRU (Replicate 2) | P1 (all involving) | 34 | 26 | 8 | **0.7647** |
| `gru_seed44` | Primary GRU (Replicate 2) | P2 (all involving) | 22 | 22 | 0 | **1.0000** |
| `gru_seed44` | Primary GRU (Replicate 2) | P3 (all involving) | 8 | 7 | 1 | **0.8750** |
| `gru_seed44` | Primary GRU (Replicate 2) | Multi-process (>1 process) | 14 | 14 | 0 | **1.0000** |
| `gru_seed44` | Primary GRU (Replicate 2) | **All Attacks** | **50** | **41** | **9** | **0.8200** |

### 1.3 Key Process Insights
1. **P2 Subsystem Universality:** Every model except Persistence achieved $100\%$ recall ($22/22$) on attacks involving P2. P2 attacks typically induce abrupt, large-magnitude valve and flow transitions that violate static inter-channel correlation structures.
2. **Multi-Process Superiority:** GRU models across all seeds detected $100\%$ ($14/14$) of multi-process attack events, outperforming Ridge ($13/14$, $92.86\%$) and Persistence ($10/14$, $71.43\%$).
3. **P1 Differentiation:** P1 contains the largest number of subtle single-loop attacks. On P1-only attacks ($N=23$), Persistence achieved only $17.39\%$ recall ($4/23$), Ridge achieved $47.83\%$ ($11/23$), while canonical GRU seed 42 achieved $65.22\%$ ($15/23$)—a $+36.4\%$ relative recall improvement over Ridge.
4. **P3 Boiler Attenuation:** Out of 2 isolated P3 attacks, both Persistence and Ridge detected 0. Canonical GRU seed 42 missed both at $Q_{99.9}$, whereas GRU seed 44 detected `test2_atk_19` (delay: 101s).

---

## 2. Common-Detected Paired Delay Comparison (Selection Bias Correction)

In the initial test summary report, the unconditional median detection delay was reported as:
- **Baseline 1 (Ridge):** $13.0\text{ s}$ (based on $N=35$ detected attacks)
- **Primary GRU (Seed 42):** $22.0\text{ s}$ (based on $N=40$ detected attacks)

This unconditional comparison is subject to severe **selection bias (survival bias)**: Ridge detected 35 attacks, which were heavily weighted toward abrupt, instantaneous transients with zero or near-zero detection delay. GRU detected 40 attacks, successfully uncovering 9 additional attacks that Ridge missed completely—many of which were slow-developing transients with naturally larger onset delays.

### 2.1 Paired Delay Evaluation
To eliminate selection bias, we restrict evaluation to the exact subset of attacks detected by **both** models in each pair (persisted in [`results/metrics/final_common_detected_delay.csv`](file:///home/kali/DT049/results/metrics/final_common_detected_delay.csv)):

| Model Pair ($A$ vs $B$) | Common Detected Attacks ($N$) | Model $A$ Mean Delay | Model $A$ Median Delay | Model $B$ Mean Delay | Model $B$ Median Delay | Mean Paired Diff ($B - A$) | Median Paired Diff ($B - A$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence vs Ridge** | 22 | 68.50 s | 40.50 s | 38.09 s | 25.50 s | **-30.41 s** | **-0.50 s** |
| **Persistence vs GRU seed42** | 22 | 68.50 s | 40.50 s | 31.59 s | 25.50 s | **-36.91 s** | **-0.50 s** |
| **Ridge vs GRU seed42** | 31 | 37.42 s | 21.00 s | 28.58 s | 21.00 s | **-8.84 s** | **0.00 s** |

### 2.2 Findings on Detection Latency
1. **On Common Attacks, GRU is Faster on Average:** When restricted to the 31 attacks detected by both Ridge and GRU seed 42, GRU's mean detection delay is **$28.58\text{ s}$ vs Ridge's $37.42\text{ s}$**—an average speedup of **$8.84\text{ seconds}$** in favor of GRU.
2. **Median Paired Delay is Identical:** The paired median delay on common detected attacks is **$21.0\text{ s}$** for both Ridge and GRU seed 42, yielding a median paired difference of exactly **$0.00\text{ s}$**.
3. **Apparent Ridge Latency Advantage Was Selection Bias:** Ridge's apparent unconditional $13.0\text{ s}$ median was driven entirely by 4 abrupt attacks that GRU missed at $Q_{99.9}$, while omitting the 9 harder attacks that GRU captured. On the shared detection envelope, GRU reacts as fast as or faster than Ridge.

---

## 3. Factual Characteristics of Missed Attacks (GRU Seed 42)

Canonical GRU seed 42 missed exactly 10 of the 50 evaluable attacks (persisted in [`results/metrics/final_missed_attack_analysis.csv`](file:///home/kali/DT049/results/metrics/final_missed_attack_analysis.csv)). Rather than speculating on unverified causal theories (e.g., closed-loop compensation), we document the factual telemetry and residual metrics:

| Attack Event ID | Session | Duration | Process Set | Peak GRU Score | Frozen Thresh $\theta^*$ | Score / $\theta^*$ Ratio | Ridge Det? | Persist Det? | Nearest Alarm Before | Nearest Alarm After | Factual Telemetry & Residual Observation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `test2_atk_05` | `test2` | 123 s | P1 | 0.8217 | 1.058582 | 0.7763 | **Yes** (12s) | No | 2092 s | 2319 s | Ridge barely exceeded threshold ($1.4438$ vs $1.4009$, ratio $1.031$); GRU score peaked at $0.8217$ ($77.6\%$ of threshold). |
| `test2_atk_06` | `test2` | 256 s | P1 | 0.8848 | 1.058582 | 0.8358 | No | No | 711 s | 2711 s | Missed by all models. Residual remained low across all 79 features; peak GRU score reached $83.6\%$ of threshold. |
| `test2_atk_09` | `test2` | 199 s | P1 | 1.0481 | 1.058582 | **0.9901** | No | No | 481 s | 122 s | **Near miss:** Max score reached $1.0481$ ($99.01\%$ of threshold), falling short of alarm threshold by only $0.0105$ RMSE. |
| `test2_atk_10` | `test2` | 421 s | P1 | 0.7317 | 1.058582 | 0.6912 | **Yes** (187s)| No | 4560 s | 1173 s | Ridge triggered late alarm at $t=187\text{s}$ (peak $1.4601$ vs $1.4009$, ratio $1.042$); GRU score stayed stable at $0.7317$ ($69.1\%$). |
| `test2_atk_14` | `test2` | 152 s | P1 | 0.7865 | 1.058582 | 0.7429 | No | No | 861 s | 2639 s | Missed by all models. Peak GRU score $0.7865$ ($74.3\%$ of threshold); sensor excursions did not exceed normal operating variance. |
| `test2_atk_16` | `test2` | 65 s | P1 | 0.7167 | 1.058582 | 0.6770 | **Yes** (13s) | No | 995 s | 290 s | Ridge detected at $t=13\text{s}$ (peak $1.4604$ vs $1.4009$, ratio $1.042$); GRU peak score reached $0.7167$ ($67.7\%$). |
| `test2_atk_19` | `test2` | 119 s | P3 | 0.6559 | 1.058582 | 0.6196 | No | No | 2484 s | 709 s | Missed by Persistence, Ridge, and GRU seed 42/43; detected by GRU seed 44 at $t=101\text{s}$. GRU seed 42 peak was $0.6559$ ($62.0\%$). |
| `test2_atk_20` | `test2` | 119 s | P3 | 0.8685 | 1.058582 | 0.8204 | No | No | 343 s | 448 s | Missed by all models and all seeds. P3 boiler disturbance; GRU seed 42 peak reached $0.8685$ ($82.0\%$ of threshold). |
| `test3_atk_01` | `test3` | 132 s | P1 | 0.9772 | 1.058582 | **0.9231** | No | No | 12 s | 32 s | Missed by all models. High background residual, peak $0.9772$ ($92.3\%$ threshold), bounded by normal alarms 12s before and 32s after. |
| `test5_atk_09` | `test5` | 241 s | P1 | 1.0193 | 1.058582 | **0.9629** | **Yes** (10s) | No | 1160 s | 3660 s | **Near miss:** GRU peak reached $1.0193$ ($96.29\%$ of threshold). Ridge detected at $t=10\text{s}$ (peak $1.4529$ vs $1.4009$, ratio $1.037$). |

### 3.1 Synthesis of Missed Attacks
- **Near Misses:** Two attacks (`test2_atk_09` at $99.0\%$ and `test5_atk_09` at $96.3\%$) fell just short of the frozen $Q_{99.9}$ threshold. At the secondary $Q_{99.5}$ threshold ($\theta^*=0.6957$), both were detected.
- **Universal Misses:** 5 attacks (`test2_atk_06`, `test2_atk_09`, `test2_atk_14`, `test2_atk_20`, `test3_atk_01`) were missed by Persistence, Ridge, and canonical GRU seed 42.
- **Baseline Detections ($N=4$):** In all 4 attacks where Ridge alarmed but GRU seed 42 did not (`test2_atk_05`, `test2_atk_10`, `test2_atk_16`, `test5_atk_09`), Ridge's anomaly score exceeded its threshold by only $3.1\%$ to $4.2\%$. Ridge's 1-step linear formulation reacts to instantaneous 1-sample transitions between $t-1$ and $t$, whereas GRU integrates a 30-second context window, dampening transient single-step anomalies.

---

## 4. Event Chatter Decomposition & Low Event Precision

Across all detectors, Event Precision is low ($0.0487$ to $0.1297$) despite high recall ($0.44$ to $0.82$). This phenomenon is completely explained by **event chatter** and the strict 1-to-1 event matching rule.

### 4.1 Alarm Event Decomposition Identity
Under strict 1-to-1 matching, every predicted alarm event belongs to exactly one of three mutually exclusive categories:
$$\text{Total Predicted Alarms} = \text{Matched TP Alarms} + \text{Redundant In-Attack Alarms} + \text{Normal False Alarms}$$
$$\text{Event-Level FP} = \text{Redundant In-Attack Alarms} + \text{Normal False Alarms}$$

| Model ID | Total Predicted Alarm Events ($N_{alm}$) | Matched $TP_{events}$ | Redundant In-Attack Alarms ($FP_{red}$) | Normal False Alarms ($FP_{norm}$) | Event-Level FP ($FP_{ev}$) | Event Precision | Mean Alarm Duration | Median Alarm Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `persistence` | 452 | 22 | 26 | 404 | 430 | 0.0487 | 1.13 s | 1.0 s |
| `ridge` | 550 | 35 | 26 | 489 | 515 | 0.0636 | 4.34 s | 1.0 s |
| `gru_seed42` | 375 | 40 | 27 | 308 | 335 | 0.1067 | 9.63 s | 1.0 s |
| `gru_seed43` | 356 | 39 | 20 | 297 | 317 | 0.1096 | 9.60 s | 1.0 s |
| `gru_seed44` | 316 | 41 | 25 | 250 | 275 | 0.1297 | 11.13 s | 1.0 s |

### 4.2 Structural Mechanisms Driving Low Event Precision
1. **1-Second Median Alarm Pulse (Chatter):** Across all models, the median alarm duration is exactly **1.0 second**. Anomaly scores oscillate across the threshold boundary during an attack, generating multiple discrete 1-second alarm pulses rather than a single sustained alarm.
2. **Redundant Alarm Penalization:** When an attack lasts 200 seconds and triggers 5 separate alarm pulses:
   - The first pulse matches the attack ($TP = 1$).
   - The remaining 4 pulses cannot match (1-to-1 rule) and are tallied as **Event-Level False Positives** ($FP = 4$), even though they occurred entirely inside the ground-truth attack window.
   - For canonical GRU seed 42, 27 alarm events ($8.1\%$ of all event-level FPs) were redundant pulses during ongoing attacks.
3. **High Normal Exposure Time:** The test dataset spans **$109.1411\text{ hours}$** of normal operation ($392,908\text{ normal scored samples}$). At $Q_{99.9}$, by mathematical definition, $0.1\%$ of normal time steps may exceed the threshold. For uncorrelated noise, this generates up to $3.6\text{ alarms/hour}$. GRU generates $2.82\text{ normal alarm events/hour}$, producing 308 normal alarm events across 109.1411 hours. Compared to only 50 total attacks, this yields an event-level precision of $\approx 0.11$.

---

## 5. False-Alarm Tradeoff: Event Frequency vs Temporal Occupancy

A critical operational distinction emerges when comparing discrete false alarm frequency against continuous alarm temporal occupancy:

| Model ID | Normal FA Events / Hour | Total FP Alarm Seconds / Hour | Point False Positive Rate (FPR) | Mean Normal Alarm Duration |
| :--- | :---: | :---: | :---: | :---: |
| `persistence` | 3.7016 ev/h | **3.9215 s/h** | **0.001089** | 1.06 s |
| `ridge` | 4.4804 ev/h | 6.5237 s/h | 0.001812 | 1.46 s |
| `gru_seed42` | **2.8220 ev/h** | 5.8915 s/h | 0.001637 | 2.08 s |
| `gru_seed43` | **2.7212 ev/h** | 5.5433 s/h | 0.001540 | 2.04 s |
| `gru_seed44` | **2.2906 ev/h** | 5.3142 s/h | 0.001476 | 2.32 s |
| **GRU 3-Seed Mean** | **$2.6113 \pm 0.2305$** | **$5.5830 \pm 0.2384$** | **$0.001551 \pm 0.000066$** | **$2.15\text{ s}$** |

### 5.1 Factual Operational Tradeoff
- **Discrete Alarm Events (Operator Interruptions):** GRU models demonstrate clear superiority, generating **$2.61\text{ events/hour}$** ($29.5\%$ fewer interruptions than Persistence at $3.70\text{ ev/h}$, and $41.7\%$ fewer than Ridge at $4.48\text{ ev/h}$).
- **Continuous Alarm Occupancy (Total Time in Alarm):** Persistence achieves the lowest total false-alarm duration (**$3.92\text{ seconds/hour}$**, point FPR $= 0.001089$), because Persistence false alarms almost never persist beyond a single isolated second (mean duration $1.06\text{s}$). In contrast, GRU's temporal memory causes false alarm scores to occasionally remain above threshold for 2–3 consecutive seconds (mean normal duration $2.08\text{s}$), leading to **$5.89\text{ seconds/hour}$** of total alarm occupancy.

> [!IMPORTANT]
> GRU minimizes discrete operator alert fatigue (fewer alert notifications per shift), but Persistence minimizes total alarm occupancy time on normal data. Industrial system operators must evaluate this tradeoff according to whether their alerting pipeline penalizes event triggers or alarm state duration.

---

## 6. Sensitivity Analysis across Operating Points ($Q_{99.5}, Q_{99.9}, Q_{99.99}$)

The full suite of sensitivity metrics across all three predetermined operating points is presented below (persisted in [`results/metrics/final_test_sensitivity_complete.csv`](file:///home/kali/DT049/results/metrics/final_test_sensitivity_complete.csv)):

| Model ID | Operating Point | Threshold $\theta^*$ | $TP$ | $FN$ | $FP_{ev}$ | Event Prec | Event Rec | Event F1 | Normal FA ev/h | FP Alarm s/h | Point FPR | Median Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `persistence` | $Q_{99.5}$ | 1.293395 | 37 | 13 | 1293 | 0.0278 | 0.7400 | 0.0536 | 11.0682 | 18.8930 | 0.005248 | 41.0 s |
| `persistence` | **$Q_{99.9}$ (Primary)** | 1.680523 | 22 | 28 | 430 | 0.0487 | 0.4400 | 0.0876 | 3.7016 | 3.9215 | 0.001089 | 40.5 s |
| `persistence` | $Q_{99.99}$ | 1.866531 | 17 | 33 | 64 | 0.2099 | 0.3400 | 0.2595 | 0.4306 | 0.5681 | 0.000158 | 30.0 s |
| `ridge` | $Q_{99.5}$ | 0.851713 | 50 | 0 | 1948 | 0.0250 | 1.0000 | 0.0488 | 17.0513 | 19.3053 | 0.005363 | 18.0 s |
| `ridge` | **$Q_{99.9}$ (Primary)** | 1.400928 | 35 | 15 | 515 | 0.0636 | 0.7000 | 0.1167 | 4.4804 | 6.5237 | 0.001812 | 13.0 s |
| `ridge` | $Q_{99.99}$ | 1.484236 | 29 | 21 | 112 | 0.2057 | 0.5800 | 0.3037 | 0.8338 | 2.8678 | 0.000797 | 13.0 s |
| `gru_seed42` | $Q_{99.5}$ | 0.695715 | 49 | 1 | 1772 | 0.0269 | 0.9800 | 0.0524 | 14.8707 | 19.6809 | 0.005467 | 15.0 s |
| `gru_seed42` | **$Q_{99.9}$ (Primary)** | 1.058582 | 40 | 10 | 335 | 0.1067 | 0.8000 | 0.1882 | 2.8220 | 5.8915 | 0.001637 | 22.0 s |
| `gru_seed42` | $Q_{99.99}$ | 1.376288 | 36 | 14 | 54 | **0.4000** | 0.7200 | **0.5143** | **0.3390** | 2.9686 | 0.000825 | 24.5 s |
| `gru_seed43` | $Q_{99.5}$ | 0.803722 | 49 | 1 | 1813 | 0.0263 | 0.9800 | 0.0513 | 15.7594 | 19.5985 | 0.005444 | 17.0 s |
| `gru_seed43` | **$Q_{99.9}$ (Primary)** | 1.195869 | 39 | 11 | 317 | 0.1096 | 0.7800 | 0.1921 | 2.7212 | 5.5433 | 0.001540 | 21.0 s |
| `gru_seed43` | $Q_{99.99}$ | 1.359241 | 36 | 14 | 54 | **0.4000** | 0.7200 | **0.5143** | **0.3573** | 2.9686 | 0.000825 | 24.5 s |
| `gru_seed44` | $Q_{99.5}$ | 0.713679 | 49 | 1 | 1759 | 0.0271 | 0.9800 | 0.0527 | 15.0173 | 19.8184 | 0.005505 | 17.0 s |
| `gru_seed44` | **$Q_{99.9}$ (Primary)** | 1.076832 | 41 | 9 | 275 | 0.1297 | 0.8200 | 0.2240 | 2.2906 | 5.3142 | 0.001476 | 24.0 s |
| `gru_seed44` | $Q_{99.99}$ | 1.366715 | 36 | 14 | 57 | **0.3871** | 0.7200 | **0.5035** | **0.3573** | 2.9411 | 0.000817 | 25.5 s |

### 6.1 Sensitivity Findings
1. **$Q_{99.9}$ Remains the Frozen Primary Operating Point:** In accordance with protocol, $Q_{99.9}$ remains the single frozen operating threshold. It balances high recall ($0.80$) with a manageable false-alarm rate ($2.82\text{ ev/h}$).
2. **$Q_{99.5}$ High-Recall Profile:** Lowering the threshold to $Q_{99.5}$ elevates GRU recall to **$98.0\%$** ($49/50$ detected), missing only `test2_atk_19`. However, normal false alarms increase to $\sim 15\text{ ev/h}$ ($19.7\text{ s/h}$).
3. **$Q_{99.99}$ High-Precision Profile:** Increasing the threshold to $Q_{99.99}$ drastically suppresses false alarms to **$0.34\text{ ev/h}$** (one false alarm every 3 hours), driving Event Precision to **$0.4000$** and Event F1 to **$0.5143$**, while preserving a remarkable **$72.0\%$ recall** ($36/50$ detected).

---

## 7. Representative Cases: Telemetry & Residual Trajectories

Four representative attack cases were analyzed and visualized under [`results/figures/error_analysis/`](file:///home/kali/DT049/results/figures/error_analysis/):

### Case 1: Detected by All Models (`test1_atk_04`)
- **Subsystem:** P2 (Pumping/Storage), Duration: $60\text{ seconds}$
- **Detection Delays:** Persistence: $0.0\text{s}$, Ridge: $0.0\text{s}$, GRU seed 42: $0.0\text{s}$
- **Telemetry Observation:** Immediate, sharp deviation in pump discharge and valve position sensors. All models exceeded their respective thresholds instantaneously at $t=0\text{s}$.
- **Figure:** [`fig_ea_06_case_all_detected.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_06_case_all_detected.png)

### Case 2: GRU-Only Detection (`test1_atk_02`)
- **Subsystem:** P1 (Water Treatment), Duration: $240\text{ seconds}$
- **Detection Delays:** Persistence: Missed, Ridge: Missed, GRU seed 42: Detected ($84.0\text{s}$)
- **Telemetry Observation:** A gradual, cumulative drift in water tank level and pressure. 1-step linear predictors (Persistence and Ridge) adjusted locally to the slow drift and remained below threshold ($1.68$ and $1.40$). GRU accumulated the 30-second temporal trajectory, driving its RMSE score above $\theta^* = 1.0586$ at $t=84\text{s}$.
- **Figure:** [`fig_ea_07_case_gru_only.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_07_case_gru_only.png)

### Case 3: Missed by All Models (`test2_atk_06`)
- **Subsystem:** P1, Duration: $256\text{ seconds}$
- **Telemetry Observation:** Subtle manipulation of flow control parameters. Sensor deviations remained entirely within the envelope of normal operating variance. GRU score peaked at $0.8848$ ($83.6\%$ of threshold).
- **Figure:** [`fig_ea_08_case_all_missed.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_08_case_all_missed.png)

### Case 4: GRU Miss but Ridge Detected (`test2_atk_05`)
- **Subsystem:** P1, Duration: $123\text{ seconds}$
- **Detection Delays:** Persistence: Missed, Ridge: Detected ($12.0\text{s}$), GRU seed 42: Missed
- **Telemetry Observation:** Ridge produced a transient spike to $1.4438$ ($3.1\%$ above its $1.4009$ threshold) at $t=12\text{s}$. GRU's recurrent hidden state absorbed the transient fluctuation, keeping the prediction error at $0.8217$ ($77.6\%$ of threshold).
- **Figure:** [`fig_ea_09_case_gru_miss_ridge_detected.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_09_case_gru_miss_ridge_detected.png)

---

## 8. Audit of Previous Report Claims & Required Softening

In strict adherence to scientific rigor, the following claims made in prior preliminary summaries must be formally softened or corrected based on empirical findings:

1. **Claim on Detection Delay Advantage:**
   - *Previous Statement:* "Ridge detects faster than GRU (median delay 13.0s vs 22.0s)."
   - *Correction / Softening:* This apparent advantage was an artifact of **selection bias**. When evaluating the 31 attacks detected by **both** models, GRU has a lower mean delay ($28.58\text{s}$ vs $37.42\text{s}$) and an identical median delay ($21.0\text{s}$). Ridge's lower unconditional median was driven by the fact that it only detected abrupt attacks while failing on the harder, slower attacks that GRU successfully captured.
2. **Claim on P3 Boiler Detection:**
   - *Previous Statement:* "P3 attacks `test2_atk_19` and `test2_atk_20` were missed by all models."
   - *Correction / Softening:* Only `test2_atk_20` was missed by all models and all seeds. Replicate GRU seed 44 successfully detected `test2_atk_19` at $t=101\text{s}$.
3. **Claim on False-Alarm Superiority:**
   - *Previous Statement:* "GRU outperforms baselines across false-alarm metrics."
   - *Correction / Softening:* GRU is superior in **discrete false-alarm event frequency** ($2.61\text{ ev/h}$ vs $3.70\text{ ev/h}$ for Persistence and $4.48\text{ ev/h}$ for Ridge), minimizing discrete alert fatigue. However, Persistence achieves lower **total false-alarm temporal occupancy** ($3.92\text{ s/h}$ vs $5.58\text{ s/h}$ for GRU) because its false alarms last almost strictly 1 second.
4. **Physical Mechanism Inferences:**
   - *Previous Statement:* References to "closed-loop compensation" or "actuator masking" as causes for misses.
   - *Correction / Softening:* Physical feedback cannot be conclusively asserted without controller setpoint telemetry. We restrict conclusions strictly to factual residual evidence: in all missed attacks, sensor and residual excursions remained within the normal operating distribution.

---

## 9. Generated Artifacts Index

All post-test error analysis artifacts are archived in the repository:

### Metrics & Data Tables:
- [`results/metrics/final_error_analysis_by_process.csv`](file:///home/kali/DT049/results/metrics/final_error_analysis_by_process.csv)
- [`results/metrics/final_common_detected_delay.csv`](file:///home/kali/DT049/results/metrics/final_common_detected_delay.csv)
- [`results/metrics/final_missed_attack_analysis.csv`](file:///home/kali/DT049/results/metrics/final_missed_attack_analysis.csv)
- [`results/metrics/final_test_sensitivity_complete.csv`](file:///home/kali/DT049/results/metrics/final_test_sensitivity_complete.csv)

### Diagnostic Figures:
- [`results/figures/error_analysis/fig_ea_01_process_recall.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_01_process_recall.png)
- [`results/figures/error_analysis/fig_ea_02_common_delay_paired.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_02_common_delay_paired.png)
- [`results/figures/error_analysis/fig_ea_03_event_chatter_breakdown.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_03_event_chatter_breakdown.png)
- [`results/figures/error_analysis/fig_ea_04_false_alarm_tradeoff.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_04_false_alarm_tradeoff.png)
- [`results/figures/error_analysis/fig_ea_05_sensitivity_curves.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_05_sensitivity_curves.png)
- [`results/figures/error_analysis/fig_ea_06_case_all_detected.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_06_case_all_detected.png)
- [`results/figures/error_analysis/fig_ea_07_case_gru_only.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_07_case_gru_only.png)
- [`results/figures/error_analysis/fig_ea_08_case_all_missed.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_08_case_all_missed.png)
- [`results/figures/error_analysis/fig_ea_09_case_gru_miss_ridge_detected.png`](file:///home/kali/DT049/results/figures/error_analysis/fig_ea_09_case_gru_miss_ridge_detected.png)
