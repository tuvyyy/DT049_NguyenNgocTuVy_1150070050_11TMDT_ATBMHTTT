#!/usr/bin/env python3
"""
DT049 — Multivariate Time-Series Anomaly Detection on HAI 21.03
==============================================================
Official Streamlit Demo Dashboard for DT049.
Displays frozen official test results, post-test error analyses,
and post-hoc Dynamic57 ablation without retraining or re-running inference.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "demo"))

from ui_utils import load_csv, load_json, display_figure, format_delay, format_rate

# Configure Streamlit page
st.set_page_config(
    page_title="DT049 — HAI 21.03 Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
st.sidebar.title("🛡️ DT049 ICS Anomaly Detection")
st.sidebar.markdown(
    """
    **Dataset:** HAI 21.03 Testbed  
    **Evaluation:** Frozen Official Test  
    **Environment:** Kali Linux VM  
    """
)
st.sidebar.divider()

navigation_options = [
    "A. Overview",
    "B. Model Comparison",
    "C. Attack Explorer",
    "D. Error Analysis",
    "E. Threshold Sensitivity",
    "F. Dynamic57 Ablation",
    "G. Reproducibility",
]

page = st.sidebar.radio("Navigation", navigation_options, index=0)

st.sidebar.divider()
st.sidebar.info(
    "🔒 **Frozen State:** Official test evaluation permanently tagged at `dt049-final-test-v1` "
    "(commit `331cc07cb72b8b3f30a64bbe7af52248bd0cd9ba`). All artifacts are read-only."
)


# =============================================================================
# A. OVERVIEW
# =============================================================================
if page == "A. Overview":
    st.title("DT049 — Multivariate Time-Series Anomaly Detection on HAI 21.03")
    st.subheader("Leakage-aware anomaly detection using Persistence, Ridge, and GRU predictors")
    st.caption("Frozen Official Evaluation on 5 Unseen Test Sessions (`test1`–`test5`), Operating Point: Validation Quantile $Q_{99.9}$")

    st.markdown("---")

    # Top Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Total Attacks", "50", help="100% evaluable attacks on common timeline (t >= 30)")
    with c2:
        st.metric("GRU s42 Detected", "40 / 50", delta="80.0% Recall")
    with c3:
        st.metric("3-Seed Mean Recall", "0.8000", delta="± 0.0163")
    with c4:
        st.metric("3-Seed Mean F1", "0.2015", delta="± 0.0160")
    with c5:
        st.metric("Normal FA Rate", "2.61 ev/h", delta="-41.7% vs Ridge", delta_color="inverse")
    with c6:
        st.metric("Canonical GRU Delay", "22.0 s", delta="3-seed: ≈ 22.33 s", help="Canonical GRU seed42 median detection delay = 22.0 s. GRU three-seed mean of per-seed median delays ≈ 22.33 s.")

    st.markdown("---")

    # Primary Comparison Figure
    col_fig, col_text = st.columns([1.6, 1.0])
    with col_fig:
        st.markdown("#### Primary Model Performance Comparison")
        display_figure(
            "results/figures/final_test/fig_01_final_model_comparison.png",
            caption="Figure 1: Event-level F1/Precision/Recall, Normal False Alarm Rate, and Detection Latency across all models."
        )

    with col_text:
        st.markdown("#### Scientific & Operational Interpretation")
        st.markdown(
            r"""
            - **Primary Architecture:** Compact 1-layer GRU (Hidden=64, 32,975 parameters) predicting the 79-channel next-second state from a 30-second window ($W=30$).
            - **Detection Sensitivity:** The primary GRU captures **$80.0\%$** of ground-truth attacks ($40/50$), significantly outperforming Baseline 0 Persistence ($44.0\%$) and Baseline 1 Ridge ($70.0\%$).
            - **False-Alarm Mitigation:** GRU suppresses discrete operator false-alarm interruptions to **$2.61\text{ events/hour}$** ($29.5\%$ reduction vs Persistence, $41.7\%$ reduction vs Ridge).
            - **Detection Delay:** Canonical GRU seed42 achieves a median detection delay of **$22.0\text{ s}$** (three-seed mean of per-seed medians $\approx 22.33\text{ s}$).
            - **Event Precision Context:** Event precision remains modest ($\sim 0.11$) because of strict 1-to-1 event matching without artificial hysteresis or event merging across 109.14 hours of normal operation.
            - **Operational Reality:** *This system demonstrates strong research benchmark performance on HAI 21.03; it does not claim zero error or unassisted production readiness.*
            """
        )


# =============================================================================
# B. MODEL COMPARISON
# =============================================================================
elif page == "B. Model Comparison":
    st.title("Model Comparison: Persistence vs. Ridge vs. Primary GRU")
    st.markdown("Detailed side-by-side evaluation across all 5 test sessions (401,855 scored transitions, 109.1411 normal exposure hours).")

    summary_df = load_csv("results/metrics/final_test_summary.csv")
    multiseed_df = load_csv("results/metrics/final_test_multiseed.csv")

    if summary_df is not None:
        st.markdown("### Official Benchmark Summary Table ($Q_{99.9}$ Operating Point)")

        display_cols = [
            "model_name", "tp_detected", "fn_missed", "event_level_fp",
            "event_precision", "event_recall", "event_f1",
            "normal_fa_events_per_hour", "false_positive_alarm_seconds_per_hour",
            "point_fpr", "delay_median_sec", "delay_mean_sec"
        ]
        rename_map = {
            "model_name": "Model",
            "tp_detected": "Detected (TP)",
            "fn_missed": "Missed (FN)",
            "event_level_fp": "Event FP",
            "event_precision": "Precision",
            "event_recall": "Recall",
            "event_f1": "Event F1",
            "normal_fa_events_per_hour": "Normal FA (ev/h)",
            "false_positive_alarm_seconds_per_hour": "FP Time (s/h)",
            "point_fpr": "Point FPR",
            "delay_median_sec": "Median Delay (s)",
            "delay_mean_sec": "Mean Delay (s)",
        }

        table_data = summary_df[display_cols].rename(columns=rename_map)
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    if multiseed_df is not None:
        st.markdown("### Primary GRU Multi-Seed Stability (Seeds 42, 43, 44)")
        ms_cols = [
            "seed", "event_recall", "event_precision", "event_f1",
            "normal_fa_events_per_hour", "false_positive_alarm_seconds_per_hour",
            "point_fpr", "delay_median_sec"
        ]
        ms_rename = {
            "seed": "Seed",
            "event_recall": "Recall",
            "event_precision": "Precision",
            "event_f1": "Event F1",
            "normal_fa_events_per_hour": "Normal FA (ev/h)",
            "false_positive_alarm_seconds_per_hour": "FP Time (s/h)",
            "point_fpr": "Point FPR",
            "delay_median_sec": "Median Delay (s)",
        }
        st.dataframe(multiseed_df[ms_cols].rename(columns=ms_rename), use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Detection Delay Distribution")
        display_figure(
            "results/figures/final_test/fig_03_detection_delay_distribution.png",
            caption="Figure 3: Detection delay distribution (boxplots and cumulative onset latency)."
        )
    with c2:
        st.markdown("#### Normal False-Alarm Rate Analysis")
        display_figure(
            "results/figures/final_test/fig_05_normal_false_alarm_analysis.png",
            caption="Figure 5: False-alarm event rate per hour and point false-positive rate."
        )

    st.info(
        "📌 **Factual Note on Operational Tradeoff:** "
        "GRU reduces discrete false-alarm event frequency relative to both baselines (2.61 ev/h vs 3.70 for Persistence and 4.48 for Ridge), "
        "but Persistence has lower false-positive alarm time occupancy (3.92 s/h vs 5.58 s/h) because its alarms last almost strictly 1 second."
    )


# =============================================================================
# C. ATTACK EXPLORER
# =============================================================================
elif page == "C. Attack Explorer":
    st.title("Attack Event Explorer")
    st.markdown("Interactive inspection of all 50 ground-truth attacks across official test sessions.")

    atks_df = load_csv("results/metrics/final_attack_events.csv")

    if atks_df is not None:
        # Filters
        st.markdown("#### Filter Criteria")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            session_filter = st.selectbox("Test Session", ["All"] + sorted(atks_df["session"].unique().tolist()))
        with col_f2:
            process_filter = st.selectbox("Subsystem / Process Set", ["All", "P1", "P2", "P3", "Multi-process"])
        with col_f3:
            detector_filter = st.selectbox(
                "Detector to Inspect",
                ["gru_seed42", "gru_seed43", "gru_seed44", "ridge", "persistence"],
                format_func=lambda x: {
                    "gru_seed42": "GRU Seed 42 (Canonical)",
                    "gru_seed43": "GRU Seed 43 (Replicate 1)",
                    "gru_seed44": "GRU Seed 44 (Replicate 2)",
                    "ridge": "Ridge Baseline 1",
                    "persistence": "Persistence Baseline 0",
                }.get(x, x)
            )
        with col_f4:
            status_filter = st.selectbox("Detection Status", ["All", "Detected", "Missed"])

        # Apply filtering
        filtered_df = atks_df.copy()
        if session_filter != "All":
            filtered_df = filtered_df[filtered_df["session"] == session_filter]

        if process_filter == "P1":
            filtered_df = filtered_df[filtered_df["process_set"] == "P1"]
        elif process_filter == "P2":
            filtered_df = filtered_df[filtered_df["process_set"] == "P2"]
        elif process_filter == "P3":
            filtered_df = filtered_df[filtered_df["process_set"] == "P3"]
        elif process_filter == "Multi-process":
            filtered_df = filtered_df[filtered_df[["attack_P1", "attack_P2", "attack_P3"]].sum(axis=1) > 1]

        det_col = f"{detector_filter}_detected"
        delay_col = f"{detector_filter}_delay_seconds"

        if status_filter == "Detected":
            filtered_df = filtered_df[filtered_df[det_col] == True]
        elif status_filter == "Missed":
            filtered_df = filtered_df[filtered_df[det_col] == False]

        st.markdown(f"**Showing {len(filtered_df)} of 50 attacks:**")

        table_cols = [
            "attack_event_id", "session", "orig_start", "orig_end",
            "duration_seconds", "process_set",
            "persistence_detected", "ridge_detected", "gru_seed42_detected",
            f"{detector_filter}_delay_seconds"
        ]
        rename_table = {
            "attack_event_id": "Attack ID",
            "session": "File",
            "orig_start": "Start Time",
            "orig_end": "End Time",
            "duration_seconds": "Duration (s)",
            "process_set": "Subsystems",
            "persistence_detected": "Persist Det?",
            "ridge_detected": "Ridge Det?",
            "gru_seed42_detected": "GRU s42 Det?",
            f"{detector_filter}_delay_seconds": f"{detector_filter} Delay (s)",
        }
        st.dataframe(filtered_df[table_cols].rename(columns=rename_table), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Attack Detection Matrix & Timeline")
    display_figure(
        "results/figures/final_test/fig_04_attack_detection_matrix.png",
        caption="Figure 4: Full binary detection matrix across all 50 attacks for Persistence, Ridge, and GRU seeds."
    )
    display_figure(
        "results/figures/final_test/fig_02_test_attack_timeline_seed42.png",
        caption="Figure 2: Complete chronological attack timeline and anomaly score trajectories for GRU seed 42 across official test sessions."
    )


# =============================================================================
# D. ERROR ANALYSIS
# =============================================================================
elif page == "D. Error Analysis":
    st.title("Post-Test Diagnostic Error Analysis")
    st.markdown("In-depth analysis of subsystem detection profiles, paired latency comparison, and alarm chatter decomposition.")

    proc_df = load_csv("results/metrics/final_error_analysis_by_process.csv")
    paired_df = load_csv("results/metrics/final_common_detected_delay.csv")
    missed_df = load_csv("results/metrics/final_missed_attack_analysis.csv")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Subsystem Detection", "Paired Latency (Selection Bias)", "Event Chatter & Precision", "Missed Attacks Explorer"
    ])

    with tab1:
        st.markdown("### Process-Level Anomaly Detection Breakdown")
        if proc_df is not None:
            st.dataframe(proc_df, use_container_width=True, hide_index=True)
        display_figure(
            "results/figures/error_analysis/fig_ea_01_process_recall.png",
            caption="Figure EA-1: Detection recall across process groups: P1 (Water Treatment), P2 (Pumping), P3 (Boiler), Multi-process."
        )

    with tab2:
        st.markdown("### Paired Detection Delay Comparison (Selection Bias Correction)")
        st.markdown(
            """
            *Note:* Unconditional median delays (Ridge 13.0s vs GRU 22.0s) suffered from selection bias because Ridge only detected 35 easier,
            abrupt attacks. When evaluated on the **exact common detected event sets**, GRU detects faster on average:
            """
        )
        if paired_df is not None:
            st.dataframe(paired_df, use_container_width=True, hide_index=True)
        display_figure(
            "results/figures/error_analysis/fig_ea_02_common_delay_paired.png",
            caption="Figure EA-2: Paired latency comparison on common detected attacks (scatter and boxplot)."
        )

    with tab3:
        st.markdown("### Why Event Precision is Low")
        st.markdown(
            """
            Under deterministic 1-to-1 event matching, every predicted alarm event falls into exactly one category:
            $$\\text{Total Predicted Alarms} = \\text{Matched TP Alarms} + \\text{Redundant In-Attack Alarms} + \\text{Normal False Alarms}$$
            $$\\text{Event-Level FP} = \\text{Redundant In-Attack Alarms} + \\text{Normal False Alarms}$$
            
            1. **Alarm Chatter (1-second pulses):** Median alarm duration across all models is strictly **1.0 second**. Oscillations across threshold generate multiple brief pulses during ongoing attacks.
            2. **Redundant Alarm Penalization:** Alarms occurring inside an attack after the first pulse cannot be matched under the 1-to-1 rule and are tallied as False Positives ($20$ to $27$ per detector).
            3. **Long Normal Exposure:** 109.1411 hours ($392,908$ normal seconds) naturally produces $\\sim 2.6 - 4.5$ false alarms per hour at $Q_{99.9}$, capping Event Precision near $0.11$.
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            display_figure(
                "results/figures/error_analysis/fig_ea_03_event_chatter_breakdown.png",
                caption="Figure EA-3: Alarm event decomposition and duration CDF."
            )
        with c2:
            display_figure(
                "results/figures/error_analysis/fig_ea_04_false_alarm_tradeoff.png",
                caption="Figure EA-4: Discrete false alarm frequency vs continuous temporal occupancy."
            )

    with tab4:
        st.markdown("### Canonical GRU Missed Attack Diagnostic")
        st.markdown("Factual telemetry and residual characteristics of the 10 attacks missed by GRU seed 42:")
        if missed_df is not None:
            st.dataframe(missed_df, use_container_width=True, hide_index=True)

            selected_miss = st.selectbox(
                "Select a Missed Attack to Inspect",
                missed_df["attack_event_id"].tolist()
            )
            miss_row = missed_df[missed_df["attack_event_id"] == selected_miss].iloc[0]
            st.write(f"**Selected Attack:** `{selected_miss}` (Session: `{miss_row['session']}`, Duration: `{miss_row['duration_seconds']}s`, Subsystem: `{miss_row['process_set']}`)")
            st.write(f"- Peak Score: `{miss_row['max_gru_score_inside']:.4f}` | Frozen Threshold: `{miss_row['frozen_threshold']:.4f}` | Score/Threshold Ratio: `{miss_row['score_to_threshold_ratio']:.3f}`")
            st.write(f"- Ridge Detected: `{miss_row['ridge_detected']}` | Persistence Detected: `{miss_row['persistence_detected']}`")
            st.write(f"- Nearest Alarms: Before: `{miss_row['nearest_alarm_before_sec']}s` | After: `{miss_row['nearest_alarm_after_sec']}s`")
            st.info(f"Factual Telemetry Observation: {miss_row['factual_telemetry_observation']}")

        st.markdown("#### Representative Attack Case Studies")
        c_case1, c_case2 = st.columns(2)
        with c_case1:
            display_figure("results/figures/error_analysis/fig_ea_06_case_all_detected.png", caption="Case 1: Detected by All Models (test1_atk_04)")
            display_figure("results/figures/error_analysis/fig_ea_08_case_all_missed.png", caption="Case 3: Missed by All Models (test2_atk_06)")
        with c_case2:
            display_figure("results/figures/error_analysis/fig_ea_07_case_gru_only.png", caption="Case 2: GRU-Only Detection (test1_atk_02)")
            display_figure("results/figures/error_analysis/fig_ea_09_case_gru_miss_ridge_detected.png", caption="Case 4: GRU Miss but Ridge Detected (test2_atk_05)")


# =============================================================================
# E. THRESHOLD SENSITIVITY
# =============================================================================
elif page == "E. Threshold Sensitivity":
    st.title("Predefined Threshold Sensitivity Analysis")
    st.markdown("Evaluation of detectors across predetermined quantiles ($Q_{99.5}$, $Q_{99.9}$, $Q_{99.99}$).")

    sens_df = load_csv("results/metrics/final_test_sensitivity_complete.csv")

    st.warning(
        "⚠️ **Operating Point Designation:**\n"
        "- **$Q_{99.9}$** = **FROZEN PRIMARY OPERATING POINT** (Calibrated strictly on normal validation data).\n"
        "- **$Q_{99.5}$ and $Q_{99.99}$** = Predefined sensitivity bounds only. Even though Event F1 is higher at $Q_{99.99}$, it is not selected as a post-hoc replacement winner."
    )

    if sens_df is not None:
        # Restrict strictly to predefined frozen quantiles
        valid_points = ["Q99.5", "Q99.9", "Q99.99"]
        sens_df = sens_df[sens_df["operating_point"].isin(valid_points)].copy()

        q_options = ["Q99.5", "Q99.9 — FROZEN PRIMARY", "Q99.99"]
        q_choice = st.radio("Select Predefined Operating Point to Inspect:", q_options, index=1)
        selected_q = "Q99.9" if "Q99.9" in q_choice else q_choice.strip()

        filtered_sens = sens_df[sens_df["operating_point"] == selected_q].copy()

        disp_cols = [
            "model_name", "threshold_value", "tp_detected", "fn_missed", "event_level_fp",
            "event_precision", "event_recall", "event_f1",
            "normal_fa_events_per_hour", "fp_alarm_seconds_per_hour", "point_fpr",
            "median_detection_delay_sec"
        ]
        disp_rename = {
            "model_name": "Model",
            "threshold_value": "Threshold",
            "tp_detected": "TP",
            "fn_missed": "FN",
            "event_level_fp": "Event FP",
            "event_precision": "Precision",
            "event_recall": "Recall",
            "event_f1": "Event F1",
            "normal_fa_events_per_hour": "Normal FA (ev/h)",
            "fp_alarm_seconds_per_hour": "FP Time (s/h)",
            "point_fpr": "Point FPR",
            "median_detection_delay_sec": "Median Delay (s)",
        }
        st.dataframe(filtered_sens[disp_cols].rename(columns=disp_rename), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Predefined Sensitivity Tradeoff Curves")
    display_figure(
        "results/figures/error_analysis/fig_ea_05_sensitivity_curves.png",
        caption="Figure EA-5: Tradeoff curves across predetermined quantiles Q99.5, Q99.9 (frozen primary), and Q99.99 (Recall vs FA rate, F1 vs FA rate, Recall vs FP Time)."
    )


# =============================================================================
# F. DYNAMIC57 ABLATION
# =============================================================================
elif page == "F. Dynamic57 Ablation":
    st.title("Train-Dynamic Feature Score Ablation")
    st.markdown("Post-hoc exploratory variant testing the contribution of the 22 training-constant channels.")

    st.info(
        "📌 **Ablation Finding:** "
        "The 22 training-constant channels did not alter detector decisions under this post-hoc Q99.9 ablation on HAI 21.03. "
        "On the official test set, some excluded channels become active, so exact score scaling need not hold. "
        "Empirically, their contribution was not large enough to change any Q99.9 alarm decision, attack detection status, or detection delay in this experiment."
    )

    st.markdown(
        r"""
        #### Mathematical Derivation of Score Scaling
        - **Correct Scale Factor:**
          $$\kappa = \sqrt{\frac{79}{57}} \approx 1.177270 \quad \left(\text{Note: } \kappa = \sqrt{\frac{79}{57}} \approx 1.177270, \text{ NOT } \frac{79}{57} \approx 1.177270\right)$$
        - **Validation Normal Invariance:**
          Where the 22 excluded channels have zero residual ($e_j(t) = 0$):
          $$\text{RMSE}_{57}(t) = \sqrt{\frac{79}{57}} \cdot \text{RMSE}_{79}(t) = \kappa \cdot \text{RMSE}_{79}(t)$$
        - **Official Test Caveat:**
          *Exact constant rescaling applies on normal validation because the 22 excluded channels have zero residual there. On the official test set, some excluded channels become active, so exact score scaling need not hold. Empirically, their contribution was not large enough to change any Q99.9 alarm decision, attack detection status, or detection delay in this experiment.*
        """
    )

    ab_sum_df = load_csv("results/metrics/ablation_dynamic57_summary.csv")
    ab_ms_df = load_csv("results/metrics/ablation_dynamic57_multiseed.csv")
    ab_ch_df = load_csv("results/metrics/ablation_dynamic57_changed_events.csv")

    if ab_sum_df is not None:
        st.markdown("### Primary (79 Features) vs. Ablation (57 Dynamic Features) at $Q_{99.9}$")
        q999_ab = ab_sum_df[ab_sum_df["operating_point"] == "Q99.9"].copy()
        cols_show = [
            "model_name", "feature_configuration", "threshold_value",
            "tp_detected", "fn_missed", "event_precision", "event_recall", "event_f1",
            "normal_fa_events_per_hour", "median_detection_delay_sec"
        ]
        rename_ab = {
            "model_name": "Model",
            "feature_configuration": "Feature Set",
            "threshold_value": "Threshold",
            "tp_detected": "TP",
            "fn_missed": "FN",
            "event_precision": "Precision",
            "event_recall": "Recall",
            "event_f1": "Event F1",
            "normal_fa_events_per_hour": "Normal FA (ev/h)",
            "median_detection_delay_sec": "Median Delay (s)",
        }
        st.dataframe(q999_ab[cols_show].rename(columns=rename_ab), use_container_width=True, hide_index=True)

    if ab_ms_df is not None:
        st.markdown("### GRU Multi-Seed Aggregate (79 vs. 57)")
        q_ms = ab_ms_df[ab_ms_df["operating_point"] == "Q99.9"].copy()
        st.dataframe(q_ms, use_container_width=True, hide_index=True)

    if ab_ch_df is not None:
        changed_count = ab_ch_df["changed_status_any_model"].sum()
        st.metric("Total Attacks with Changed Detection Status", f"{changed_count} / 50", delta="0.0% Change")

    st.markdown("---")
    st.markdown("### Ablation Comparison Figures")
    c1, c2 = st.columns(2)
    with c1:
        display_figure(
            "results/figures/ablation_dynamic57/fig_ab_01_primary_vs_dynamic57.png",
            caption=r"Figure AB-1: Recall, F1, and False-Alarm rate comparison between 79-feature primary and 57-feature dynamic ablation ($\kappa = \sqrt{79/57} \approx 1.177270$)."
        )
        display_figure(
            "results/figures/ablation_dynamic57/fig_ab_03_false_alarm_tradeoff.png",
            caption="Figure AB-3: False-alarm event rate vs temporal occupancy."
        )
    with c2:
        display_figure(
            "results/figures/ablation_dynamic57/fig_ab_02_detection_changes.png",
            caption="Figure AB-2: Binary detection status grid showing identical detection across all 50 attacks."
        )


# =============================================================================
# G. REPRODUCIBILITY
# =============================================================================
elif page == "G. Reproducibility":
    st.title("Reproducibility & Freeze Manifest")
    st.markdown("Complete audit trail of the frozen DT049 development and final test evaluation pipeline.")

    pretest_manifest = load_json("results/audit/pretest_freeze_manifest.json")
    exec_audit = load_json("results/audit/final_test_execution.json")

    col_meta1, col_meta2 = st.columns(2)

    with col_meta1:
        st.markdown("#### Git Repository State")
        st.write("- **HAI Upstream Commit:** `2a814cebc9a66b06c9e5cd545e2d72e65d383737`")
        st.write("- **Development Freeze Commit:** `6ce11fb8bcb682c10e125219d0b964f394b8a5d2`")
        st.write("- **Protocol Commit:** `6a0613b87a0a9d89c89b6ae848350b44c1aa1740`")
        st.write("- **Official Final Test Tag:** `dt049-final-test-v1` (`331cc07cb72b8b3f30a64bbe7af52248bd0cd9ba`)")
        st.write("- **Analysis & Ablation Commit:** `4104576d1c4ad69edd87cc97ea6463974becf460`")

        st.markdown("#### Software Environment")
        if exec_audit and "software_environment" in exec_audit:
            env = exec_audit["software_environment"]
            st.write(f"- **Python:** `{env.get('python', '3.14.7')}`")
            st.write(f"- **PyTorch:** `{env.get('torch', '2.14.0+cpu')}` (CPU build)")
            st.write(f"- **scikit-learn:** `{env.get('scikit_learn', '1.9.1')}`")
            st.write(f"- **NumPy:** `{env.get('numpy', '2.5.3')}`")
            st.write(f"- **Pandas:** `{env.get('pandas', '3.0.6')}`")
            st.write(f"- **Joblib:** `{env.get('joblib', '1.6.0')}`")
            st.write("- **Streamlit:** `1.65.0`")

    with col_meta2:
        st.markdown("#### Primary Model Architecture & Hyperparameters")
        st.write("- **Selected Input Window ($W$):** 30 seconds")
        st.write("- **Hidden Units:** 64 (1 Layer GRU)")
        st.write("- **Parameter Count:** 32,975 parameters")
        st.write("- **Final Stability Seeds:** `[42, 43, 44]` (Canonical: 42)")
        st.write("- **Ridge Baseline 1:** $\\alpha = 300.0$")
        st.write("- **Feature Count:** 79 standardized telemetry channels")

        st.markdown("#### Frozen Validation Thresholds ($Q_{99.9}$)")
        st.write("- Persistence Baseline 0: `1.680523`")
        st.write("- Ridge Baseline 1 ($\alpha=300$): `1.400928`")
        st.write("- GRU Seed 42: `1.058582`")
        st.write("- GRU Seed 43: `1.195869`")
        st.write("- GRU Seed 44: `1.076832`")

    st.markdown("---")
    st.markdown("#### Checkpoint SHA256 Hashes")
    if pretest_manifest and "frozen_models" in pretest_manifest:
        models_meta = pretest_manifest["frozen_models"]
        st.write("- **StandardScaler:** `cbc43757fb1155a36285b711d9b134d387e20e9775a7ab19c8776749a9000d26`")
        st.write(f"- **Ridge Baseline 1:** `{models_meta['ridge_baseline_1']['model_sha256']}`")
        st.write(f"- **GRU Seed 42 Checkpoint:** `{models_meta['gru_primary']['checkpoints']['seed_42']['sha256']}`")
        st.write(f"- **GRU Seed 43 Checkpoint:** `{models_meta['gru_primary']['checkpoints']['seed_43']['sha256']}`")
        st.write(f"- **GRU Seed 44 Checkpoint:** `{models_meta['gru_primary']['checkpoints']['seed_44']['sha256']}`")

    st.markdown("---")
    st.markdown("#### Verification Audit Log (`final_test_execution.json`)")
    if exec_audit:
        st.json(exec_audit)
