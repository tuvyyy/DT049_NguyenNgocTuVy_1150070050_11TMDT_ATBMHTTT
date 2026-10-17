#!/usr/bin/env python3
"""
DT049 — Multivariate Time-Series Anomaly Detection on HAI 21.03
==============================================================
Official Streamlit Demonstration Dashboard for DT049.
Official benchmark pages display frozen results and post-test analyses.
The GRU workspace executes new jobs in isolated runs/gru folders.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "demo"))

from ui_utils import (
    load_csv,
    load_json,
    display_figure,
    format_delay,
    format_rate,
    check_artifact_health,
    render_download_button,
)
from overview import render_overview
from sidebar import render_sidebar
from dnn_v2 import render_dnn_v2
from gru_presentation import render_gru_presentation

# Configure Streamlit page layout and appearance
st.set_page_config(
    page_title="DT049 — HAI 21.03 Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR NAVIGATION & PRODUCT METADATA
# =============================================================================
page = render_sidebar()

if page not in ("Overview", "DNN Replication V2"):
    st.title("DT049 — Multivariate Time-Series Anomaly Detection on HAI 21.03")
    st.markdown("**Persistence • Ridge • GRU** | *Leakage-aware event detection and detection-delay analysis*")
    col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
    with col_b1:
        st.info("🏷️ **Dataset:** HAI 21.03", icon=None)
    with col_b2:
        st.info("🔒 **Primary Threshold:** $Q_{99.9}$", icon=None)
    with col_b3:
        st.info("🎯 **Attack Events:** 50", icon=None)
    with col_b4:
        st.info("🧠 **Primary Model:** GRU $W=30$", icon=None)
    with col_b5:
        st.success("✅ **Status:** Frozen Final Test", icon=None)
    st.markdown("---")


# =============================================================================
# 1. OVERVIEW
# =============================================================================
if page == "Overview":
    with st.container(key="overview_mode_bar"):
        overview_mode = st.radio(
            "Chế độ xem", ["Thực hiện GRU", "Tổng quan nghiên cứu"],
            horizontal=True, label_visibility="collapsed", key="overview_mode",
        )
    if overview_mode == "Thực hiện GRU":
        render_gru_presentation()
    else:
        render_overview()


# =============================================================================
# 2. MODEL COMPARISON
# =============================================================================
elif page == "Model Comparison":
    st.header("Comparative Model Performance ($Q_{99.9}$ Operating Point)")
    st.markdown("Side-by-side evaluation across all 5 official test sessions (401,855 scored transitions, 109.1411 normal exposure hours).")

    summary_df = load_csv("results/metrics/final_test_summary.csv")

    if summary_df is not None:
        # Interactive metric selector
        st.subheader("Interactive Metric Comparison")
        metric_options = {
            "Event F1": "event_f1",
            "Event Recall": "event_recall",
            "Event Precision": "event_precision",
            "Normal FA events/hour": "normal_fa_events_per_hour",
            "FP alarm seconds/hour": "false_positive_alarm_seconds_per_hour",
            "Median detection delay (s)": "delay_median_sec",
        }
        selected_metric_label = st.selectbox("Select Metric to Visualize Across Models:", list(metric_options.keys()), index=0)
        selected_col = metric_options[selected_metric_label]

        # Compact bar chart
        chart_data = summary_df.set_index("model_name")[selected_col]
        st.bar_chart(chart_data)

        # Full Official Benchmark Table
        st.subheader("Complete Official Benchmark Summary Table")
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
        st.dataframe(
            summary_df[display_cols].rename(columns=rename_map),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.subheader("Benchmark Comparison Visualizations")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        display_figure(
            "results/figures/final_test/fig_01_final_model_comparison.png",
            caption="Figure 1: Benchmark metric summary across Persistence, Ridge, and GRU."
        )
    with col_p2:
        display_figure(
            "results/figures/final_test/fig_02_test_attack_timeline_seed42.png",
            caption="Figure 2: Attack timeline and alarm occurrences for Canonical GRU Seed 42."
        )


# =============================================================================
# 3. ATTACK EXPLORER
# =============================================================================
elif page == "Attack Explorer":
    st.header("Attack Event Catalog & Interactive Case Explorer")
    st.markdown("Inspect all 50 ground-truth physical attack events across test sessions `test1` to `test5`.")

    atk_df = load_csv("results/metrics/final_attack_events.csv")
    missed_df = load_csv("results/metrics/final_missed_attack_analysis.csv")

    if atk_df is not None:
        # Merge missed analysis info if available
        if missed_df is not None:
            merge_cols = ["attack_event_id", "max_gru_score_inside", "frozen_threshold", "score_to_threshold_ratio", "factual_telemetry_observation"]
            merged_atk = pd.merge(atk_df, missed_df[merge_cols], on="attack_event_id", how="left")
        else:
            merged_atk = atk_df.copy()

        # Filter row 1: Test file & Process
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            session_options = ["All"] + sorted(merged_atk["session"].unique().tolist())
            selected_session = st.selectbox("Filter by Test File:", session_options, index=0)
        with col_f2:
            process_options = ["All"] + sorted(merged_atk["process_set"].unique().tolist())
            selected_process = st.selectbox("Filter by Process / Subsystem:", process_options, index=0)
        with col_f3:
            status_options = [
                "All",
                "Detected by GRU s42",
                "Missed by GRU s42",
                "Detected by All Models",
                "Missed by All Models",
                "Detected by Ridge Only",
            ]
            selected_status = st.selectbox("Filter by Detection Status:", status_options, index=0)
        with col_f4:
            search_query = st.text_input("Search Attack ID (e.g. atk_02):", "").strip()

        # Apply filters
        filtered_df = merged_atk.copy()
        if selected_session != "All":
            filtered_df = filtered_df[filtered_df["session"] == selected_session]
        if selected_process != "All":
            filtered_df = filtered_df[filtered_df["process_set"] == selected_process]

        if selected_status == "Detected by GRU s42":
            filtered_df = filtered_df[filtered_df["gru_seed42_detected"] == True]
        elif selected_status == "Missed by GRU s42":
            filtered_df = filtered_df[filtered_df["gru_seed42_detected"] == False]
        elif selected_status == "Detected by All Models":
            filtered_df = filtered_df[(filtered_df["persistence_detected"] == True) & (filtered_df["ridge_detected"] == True) & (filtered_df["gru_seed42_detected"] == True)]
        elif selected_status == "Missed by All Models":
            filtered_df = filtered_df[(filtered_df["persistence_detected"] == False) & (filtered_df["ridge_detected"] == False) & (filtered_df["gru_seed42_detected"] == False)]
        elif selected_status == "Detected by Ridge Only":
            filtered_df = filtered_df[(filtered_df["ridge_detected"] == True) & (filtered_df["gru_seed42_detected"] == False)]

        if search_query:
            filtered_df = filtered_df[filtered_df["attack_event_id"].str.contains(search_query, case=False, na=False)]

        st.caption(f"Showing **{len(filtered_df)}** of **{len(merged_atk)}** total attack events matching active filters.")

        # Display table of filtered attacks
        preview_cols = [
            "attack_event_id", "session", "process_set", "duration_seconds",
            "persistence_detected", "ridge_detected", "gru_seed42_detected",
            "gru_seed42_delay_seconds"
        ]
        rename_preview = {
            "attack_event_id": "Attack ID",
            "session": "File",
            "process_set": "Process",
            "duration_seconds": "Duration (s)",
            "persistence_detected": "Persistence",
            "ridge_detected": "Ridge",
            "gru_seed42_detected": "GRU s42",
            "gru_seed42_delay_seconds": "GRU s42 Delay (s)",
        }
        st.dataframe(
            filtered_df[preview_cols].rename(columns=rename_preview),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Detail Inspection Panel
        st.subheader("Attack Event Detail Panel")
        if not filtered_df.empty:
            chosen_atk_id = st.selectbox(
                "Select Attack Event to Inspect in Detail:",
                filtered_df["attack_event_id"].tolist(),
                index=0
            )
            atk_row = merged_atk[merged_atk["attack_event_id"] == chosen_atk_id].iloc[0]

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.write(f"**Attack ID:** `{atk_row['attack_event_id']}`")
                st.write(f"**Source File:** `{atk_row['session']}`")
                st.write(f"**Subsystem:** `{atk_row['process_set']}`")
            with d2:
                st.write(f"**Duration:** `{atk_row['duration_seconds']} s`")
                st.write(f"**Effective Start:** `{atk_row['effective_start']}`")
                st.write(f"**Original End:** `{atk_row['orig_end']}`")
            with d3:
                p_status = "✅ Detected" if atk_row["persistence_detected"] else "❌ Missed"
                r_status = "✅ Detected" if atk_row["ridge_detected"] else "❌ Missed"
                g_status = "✅ Detected" if atk_row["gru_seed42_detected"] else "❌ Missed"
                st.write(f"**Persistence:** {p_status} ({format_delay(atk_row['persistence_delay_seconds'])})")
                st.write(f"**Ridge:** {r_status} ({format_delay(atk_row['ridge_delay_seconds'])})")
                st.write(f"**GRU s42:** {g_status} ({format_delay(atk_row['gru_seed42_delay_seconds'])})")
            with d4:
                if pd.notna(atk_row.get("score_to_threshold_ratio")):
                    st.write(f"**Max Score (GRU):** `{atk_row['max_gru_score_inside']:.4f}`")
                    st.write(f"**Threshold ($Q_{{99.9}}$):** `{atk_row['frozen_threshold']:.4f}`")
                    st.write(f"**Ratio:** `{atk_row['score_to_threshold_ratio']:.3f}`")
                else:
                    st.write(f"**Status:** Detected by Canonical GRU")
                    st.write(f"**Delay:** `{format_delay(atk_row['gru_seed42_delay_seconds'])}`")

            # Dedicated Diagnostic Figures for the 4 Representative Case Studies
            case_figures = {
                "test1_atk_04": ("results/figures/error_analysis/fig_ea_06_case_all_detected.png", "Case 1: Detected by All Models (test1_atk_04)"),
                "test1_atk_02": ("results/figures/error_analysis/fig_ea_07_case_gru_only.png", "Case 2: GRU-Only Detection (test1_atk_02)"),
                "test2_atk_06": ("results/figures/error_analysis/fig_ea_08_case_all_missed.png", "Case 3: Missed by All Models (test2_atk_06)"),
                "test2_atk_05": ("results/figures/error_analysis/fig_ea_09_case_gru_miss_ridge_detected.png", "Case 4: GRU Missed, Ridge Detected (test2_atk_05)"),
            }
            if chosen_atk_id in case_figures:
                fig_path, fig_cap = case_figures[chosen_atk_id]
                st.markdown("#### Dedicated Case Study Diagnostic Figure")
                display_figure(fig_path, caption=fig_cap)
        else:
            st.warning("No attacks match the selected filters.")


# =============================================================================
# 4. ERROR ANALYSIS
# =============================================================================
elif page == "Error Analysis":
    st.header("Post-Test Error Analysis & Operational Characterization")
    st.markdown("In-depth diagnostic breakdown of process attribution, common-detected latency, missed attacks, and false-alarm taxonomy.")

    # 4A. Process-Level Detection
    st.subheader("A. Process-Level Detection Breakdown")
    proc_df = load_csv("results/metrics/final_error_analysis_by_process.csv")
    if proc_df is not None:
        col_pr1, col_pr2 = st.columns([1.4, 1.0])
        with col_pr1:
            st.dataframe(proc_df, use_container_width=True, hide_index=True)
        with col_pr2:
            display_figure(
                "results/figures/error_analysis/fig_ea_01_process_recall.png",
                caption="Figure EA-1: Event recall by process subsystem (P1 Boiler, P2 Turbine, P3 Water Treatment, Multi-process)."
            )

    st.markdown("---")

    # 4B. Paired Detection Delay Comparison
    st.subheader("B. Paired Detection Delay Comparison (Selection Bias Correction)")
    st.markdown(
        "*Note:* Unconditional median delays (Ridge 13.0s vs GRU 22.0s) suffered from selection bias because Ridge only detected 35 easier, "
        "faster-onset attacks. Comparing models strictly on attacks detected by **both** provides an unbiased evaluation:"
    )
    paired_df = load_csv("results/metrics/final_common_detected_delay.csv")
    if paired_df is not None:
        col_pd1, col_pd2 = st.columns([1.4, 1.0])
        with col_pd1:
            st.dataframe(paired_df, use_container_width=True, hide_index=True)
        with col_pd2:
            display_figure(
                "results/figures/error_analysis/fig_ea_02_common_delay_paired.png",
                caption="Figure EA-2: Paired delay distributions on commonly detected attack sets."
            )

    st.markdown("---")

    # 4C. Missed Attack Diagnostics (Requirement 7)
    st.subheader("C. Missed Attack Diagnostics (GRU Seed 42)")
    st.markdown("Factual telemetry and residual characteristics of the 10 attacks missed by canonical GRU seed 42:")
    missed_df = load_csv("results/metrics/final_missed_attack_analysis.csv")
    if missed_df is not None:
        disp_miss_cols = [
            "attack_event_id", "session", "process_set", "duration_seconds",
            "max_gru_score_inside", "score_to_threshold_ratio",
            "ridge_detected", "persistence_detected"
        ]
        rename_miss = {
            "attack_event_id": "Attack ID",
            "session": "File",
            "process_set": "Process",
            "duration_seconds": "Duration (s)",
            "max_gru_score_inside": "Peak Score",
            "score_to_threshold_ratio": "Threshold Ratio",
            "ridge_detected": "Ridge Detected",
            "persistence_detected": "Persistence Detected",
        }
        st.dataframe(missed_df[disp_miss_cols].rename(columns=rename_miss), use_container_width=True, hide_index=True)

        st.info(
            "📌 **Factual Near-Miss Observations (Score/Threshold close to 1.0):**\n"
            "- **`test2_atk_09`**: Peak GRU score reached `1.0481` (**99.0%** of threshold `1.0586`), falling short by only `0.0105` RMSE.\n"
            "- **`test5_atk_09`**: Peak GRU score reached `1.0193` (**96.3%** of threshold `1.0586`).\n"
            "- **`test3_atk_01`**: Peak GRU score reached `0.9772` (**92.3%** of threshold `1.0586`), situated between normal false alarms.\n"
            "*Observation: No physical causes are inferred solely from telemetry labels.*"
        )

    st.markdown("---")

    # 4D. False Alarm View (Requirement 8)
    st.subheader("D. Operational False-Alarm Characterization")
    st.markdown(
        """
        Industrial evaluation must distinguish four distinct dimensions of false alarms:
        1. **Event-Level FP:** Total discrete predicted alarm events that do not match a new attack (sum of redundant in-attack alarms and normal false alarms).
        2. **Normal False-Alarm Events/Hour:** Spurious discrete alarm events during attack-free operation that interrupt human operators.
        3. **FP Alarm Seconds/Hour:** Total cumulative duration the system was actively alarming during normal exposure.
        4. **Point FPR:** Proportion of normal seconds misclassified as anomalous ($t_{\\text{alarm}} / t_{\\text{normal}}$).
        """
    )
    st.warning(
        "⚖️ **Operational Tradeoff Finding:**\n\n"
        "**GRU has fewer discrete normal false-alarm events** ($2.61\\text{ ev/h}$ vs $3.70\\text{ ev/h}$ Persistence and $4.48\\text{ ev/h}$ Ridge), "
        "**while Persistence has lower false-positive alarm time occupancy** ($3.92\\text{ s/h}$ vs $5.58\\text{ s/h}$ GRU and $6.52\\text{ s/h}$ Ridge).\n\n"
        "*Explanation:* Persistence residuals drop back to zero in 1–2 seconds after an impulse, while GRU multi-step hidden state ring down over several seconds."
    )
    col_fa1, col_fa2 = st.columns(2)
    with col_fa1:
        display_figure(
            "results/figures/error_analysis/fig_ea_03_event_chatter_breakdown.png",
            caption="Figure EA-3: Event chatter and redundant in-attack alarm breakdown across detectors."
        )
    with col_fa2:
        display_figure(
            "results/figures/error_analysis/fig_ea_04_false_alarm_tradeoff.png",
            caption="Figure EA-4: False-alarm event rate vs temporal alarm occupancy tradeoff."
        )


# =============================================================================
# 5. THRESHOLD SENSITIVITY
# =============================================================================
elif page == "Threshold Sensitivity":
    st.header("Predefined Threshold Sensitivity Analysis")
    st.markdown("Evaluation of detectors across strictly predetermined quantiles ($Q_{99.5}$, $Q_{99.9}$, $Q_{99.99}$).")

    st.warning(
        "⚠️ **Operating Point Designation:**\n"
        "- **$Q_{99.9}$** = **FROZEN PRIMARY OPERATING POINT** (Calibrated strictly on normal validation data `train3`).\n"
        "- **$Q_{99.5}$ and $Q_{99.99}$** = Predefined sensitivity bounds only.\n\n"
        "**Sensitivity operating points are not replacements for the frozen $Q_{99.9}$ primary result**, "
        "even though $Q_{99.99}$ yields higher Event F1 ($0.5143$) on test data."
    )

    sens_df = load_csv("results/metrics/final_test_sensitivity_complete.csv")

    if sens_df is not None:
        # Restrict strictly to predefined frozen quantiles (never allow Q99.0)
        valid_points = ["Q99.5", "Q99.9", "Q99.99"]
        sens_df = sens_df[sens_df["operating_point"].isin(valid_points)].copy()

        q_options = ["Q99.5 (Sensitivity Lower)", "Q99.9 — FROZEN PRIMARY", "Q99.99 (Sensitivity Upper)"]
        q_choice = st.radio("Select Predefined Operating Point to Inspect:", q_options, index=1)
        selected_q = "Q99.9" if "Q99.9" in q_choice else ("Q99.5" if "Q99.5" in q_choice else "Q99.99")

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
    st.subheader("Sensitivity Tradeoff Curves")
    display_figure(
        "results/figures/error_analysis/fig_ea_05_sensitivity_curves.png",
        caption="Figure EA-5: Tradeoff curves across predetermined quantiles Q99.5, Q99.9 (frozen primary), and Q99.99 (Recall vs FA rate, F1 vs FA rate, Recall vs FP Time)."
    )


# =============================================================================
# 6. DYNAMIC57 ABLATION
# =============================================================================
elif page == "Dynamic57 Ablation":
    st.header("Train-Dynamic Feature Score Ablation")
    st.markdown("Post-hoc exploratory variant testing the contribution of the 22 training-constant channels.")

    col_ab1, col_ab2, col_ab3 = st.columns(3)
    with col_ab1:
        st.metric("Primary Score Features", "79", help="All standardized candidate features")
    with col_ab2:
        st.metric("Post-Hoc Ablation Features", "57", help="Channels with variance > 0 during training")
    with col_ab3:
        st.metric("Excluded Features", "22", help="Channels with zero variance during training")

    st.info(
        "📌 **Ablation Finding & Official Conclusion:**\n\n"
        "**No Q99.9 attack decision or detection delay changed in this experiment.** "
        "The 22 training-constant channels did not alter detector decisions under this post-hoc Q99.9 ablation on HAI 21.03. "
        "*(Note: These 22 channels are not termed 'useless' or 'redundant'; some become active during physical testbed disruptions, "
        "but their residual magnitude was not large enough to change alarm thresholds or decisions.)*"
    )

    st.markdown(
        r"""
        #### Mathematical Derivation of Score Scaling
        - **Scale Factor Definition:**
          $$\kappa = \sqrt{\frac{79}{57}} \approx 1.177270$$
          *(Dimension ratio: $\frac{79}{57} \approx 1.385965 \implies \text{scale factor } \kappa = \sqrt{\frac{79}{57}} \approx 1.177270$)*
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
        st.subheader("Primary (79 Features) vs. Ablation (57 Dynamic Features) at $Q_{99.9}$")
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
        st.subheader("GRU Multi-Seed Aggregate (79 vs. 57)")
        q_ms = ab_ms_df[ab_ms_df["operating_point"] == "Q99.9"].copy()
        st.dataframe(q_ms, use_container_width=True, hide_index=True)

    if ab_ch_df is not None:
        changed_count = ab_ch_df["changed_status_any_model"].sum()
        st.metric("Total Attacks with Changed Detection Status", f"{changed_count} / 50", delta="0.0% Change")

    st.markdown("---")
    st.subheader("Ablation Diagnostic Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        display_figure(
            "results/figures/ablation_dynamic57/fig_ab_01_primary_vs_dynamic57.png",
            caption=r"Figure AB-1: Recall, F1, and False-Alarm rate comparison between 79-feature primary and 57-feature dynamic ablation ($\kappa = \sqrt{79/57} \approx 1.177270$)."
        )
        display_figure(
            "results/figures/ablation_dynamic57/fig_ab_03_false_alarm_tradeoff.png",
            caption="Figure AB-3: False-alarm event rate vs temporal occupancy for 79 vs 57 features."
        )
    with c2:
        display_figure(
            "results/figures/ablation_dynamic57/fig_ab_02_detection_changes.png",
            caption="Figure AB-2: Binary detection status grid showing identical detection across all 50 attacks."
        )


# =============================================================================
# 7. REPRODUCIBILITY & INTEGRITY
# =============================================================================
elif page == "Reproducibility":
    st.header("Reproducibility, Artifact Health & Export")
    st.markdown("Audit trail, pipeline integrity health check, and official result artifact downloads.")

    # 7A. Product Health Panel (Requirement 11)
    st.subheader("A. Product Health Panel")
    health = check_artifact_health()
    if health["is_healthy"]:
        st.success(f"✅ **Artifact Health: {health['available']} / {health['total']} available** (All frozen models and metric tables verified)")
    else:
        st.warning(f"⚠️ **Artifact Health: {health['available']} / {health['total']} available** (Some required files are missing)")

    health_table = pd.DataFrame(health["details"])
    health_table["status"] = health_table["exists"].apply(lambda x: "Available" if x else "Missing")
    st.dataframe(
        health_table[["name", "path", "status", "size_bytes"]].rename(
            columns={"name": "Artifact", "path": "Relative Path", "status": "Status", "size_bytes": "File Size (Bytes)"}
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # 7B. Git Identifiers & Audit Records
    st.subheader("B. Frozen Audit Checkpoints")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Git Commit Checkpoints")
        st.write("- **Upstream HAI Commit:** `2a814cebc9a66b06c9e5cd545e2d72e65d383737`")
        st.write("- **Development Freeze:** `6ce11fb8bcb682c10e125219d0b964f394b8a5d2`")
        st.write("- **Protocol Freeze:** `6a0613b87a0a9d89c89b6ae848350b44c1aa1740`")
        st.write("- **Official Final-Test Tag:** `dt049-final-test-v1`")
        st.write("- **Final-Test Execution Commit:** `331cc07cb72b8b3f30a64bbe7af52248bd0cd9ba`")
        st.write("- **Analysis Commit:** `4104576d1c4ad69edd87cc97ea6463974becf460`")
        st.write("- **UI Commit:** `a9f03c940b066bd3bea46007a21f372915f45db9`")
    with col_g2:
        st.markdown("#### Frozen Model Architecture & Hyperparameters")
        st.write("- **Input Features:** 79 continuous telemetry channels")
        st.write("- **GRU Lookback Window:** $W = 30\\text{ seconds}$")
        st.write("- **GRU Hidden Units:** 64 (1 recurrent layer, 32,975 parameters)")
        st.write("- **Ridge Baseline 1:** $\\alpha = 300.0$")
        st.write("- **Validation Normal Hours:** 133.00 hours (`train3.csv.gz`)")
        st.write("- **Official Test Hours:** 111.67 hours (`test1`–`test5`)")

    st.markdown("---")

    # 7C. Download / Export Functions (Requirement 12)
    st.subheader("C. Result Artifact Downloads")
    st.markdown(
        "Download official frozen metric tables for independent verification. "
        "*(Note: Raw HAI telemetry archives and model checkpoint binaries are excluded from dashboard export.)*"
    )

    down_col1, down_col2, down_col3 = st.columns(3)
    with down_col1:
        render_download_button(
            "results/metrics/final_test_summary.csv",
            "📥 Download Final Test Summary CSV",
            "final_test_summary.csv"
        )
        render_download_button(
            "results/metrics/final_per_file.csv",
            "📥 Download Per-File Metrics CSV",
            "final_per_file.csv"
        )
    with down_col2:
        render_download_button(
            "results/metrics/final_attack_events.csv",
            "📥 Download Attack Events CSV",
            "final_attack_events.csv"
        )
        render_download_button(
            "results/metrics/final_missed_attack_analysis.csv",
            "📥 Download Missed Attack Analysis CSV",
            "final_missed_attack_analysis.csv"
        )
    with down_col3:
        render_download_button(
            "results/metrics/final_test_sensitivity_complete.csv",
            "📥 Download Complete Sensitivity CSV",
            "final_test_sensitivity_complete.csv"
        )
        render_download_button(
            "results/metrics/ablation_dynamic57_summary.csv",
            "📥 Download Dynamic57 Summary CSV",
            "ablation_dynamic57_summary.csv"
        )


# =============================================================================
# 8. DNN REPLICATION V2 (Kuchar & Fujdiak 2025 Adapted Reproduction)
# =============================================================================
elif page == "DNN Replication V2":
    render_dnn_v2()
