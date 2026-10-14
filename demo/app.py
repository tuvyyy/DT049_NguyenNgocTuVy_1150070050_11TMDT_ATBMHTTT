#!/usr/bin/env python3
"""
DT049 — Multivariate Time-Series Anomaly Detection on HAI 21.03
==============================================================
Official Streamlit Demonstration Dashboard for DT049.
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

if page != "Overview":
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
    st.header("DNN Replication V2 — Tái lập Có Điều chỉnh Kuchař & Fujdiak (2025) trên HAI 21.03")
    st.caption(
        "Thí nghiệm V2 (`DT049_V2_SUPERVISED_DNN`): Phân loại nhị phân Normal/Attack có giám sát bằng mạng Feedforward DNN 6 tầng ẩn "
        "và đối chiếu với biến thể dự báo chuỗi thời gian không giám sát GRU V1 trên tập Held-out `test4 + test5` (17 sự kiện, t >= 30)."
    )

    # Load frozen V2 artifacts safely (read-only from disk, zero model inference)
    v2_summary_df = load_csv("results/v2_dnn/v2_heldout_test4_test5_summary.csv")
    v2_multiseed_df = load_csv("results/v2_dnn/v2_heldout_multiseed_stats.csv")
    v2_events_df = load_csv("results/v2_dnn/v2_heldout_17_attack_events.csv")
    v2_per_file_df = load_csv("results/v2_dnn/v2_heldout_per_file.csv")
    v2_cm_df = load_csv("results/v2_dnn/v2_heldout_confusion_matrices.csv")
    v2_subsys_df = load_csv("results/v2_dnn/v2_heldout_subsystem_breakdown.csv")
    v2_val_df = load_csv("results/v2_dnn/v2_dnn_validation_summary.csv")
    v2_freeze_json = load_json("results/v2_dnn/v2_pre_evaluation_freeze_manifest.json")
    v2_audit_json = load_json("results/v2_dnn/v2_heldout_evaluation_audit.json")

    missing_v2 = [
        name for name, obj in [
            ("v2_heldout_test4_test5_summary.csv", v2_summary_df),
            ("v2_heldout_multiseed_stats.csv", v2_multiseed_df),
            ("v2_heldout_17_attack_events.csv", v2_events_df),
            ("v2_heldout_per_file.csv", v2_per_file_df),
            ("v2_heldout_confusion_matrices.csv", v2_cm_df),
            ("v2_pre_evaluation_freeze_manifest.json", v2_freeze_json),
            ("v2_heldout_evaluation_audit.json", v2_audit_json),
        ]
        if obj is None
    ]
    if missing_v2:
        st.warning(
            f"⚠️ **Cảnh báo thiếu tệp kết quả V2:** Các tệp sau chưa sẵn sàng trong `results/v2_dnn/`: "
            f"`{', '.join(missing_v2)}`. Giao diện vẫn hoạt động an toàn ở chế độ hiển thị phần dữ liệu hiện có."
        )

    # -------------------------------------------------------------------------
    # SECTION 1: RESEARCH BACKGROUND
    # -------------------------------------------------------------------------
    st.subheader("1. RESEARCH BACKGROUND — Bối cảnh Nghiên cứu & Phạm vi Tái lập Có Điều chỉnh")

    arch_info = v2_freeze_json.get("architecture_contract", {}) if v2_freeze_json else {}
    hidden_layers_str = str(arch_info.get("hidden_layers", [100, 70, 100, 80, 60, 10]))
    hidden_acts_str = " → ".join(arch_info.get("hidden_activations", ["relu", "tanh", "relu", "tanh", "tanh", "tanh"]))
    param_count = int(arch_info.get("trainable_parameters", 35731))

    col_bg1, col_bg2 = st.columns([1.25, 1.0])
    with col_bg1:
        st.markdown(
            f"""
            - **Bài báo tham chiếu gốc:** Kuchař & Fujdiak (2025), *"Analyzing anomalies in industrial networks: A data-driven approach to enhance security in manufacturing processes"*, *Computers & Security*, DOI `10.1016/j.cose.2025.104395`.
            - **Bài toán gốc của bài báo:** Sử dụng mạng truyền thẳng (Feedforward DNN) để phân loại **4 trạng thái**: `Normal`, `Security`, `Operational`, `Service`.
            - **Đặc thù bộ dữ liệu HAI 21.03:** Không có 4 trạng thái tương đương với bộ dữ liệu nội bộ của bài báo; HAI 21.03 chỉ cung cấp nhãn nhị phân `attack` (`0 = Normal`, `1 = Attack`) và các nhãn phân hệ vật lý `attack_P1`, `attack_P2`, `attack_P3`.
            - **Kiến trúc DNN V2 đã xác minh theo Listing 1:** Giữ nguyên **6 tầng ẩn Dense `{hidden_layers_str}`**, chuỗi kích hoạt **`{hidden_acts_str}`**, và **2 lớp `Dropout(0.3)`** đặt sau `Dense(60)` và `Dense(10)` (tổng cộng **`{param_count:,}` tham số**).
            - **Điều chỉnh sang HAI 21.03:** Đầu vào gồm **79 đặc trưng** cảm biến/cơ cấu chấp hành (sử dụng bộ chuẩn hóa `StandardScaler` đã đóng băng từ V1) và đầu ra nhị phân **1 logit (`Normal` / `Attack`)** kết hợp `BCEWithLogitsLoss` / `Sigmoid`.
            - **Vai trò của GRU V1:** Mô hình **GRU $W=30$ (`32,975` tham số)** được giữ nguyên làm **biến thể dự báo chuỗi thời gian không giám sát** (Unsupervised Time-Series Forecasting Variant).
            """
        )
    with col_bg2:
        st.info(
            "📌 **Tuyên bố Minh bạch Học thuật (Adaptation Disclaimer):**\n\n"
            "- **DNN V2 là bản tái lập kiến trúc có điều chỉnh sang HAI 21.03**, **KHÔNG tuyên bố tái lập nguyên bản 100% bài báo gốc** "
            "và **KHÔNG tuyên bố đạt độ chính xác 4 lớp `99,72%`** của bài báo (do khác tập dữ liệu, khác hệ thống nhãn, khác số chiều đầu vào `79 đặc trưng HAI` so với sự không nhất quán giữa 87 đặc trưng mô tả trong bài báo và `input_dim=10` trong Listing 1, "
            "và chưa có định nghĩa chi tiết của `norm_layer` trong Listing 1).\n"
            "- Mục tiêu của trang này là giúp sinh viên trình bày rõ ràng sự khác biệt giữa **Phân loại mẫu đơn có giám sát (DNN V2)** "
            "và **Dự báo chuỗi thời gian không giám sát (GRU V1)** trên cùng một tập kiểm tra giữ ngoài."
        )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 2: DATA SPLIT
    # -------------------------------------------------------------------------
    st.subheader("2. DATA SPLIT — Giao thức Phân chia Dữ liệu Thí nghiệm V2")

    v2_pipe_json = load_json("results/v2_dnn/v2_data_pipeline_audit.json")
    splits_cfg = v2_freeze_json.get("splits", {}) if v2_freeze_json else {}
    scaler_fit_files = " + ".join(splits_cfg.get("scaler_fit_only", ["train1.csv.gz", "train2.csv.gz"]))
    train_files = " + ".join(splits_cfg.get("dnn_train", ["test1", "test2"]))
    val_files = " + ".join(splits_cfg.get("dnn_validation", ["test3"]))
    eval_files = " + ".join(splits_cfg.get("dnn_heldout_evaluation_v2", ["test4", "test5"]))

    scaler_n = int(v2_pipe_json["scaler_verification"]["fit_sample_count"]) if v2_pipe_json else 442802
    tr_n = int(v2_pipe_json["splits"]["dnn_train"]["total_rows"]) if v2_pipe_json else 162002
    tr_atk_s = int(v2_pipe_json["splits"]["dnn_train"]["attack_rows"]) if v2_pipe_json else 4078
    val_n = int(v2_pipe_json["splits"]["dnn_validation"]["total_rows"]) if v2_pipe_json else 108001
    val_atk_s = int(v2_pipe_json["splits"]["dnn_validation"]["attack_rows"]) if v2_pipe_json else 1535
    ev_n = int(v2_summary_df["scored_transitions"].iloc[0]) if v2_summary_df is not None else 131942
    ev_norm_h = float(v2_summary_df["normal_exposure_hours"].iloc[0]) if v2_summary_df is not None else 35.7244
    ev_atk_cnt = int(v2_summary_df["evaluable_attacks"].iloc[0]) if v2_summary_df is not None else 17

    sp1, sp2, sp3, sp4 = st.columns(4)
    with sp1:
        st.metric("1. Scaler Fit Only", "train1 + train2", help=f"{scaler_n:,} bản ghi normal — chỉ dùng scaler.joblib đã đóng băng, không fit lại")
    with sp2:
        st.metric("2. DNN Supervised Train", train_files, help=f"{tr_n:,} mẫu ({tr_atk_s:,}s Attack) — tối ưu trọng số DNN & tính pos_weight")
    with sp3:
        st.metric("3. DNN Validation", val_files, help=f"{val_n:,} mẫu ({val_atk_s:,}s Attack) — early stopping & hiệu chuẩn ngưỡng τ*")
    with sp4:
        st.metric("4. DNN Held-out Eval", eval_files, help=f"{ev_n:,} mẫu tại t >= 30 ({ev_atk_cnt} sự kiện tấn công, {ev_norm_h:.4f} giờ normal)")

    split_table_data = [
        {
            "Vai trò trong Giao thức V2": "Fit bộ chuẩn hóa (Frozen Scaler)",
            "Các phiên dữ liệu (Files)": scaler_fit_files,
            "Số dòng / Mẫu chấm điểm": f"{scaler_n:,} (100% Normal)",
            "Số sự kiện tấn công": 0,
            "Quy tắc chống rò rỉ (Leakage Control)": "Chỉ dùng lại scaler.joblib từ V1; không tham gia huấn luyện DNN V2.",
        },
        {
            "Vai trò trong Giao thức V2": "Huấn luyện có giám sát (DNN Supervised Train)",
            "Các phiên dữ liệu (Files)": train_files,
            "Số dòng / Mẫu chấm điểm": f"{tr_n:,} ({tr_atk_s:,} giây Attack)",
            "Số sự kiện tấn công": 25,
            "Quy tắc chống rò rỉ (Leakage Control)": "Tối ưu trọng số mạng DNN và tính pos_weight = 6.223009.",
        },
        {
            "Vai trò trong Giao thức V2": "Kiểm định & Chọn ngưỡng (DNN Validation)",
            "Các phiên dữ liệu (Files)": val_files,
            "Số dòng / Mẫu chấm điểm": f"{val_n:,} ({val_atk_s:,} giây Attack)",
            "Số sự kiện tấn công": 8,
            "Quy tắc chống rò rỉ (Leakage Control)": "Chọn best epoch theo val_pr_auc và hiệu chuẩn ngưỡng đóng băng τ*.",
        },
        {
            "Vai trò trong Giao thức V2": "Đánh giá giữ ngoài (DNN Held-out Evaluation)",
            "Các phiên dữ liệu (Files)": f"{eval_files} (t >= 30)",
            "Số dòng / Mẫu chấm điểm": f"{ev_n:,} ({ev_norm_h:.4f}h Normal)",
            "Số sự kiện tấn công": ev_atk_cnt,
            "Quy tắc chống rò rỉ (Leakage Control)": "Tuyệt đối không dùng để fit scaler, chọn mô hình hay tinh chỉnh ngưỡng τ*.",
        },
    ]
    st.dataframe(pd.DataFrame(split_table_data), use_container_width=True, hide_index=True)

    st.warning(
        "⚠️ **Ghi chú Minh bạch về Tính Độc lập của Tập Đánh giá (`test4 + test5`):**\n"
        "`test4` và `test5` hoàn toàn không tham gia vào quá trình huấn luyện, chọn epoch hay hiệu chuẩn ngưỡng của **DNN V2**. "
        "Tuy nhiên, vì hai phiên này đã từng được đánh giá và phân tích lỗi trong báo cáo **V1**, chúng **không phải là tập dữ liệu mù hoàn toàn (not completely blind) ở cấp độ toàn bộ dự án DT049**."
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 3: CANONICAL PERFORMANCE (SEED 42 ON TEST4 + TEST5)
    # -------------------------------------------------------------------------
    st.subheader("3. CANONICAL PERFORMANCE — Hiệu năng Chính tắc (Seed 42 trên `test4 + test5`, $t \\ge 30$)")

    if v2_summary_df is not None:
        dnn42_rows = v2_summary_df[v2_summary_df["model_id"] == "dnn_v2_seed42_tau_star"]
        gru42_rows = v2_summary_df[v2_summary_df["model_id"] == "gru_seed42"]

        if not dnn42_rows.empty and not gru42_rows.empty:
            d42 = dnn42_rows.iloc[0]
            g42 = gru42_rows.iloc[0]

            st.markdown(
                f"**A. Mô hình Tái lập Có Điều chỉnh: `Feedforward DNN V2 — Seed 42 (Canonical)`** "
                f"*(Ngưỡng đóng băng $\\tau^* = {float(d42['threshold']):.6f}$ chọn trên `test3`)*"
            )
            c_k1, c_k2, c_k3, c_k4, c_k5, c_k6 = st.columns(6)
            with c_k1:
                st.metric("Attack Events", f"{int(d42['evaluable_attacks'])}")
            with c_k2:
                st.metric("Detected", f"{int(d42['tp_events'])}/{int(d42['evaluable_attacks'])}")
            with c_k3:
                st.metric("Event Recall", f"{float(d42['event_recall']) * 100:.0f}%")
            with c_k4:
                st.metric("Event F1", f"{float(d42['event_f1']):.6f}")
            with c_k5:
                st.metric("Normal FA Events/hour", f"{float(d42['normal_fa_events_per_hour']):.4f}")
            with c_k6:
                st.metric("Median Detection Delay", f"{float(d42['median_delay_sec']):.1f} seconds")

            st.markdown(
                f"**B. Mô hình Biến thể Chuỗi thời gian: `Primary GRU W=30 — Seed 42 (V1, Canonical)` trên chính `test4 + test5` ($t \\ge 30$)** "
                f"*(Ngưỡng đóng băng $Q_{{99.9}} = {float(g42['threshold']):.6f}$ chọn trên `train3`)*"
            )
            g_k1, g_k2, g_k3, g_k4 = st.columns(4)
            with g_k1:
                st.metric(
                    "GRU Seed 42 Detected",
                    f"{int(g42['tp_events'])}/{int(g42['evaluable_attacks'])}",
                    delta=f"{float(g42['event_recall']) * 100:.2f}% Recall",
                )
            with g_k2:
                st.metric("GRU Seed 42 Event F1", f"{float(g42['event_f1']):.6f}")
            with g_k3:
                st.metric("GRU Seed 42 Normal FA/hour", f"{float(g42['normal_fa_events_per_hour']):.4f}")
            with g_k4:
                st.metric("GRU Seed 42 Median Delay", f"{float(g42['median_delay_sec']):.1f} seconds")
        else:
            st.warning("Không tìm thấy dòng dữ liệu `dnn_v2_seed42_tau_star` hoặc `gru_seed42` trong bảng tổng hợp.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 4: MODEL COMPARISON — COMMON 17 ATTACKS
    # -------------------------------------------------------------------------
    st.subheader("4. MODEL COMPARISON — COMMON 17 ATTACKS (`test4 + test5`, $t \\ge 30$)")
    st.markdown(
        "Bảng đối chiếu trực tiếp giữa các mô hình **V1 (`Persistence`, `Ridge`, `GRU seeds 42/43/44`)** và "
        "**V2 (`DNN seeds 42/43/44` tại ngưỡng đóng băng $\\tau^*$)** trên đúng cùng tập `test4 + test5` ($131.942$ mẫu tại $t \\ge 30$, $17$ sự kiện tấn công). "
        "**Tuyệt đối không trộn lẫn với bảng 50 sự kiện của toàn bộ `test1–test5` trong V1.**"
    )

    if v2_summary_df is not None:
        primary_model_ids = [
            "persistence",
            "ridge",
            "gru_seed42",
            "gru_seed43",
            "gru_seed44",
            "dnn_v2_seed42_tau_star",
            "dnn_v2_seed43_tau_star",
            "dnn_v2_seed44_tau_star",
        ]
        comp_df = v2_summary_df[v2_summary_df["model_id"].isin(primary_model_ids)].copy()
        comp_df["event_tp_17"] = comp_df.apply(
            lambda r: f"{int(r['tp_events'])}/{int(r['evaluable_attacks'])}", axis=1
        )

        comp_cols = [
            "model_name",
            "paradigm",
            "threshold",
            "event_tp_17",
            "event_precision",
            "event_recall",
            "event_f1",
            "normal_fa_events_per_hour",
            "median_delay_sec",
            "point_f1",
            "point_pr_auc",
        ]
        comp_rename = {
            "model_name": "Model",
            "paradigm": "Supervision Regime",
            "threshold": "Threshold",
            "event_tp_17": "Event TP/17",
            "event_precision": "Event Precision",
            "event_recall": "Event Recall",
            "event_f1": "Event F1",
            "normal_fa_events_per_hour": "Normal FA/h",
            "median_delay_sec": "Median Delay (s)",
            "point_f1": "Point F1",
            "point_pr_auc": "Point PR-AUC",
        }
        st.dataframe(
            comp_df[comp_cols].rename(columns=comp_rename),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "⚖️ **Chú thích Bắt buộc về Chế độ Giám sát (Supervision Regime Disclaimer):**\n"
            "- **DNN V2 (Có giám sát):** Huấn luyện trực tiếp với nhãn tấn công `attack = 0/1` trên `test1 + test2` (25 cuộc tấn công) và hiệu chuẩn ngưỡng $\\tau^*$ có giám sát trên `test3` (8 cuộc tấn công).\n"
            "- **Persistence, Ridge, GRU V1 (Không giám sát):** Chỉ huấn luyện trên dữ liệu vận hành bình thường `train1 + train2` (0 cuộc tấn công) và chọn ngưỡng phân vị $Q_{99.9}$ hoàn toàn không dùng nhãn tấn công trên `train3`.\n"
            "- Do đó, sự khác biệt kết quả phản ánh **sự đánh đổi giữa hai mô thức học và chế độ giám sát nhãn**, không phải phép so sánh cô lập riêng kiến trúc mạng."
        )

        if v2_per_file_df is not None:
            with st.expander("📂 Xem bảng kết quả tách riêng từng phiên (`test4` gồm 5 sự kiện vs `test5` gồm 12 sự kiện)"):
                pf_sub = v2_per_file_df[v2_per_file_df["model_id"].isin(primary_model_ids)].copy()
                pf_cols = [
                    "session", "model_name", "threshold", "tp_events", "fn_events",
                    "event_level_fp", "event_precision", "event_recall", "event_f1",
                    "normal_fa_events_per_hour", "median_delay_sec", "point_f1", "point_pr_auc"
                ]
                st.dataframe(pf_sub[pf_cols], use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 5: MULTISEED STABILITY
    # -------------------------------------------------------------------------
    st.subheader("5. MULTISEED STABILITY — Độ Ổn định Đa Seed (`Seeds 42, 43, 44`)")

    if v2_summary_df is not None:
        dnn_seeds_ids = ["dnn_v2_seed42_tau_star", "dnn_v2_seed43_tau_star", "dnn_v2_seed44_tau_star"]
        dnn_seeds_df = v2_summary_df[v2_summary_df["model_id"].isin(dnn_seeds_ids)].copy()

        if len(dnn_seeds_df) == 3:
            s42_row = dnn_seeds_df[dnn_seeds_df["model_id"] == "dnn_v2_seed42_tau_star"].iloc[0]
            s43_row = dnn_seeds_df[dnn_seeds_df["model_id"] == "dnn_v2_seed43_tau_star"].iloc[0]
            s44_row = dnn_seeds_df[dnn_seeds_df["model_id"] == "dnn_v2_seed44_tau_star"].iloc[0]

            ms_c1, ms_c2, ms_c3 = st.columns(3)
            with ms_c1:
                st.markdown("##### 🔒 Seed 42 (Canonical — $\\tau^*$)")
                st.metric(
                    "Detected / Event F1",
                    f"{int(s42_row['tp_events'])}/{int(s42_row['evaluable_attacks'])}  |  F1: {float(s42_row['event_f1']):.6f}",
                    delta=f"Recall: {float(s42_row['event_recall'])*100:.2f}% | τ*={float(s42_row['threshold']):.6f}",
                )
                st.caption(
                    f"Normal FA/h: `{float(s42_row['normal_fa_events_per_hour']):.4f}` | "
                    f"Median Delay: `{float(s42_row['median_delay_sec']):.1f} s` | "
                    f"Point PR-AUC: `{float(s42_row['point_pr_auc']):.6f}`"
                )
            with ms_c2:
                st.markdown("##### 🧪 Seed 43 (Replicate 1 — $\\tau^*$)")
                st.metric(
                    "Detected / Event F1",
                    f"{int(s43_row['tp_events'])}/{int(s43_row['evaluable_attacks'])}  |  F1: {float(s43_row['event_f1']):.6f}",
                    delta=f"Recall: {float(s43_row['event_recall'])*100:.2f}% | τ*={float(s43_row['threshold']):.6f}",
                )
                st.caption(
                    f"Normal FA/h: `{float(s43_row['normal_fa_events_per_hour']):.4f}` | "
                    f"Median Delay: `{float(s43_row['median_delay_sec']):.1f} s` | "
                    f"Point PR-AUC: `{float(s43_row['point_pr_auc']):.6f}`"
                )
            with ms_c3:
                st.markdown("##### 🧪 Seed 44 (Replicate 2 — $\\tau^*$)")
                st.metric(
                    "Detected / Event F1",
                    f"{int(s44_row['tp_events'])}/{int(s44_row['evaluable_attacks'])}  |  F1: {float(s44_row['event_f1']):.6f}",
                    delta=f"Recall: {float(s44_row['event_recall'])*100:.2f}% | τ*={float(s44_row['threshold']):.6f}",
                    delta_color="off",
                )
                st.caption(
                    f"Normal FA/h: `{float(s44_row['normal_fa_events_per_hour']):.4f}` | "
                    f"Median Delay: `{float(s44_row['median_delay_sec']):.1f} s` | "
                    f"Point PR-AUC: `{float(s44_row['point_pr_auc']):.6f}`"
                )

    if v2_multiseed_df is not None:
        st.markdown("#### Thống kê Tổng hợp 3 Seeds (`Mean ± Std` từ `v2_heldout_multiseed_stats.csv`)")
        ms_table = pd.DataFrame({
            "Nhóm mô hình (3 Seeds: 42, 43, 44)": v2_multiseed_df["model_group"],
            "TP / 17 (Mean ± Std)": v2_multiseed_df.apply(lambda r: f"{r['tp_events_mean']:.2f} ± {r['tp_events_std']:.2f}", axis=1),
            "Event Recall (Mean ± Std)": v2_multiseed_df.apply(lambda r: f"{r['event_recall_mean']*100:.2f}% ± {r['event_recall_std']*100:.2f}%", axis=1),
            "Event Precision (Mean ± Std)": v2_multiseed_df.apply(lambda r: f"{r['event_precision_mean']*100:.2f}% ± {r['event_precision_std']*100:.2f}%", axis=1),
            "Event F1 (Mean ± Std)": v2_multiseed_df.apply(lambda r: f"{r['event_f1_mean']:.4f} ± {r['event_f1_std']:.4f}", axis=1),
            "Normal FA/h (Mean ± Std)": v2_multiseed_df.apply(lambda r: f"{r['normal_fa_events_per_hour_mean']:.4f} ± {r['normal_fa_events_per_hour_std']:.4f}", axis=1),
            "Median Delay s (Mean ± Std)": v2_multiseed_df.apply(lambda r: f"{r['median_delay_sec_mean']:.2f} ± {r['median_delay_sec_std']:.2f}", axis=1),
            "Point F1 (Mean ± Std)": v2_multiseed_df.apply(lambda r: f"{r['point_f1_mean']:.4f} ± {r['point_f1_std']:.4f}", axis=1),
            "Point ROC-AUC (Mean ± Std)": v2_multiseed_df.apply(lambda r: f"{r['point_roc_auc_mean']:.4f} ± {r['point_roc_auc_std']:.4f}", axis=1),
            "Point PR-AUC (Mean ± Std)": v2_multiseed_df.apply(lambda r: f"{r['point_pr_auc_mean']:.4f} ± {r['point_pr_auc_std']:.4f}", axis=1),
        })
        st.dataframe(ms_table, use_container_width=True, hide_index=True)

    st.warning(
        "📌 **Giải thích về Seed 44 và Nguyên tắc Giữ Cố định Seed 42 làm Canonical:**\n"
        "- **Tại sao Seed 44 có `Event F1 = 0.444444` cao hơn nhưng `Event Recall` giảm mạnh xuống `47.06%` (`8/17`)?** "
        "Do ngưỡng đóng băng của Seed 44 rất sát biên trên ($\\tau^* = 0.958462$), mô hình chỉ phát sinh tổng cộng `11` cảnh báo dư mức sự kiện (`Normal FA = 3` sự kiện, tức `0.0840 FA/h`), "
        "làm `Event Precision` tăng lên `42.11%` (`8/19`) và kéo trung bình điều hòa `Event F1` lên cao dù bỏ sót tới `9/17` cuộc tấn công (và `Point Recall` chỉ còn `1.53%`).\n"
        "- **Không chọn lọc hậu nghiệm (No Post-hoc Cherry-Picking):** `Seed 42` được cố định làm mô hình **Canonical** trước khi mở tập đánh giá `test4 + test5`. "
        "Dự án tuyệt đối không lấy `Seed 43` hay `Seed 44` có `Event F1` held-out cao hơn để thay thế kết quả đại diện chính."
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 6: ATTACK EVENT EXPLORER V2
    # -------------------------------------------------------------------------
    st.subheader("6. ATTACK EVENT EXPLORER V2 — Khám phá 17 Sự kiện Tấn công (`test4 + test5`)")

    if v2_events_df is not None:
        model_filter_map = {
            "DNN Seed 42 (Primary τ*, Canonical)": ("dnn_seed42_tau_star_detected", "dnn_seed42_tau_star_delay_seconds", "dnn_seed42_tau_star_peak_score", "dnn_seed42_tau_star_threshold", "dnn_seed42_tau_star_peak_to_threshold_ratio"),
            "DNN Seed 43 (Primary τ*)": ("dnn_seed43_tau_star_detected", "dnn_seed43_tau_star_delay_seconds", "dnn_seed43_tau_star_peak_score", "dnn_seed43_tau_star_threshold", "dnn_seed43_tau_star_peak_to_threshold_ratio"),
            "DNN Seed 44 (Primary τ*)": ("dnn_seed44_tau_star_detected", "dnn_seed44_tau_star_delay_seconds", "dnn_seed44_tau_star_peak_score", "dnn_seed44_tau_star_threshold", "dnn_seed44_tau_star_peak_to_threshold_ratio"),
            "GRU Seed 42 (V1, Canonical)": ("gru_seed42_detected", "gru_seed42_delay_seconds", "gru_seed42_peak_score", "gru_seed42_threshold", "gru_seed42_peak_to_threshold_ratio"),
            "GRU Seed 43 (V1)": ("gru_seed43_detected", "gru_seed43_delay_seconds", "gru_seed43_peak_score", "gru_seed43_threshold", "gru_seed43_peak_to_threshold_ratio"),
            "GRU Seed 44 (V1)": ("gru_seed44_detected", "gru_seed44_delay_seconds", "gru_seed44_peak_score", "gru_seed44_threshold", "gru_seed44_peak_to_threshold_ratio"),
            "Ridge α=300 (V1)": ("ridge_detected", "ridge_delay_seconds", "ridge_peak_score", "ridge_threshold", "ridge_peak_to_threshold_ratio"),
            "Persistence (V1)": ("persistence_detected", "persistence_delay_seconds", "persistence_peak_score", "persistence_threshold", "persistence_peak_to_threshold_ratio"),
            "DNN Seed 42 (Secondary τ=0.50)": ("dnn_seed42_tau_050_detected", "dnn_seed42_tau_050_delay_seconds", "dnn_seed42_tau_050_peak_score", "dnn_seed42_tau_050_threshold", "dnn_seed42_tau_050_peak_to_threshold_ratio"),
            "DNN Seed 43 (Secondary τ=0.50)": ("dnn_seed43_tau_050_detected", "dnn_seed43_tau_050_delay_seconds", "dnn_seed43_tau_050_peak_score", "dnn_seed43_tau_050_threshold", "dnn_seed43_tau_050_peak_to_threshold_ratio"),
            "DNN Seed 44 (Secondary τ=0.50)": ("dnn_seed44_tau_050_detected", "dnn_seed44_tau_050_delay_seconds", "dnn_seed44_tau_050_peak_score", "dnn_seed44_tau_050_threshold", "dnn_seed44_tau_050_peak_to_threshold_ratio"),
        }

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            v2_file_opts = ["All"] + sorted(v2_events_df["session"].unique().tolist())
            sel_v2_file = st.selectbox("1. Lọc theo Test File:", v2_file_opts, index=0, key="v2_file_filter")
        with f_col2:
            v2_proc_opts = [
                "All",
                "Involves P1 (Boiler)",
                "Involves P2 (Turbine)",
                "Involves P3 (Water-Treatment)",
                "Single-Process Only",
                "Multi-Process Only",
            ] + sorted(v2_events_df["process_set"].unique().tolist())
            sel_v2_proc = st.selectbox("2. Lọc theo Phân hệ P1/P2/P3:", v2_proc_opts, index=0, key="v2_proc_filter")
        with f_col3:
            sel_v2_model = st.selectbox("3. Chọn Mô hình / Seed:", list(model_filter_map.keys()), index=0, key="v2_model_filter")
        with f_col4:
            v2_status_opts = [
                "All",
                "Detected (Phát hiện bởi mô hình đang chọn)",
                "Missed (Bỏ sót bởi mô hình đang chọn)",
                "DNN s42 (τ*) phát hiện nhưng GRU s42 bỏ sót",
            ]
            sel_v2_status = st.selectbox("4. Lọc theo Tình trạng Phát hiện:", v2_status_opts, index=0, key="v2_status_filter")

        det_col, del_col, peak_col, thr_col, ratio_col = model_filter_map[sel_v2_model]

        sub_ev = v2_events_df.copy()
        if sel_v2_file != "All":
            sub_ev = sub_ev[sub_ev["session"] == sel_v2_file]

        if sel_v2_proc == "Involves P1 (Boiler)":
            sub_ev = sub_ev[sub_ev["attack_P1"] == True]
        elif sel_v2_proc == "Involves P2 (Turbine)":
            sub_ev = sub_ev[sub_ev["attack_P2"] == True]
        elif sel_v2_proc == "Involves P3 (Water-Treatment)":
            sub_ev = sub_ev[sub_ev["attack_P3"] == True]
        elif sel_v2_proc == "Single-Process Only":
            sub_ev = sub_ev[sub_ev["is_multiprocess"] == False]
        elif sel_v2_proc == "Multi-Process Only":
            sub_ev = sub_ev[sub_ev["is_multiprocess"] == True]
        elif sel_v2_proc != "All":
            sub_ev = sub_ev[sub_ev["process_set"] == sel_v2_proc]

        if sel_v2_status == "Detected (Phát hiện bởi mô hình đang chọn)":
            sub_ev = sub_ev[sub_ev[det_col] == True]
        elif sel_v2_status == "Missed (Bỏ sót bởi mô hình đang chọn)":
            sub_ev = sub_ev[sub_ev[det_col] == False]
        elif sel_v2_status == "DNN s42 (τ*) phát hiện nhưng GRU s42 bỏ sót":
            sub_ev = sub_ev[(sub_ev["dnn_seed42_tau_star_detected"] == True) & (sub_ev["gru_seed42_detected"] == False)]

        st.caption(
            f"Đang hiển thị **{len(sub_ev)} / {len(v2_events_df)}** sự kiện tấn công trên `test4 + test5` "
            f"(Mô hình đang chọn: **{sel_v2_model}**)."
        )

        ev_display_df = pd.DataFrame({
            "Attack ID": sub_ev["attack_event_id"],
            "File": sub_ev["session"],
            "Start Time": sub_ev["effective_start"],
            "End Time": sub_ev["orig_end"],
            "Duration (s)": sub_ev["duration_seconds"],
            "Process": sub_ev["process_set"],
            "Multi-Process": sub_ev["is_multiprocess"],
            "Selected Model Detected": sub_ev[det_col],
            "Selected Model Delay (s)": sub_ev[del_col],
            "Selected Peak Score": sub_ev[peak_col],
            "Selected Threshold": sub_ev[thr_col],
            "Peak / Threshold Ratio": sub_ev[ratio_col],
            "GRU s42 Delay (s)": sub_ev["gru_seed42_delay_seconds"],
            "DNN s42 (τ*) Delay (s)": sub_ev["dnn_seed42_tau_star_delay_seconds"],
            "DNN s43 (τ*) Delay (s)": sub_ev["dnn_seed43_tau_star_delay_seconds"],
            "DNN s44 (τ*) Delay (s)": sub_ev["dnn_seed44_tau_star_delay_seconds"],
        })
        st.dataframe(ev_display_df, use_container_width=True, hide_index=True)

        if v2_subsys_df is not None:
            with st.expander("📊 Xem bảng tổng hợp số sự kiện phát hiện theo từng Phân hệ Vật lý (`P1, P2, P3`, Đơn trình vs Đa trình)"):
                st.dataframe(v2_subsys_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 7: ERROR ANALYSIS & THRESHOLD SENSITIVITY
    # -------------------------------------------------------------------------
    st.subheader("7. ERROR ANALYSIS & THRESHOLD SENSITIVITY — Phân tích Lỗi & Độ nhạy Ngưỡng")

    st.warning(
        "⚠️ **Phân biệt Chế độ Ngưỡng Chính ($\\tau^*$) và Phân tích Độ nhạy Phụ ($\\tau = 0.50$):**\n"
        "- **Ngưỡng chính $\\tau^*$ (`0.953203` cho Seed 42, `0.958436` cho Seed 43, `0.958462` cho Seed 44):** Được chọn trên tập Validation `test3` và đóng băng trước khi đánh giá `test4 + test5`.\n"
        "- **Ngưỡng $\\tau = 0.50$:** Chỉ là phân tích độ nhạy phụ để minh họa hành vi xác suất chưa hiệu chuẩn (tại $\\tau = 0.50$, tỷ lệ báo động giả vọt lên `26.70 – 72.86 FA/h` và `Event F1` giảm còn `0.0127 – 0.0334`). **Tuyệt đối không dùng $\\tau = 0.50$ để thay thế kết quả benchmark chính.**"
    )

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        display_figure(
            "results/v2_dnn/figures/v2_event_metrics_comparison.png",
            caption="Hình V2-1: So sánh Event Recall, Event Precision, Event F1 và Normal FA/h trên test4 + test5 (17 sự kiện).",
        )
        display_figure(
            "results/v2_dnn/figures/v2_attack_event_heatmap_17.png",
            caption="Hình V2-3: Ma trận phát hiện và độ trễ (giây) trên toàn bộ 17 cuộc tấn công của test4 + test5.",
        )
    with fig_col2:
        display_figure(
            "results/v2_dnn/figures/v2_confusion_matrices_seed42.png",
            caption="Hình V2-2: Ma trận nhầm lẫn mức thời điểm (Point-level Confusion Matrix, N = 131,942) của GRU Seed 42, DNN Seed 42 (τ*) và DNN Seed 42 (τ = 0.50).",
        )
        display_figure(
            "results/v2_dnn/figures/v2_probability_saturation_analysis.png",
            caption="Hình V2-4: Phân phối xác suất dự đoán P(attack=1|x) của DNN V2 Seed 42 trên Validation (test3) và Held-out (test4+test5).",
        )

    if v2_cm_df is not None:
        with st.expander("🔢 Xem bảng Ma trận Nhầm lẫn Mức điểm (Point-Level Confusion Matrices) & Chỉ số Độ nhạy Đầy đủ"):
            st.dataframe(v2_cm_df, use_container_width=True, hide_index=True)

    st.info(
        "🔬 **Phân tích Khoa học: Tách biệt Quan sát Thực nghiệm Trực tiếp và Giả thuyết Cơ chế**\n\n"
        "**1. Quan sát trực tiếp từ dữ liệu thực nghiệm (Direct Empirical Observations):**\n"
        "- **Sự tập trung xác suất sát biên trên:** Từ dữ liệu điểm và *Hình V2-4*, xác suất dự đoán $p_t = \\sigma(z_t)$ của DNN V2 nén dày đặc trong dải hẹp sát biên trên $\\approx [0.953;\\; 0.959]$.\n"
        "- **Độ nhạy ngưỡng cực lớn giữa các seed:** Trên `test4 + test5`, cả 3 seed của DNN V2 có khả năng xếp hạng liên tục rất gần nhau (`Point ROC-AUC` $= 0.8472 \\pm 0.0012$). "
        "Tuy nhiên, tại Seed 44 ($\\tau^* = 0.958462$), **8 trong 9 sự kiện bị bỏ sót có đỉnh xác suất đạt từ `0.958428` đến `0.958458`** (`Peak/Threshold Ratio` $= 0.999964 \\div 0.999995$), tức chỉ thấp hơn ngưỡng đóng băng từ $4 \\times 10^{-6}$ đến $3.4 \\times 10^{-5}$.\n"
        "- **Cách đếm `Event FP` (`event_level_fp`) và `Normal FA Events` (`normal_fa_events`):** Trong mã đánh giá (`src/evaluate_v2_heldout.py`), "
        "`Event FP` đếm tổng số cụm cảnh báo (`score > τ`) trên toàn miền $t \\ge 30$ không ghép cặp 1–1 với sự kiện tấn công mới (`total_alarm_events - tp_events`), "
        "trong khi `Normal FA Events` đếm các cụm cảnh báo sau khi đã che toàn bộ thời điểm tấn công (`score > τ AND attack == 0`). "
        "Do hai chỉ số được trích xuất theo hai định nghĩa khoảng thời gian khác nhau, **không thể tự động đồng nhất toàn bộ hiệu số `Event FP - Normal FA Events` là alarm chatter trong vùng tấn công nếu không phân rã từng khoảng cảnh báo**.\n\n"
        "**2. Giả thuyết cơ chế (Mechanistic Hypotheses — Chưa kiểm chứng tách biệt bằng Ablation riêng):**\n"
        "- Hiện tượng xác suất bị nén sát biên trên và dao động ngưỡng giữa 3 seeds có thể liên quan đến sự kết hợp của nhiều yếu tố: "
        "(a) tầng ẩn thứ 6 `Dense(10, activation='tanh')` giới hạn đầu vào lớp logit cuối trong $(-1, 1)$, "
        "(b) việc dừng sớm (Early Stopping) tại Epoch 1–2 khi biên độ trọng số lớp đầu ra còn hạn chế, "
        "(c) hệ số phạt lớp dương `pos_weight = 6.223009`, và "
        "(d) số lượng sự kiện tấn công ít trên tập Validation `test3` (chỉ có 8 sự kiện). "
        "**Do giữ nguyên kiến trúc Listing 1 và không thực hiện thí nghiệm ablation thay đổi hàm kích hoạt, cơ chế `Tanh(10)` được trình bày như một giả thuyết giải thích hợp lý, không tuyên bố là nguyên nhân duy nhất.**"
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 8: RESEARCH LIMITATIONS
    # -------------------------------------------------------------------------
    st.subheader("8. RESEARCH LIMITATIONS — Các Giới hạn Nghiên cứu & Lưu ý Khi Thuyết trình")

    st.markdown(
        """
        1. **Không tái lập toàn bộ mô hình và bài toán gốc của bài báo:** DNN V2 chỉ tái lập có điều chỉnh cấu trúc 6 tầng ẩn Dense từ Listing 1 của Kuchař & Fujdiak (2025) sang bài toán nhị phân `Normal/Attack` với 79 đặc trưng trên HAI 21.03; không tái lập 4 trạng thái `Normal/Security/Operational/Service`, không tái lập lớp `norm_layer` chưa rõ định nghĩa, và không tuyên bố đạt độ chính xác `99,72%` của bài báo.
        2. **Hai mô hình DNN V2 và GRU V1 được huấn luyện dưới hai chế độ giám sát khác nhau:** DNN V2 sử dụng nhãn tấn công thật trong quá trình huấn luyện (`test1 + test2`) và chọn ngưỡng (`test3`), trong khi GRU/Ridge/Persistence V1 hoàn toàn không sử dụng nhãn tấn công khi huấn luyện (`train1 + train2`) và chọn ngưỡng $Q_{99.9}$ (`train3`).
        3. **Dao động lớn giữa các seeds của DNN V2 tại ngưỡng đóng băng $\\tau^*$:** Dù diện tích dưới đường cong `Point ROC-AUC` và `Point PR-AUC` ổn định, `Event Recall` tại ngưỡng cứng $\\tau^*$ biến thiên mạnh từ `100%` (Seed 42) và `94.12%` (Seed 43) xuống `47.06%` (Seed 44).
        4. **Ngưỡng V2 được hiệu chỉnh hậu nghiệm dựa trên tập Validation `test3`:** Việc bổ sung trần `max_point_fpr_cap = 0.05`, sàn `min_event_recall_floor = 0.75` và lưới phân vị thực nghiệm là bước điều chỉnh hậu nghiệm sau khi quan sát hiện tượng cảnh báo luôn bật ở ngưỡng thấp trên `test3`, không phải quy tắc tiền định hoàn toàn từ trước.
        5. **Tập Held-out V2 (`test4 + test5`) không hoàn toàn mù (not completely blind) ở cấp độ toàn bộ dự án:** Mặc dù `test4 + test5` không tham gia vào quá trình huấn luyện hay chọn ngưỡng của DNN V2, hai phiên này đã được phân tích trong thí nghiệm V1 trước đó.
        6. **Kết quả thực nghiệm không chứng minh sẵn sàng triển khai trên hệ thống ICS thực tế:** Mức báo động giả và độ nhạy ngưỡng trên tập dữ liệu học thuật HAI 21.03 cho thấy hệ thống hiện tại phục vụ mục tiêu nghiên cứu và giảng dạy, chưa đủ điều kiện vận hành tự động trong môi trường công nghiệp thực tế.
        """
    )
