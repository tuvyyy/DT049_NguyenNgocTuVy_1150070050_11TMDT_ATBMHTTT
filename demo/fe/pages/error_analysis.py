"""Error Analysis view; display existing research artifacts."""

import streamlit as st
from fe.ui_utils import load_csv, display_figure


def render_error_analysis():
    st.header("Post-Test Error Analysis & Operational Characterization")
    st.markdown("In-depth diagnostic breakdown of process attribution, common-detected latency, missed attacks, and false-alarm taxonomy.")

    # 4A. Process-Level Detection
    st.subheader("A. Process-Level Detection Breakdown")
    proc_df = load_csv("results/metrics/final_error_analysis_by_process.csv")
    if proc_df is not None:
        col_pr1, col_pr2 = st.columns([1.4, 1.0])
        with col_pr1:
            st.dataframe(proc_df, width="stretch", hide_index=True)
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
            st.dataframe(paired_df, width="stretch", hide_index=True)
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
        st.dataframe(missed_df[disp_miss_cols].rename(columns=rename_miss), width="stretch", hide_index=True)

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
