"""Model Comparison view; display existing research artifacts."""

import streamlit as st
from fe.ui_utils import load_csv, display_figure


def render_comparison():
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
