"""Threshold Sensitivity view; display existing research artifacts."""

import streamlit as st
from fe.ui_utils import load_csv, display_figure


def render_threshold():
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
