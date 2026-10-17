"""Dynamic57 Ablation view; display existing research artifacts."""

import streamlit as st
from fe.ui_utils import load_csv, display_figure


def render_ablation():
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
        st.dataframe(q999_ab[cols_show].rename(columns=rename_ab), width="stretch", hide_index=True)

    if ab_ms_df is not None:
        st.subheader("GRU Multi-Seed Aggregate (79 vs. 57)")
        q_ms = ab_ms_df[ab_ms_df["operating_point"] == "Q99.9"].copy()
        st.dataframe(q_ms, width="stretch", hide_index=True)

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
