"""Attack Explorer view; display existing research artifacts."""

import streamlit as st
import pandas as pd
from fe.ui_utils import load_csv, display_figure, format_delay


def render_attacks():
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
            selected_session = st.selectbox("Phiên kiểm thử", session_options, index=0, key="attack_session")
        with col_f2:
            process_options = ["All"] + sorted(merged_atk["process_set"].unique().tolist())
            selected_process = st.selectbox("Hệ thống con", process_options, index=0, key="attack_process")
        with col_f3:
            status_options = [
                "All",
                "Detected by GRU s42",
                "Missed by GRU s42",
                "Detected by All Models",
                "Missed by All Models",
                "Ridge detected, GRU missed",
            ]
            selected_status = st.selectbox("Tình trạng phát hiện", status_options, index=0, key="attack_status")
        with col_f4:
            search_query = st.text_input("Tìm mã sự kiện", "", placeholder="Ví dụ: atk_02", key="attack_search").strip()

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
        elif selected_status == "Ridge detected, GRU missed":
            filtered_df = filtered_df[(filtered_df["ridge_detected"] == True) & (filtered_df["gru_seed42_detected"] == False)]

        if search_query:
            filtered_df = filtered_df[filtered_df["attack_event_id"].str.contains(search_query, case=False, na=False, regex=False)]

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
            width="stretch",
            hide_index=True
        )

        st.markdown("---")

        # Detail Inspection Panel
        st.subheader("Attack Event Detail Panel")
        if not filtered_df.empty:
            chosen_atk_id = st.selectbox(
                "Sự kiện cần xem",
                filtered_df["attack_event_id"].tolist(),
                index=0, key="attack_detail"
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
