"""Reproducibility view; display existing research artifacts."""

import streamlit as st
import pandas as pd
from fe.ui_utils import check_artifact_health, render_download_button


def render_reproducibility():
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
        width="stretch",
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
