"""Frontend entry: application layout and page routing."""

import streamlit as st

from fe.gru_workflow import render_gru_workflow
from fe.overview import render_overview
from fe.sidebar import render_sidebar
from fe.pages.comparison import render_comparison
from fe.pages.attacks import render_attacks
from fe.pages.error_analysis import render_error_analysis
from fe.pages.threshold import render_threshold
from fe.pages.ablation import render_ablation
from fe.pages.reproducibility import render_reproducibility

PAGES = {
    "Model Comparison": render_comparison,
    "Attack Explorer": render_attacks,
    "Error Analysis": render_error_analysis,
    "Threshold Sensitivity": render_threshold,
    "Dynamic57 Ablation": render_ablation,
    "Reproducibility": render_reproducibility,
}


def _benchmark_header():
    st.title("DT049 — Multivariate Time-Series Anomaly Detection on HAI 21.03")
    st.markdown("**Persistence • Ridge • GRU** | *Leakage-aware event detection and detection-delay analysis*")
    columns = st.columns(5)
    for column, text in zip(columns, ["🏷️ **Dataset:** HAI 21.03", "🔒 **Primary Threshold:** $Q_{99.9}$",
                                      "🎯 **Attack Events:** 50", "🧠 **Primary Model:** GRU $W=30$"]):
        column.info(text)
    columns[4].success("✅ **Status:** Frozen Final Test")
    st.markdown("---")


def run_app():
    st.set_page_config(page_title="DT049 — HAI 21.03 Anomaly Detection", page_icon="🛡️",
                       layout="wide", initial_sidebar_state="expanded")
    page = render_sidebar()
    if page == "Overview":
        with st.container(key="overview_mode_bar"):
            mode = st.radio("Chế độ xem", ["Thực hiện GRU", "Tổng quan nghiên cứu"],
                            horizontal=True, label_visibility="collapsed", key="overview_mode")
        if mode == "Thực hiện GRU":
            render_gru_workflow()
        else:
            render_overview()
    elif page == "DNN Replication V2":
        # This extension is available only after restoring the end-of-term backup.
        from dnn_v2 import render_dnn_v2
        render_dnn_v2()
    else:
        _benchmark_header()
        PAGES[page]()
