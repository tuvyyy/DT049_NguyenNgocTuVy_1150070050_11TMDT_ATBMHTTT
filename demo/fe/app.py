"""Frontend entry: application layout and page routing."""

from pathlib import Path

import streamlit as st

from fe.gru_workflow import render_gru_workflow
from fe.navigation import navigation
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
    st.markdown('<div class="db-context"><span>DT049 / HAI 21.03 · Persistence, Ridge, GRU</span>'
                '<span>50 sự kiện · Q99.9 · Kết quả đã lưu</span></div>', unsafe_allow_html=True)


def run_app():
    st.set_page_config(page_title="DT049 — HAI 21.03 Anomaly Detection", page_icon="🛡️",
                       layout="wide", initial_sidebar_state="expanded")
    css = (Path(__file__).parent / "styles/dashboard.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    page = render_sidebar()
    if page == "Overview":
        with st.container(key="overview_mode_bar"):
            mode = navigation(["Thực hiện GRU", "Tổng quan nghiên cứu"], key="overview_mode")
        if mode == "Thực hiện GRU":
            render_gru_workflow()
        else:
            render_overview()
    elif page == "DNN Replication V2":
        # This extension is available only after restoring the end-of-term backup.
        from dnn_v2 import render_dnn_v2
        render_dnn_v2()
    else:
        with st.container(key="research_page"):
            _benchmark_header()
            PAGES[page]()
