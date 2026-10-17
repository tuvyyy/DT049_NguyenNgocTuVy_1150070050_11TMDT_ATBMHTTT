"""Shared, grouped navigation for the research dashboard."""

from pathlib import Path

import streamlit as st


NAVIGATION = (
    ("Workspace", (
        ("Overview", "overview", "space_dashboard"),
        ("Model Comparison", "models", "bar_chart"),
        ("Attack Explorer", "attacks", "radar"),
        ("Error Analysis", "errors", "troubleshoot"),
    )),
    ("Experiments", (
        ("Threshold Sensitivity", "threshold", "tune"),
        ("Dynamic57 Ablation", "ablation", "layers"),
        ("DNN Replication V2", "dnn", "account_tree"),
    )),
    ("Project", (
        ("Reproducibility", "reproducibility", "description"),
    )),
)


def _select_page(page):
    st.session_state["navigation"] = page


def render_sidebar():
    pages = {page for _, items in NAVIGATION for page, _, _ in items}
    if st.session_state.get("navigation") not in pages:
        st.session_state["navigation"] = "Overview"
    current = st.session_state["navigation"]
    css = Path(__file__).with_name("sidebar.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            '<div class="sb-brand"><div class="sb-monogram" aria-hidden="true">dt<span>.</span></div>'
            '<div><div class="sb-brand-name">DT049</div>'
            '<div class="sb-brand-subtitle">Industrial security research</div></div></div>',
            unsafe_allow_html=True,
        )
        with st.container(key="sidebar_nav"):
            for group, items in NAVIGATION:
                st.markdown(f'<div class="sb-group-label">{group}</div>', unsafe_allow_html=True)
                for page, key, icon in items:
                    st.button(
                        page,
                        key=f"nav_{key}",
                        icon=f":material/{icon}:",
                        type="primary" if current == page else "secondary",
                        width="stretch",
                        on_click=_select_page,
                        args=(page,),
                    )

        with st.container(key="sidebar_about"):
            st.markdown(
                '<div class="sb-dataset"><span>HAI 21.03</span><span class="sb-version">v1.0</span></div>'
                '<div class="sb-dataset-note"><i></i>Frozen benchmark</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Project details", icon=":material/info:"):
                st.markdown(
                    "**DT049 · Research demo**\n\n"
                    "Official benchmark pages read saved artifacts. The Thực hiện GRU workspace runs training and inference in isolated folders.\n\n"
                    "**Evaluation protocol**\n\n"
                    "Unseen official test sessions; frozen at commit `6a0613b`.\n\n"
                    "**Release**\n\n"
                    "`dt049-final-test-v1`"
                )
    return current
