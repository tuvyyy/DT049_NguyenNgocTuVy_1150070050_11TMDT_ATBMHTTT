"""Presentation helpers; artifact reading is delegated to the backend."""

import pandas as pd
import streamlit as st

from be import artifacts
from be.artifacts import REPO_ROOT, check_artifact_health


@st.cache_data
def load_csv(relative):
    try:
        return artifacts.read_csv(relative)
    except FileNotFoundError:
        st.warning(f"Artifact not found: `{relative}`")
    except Exception as error:
        st.error(f"Error loading `{relative}`: {error}")
    return None


@st.cache_data
def load_json(relative):
    try:
        return artifacts.read_json(relative)
    except FileNotFoundError:
        st.warning(f"Artifact not found: `{relative}`")
    except Exception as error:
        st.error(f"Error loading `{relative}`: {error}")
    return None


def display_figure(relative, caption=None):
    path = REPO_ROOT / relative
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Figure not found: `{relative}`")


def format_delay(value):
    return "N/A" if pd.isna(value) else f"{float(value):.1f} s"


def render_download_button(relative, label, file_name=None, mime="text/csv"):
    try:
        data = artifacts.metric_download(relative)
    except (OSError, ValueError) as error:
        st.warning(f"File not available for download: `{relative}` ({error})")
        return
    st.download_button(label, data=data, file_name=file_name or relative.rsplit("/", 1)[-1],
                       mime=mime, width="stretch")
