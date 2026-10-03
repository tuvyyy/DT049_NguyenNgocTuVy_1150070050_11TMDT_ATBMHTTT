#!/usr/bin/env python3
"""
DT049 Streamlit UI Utilities
============================
Helper functions for loading cached metrics, handling missing artifacts,
and rendering figures safely in the Streamlit application.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent


@st.cache_data
def load_csv(rel_path: str) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV artifact relative to repository root with Streamlit caching.
    """
    p = REPO_ROOT / rel_path
    if not p.exists():
        st.warning(f"Artifact not found: `{rel_path}`")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Error loading `{rel_path}`: {e}")
        return None


@st.cache_data
def load_json(rel_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely load a JSON artifact relative to repository root with Streamlit caching.
    """
    p = REPO_ROOT / rel_path
    if not p.exists():
        st.warning(f"Artifact not found: `{rel_path}`")
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading `{rel_path}`: {e}")
        return None


def display_figure(rel_path: str, caption: Optional[str] = None):
    """
    Safely display a figure PNG relative to repository root.
    """
    p = REPO_ROOT / rel_path
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure not found: `{rel_path}`")


def format_delay(val) -> str:
    """Format delay in seconds or N/A."""
    if pd.isna(val):
        return "N/A"
    return f"{float(val):.1f} s"


def format_rate(val) -> str:
    """Format rate in events/hour."""
    if pd.isna(val):
        return "N/A"
    return f"{float(val):.4f}"
