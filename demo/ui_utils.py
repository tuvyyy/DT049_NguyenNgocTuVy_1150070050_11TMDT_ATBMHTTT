#!/usr/bin/env python3
"""
DT049 Streamlit UI Utilities
============================
Helper functions for loading cached metrics, handling missing artifacts,
checking artifact health, rendering figures, and offering secure downloads
in the Streamlit application.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent

HEALTH_ARTIFACTS = [
    ("Final Test Summary CSV", "results/metrics/final_test_summary.csv"),
    ("Final Attack Events CSV", "results/metrics/final_attack_events.csv"),
    ("Final Test Execution Audit JSON", "results/audit/final_test_execution.json"),
    ("Pre-Test Freeze Manifest JSON", "results/audit/pretest_freeze_manifest.json"),
    ("Frozen Ridge Model Checkpoint", "models/baselines/ridge.joblib"),
    ("Primary GRU Seed 42 Checkpoint", "models/primary/gru_w30_seed42.pt"),
    ("Primary GRU Seed 43 Checkpoint", "models/primary/gru_w30_seed43.pt"),
    ("Primary GRU Seed 44 Checkpoint", "models/primary/gru_w30_seed44.pt"),
]


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
    Safely display a figure PNG relative to repository root without crashing if missing.
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


def check_artifact_health() -> Dict[str, Any]:
    """
    Non-destructively check the presence and size of the 8 required health artifacts.
    """
    details: List[Dict[str, Any]] = []
    available = 0
    for name, rel_path in HEALTH_ARTIFACTS:
        p = REPO_ROOT / rel_path
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        if exists:
            available += 1
        details.append({
            "name": name,
            "path": rel_path,
            "exists": exists,
            "size_bytes": size,
        })
    return {
        "total": len(HEALTH_ARTIFACTS),
        "available": available,
        "is_healthy": available == len(HEALTH_ARTIFACTS),
        "details": details,
    }


def render_download_button(rel_path: str, label: str, file_name: Optional[str] = None, mime: str = "text/csv"):
    """
    Safely render a Streamlit download button for an existing result file.
    Does not allow downloading files outside results/metrics/.
    """
    p = REPO_ROOT / rel_path
    if not p.exists():
        st.warning(f"File not available for download: `{rel_path}`")
        return

    # Security check: only allow results/metrics/ files
    if not str(rel_path).startswith("results/metrics/"):
        st.error(f"Download restricted: `{rel_path}` is outside allowed metrics directory.")
        return

    out_name = file_name or p.name
    try:
        with open(p, "rb") as f:
            data = f.read()
        st.download_button(
            label=label,
            data=data,
            file_name=out_name,
            mime=mime,
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Error reading `{rel_path}`: {e}")
