"""Read official artifacts without importing Streamlit or running models."""

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def read_csv(relative):
    return pd.read_csv(REPO_ROOT / relative)


def read_json(relative):
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def check_artifact_health():
    details = []
    for name, relative in HEALTH_ARTIFACTS:
        path = REPO_ROOT / relative
        exists = path.is_file()
        details.append(dict(name=name, path=relative, exists=exists,
                            size_bytes=path.stat().st_size if exists else 0))
    available = sum(item["exists"] for item in details)
    return dict(total=len(details), available=available,
                is_healthy=available == len(details), details=details)


def metric_download(relative):
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to((REPO_ROOT / "results/metrics").resolve()):
        raise ValueError("Only result metric files can be exported.")
    return path.read_bytes()
