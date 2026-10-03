#!/usr/bin/env python3
"""
DT049 Product Health & Integrity Check
======================================
Verifies all source modules, result CSVs, diagnostic figures,
audit logs, model checkpoints, and runtime imports required for
the DT049 Streamlit product dashboard.

Non-destructive: Performs zero raw dataset loading, zero model inference,
and zero file modifications.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

# 1. Required Dashboard Scripts
DASHBOARD_SOURCES = [
    "demo/app.py",
    "demo/ui_utils.py",
    "demo/run_dashboard.sh",
    "demo/stop_dashboard.sh",
]

# 2. Required Result Metrics CSVs
METRICS_CSVS = [
    "results/metrics/final_test_summary.csv",
    "results/metrics/final_test_multiseed.csv",
    "results/metrics/final_attack_events.csv",
    "results/metrics/final_per_file.csv",
    "results/metrics/final_missed_attack_analysis.csv",
    "results/metrics/final_test_sensitivity_complete.csv",
    "results/metrics/ablation_dynamic57_summary.csv",
    "results/metrics/ablation_dynamic57_multiseed.csv",
    "results/metrics/ablation_dynamic57_changed_events.csv",
    "results/metrics/final_error_analysis_by_process.csv",
    "results/metrics/final_common_detected_delay.csv",
]

# 3. Required Diagnostic Figures
FIGURES_PNG = [
    "results/figures/final_test/fig_01_final_model_comparison.png",
    "results/figures/final_test/fig_02_test_attack_timeline_seed42.png",
    "results/figures/final_test/fig_03_detection_delay_distribution.png",
    "results/figures/error_analysis/fig_ea_01_process_recall.png",
    "results/figures/error_analysis/fig_ea_02_common_delay_paired.png",
    "results/figures/error_analysis/fig_ea_03_event_chatter_breakdown.png",
    "results/figures/error_analysis/fig_ea_04_false_alarm_tradeoff.png",
    "results/figures/error_analysis/fig_ea_05_sensitivity_curves.png",
    "results/figures/error_analysis/fig_ea_06_case_all_detected.png",
    "results/figures/error_analysis/fig_ea_07_case_gru_only.png",
    "results/figures/error_analysis/fig_ea_08_case_all_missed.png",
    "results/figures/error_analysis/fig_ea_09_case_gru_miss_ridge_detected.png",
    "results/figures/ablation_dynamic57/fig_ab_01_primary_vs_dynamic57.png",
    "results/figures/ablation_dynamic57/fig_ab_02_detection_changes.png",
    "results/figures/ablation_dynamic57/fig_ab_03_false_alarm_tradeoff.png",
]

# 4. Audit Manifests
AUDIT_JSONS = [
    "results/audit/final_test_execution.json",
    "results/audit/pretest_freeze_manifest.json",
]

# 5. Frozen Model Artifacts
MODEL_CHECKPOINTS = [
    "models/preprocessing/scaler.joblib",
    "models/baselines/ridge.joblib",
    "models/primary/gru_w30_seed42.pt",
    "models/primary/gru_w30_seed43.pt",
    "models/primary/gru_w30_seed44.pt",
]

# 6. Required Python Libraries
REQUIRED_IMPORTS = [
    "streamlit",
    "pandas",
    "numpy",
    "matplotlib",
]


def check_file_group(group_name: str, file_list: list) -> tuple:
    """Verify presence of files in list."""
    passed = 0
    missing = []
    for rel_path in file_list:
        p = REPO_ROOT / rel_path
        if p.exists() and p.stat().st_size > 0:
            passed += 1
        else:
            missing.append(rel_path)
    return passed, len(file_list), missing


def check_imports() -> tuple:
    """Verify runtime importability of core packages."""
    passed = 0
    failed = []
    for mod in REQUIRED_IMPORTS:
        try:
            __import__(mod)
            passed += 1
        except ImportError as e:
            failed.append(f"{mod} ({e})")
    return passed, len(REQUIRED_IMPORTS), failed


def main():
    print("==================================================")
    print("DT049 Product Health & Integrity Verification")
    print("==================================================")
    print(f"Repository Root: {REPO_ROOT}\n")

    overall_status = "PASS"

    checks = [
        ("Dashboard Sources", DASHBOARD_SOURCES),
        ("Result Metrics CSVs", METRICS_CSVS),
        ("Diagnostic Figures PNG", FIGURES_PNG),
        ("Audit Manifest JSONs", AUDIT_JSONS),
        ("Frozen Model Artifacts", MODEL_CHECKPOINTS),
    ]

    for label, files in checks:
        passed, total, missing = check_file_group(label, files)
        status = "PASS" if passed == total else ("WARN" if passed > 0 else "FAIL")
        if status != "PASS":
            overall_status = "WARN" if overall_status == "PASS" else overall_status
        print(f"[{status}] {label:24s} : {passed:2d} / {total:2d} present")
        for m in missing:
            print(f"       MISSING: {m}")

    # Import Check
    p_imp, t_imp, f_imp = check_imports()
    imp_status = "PASS" if p_imp == t_imp else "FAIL"
    if imp_status != "PASS":
        overall_status = "FAIL"
    print(f"[{imp_status}] {'Python Runtime Imports':24s} : {p_imp:2d} / {t_imp:2d} imported successfully")
    for fi in f_imp:
        print(f"       IMPORT ERROR: {fi}")

    print("\n--------------------------------------------------")
    print(f"OVERALL PRODUCT HEALTH STATUS: [{overall_status}]")
    print("--------------------------------------------------")

    if overall_status == "PASS":
        print("All dashboard sources, metrics, figures, checkpoints, and dependencies verified.")
        sys.exit(0)
    elif overall_status == "WARN":
        print("Some optional components or files are missing. Product may have reduced functionality.")
        sys.exit(1)
    else:
        print("Critical product dependencies or files are missing.")
        sys.exit(2)


if __name__ == "__main__":
    main()
