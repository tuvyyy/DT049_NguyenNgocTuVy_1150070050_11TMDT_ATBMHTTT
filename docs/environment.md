# Runtime & Computational Environment: DT049

This document records the exact computational environment, hardware constraints, operating system, and software versions used throughout the development, frozen evaluation, and dashboard serving of the DT049 project.

---

## 1. System & Hardware Specifications

| Component | Specification |
| :--- | :--- |
| **Operating System** | Kali GNU/Linux Rolling (Release 2026.2) |
| **Kernel** | Linux 6.19.14+kali-amd64 (x86_64) |
| **Virtualization** | Dedicated Kali Linux Virtual Machine |
| **System Memory (RAM)** | 3,881 MB physical RAM (~4.0 GB) + 953 MB swap |
| **Compute Device** | CPU-only (Strictly no GPU / CUDA dependencies) |
| **Security & Privacy** | Zero private credentials, SSH keys, tokens, or host paths exposed |

---

## 2. Python Runtime & Virtual Environment

- **Python Version**: `3.14.7`
- **Virtual Environment Path**: `/home/kali/DT049/.venv`
- **Lockfile**: [`requirements-lock.txt`](../requirements-lock.txt)

### Core Dependency Versions

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `numpy` | `2.5.3` | Array operations and memory-mapped scoring |
| `pandas` | `3.0.6` | Tabular data manipulation and CSV parsing |
| `scipy` | `1.18.1` | Scientific routines and statistical distributions |
| `scikit-learn` | `1.9.1` | StandardScaler and multi-output Ridge regression |
| `matplotlib` | `3.11.2` | Publication figures and evaluation diagnostics |
| `PyYAML` | `6.0.3` | Protocol configuration loading |
| `torch` | `2.14.0+cpu` | GRU recurrent model architecture and training |
| `streamlit` | `1.65.0` | Interactive web dashboard |
| `joblib` | `1.6.0` | Model and scaler persistence |
| `tqdm` | `4.70.1` | Progress monitoring for batch routines |
| `psutil` | `7.2.2` | Resident set size (RSS) tracking and memory safety |

---

## 3. Peak Memory (RSS) Measurements

All pipelines were designed with explicit memory-safety guardrails for a 4 GB RAM ceiling. The table below records the measured peak Resident Set Size (RSS) during each major phase:

| Phase | Script Entry Point | Peak RSS (MB) | Safety Margin vs 4 GB |
| :--- | :--- | :---: | :---: |
| **Baseline 0 (Persistence)** | `src/baseline_persistence.py` | `270.27 MB` | ~93.2% free |
| **Baseline 1 (Ridge $\alpha=300$)** | `src/baseline_ridge.py` | `669.70 MB` | ~83.2% free |
| **Primary GRU (Training)** | `src/train_gru.py` | `865.34 MB` | ~78.3% free |
| **Official Final-Test Evaluation** | `src/evaluate_final_test.py` | `850.06 MB` | ~78.7% free |
| **Post-Hoc Dynamic57 Ablation** | `src/run_ablation_dynamic57.py` | `738.12 MB` | ~81.5% free |
| **Streamlit Interactive Demo** | `demo/app.py` | `~61.40 MB` | ~98.5% free |

---

## 4. Interactive Dashboard Endpoint

The interactive demonstration runs locally without network egress:

```bash
# Activation & launch command
cd /home/kali/DT049
source .venv/bin/activate
streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501
```

- **Local Access URL**: `http://127.0.0.1:8501`
- **Execution Mode**: Read-only static artifact rendering; strictly zero model retraining or test recomputation during runtime.
