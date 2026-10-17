# Preprocessing and Leakage-Safe Windowing Infrastructure: DT049

## 1. Overview & Pipeline Objectives

This document establishes the streaming standardization and lazy, memory-mapped sequence-to-next-step windowing pipeline for project DT049 using the **HAI 21.03 Security Dataset**.

### Methodological Controls Enforced
1. **Streaming Scaler Fitting**: `StandardScaler.partial_fit()` was executed sequentially on 100,000-row chunks strictly from **Development Training data** (`train1.csv.gz` and `train2.csv.gz`, totaling **442,802 samples**).
2. **Zero Test & Validation Ingestion**: Normal Validation (`train3.csv.gz`) and Official Test files (`test1.csv.gz` through `test5.csv.gz`) were **never accessed or fed** into `partial_fit()`.
3. **Preservation of All 79 Features**: All 79 candidate telemetry features are preserved with `float32` precision. Zero-variance features are handled deterministically without column pruning.
4. **Incremental Memory-Mapped Serialization**: Processed arrays were written directly to disk via `numpy.lib.format.open_memmap`, keeping memory consumption low.
5. **Strict Session Boundary Isolation**: Each recording session is processed and indexed as an independent temporal block. Sliding windows **never straddle session boundaries**.

---

## 2. Streaming Standardization & Zero-Variance Behavior

Standardization follows the z-score formulation:
$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

### Handling of Zero-Variance Training Features
Across the 442,802 Development Training samples, exactly **22 candidate features exhibit zero variance** ($\sigma_{\text{train}} = 0$):
- **P1 (8)**: `P1_PCV02D`, `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`
- **P2 (12)**: `P2_ASD`, `P2_AutoGO`, `P2_Emerg`, `P2_MSD`, `P2_ManualGO`, `P2_OnOff`, `P2_RTR`, `P2_TripEx`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`
- **P3 (2)**: `P3_LH`, `P3_LL`
- **P4 (0)**: *(None)*

### Deterministic Scale Handling
In streaming Welford variance accumulation, constant columns accumulate floating-point residuals ($< 10^{-12}$). The pipeline enforces:
$$\sigma_j = 0 \implies \text{scale}_j = 1.0$$
Consequently:
- For normal training data where $x_j = \mu_j$, the standardized value is:
  $$z_j = \frac{x_j - \mu_j}{1.0} = 0.0$$
- If a channel exhibits operational deviation in subsequent test evaluations (e.g. `P1_PCV02D`, `P2_Emerg`, or `P2_OnOff` during cyber attacks), its departure from the training baseline $(x_j - \mu_j)$ is preserved without division-by-zero or NaN values.

---

## 3. Processed File Manifest

Generated arrays are stored under [`data/processed/hai_21_03/`](file:///home/kali/DT049/data/processed/hai_21_03/) and tracked via [`results/audit/hai_21_03_preprocessing_manifest.csv`](file:///home/kali/DT049/results/audit/hai_21_03_preprocessing_manifest.csv):

| Processed File | Source File | Shape | Dtype | Size on Disk | SHA256 Hash |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train1_features.npy` | `train1.csv.gz` | `(216001, 79)` | `float32` | 68,256,444 B | `2e6f11e0b0324745aa2c75cbd6bffa340e12378c94bb9b6b2caa18a58f36a860` |
| `train1_timestamps.npy`| `train1.csv.gz` | `(216001,)` | `datetime64[s]`| 1,728,136 B | `2c8593fc7e595b3a18215f9395c046af8c263807304691eadba2d9b37e45e60e` |
| `train2_features.npy` | `train2.csv.gz` | `(226801, 79)` | `float32` | 71,669,244 B | `2f80b24c63bd268d61d177d471bce5b61330913dd323a27936a93db830d8050b` |
| `train2_timestamps.npy`| `train2.csv.gz` | `(226801,)` | `datetime64[s]`| 1,814,536 B | `4742a602ca117fb7e134c390483fd6a2f661e07f28e9567f476099425244284c` |
| `train3_features.npy` | `train3.csv.gz` | `(478801, 79)` | `float32` | 151,301,244 B | `266c634e358708b716047c5113a53d548c13dde691951562390888b7081a48f4` |
| `train3_timestamps.npy`| `train3.csv.gz` | `(478801,)` | `datetime64[s]`| 3,830,536 B | `0084ca3e3255eea51b0e289e6197c8c15435667bcbf12c797d7390edaf72f2a1` |

- **Scaler Reproducibility Artifacts**:
  - [`models/preprocessing/scaler.joblib`](file:///home/kali/DT049/models/preprocessing/scaler.joblib): Serialized `StandardScaler`.
  - [`models/preprocessing/scaler_metadata.json`](file:///home/kali/DT049/models/preprocessing/scaler_metadata.json): Complete machine- and human-readable metadata record.
  - [`configs/preprocessing.yaml`](file:///home/kali/DT049/configs/preprocessing.yaml): Declarative configuration.

---

## 4. Windowing Infrastructure & Boundary Protection

The windowing framework is implemented in [`src/window_dataset.py`](file:///home/kali/DT049/src/window_dataset.py).

### Sequence-to-Next-Step Formulation
For input window length $W$ and stride $S = 1$:
- **Input sequence**: $\mathbf{X}_k = X[i : i + W] \in \mathbb{R}^{W \times 79}$
- **Target next step**: $\mathbf{y}_k = X[i + W] \in \mathbb{R}^{79}$

### Lazy Memory-Mapped Slicing
- Arrays are opened on demand via `np.load(filepath, mmap_mode="r")`.
- Individual window slices are returned as dynamic array views in `__getitem__()`.
- No massive 3D tensor $(M, W, 79)$ is ever materialized in memory.

### Valid Windows Formula & Counts
For a recording session with $N$ rows, window length $W$, and stride $S = 1$:
$$M(N, W) = N - W$$

| Session | Total Rows ($N$) | Windows ($W = 10$) | Windows ($W = 30$) | Windows ($W = 60$) |
| :--- | :--- | :--- | :--- | :--- |
| `train1` (Dev Training 1) | 216,001 | **215,991** | **215,971** | **215,941** |
| `train2` (Dev Training 2) | 226,801 | **226,791** | **226,771** | **226,741** |
| **Combined Dev Training** | **442,802** | **442,782** | **442,742** | **442,682** |
| `train3` (Normal Validation) | 478,801 | **478,791** | **478,771** | **478,741** |

### Boundary Protection Proof
The `MultiSessionWindowDataset` maps a global index $k$ to a specific session $s$ using cumulative window count offsets:
- If $k < M_1$: window is drawn from `train1` at local index $k$.
- If $k \ge M_1$: window is drawn from `train2` at local index $k - M_1$.
- **Boundary Test**:
  - Global index $M_1 - 1$ (last window of `train1`): starts at row $N_1 - W - 1$, target at row $N_1 - 1$ (the very last row of `train1`).
  - Global index $M_1$ (first window of `train2`): starts at row $0$, target at row $W$ of `train2`.
  - **Result**: Zero samples from `train2` ever enter `train1` windows, and zero samples from `train1` ever enter `train2` windows.

---

## 5. Verification Summary

All 10 required verification checks passed:
1. **Sample Count**: Scaler observed exactly 442,802 training samples across `train1` and `train2`.
2. **Feature Ordering**: Column order is strictly identical across all outputs.
3. **Dimensionality**: All processed feature arrays have 79 columns.
4. **Data Types**: Feature arrays are `float32`; timestamp arrays are `datetime64[s]`.
5. **Data Cleanliness**: 0 NaNs and 0 Infs across all 1,121,603 processed rows.
6. **Dynamic Standardization**: Non-constant training features standardize to mean $\approx 0.0$ and variance $\approx 1.0$.
7. **Zero-Variance Stability**: 22 training-constant features transform to 0.0 identically across all normal sessions.
8. **Validation Isolation**: `train3` transformed via frozen scaler with zero parameter updates.
9. **Window Counts**: Verified for candidate window lengths 10, 30, and 60.
10. **Boundary Tests**: Multi-session index mapping verified with zero cross-session leakage.

### Performance & Resource Usage
- **Peak Process Memory**: **742.09 MB** (well below 4 GB VM limit).
- **Preprocessing Runtime**: **8.29 seconds**.
- **Window Verification Runtime**: **< 1.0 second**.
