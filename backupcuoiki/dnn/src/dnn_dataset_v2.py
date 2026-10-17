#!/usr/bin/env python3
"""
DNN V2 Pointwise Data Pipeline for DT049 (HAI 21.03)
====================================================
Implements the leakage-free data loading pipeline for Experiment V2
(Supervised Binary Feedforward DNN adapted from Kuchar & Fujdiak, 2025,
DOI: 10.1016/j.cose.2025.104395).

Key guarantees:
1. Reads pre-scaled feature arrays (`testN_features.npy`), binary labels
   (`testN_attack.npy`), and timestamps (`testN_timestamps.npy`) using
   read-only memory mapping (`mmap_mode="r"`) for < 4 GB RAM safety on Kali CPU.
2. Uses the existing frozen StandardScaler (`models/preprocessing/scaler.joblib`,
   fit on `train1 + train2` = 442,802 normal rows) without refitting.
3. Explicitly excludes `time`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`
   from the 79 model input features.
4. Enforces strict session disjointness:
   - DNN Train:      test1 + test2 (162,002 rows; 4,078 attack rows)
   - DNN Validation: test3         (108,001 rows; 1,535 attack rows)
   - DNN Eval V2:    test4 + test5 (132,002 rows; 3,334 attack rows)
5. Computes class weights exclusively from DNN Train (`test1 + test2`).
"""

from bisect import bisect_right
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dnn_v2_protocol.yaml"

FORBIDDEN_FEATURE_COLUMNS = frozenset({
    "time",
    "attack",
    "attack_P1",
    "attack_P2",
    "attack_P3",
})

CANONICAL_TRAIN_SESSIONS = ("test1", "test2")
CANONICAL_VAL_SESSIONS = ("test3",)
CANONICAL_EVAL_SESSIONS = ("test4", "test5")


def load_v2_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict:
    """Loads the V2 protocol YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_frozen_scaler_and_features(
    scaler_path: Path,
    scaler_metadata_path: Path,
    expected_feature_count: int = 79,
    expected_fit_sample_count: int = 442802,
) -> Dict:
    """
    Verifies the frozen V1 StandardScaler and feature metadata without refitting.
    Ensures that `time`, `attack`, `attack_P1`, `attack_P2`, and `attack_P3`
    are strictly excluded from the 79 input features.
    """
    if not scaler_path.exists():
        raise FileNotFoundError(f"Frozen scaler not found: {scaler_path}")
    if not scaler_metadata_path.exists():
        raise FileNotFoundError(f"Scaler metadata not found: {scaler_metadata_path}")

    with open(scaler_metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scaler = joblib.load(scaler_path)

    ordered_features = meta.get("ordered_feature_names", [])
    if len(ordered_features) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} features in metadata, found {len(ordered_features)}"
        )

    if int(getattr(scaler, "n_features_in_", -1)) != expected_feature_count:
        raise ValueError(
            f"Scaler n_features_in_={getattr(scaler, 'n_features_in_', None)} != {expected_feature_count}"
        )

    fit_count_meta = int(meta.get("fit_sample_count", -1))
    fit_count_scaler = int(getattr(scaler, "n_samples_seen_", -1))
    if fit_count_meta != expected_fit_sample_count or fit_count_scaler != expected_fit_sample_count:
        raise ValueError(
            f"Scaler fit sample count mismatch: meta={fit_count_meta}, scaler={fit_count_scaler}, "
            f"expected={expected_fit_sample_count}"
        )

    leaked_columns = FORBIDDEN_FEATURE_COLUMNS.intersection(ordered_features)
    if leaked_columns:
        raise ValueError(f"Forbidden label/time columns found in feature list: {sorted(leaked_columns)}")

    return {
        "feature_count": len(ordered_features),
        "ordered_feature_names": ordered_features,
        "fit_sample_count": fit_count_meta,
        "fit_source_files": meta.get("fit_source_files", []),
        "zero_variance_training_features_count": meta.get("zero_variance_training_features_count", 0),
        "dynamic_training_features_count": meta.get("dynamic_training_features_count", 0),
    }


def verify_split_disjointness(
    train_sessions: Sequence[str],
    val_sessions: Sequence[str],
    eval_sessions: Sequence[str],
) -> None:
    """
    Ensures Train, Validation, and Evaluation session lists are non-empty and
    strictly pairwise disjoint.
    """
    train_set = set(train_sessions)
    val_set = set(val_sessions)
    eval_set = set(eval_sessions)

    if not train_set or not val_set or not eval_set:
        raise ValueError("Train, Validation, and Evaluation session lists must all be non-empty.")

    if len(train_set) != len(train_sessions) or len(val_set) != len(val_sessions) or len(eval_set) != len(eval_sessions):
        raise ValueError("Duplicate session names detected within a single split.")

    overlap_tv = train_set.intersection(val_set)
    overlap_te = train_set.intersection(eval_set)
    overlap_ve = val_set.intersection(eval_set)

    if overlap_tv or overlap_te or overlap_ve:
        raise ValueError(
            f"Split leakage detected! Overlaps: train&val={sorted(overlap_tv)}, "
            f"train&eval={sorted(overlap_te)}, val&eval={sorted(overlap_ve)}"
        )


class SessionPointwiseDNNDatasetV2(Dataset):
    """
    Memory-mapped PyTorch Dataset for pointwise Feedforward DNN (Experiment V2).

    Reads `testN_features.npy`, `testN_attack.npy`, and `testN_timestamps.npy`
    using `mmap_mode="r"` so that underlying files cannot be mutated and RAM
    footprint remains minimal (< 4 GB).

    Parameters
    ----------
    sessions : Sequence[str]
        Ordered session names (e.g. `["test1", "test2"]`).
    processed_test_dir : Path
        Path to `data/processed/hai_21_03/final_test`.
    start_index : int, default=0
        Starting row index within each session (`0` for full session rows;
        `30` when aligning with the V1 common evaluation region `t >= 30`).
    expected_feature_dim : int, default=79
        Expected number of input features per row.
    """

    def __init__(
        self,
        sessions: Sequence[str],
        processed_test_dir: Path,
        start_index: int = 0,
        expected_feature_dim: int = 79,
    ):
        super().__init__()
        if not sessions:
            raise ValueError("At least one session must be provided.")
        if start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {start_index}")

        self.sessions: Tuple[str, ...] = tuple(sessions)
        self.processed_test_dir = Path(processed_test_dir)
        self.start_index = int(start_index)
        self.expected_feature_dim = int(expected_feature_dim)

        self.feature_arrays: List[np.ndarray] = []
        self.label_arrays: List[np.ndarray] = []
        self.timestamp_arrays: List[np.ndarray] = []
        self.session_lengths: List[int] = []
        self.cum_lengths: List[int] = []

        running_total = 0
        for sname in self.sessions:
            feat_path = self.processed_test_dir / f"{sname}_features.npy"
            atk_path = self.processed_test_dir / f"{sname}_attack.npy"
            ts_path = self.processed_test_dir / f"{sname}_timestamps.npy"

            for p in (feat_path, atk_path, ts_path):
                if not p.exists():
                    raise FileNotFoundError(f"Missing required processed array for {sname}: {p}")

            feats = np.load(feat_path, mmap_mode="r")
            atk = np.load(atk_path, mmap_mode="r")
            ts = np.load(ts_path, mmap_mode="r")

            # 1. Verify read-only protection
            if feats.flags.writeable or atk.flags.writeable or ts.flags.writeable:
                raise RuntimeError(f"Arrays for {sname} must be read-only (mmap_mode='r').")

            # 2. Verify shapes and row alignment
            if feats.ndim != 2 or feats.shape[1] != self.expected_feature_dim:
                raise ValueError(
                    f"[{sname}] Expected 2D features with {self.expected_feature_dim} columns, "
                    f"got shape {feats.shape}"
                )
            n_rows = feats.shape[0]
            if atk.shape != (n_rows,) or ts.shape != (n_rows,):
                raise ValueError(
                    f"[{sname}] Row alignment mismatch: features={feats.shape}, "
                    f"attack={atk.shape}, timestamps={ts.shape}"
                )
            if n_rows <= self.start_index:
                raise ValueError(
                    f"[{sname}] Session length {n_rows} <= start_index {self.start_index}"
                )

            # 3. Verify dtypes and binary label domain {0, 1}
            if feats.dtype != np.float32:
                raise ValueError(f"[{sname}] Expected float32 features, got {feats.dtype}")
            unique_labels = set(np.unique(atk).tolist())
            if not unique_labels.issubset({0, 1}):
                raise ValueError(
                    f"[{sname}] Target label `attack` contains invalid values outside {{0, 1}}: "
                    f"{sorted(unique_labels)}"
                )

            self.feature_arrays.append(feats)
            self.label_arrays.append(atk)
            self.timestamp_arrays.append(ts)

            eff_len = n_rows - self.start_index
            self.session_lengths.append(eff_len)
            running_total += eff_len
            self.cum_lengths.append(running_total)

        self.total_samples = running_total

    def __len__(self) -> int:
        return self.total_samples

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Maps a global dataset index to (session_idx, row_idx_in_session)."""
        if idx < 0:
            idx += self.total_samples
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.total_samples}")

        session_idx = bisect_right(self.cum_lengths, idx)
        prev_cum = 0 if session_idx == 0 else self.cum_lengths[session_idx - 1]
        local_offset = idx - prev_cum
        row_idx = self.start_index + local_offset
        return session_idx, row_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        session_idx, row_idx = self._locate(idx)
        # Copy single row (79 floats = 316 bytes) from read-only memmap to writable tensor
        x_row = np.array(self.feature_arrays[session_idx][row_idx], dtype=np.float32, copy=True)
        y_val = np.float32(self.label_arrays[session_idx][row_idx])
        return torch.from_numpy(x_row), torch.tensor(y_val, dtype=torch.float32)

    def get_session_summary(self) -> List[Dict]:
        """Returns per-session row and binary label statistics within the active index range."""
        summaries = []
        for s_idx, sname in enumerate(self.sessions):
            atk_slice = self.label_arrays[s_idx][self.start_index :]
            ts_slice = self.timestamp_arrays[s_idx][self.start_index :]
            n_total = int(len(atk_slice))
            n_attack = int(np.sum(atk_slice == 1))
            n_normal = int(n_total - n_attack)
            summaries.append({
                "session": sname,
                "start_index": self.start_index,
                "total_rows": n_total,
                "normal_rows": n_normal,
                "attack_rows": n_attack,
                "attack_ratio": round(n_attack / n_total, 8) if n_total > 0 else 0.0,
                "first_timestamp": str(ts_slice[0]),
                "last_timestamp": str(ts_slice[-1]),
            })
        return summaries

    def get_aggregate_label_distribution(self) -> Dict:
        """Returns aggregate label counts across all sessions in this dataset."""
        per_session = self.get_session_summary()
        total_rows = sum(s["total_rows"] for s in per_session)
        normal_rows = sum(s["normal_rows"] for s in per_session)
        attack_rows = sum(s["attack_rows"] for s in per_session)
        return {
            "sessions": list(self.sessions),
            "start_index": self.start_index,
            "total_rows": total_rows,
            "normal_rows": normal_rows,
            "attack_rows": attack_rows,
            "normal_ratio": round(normal_rows / total_rows, 8) if total_rows > 0 else 0.0,
            "attack_ratio": round(attack_rows / total_rows, 8) if total_rows > 0 else 0.0,
            "per_session": per_session,
        }


def compute_train_class_weights(train_dataset: SessionPointwiseDNNDatasetV2) -> Dict:
    """
    Computes binary class weights strictly from the DNN V2 Train split (`test1 + test2`).
    Raises ValueError if any validation (`test3`) or evaluation (`test4`, `test5`)
    session is present in `train_dataset`.
    """
    disallowed = set(train_dataset.sessions) - set(CANONICAL_TRAIN_SESSIONS)
    if disallowed:
        raise ValueError(
            f"Leakage violation: class weights can ONLY be computed on Train sessions "
            f"{CANONICAL_TRAIN_SESSIONS}, but got {train_dataset.sessions} "
            f"(disallowed: {sorted(disallowed)})."
        )

    dist = train_dataset.get_aggregate_label_distribution()
    n_total = dist["total_rows"]
    n_normal = dist["normal_rows"]
    n_attack = dist["attack_rows"]

    if n_attack == 0 or n_normal == 0:
        raise ValueError(f"Degenerate class distribution in Train split: normal={n_normal}, attack={n_attack}")

    pos_weight_balanced = float(n_normal / n_attack)
    pos_weight_sqrt = float(np.sqrt(pos_weight_balanced))
    weight_class_0 = float(n_total / (2.0 * n_normal))
    weight_class_1 = float(n_total / (2.0 * n_attack))

    return {
        "source_sessions": list(train_dataset.sessions),
        "start_index": train_dataset.start_index,
        "total_samples": n_total,
        "normal_samples_class0": n_normal,
        "attack_samples_class1": n_attack,
        "attack_prevalence": round(n_attack / n_total, 8),
        "pos_weight_balanced": round(pos_weight_balanced, 6),
        "pos_weight_sqrt": round(pos_weight_sqrt, 6),
        "inverse_frequency_weights": {
            "class_0_normal": round(weight_class_0, 6),
            "class_1_attack": round(weight_class_1, 6),
        },
    }


def build_v2_dataloaders(
    config_path: Path = DEFAULT_CONFIG_PATH,
    batch_size: int = 512,
    start_index: int = 0,
    include_eval: bool = False,
) -> Dict:
    """
    Constructs leakage-checked Train and Validation datasets and DataLoaders for V2.
    By default, `include_eval=False` prevents touching `test4 + test5` during
    training and hyperparameter/threshold selection.
    """
    cfg = load_v2_config(config_path)

    processed_test_dir = PROJECT_ROOT / cfg["dataset"]["processed_test_dir"]
    scaler_path = PROJECT_ROOT / cfg["dataset"]["scaler_path"]
    scaler_meta_path = PROJECT_ROOT / cfg["dataset"]["scaler_metadata_path"]
    input_dim = int(cfg["features"]["input_dim"])

    scaler_info = verify_frozen_scaler_and_features(
        scaler_path=scaler_path,
        scaler_metadata_path=scaler_meta_path,
        expected_feature_count=input_dim,
        expected_fit_sample_count=int(cfg["dataset"]["scaler_fit_source"]["total_rows"]),
    )

    train_sessions = cfg["splits"]["dnn_train"]["sessions"]
    val_sessions = cfg["splits"]["dnn_validation"]["sessions"]
    eval_sessions = cfg["splits"]["dnn_evaluation_v2"]["sessions"]

    verify_split_disjointness(train_sessions, val_sessions, eval_sessions)

    train_ds = SessionPointwiseDNNDatasetV2(
        sessions=train_sessions,
        processed_test_dir=processed_test_dir,
        start_index=start_index,
        expected_feature_dim=input_dim,
    )
    val_ds = SessionPointwiseDNNDatasetV2(
        sessions=val_sessions,
        processed_test_dir=processed_test_dir,
        start_index=start_index,
        expected_feature_dim=input_dim,
    )

    class_weights = compute_train_class_weights(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    result = {
        "config": cfg,
        "scaler_verification": scaler_info,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_class_weights": class_weights,
    }

    if include_eval:
        eval_ds = SessionPointwiseDNNDatasetV2(
            sessions=eval_sessions,
            processed_test_dir=processed_test_dir,
            start_index=start_index,
            expected_feature_dim=input_dim,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        result["eval_dataset"] = eval_ds
        result["eval_loader"] = eval_loader

    return result
