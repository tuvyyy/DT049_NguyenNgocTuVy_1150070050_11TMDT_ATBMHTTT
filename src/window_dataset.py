#!/usr/bin/env python3
"""
HAI 21.03 Window Dataset Infrastructure for DT049
=================================================
Provides framework-independent, memory-safe, lazy sequence-to-next-step windowing.

Key design guarantees:
1. Lazy evaluation: Opens processed arrays via np.load(path, mmap_mode="r").
   Windows are sliced dynamically on-the-fly without materializing a massive 3D tensor.
2. Boundary protection: Windows NEVER cross recording session boundaries.
3. Multi-session mapping: MultiSessionWindowDataset maps global indices to (session_idx, local_idx)
   using cumulative index offsets.
4. Input sequence: X[t - W + 1 : t + 1] of shape (W, 79)
   Target step:    X[t + 1] of shape (79,)
   Valid count:    N - W for a session with N rows, window size W, and stride 1.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "hai_21_03"


class LazyWindowDataset:
    """
    Single-session memory-mapped lazy window dataset for sequence-to-next-step prediction.
    """
    def __init__(
        self,
        features: Union[str, Path, np.ndarray],
        timestamps: Optional[Union[str, Path, np.ndarray]] = None,
        window_size: int = 30,
        stride: int = 1,
    ):
        self.window_size = int(window_size)
        self.stride = int(stride)

        if isinstance(features, (str, Path)):
            self.features = np.load(features, mmap_mode="r")
        else:
            self.features = features

        self.timestamps = None
        if timestamps is not None:
            if isinstance(timestamps, (str, Path)):
                self.timestamps = np.load(timestamps, mmap_mode="r")
            else:
                self.timestamps = timestamps

        self.total_rows, self.num_features = self.features.shape

        # Sequence-to-next-step requires W input steps + 1 target step
        # Maximum start index: i_max + W = total_rows - 1  =>  i_max = total_rows - W - 1
        # Number of valid windows with stride 1 is: total_rows - window_size
        if self.total_rows <= self.window_size:
            self.num_windows = 0
        else:
            self.num_windows = (self.total_rows - self.window_size - 1) // self.stride + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_windows} windows.")

        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        target_idx = end_idx

        # Slice view (zero memory copy until converted)
        window_x = self.features[start_idx:end_idx]
        target_y = self.features[target_idx]

        return window_x, target_y

    def get_window_indices(self, idx: int) -> Tuple[int, int, int]:
        """Returns (start_row_idx, end_row_idx, target_row_idx) for index verification."""
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(f"Index {idx} out of range.")
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        return start_idx, end_idx, end_idx


class MultiSessionWindowDataset:
    """
    Multi-session window dataset that wraps multiple independent recording sessions.
    Guarantees that windows NEVER cross between session boundaries.
    """
    def __init__(
        self,
        feature_paths: List[Union[str, Path]],
        timestamp_paths: Optional[List[Union[str, Path]]] = None,
        window_size: int = 30,
        stride: int = 1,
    ):
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.sessions = []
        self.session_lengths = []

        ts_paths = timestamp_paths if timestamp_paths else [None] * len(feature_paths)

        for fp, tp in zip(feature_paths, ts_paths):
            ds = LazyWindowDataset(features=fp, timestamps=tp, window_size=window_size, stride=stride)
            self.sessions.append(ds)
            self.session_lengths.append(len(ds))

        self.cumulative_lengths = np.cumsum([0] + self.session_lengths)
        self.total_windows = int(self.cumulative_lengths[-1])

    def __len__(self) -> int:
        return self.total_windows

    def _locate_session(self, idx: int) -> Tuple[int, int]:
        """Maps global window index to (session_idx, local_session_idx)."""
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Global index {idx} out of bounds for {self.total_windows} windows.")
        # Binary search for session index
        session_idx = int(np.searchsorted(self.cumulative_lengths[1:], idx, side="right"))
        local_idx = idx - self.cumulative_lengths[session_idx]
        return session_idx, local_idx

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        session_idx, local_idx = self._locate_session(idx)
        return self.sessions[session_idx][local_idx]

    def get_window_metadata(self, idx: int) -> dict:
        """Returns audit metadata verifying session boundary isolation."""
        session_idx, local_idx = self._locate_session(idx)
        ds = self.sessions[session_idx]
        start_idx, end_idx, target_idx = ds.get_window_indices(local_idx)
        return {
            "global_index": idx,
            "session_index": session_idx,
            "local_index": local_idx,
            "session_total_rows": ds.total_rows,
            "start_row": start_idx,
            "end_row": end_idx,
            "target_row": target_idx,
        }


def run_window_verification_tests():
    """
    Executes all required verification tests for windowing infrastructure:
    1. Valid window counts for W in {10, 30, 60}
    2. Explicit index boundary tests for train1, train2, train3
    3. Multi-session boundary isolation verification
    """
    print("=================================================================")
    print("Running Window Dataset Infrastructure Verification Tests")
    print("=================================================================")

    t1_path = PROCESSED_DIR / "train1_features.npy"
    t2_path = PROCESSED_DIR / "train2_features.npy"
    t3_path = PROCESSED_DIR / "train3_features.npy"

    if not t1_path.exists() or not t2_path.exists() or not t3_path.exists():
        print(f"ERROR: Processed feature arrays not found in {PROCESSED_DIR}. Run src/preprocess.py first.")
        sys.exit(1)

    t1_rows = np.load(t1_path, mmap_mode="r").shape[0]  # 216001
    t2_rows = np.load(t2_path, mmap_mode="r").shape[0]  # 226801
    t3_rows = np.load(t3_path, mmap_mode="r").shape[0]  # 478801

    candidate_windows = [10, 30, 60]

    for W in candidate_windows:
        print(f"\n--- Testing Window Size W = {W} ---")
        ds_t1 = LazyWindowDataset(t1_path, window_size=W)
        ds_t2 = LazyWindowDataset(t2_path, window_size=W)
        ds_t3 = LazyWindowDataset(t3_path, window_size=W)
        ds_train = MultiSessionWindowDataset([t1_path, t2_path], window_size=W)

        # Expected counts
        exp_t1 = t1_rows - W
        exp_t2 = t2_rows - W
        exp_t3 = t3_rows - W
        exp_train = exp_t1 + exp_t2

        assert len(ds_t1) == exp_t1, f"train1 window count mismatch: {len(ds_t1)} != {exp_t1}"
        assert len(ds_t2) == exp_t2, f"train2 window count mismatch: {len(ds_t2)} != {exp_t2}"
        assert len(ds_t3) == exp_t3, f"train3 window count mismatch: {len(ds_t3)} != {exp_t3}"
        assert len(ds_train) == exp_train, f"multi-session train window count mismatch: {len(ds_train)} != {exp_train}"

        print(f"PASSED Window Counts: train1={len(ds_t1):,}, train2={len(ds_t2):,}, DevTrain={len(ds_train):,}, train3={len(ds_t3):,}")

        # Test first window of train1
        w0_x, w0_y = ds_t1[0]
        meta0 = ds_t1.get_window_indices(0)
        assert meta0 == (0, W, W), f"train1 window 0 index error: {meta0}"
        assert w0_x.shape == (W, 79), f"train1 window 0 shape error: {w0_x.shape}"
        assert w0_y.shape == (79,), f"train1 target 0 shape error: {w0_y.shape}"

        # Test last window of train1
        last_idx = len(ds_t1) - 1
        w_last_x, w_last_y = ds_t1[last_idx]
        meta_last = ds_t1.get_window_indices(last_idx)
        assert meta_last == (t1_rows - W - 1, t1_rows - 1, t1_rows - 1), f"train1 last window index error: {meta_last}"
        print(f"PASSED train1 boundary: Window 0 starts at row 0, target at row {W}. Last window starts at row {t1_rows - W - 1}, target at row {t1_rows - 1}.")

        # Multi-session boundary isolation tests
        # Index exp_t1 - 1 is the last window of session 0 (train1)
        meta_s0_last = ds_train.get_window_metadata(exp_t1 - 1)
        assert meta_s0_last["session_index"] == 0, "Last window of session 0 wrongly mapped!"
        assert meta_s0_last["target_row"] == t1_rows - 1, f"Session 0 last target row not {t1_rows - 1}"

        # Index exp_t1 is the first window of session 1 (train2)
        meta_s1_first = ds_train.get_window_metadata(exp_t1)
        assert meta_s1_first["session_index"] == 1, "First window of session 1 wrongly mapped!"
        assert meta_s1_first["start_row"] == 0, "Session 1 first window does not start at row 0!"
        assert meta_s1_first["target_row"] == W, f"Session 1 first target row not {W}"

        print(f"PASSED Session Isolation: Global index {exp_t1 - 1:,} strictly ends in session 0 (target row {t1_rows - 1}); global index {exp_t1:,} strictly begins in session 1 (start row 0). ZERO cross-session windowing.")

    print("\n=================================================================")
    print("ALL WINDOW DATASET VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    run_window_verification_tests()
