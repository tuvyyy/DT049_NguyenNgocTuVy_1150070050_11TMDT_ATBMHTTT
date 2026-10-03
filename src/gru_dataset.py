#!/usr/bin/env python3
"""
Lazy Window Dataset with Session Boundary Isolation for DT049
============================================================
Extracts sliding windows strictly inside session boundaries:
Input: X[i : i+W] of shape (W, 79)
Target: X[i+W] of shape (79,)
No window spans across file boundaries.
"""

from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class SessionWindowDataset(Dataset):
    """
    Lazy PyTorch Dataset for session-bounded time-series windows with stride.
    """

    def __init__(self, session_arrays: List[np.ndarray], window_size: int, stride: int = 1):
        self.session_arrays = session_arrays
        self.window_size = window_size
        self.stride = stride

        # Precompute valid start indices strictly within each session
        self.indices: List[Tuple[int, int]] = []
        for session_idx, arr in enumerate(session_arrays):
            N = len(arr)
            # Target is index (start + window_size) <= N - 1
            # Therefore start <= N - 1 - window_size, i.e., range(0, N - window_size, stride)
            for start in range(0, N - window_size, stride):
                self.indices.append((session_idx, start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        session_idx, start = self.indices[idx]
        arr = self.session_arrays[session_idx]
        x = arr[start : start + self.window_size]
        y = arr[start + self.window_size]
        # x.copy() and y.copy() prevent PyTorch non-writable tensor warnings on memmaps
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    def get_target_timestamps(self, session_timestamps: List[np.ndarray]) -> np.ndarray:
        """
        Extracts target timestamps aligned 1:1 with the dataset items.
        """
        target_ts = []
        for session_idx, start in self.indices:
            target_ts.append(session_timestamps[session_idx][start + self.window_size])
        return np.array(target_ts)
