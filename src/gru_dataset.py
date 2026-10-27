#!/usr/bin/env python3
"""
Lazy Window Dataset with Session Boundary Isolation for DT049
============================================================
Extracts sliding windows strictly inside session boundaries:
Input: X[i : i+W] of shape (W, 79)
Target: X[i+W] of shape (79,)
No window spans across file boundaries.
"""

# DỮ LIỆU VÀO: mỗi dòng là 79 giá trị tín hiệu đã chuẩn hóa tại một thời điểm.
# Một mẫu học gồm W dòng quá khứ để dự đoán 79 giá trị ở dòng kế tiếp.
# Với W=30 và dữ liệu cách nhau 1 giây: đầu vào (30, 79), giá trị thật cần đoán (79,).

from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class SessionWindowDataset(Dataset):
    """
    Lazy PyTorch Dataset for session-bounded time-series windows with stride.
    """

    def __init__(self, session_arrays: List[np.ndarray], window_size: int, stride: int = 1):
        # Mỗi phần tử là mảng (số thời điểm, 79) của một phiên, ví dụ train1 hoặc train2.
        self.session_arrays = session_arrays
        # W là số thời điểm quá khứ trong một cửa sổ, không phải số tín hiệu.
        self.window_size = window_size
        # stride là bước dịch cửa sổ: học dùng 5 mẫu, đánh giá từng giây dùng 1 mẫu.
        self.stride = stride

        # Chỉ lưu (số phiên, vị trí bắt đầu); chưa sao chép toàn bộ các cửa sổ vào RAM.
        self.indices: List[Tuple[int, int]] = []
        for session_idx, arr in enumerate(session_arrays):
            N = len(arr)
            # Cần còn dòng start + W để làm giá trị thật; cửa sổ không nối qua hai phiên.
            for start in range(0, N - window_size, stride):
                self.indices.append((session_idx, start))

    def __len__(self) -> int:
        # Trả số cửa sổ hợp lệ, không phải tổng số dòng dữ liệu gốc.
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        session_idx, start = self.indices[idx]
        arr = self.session_arrays[session_idx]
        # x: W thời điểm liên tiếp, mỗi thời điểm đủ 79 tín hiệu; không có nhãn tấn công.
        x = arr[start : start + self.window_size]
        # y: 79 giá trị thật ngay sau cửa sổ, dùng so với dự đoán để tính lỗi.
        y = arr[start + self.window_size]
        # Đổi sang tensor PyTorch; copy tránh dùng trực tiếp bộ nhớ của file chỉ đọc.
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    def get_target_timestamps(self, session_timestamps: List[np.ndarray]) -> np.ndarray:
        """
        Extracts target timestamps aligned 1:1 with the dataset items.
        """
        target_ts = []
        for session_idx, start in self.indices:
            # Gắn dự đoán với giờ của dòng cần đoán, không lấy giờ bắt đầu cửa sổ.
            target_ts.append(session_timestamps[session_idx][start + self.window_size])
        return np.array(target_ts)
