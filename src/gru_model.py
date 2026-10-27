#!/usr/bin/env python3
"""
Primary Model: One-Step Multivariate GRU Architecture for DT049
==============================================================
Fixed architecture:
- input_size = 79
- hidden_size = 64
- num_layers = 1
- dropout = 0.0 (recurrent dropout does not apply for 1-layer GRU)
- output = Linear(64, 79)
Total trainable parameters: 32,975 (~131.9 KB in float32)
"""

# CẤU TRÚC GRU: chuỗi 79 tín hiệu -> trạng thái ẩn 64 giá trị -> 79 giá trị dự đoán.
# Đầu ra là dự đoán tín hiệu kế tiếp, không phải nhãn Normal/Attack hay xác suất tấn công.

import torch
import torch.nn as nn


class OneStepGRU(nn.Module):
    """
    Compact 1-layer multivariate GRU predictor.
    Input: tensor of shape (batch_size, window_len, 79)
    Output: predicted state tensor of shape (batch_size, 79)
    """

    def __init__(self, input_size: int = 79, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # nn.GRU là lớp có sẵn của PyTorch, đọc tín hiệu theo thứ tự thời gian.
        # input_size=79: số tín hiệu mỗi giây; hidden_size=64: độ dài trạng thái ghi nhớ.
        # num_layers=1: một tầng GRU; batch_first đặt nhóm mẫu ở chiều đầu tiên.
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        # Linear chuyển 64 giá trị trạng thái ẩn thành dự đoán của đủ 79 tín hiệu.
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x có dạng (số mẫu trong nhóm, W thời điểm, 79 tín hiệu).
        # out chứa trạng thái theo từng thời điểm; h_n chứa trạng thái cuối của mỗi tầng.
        out, h_n = self.gru(x)
        # h_n có dạng (số tầng, số mẫu, 64); [-1] chọn tầng cuối, đã đọc hết cửa sổ.
        last_hidden = h_n[-1]
        # pred có dạng (số mẫu, 79): dự đoán một thời điểm tiếp theo cho mỗi mẫu.
        pred = self.fc(last_hidden)
        return pred

    def count_parameters(self) -> int:
        # Đếm các trọng số và độ lệch có thể học; cấu trúc mặc định có 32.975 tham số.
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
