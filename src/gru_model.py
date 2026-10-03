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

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)
        # h_n shape: (num_layers=1, batch_size, hidden_size)
        last_hidden = h_n[-1]
        pred = self.fc(last_hidden)
        return pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
