from __future__ import annotations

import torch
from torch import nn


class LstmModel(nn.Module):
    """
    Tiny LSTM + Linear head suitable for Phase-1 baseline.
    Input:  [B, T, F]
    Output: [B, 1] (pred_5d)
    """

    def __init__(self, input_size: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, _) = self.lstm(x)  # h: [num_layers, B, hidden]
        last_h = h[-1]              # [B, hidden]
        return self.head(last_h)    # [B, 1]