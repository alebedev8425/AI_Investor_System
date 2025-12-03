# src/system/Model/trainingModels/lstm.py
from __future__ import annotations

import torch
from torch import nn


class LstmModel(nn.Module):
    """
    GINN-style LSTM:
      Input:  [B, T, F]
      Outputs:
        - ret_pred: [B] (normalized return target)
        - vol_pred: [B] (normalized log-vol / variance target)
    """

    def __init__(self, input_size: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.ret_head = nn.Linear(hidden, 1)
        self.vol_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, (h, _) = self.lstm(x)    # h: [num_layers, B, hidden]
        last_h = h[-1]                # [B, hidden]
        ret = self.ret_head(last_h).squeeze(-1)  # [B]
        vol = self.vol_head(last_h).squeeze(-1)  # [B]
        return ret, vol