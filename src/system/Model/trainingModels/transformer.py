from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


# ---------------- Transformer model config ----------------

@dataclass(frozen=True)
class TransformerConfig:
    """
    Low-level model config (built from high-level TransformerModelConfig
    in experiment_config.py).
    """
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    pooling: str = "attn"  # "attn" | "mean" | "last"


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Expects input of shape [B, T, D].
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerModel(nn.Module):
    """
    MASTER-style sequence-to-scalar Transformer model.

    Key ideas adapted to your system:
      - Input projection: per-timestep features -> d_model
      - Positional encoding over time
      - Market-status encoder: sequence-level "status" vector
      - Status-aware gating: modulates features using status
      - Temporal Transformer encoder
      - Attention pooling over time (default)
      - MLP head -> scalar 5d return (normalized)

    Input:
      x: [B, T, F]  (F = MASTER feature set: technical + corr (+ graph/sent/events later))
    Output:
      y_hat: [B]
    """

    def __init__(
        self,
        input_size: int,
        cfg: TransformerConfig | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = TransformerConfig()

        self.cfg = cfg

        # 1) Input projection
        self.input_proj = nn.Linear(input_size, cfg.d_model)

        # 2) Positional encoding
        self.pos_encoder = PositionalEncoding(cfg.d_model)

        # 3) Market-status encoder
        #    Here we approximate "market status" as a learned transform
        #    of the temporal mean representation. Later, you can extend
        #    this to also ingest explicit market-wide features.
        self.market_status_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.Tanh(),  # bounded status
        )

        # 4) Status-aware gating (market-guided modulation)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.Sigmoid(),
        )

        # 5) Temporal Transformer encoder (MASTER temporal attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,  # [B, T, D]
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )

        # 6) Temporal attention pooling
        self.attn_pool = nn.Linear(cfg.d_model, 1)

        # 7) Prediction head
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, 1),
        )

    def _temporal_pool(self, h: Tensor) -> Tensor:
        """
        Pool encoder outputs over time.

        h: [B, T, D]
        returns: [B, D]
        """
        if self.cfg.pooling == "mean":
            return h.mean(dim=1)
        if self.cfg.pooling == "last":
            return h[:, -1, :]

        # default: attention pooling
        attn_scores = self.attn_pool(h).squeeze(-1)  # [B, T]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T]
        pooled = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        return pooled

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, T, F]
        """
        if x.dim() != 3:
            raise ValueError(f"TransformerModel expected x [B,T,F], got {tuple(x.shape)}")

        # 1) Input projection
        h = self.input_proj(x)          # [B, T, D]

        # 2) Positional encoding
        h = self.pos_encoder(h)         # [B, T, D]

        # 3) Sequence summary -> market status
        seq_summary = h.mean(dim=1)     # [B, D]
        status = self.market_status_mlp(seq_summary)  # [B, D]

        # 4) Status-aware gating
        status_expanded = status.unsqueeze(1).expand_as(h)      # [B, T, D]
        gate_in = torch.cat([h, status_expanded], dim=-1)       # [B, T, 2D]
        gate = self.gate_mlp(gate_in)                          # [B, T, D]
        h = h * gate                                           # [B, T, D]

        # 5) Temporal Transformer encoder
        h = self.encoder(h)                                    # [B, T, D]

        # 6) Temporal pooling
        pooled = self._temporal_pool(h)                        # [B, D]

        # 7) Prediction head
        y = self.head(pooled).squeeze(-1)                      # [B]
        return y


