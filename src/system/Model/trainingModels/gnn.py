# src/system/Model/trainingModels/gnn.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    """
    Simple GCN-style layer:
        H_out = ReLU( A @ H_in @ W )
    where A is a fixed normalized adjacency.

    Input:
        h: [B, N, H_in]
        A: [N, N]
    Output:
        [B, N, H_out]
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # h: [B, N, H_in], A: [N, N]
        # batch-wise left-multiply by A
        Ah = torch.matmul(A, h)  # [B, N, H_in]
        return F.relu(self.lin(Ah))


class CrossAssetGNNModel(nn.Module):
    """
    MDGNN-style cross-asset temporal model (adapted to your system).

    High-level:
        For each sample:
          x: [B, N, W, F]  (from GraphSequenceDataset + collate)
            - N = #tickers (nodes)
            - W = lookback window (days)
            - F = feature dim

        1) Reinterpret x as a sequence of W daily graphs:
               x -> [B, W, N, F]

        2) For each day t in the window:
             - Per-node MLP to embed features -> H_t^0 [B, N, H]
             - L_gnn graph layers over fixed adjacency A:
                   H_t^{ℓ+1} = ReLU( A @ H_t^{ℓ} @ W_ℓ )

           This yields graph-aware daily embeddings H_t [B, N, H].

        3) Temporal Transformer (inter-day):
             - For each node u, we have a sequence over days:
                   {H_{u,0}, ..., H_{u,W-1}}.
             - Reshape to [B*N, W, H] and feed to a TransformerEncoder
               with a causal mask (no peeking into the future).
             - Take the last time step embedding as z_{u,t}.

        4) Linear head:
             z_{u,t} -> scalar prediction (e.g., 5d-ahead return).

    Notes:
        - No AR/GARCH priors are used here; those remain in the LSTM path.
        - The adjacency A is fixed for the run, built via build_mixed_adjacency
          in TrainingManager._run_gnn (blending return-corr + feature graphs).
        - We reuse the existing constructor signature so TrainingManager
          does not need structural changes.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        feature_dim: int,
        window: int,
        adj: torch.Tensor,
        temp_hidden: int = 32,
        gnn_hidden: int = 32,
        gnn_layers: int = 2,
    ) -> None:
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.feature_dim = int(feature_dim)
        self.window = int(window)

        if adj.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(
                f"CrossAssetGNNModel: adjacency shape {adj.shape} does not match "
                f"num_nodes={self.num_nodes}"
            )

        # fixed normalized adjacency (already row-normalized in builder)
        self.register_buffer("A", adj)

        # --- Intra-day graph encoder ---
        # First embed raw features into a hidden node space
        self.node_embed = nn.Linear(self.feature_dim, gnn_hidden)

        layers = []
        in_dim = gnn_hidden
        for _ in range(gnn_layers):
            layers.append(GraphLayer(in_dim, gnn_hidden))
            in_dim = gnn_hidden
        self.gnn_layers = nn.ModuleList(layers)

        # --- Inter-day temporal encoder (Transformer over days) ---
        # We use the same hidden width as the graph encoder output.
        d_model = gnn_hidden
        nhead = max(1, min(8, d_model // 8))  # simple, safe default
        self.nhead = nhead

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,  # we use [B*N, W, H]
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,  # can be tuned later if needed
        )

        # --- Output head (return-only) ---
        self.out = nn.Linear(d_model, 1)

    def _build_causal_mask(self, W: int, device: torch.device) -> torch.Tensor:
        """
        Build an upper-triangular boolean mask for causal attention:
            mask[i, j] = True  if j > i  (disallow attending to future)
        Shape: [W, W]
        """
        # True where we want to mask (future positions)
        mask = torch.triu(
            torch.ones(W, W, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, W, F]
        returns: [B, N] (per-node scalar prediction at anchor date)
        """
        if x.dim() != 4:
            raise ValueError(f"CrossAssetGNNModel: expected x [B,N,W,F], got {tuple(x.shape)}")

        B, N, W, Fdim = x.shape
        if N != self.num_nodes:
            raise ValueError(
                f"CrossAssetGNNModel: num_nodes mismatch, expected {self.num_nodes}, got {N}"
            )
        if W != self.window:
            raise ValueError(
                f"CrossAssetGNNModel: window mismatch, expected {self.window}, got {W}"
            )
        if Fdim != self.feature_dim:
            raise ValueError(
                f"CrossAssetGNNModel: feature_dim mismatch, expected {self.feature_dim}, got {Fdim}"
            )

        # Rearrange to [B, W, N, F] so dimension 1 is time
        x_bw = x.permute(0, 2, 1, 3)  # [B, W, N, F]

        # --- 1) Intra-day graph encoder for each day ---
        # Flatten batch and time so we can process all days at once:
        x_flat = x_bw.reshape(B * W, N, Fdim)  # [B*W, N, F]

        # Per-node feature embedding
        h = F.relu(self.node_embed(x_flat))  # [B*W, N, H]

        # GNN layers over fixed adjacency A
        A = self.A  # [N, N]
        for layer in self.gnn_layers:
            h = layer(h, A)  # [B*W, N, H]

        # Reshape back to [B, W, N, H]
        H = h.view(B, W, N, -1)  # [B, W, N, H]

        # --- 2) Inter-day temporal Transformer (per node) ---
        # For each node u, we want its sequence over days: [B, W, H].
        # We can batch this as [B*N, W, H] for the Transformer.
        H_bn = H.permute(0, 2, 1, 3).contiguous()  # [B, N, W, H]
        H_bn = H_bn.view(B * N, W, -1)  # [B*N, W, H]

        device = H_bn.device
        causal_mask = self._build_causal_mask(W, device)  # [W, W]

        # TransformerEncoder expects src_mask of shape [W, W]
        Z = self.temporal_encoder(H_bn, mask=causal_mask)  # [B*N, W, H]

        # Take the last time step as the representation for "today"
        Z_last = Z[:, -1, :]  # [B*N, H]

        # Reshape back to [B, N, H]
        Z_last = Z_last.view(B, N, -1)  # [B, N, H]

        # --- 3) Output head ---
        out = self.out(Z_last).squeeze(-1)  # [B, N]
        return out
