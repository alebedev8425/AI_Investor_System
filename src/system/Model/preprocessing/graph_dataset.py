# src/system/Model/preprocessing/graph_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GraphSequenceConfig:
    window: int          # lookback window in days
    target_col: str      # e.g. "target_5d"
    min_coverage: float = 0.5  # require at least this frac of non-NaN in a node's window


class GraphSequenceDataset(Dataset):
    """
    Cross-asset temporal graph dataset.

    Each sample corresponds to one "anchor" date:

        X: [N, W, F]  node features
           N = #tickers, W = window, F = #features
        y: [N]        node targets for that date (e.g. 5d-ahead return)
        mask: [N]     bool, True where target is valid for that node
        date: pd.Timestamp

    The graph structure (adjacency) is fixed for the run and passed separately
    into the GNN model; it is NOT per-sample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        feature_cols: List[str],
        cfg: GraphSequenceConfig,
    ) -> None:
        super().__init__()

        if not {"date", "ticker"}.issubset(df.columns):
            raise ValueError("GraphSequenceDataset: df must contain 'date' and 'ticker'.")

        self.tickers: List[str] = [str(t).upper() for t in tickers]
        self.ticker_to_idx: Dict[str, int] = {t: i for i, t in enumerate(self.tickers)}
        self.feature_cols = list(feature_cols)
        self.cfg = cfg

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        data["ticker"] = data["ticker"].astype(str).str.upper()

        if self.cfg.target_col not in data.columns:
            raise ValueError(f"GraphSequenceDataset: missing target column '{self.cfg.target_col}'")

        # --- dense panel [T, N, F] and [T, N] ---
        dates = data["date"].drop_duplicates().sort_values().to_list()
        self.dates = dates
        date_to_idx = {d: i for i, d in enumerate(dates)}

        N = len(self.tickers)
        T = len(dates)
        F = len(self.feature_cols)

        X = np.full((T, N, F), np.nan, dtype=np.float32)
        Y = np.full((T, N), np.nan, dtype=np.float32)

        # fill from long df
        for row in data.itertuples(index=False):
            d = getattr(row, "date")
            t = getattr(row, "ticker")
            if d not in date_to_idx:
                continue
            if t not in self.ticker_to_idx:
                continue
            di = date_to_idx[d]
            ti = self.ticker_to_idx[t]

            for j, col in enumerate(self.feature_cols):
                val = getattr(row, col, np.nan)
                if pd.notna(val):
                    X[di, ti, j] = np.float32(val)

            yv = getattr(row, self.cfg.target_col, np.nan)
            if pd.notna(yv):
                Y[di, ti] = np.float32(yv)

        # --- build samples ---
        W = int(self.cfg.window)
        samples_X: List[np.ndarray] = []
        samples_Y: List[np.ndarray] = []
        samples_mask: List[np.ndarray] = []
        samples_dates: List[pd.Timestamp] = []

        for i in range(W, T):
            end_date = dates[i]

            # lookback [i-W, ..., i-1]
            win_X = X[i - W : i, :, :]           # [W, N, F]
            node_X = np.transpose(win_X, (1, 0, 2))  # [N, W, F]

            node_y = Y[i, :]                     # [N]

            # coverage per node
            feat_notnan = np.isfinite(node_X).sum(axis=(1, 2))
            max_feat = float(W * F) if F > 0 else 1.0
            coverage = feat_notnan / max_feat

            has_target = np.isfinite(node_y)
            mask = (coverage >= self.cfg.min_coverage) & has_target

            if not mask.any():
                continue

            node_X = np.nan_to_num(node_X, nan=0.0)

            samples_X.append(node_X.astype(np.float32))
            samples_Y.append(node_y.astype(np.float32))
            samples_mask.append(mask.astype(bool))
            samples_dates.append(pd.Timestamp(end_date))

        if not samples_X:
            raise ValueError(
                "GraphSequenceDataset: no valid samples created. "
                "Check window/coverage/target."
            )

        self._X = samples_X
        self._Y = samples_Y
        self._mask = samples_mask
        self._sample_dates = samples_dates

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.Timestamp]:
        """
        Returns:
            x:    [N, W, F]  float32
            y:    [N]        float32
            mask: [N]        bool
            date: pd.Timestamp
        """
        x = torch.from_numpy(self._X[idx])
        y = torch.from_numpy(self._Y[idx])
        m = torch.from_numpy(self._mask[idx])
        d = self._sample_dates[idx]
        return x, y, m, d

    @property
    def sample_dates(self) -> List[pd.Timestamp]:
        return self._sample_dates

    @property
    def universe(self) -> List[str]:
        return self.tickers


# ---------- adjacency builders ----------

def build_corr_adjacency(
    returns: pd.DataFrame,
    tickers: List[str],
    corr_threshold: float = 0.3,
    self_loops: bool = True,
) -> np.ndarray:
    """
    Build NxN adjacency from return correlations.
    `returns` must have columns ['date','ticker','ret'].
    """
    tickers = [str(t).upper() for t in tickers]
    df = returns.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"])

    pivot = (
        df.pivot(index="date", columns="ticker", values="ret")
          .reindex(columns=tickers)
    )

    C = pivot.corr().to_numpy()
    C = np.nan_to_num(C, nan=0.0)

    A = (np.abs(C) >= corr_threshold).astype(np.float32)

    # no self edges in raw graph
    np.fill_diagonal(A, 0.0)

    if self_loops:
        A = A + np.eye(len(tickers), dtype=np.float32)

    # symmetrize
    A = np.maximum(A, A.T)

    # row-normalize
    deg = A.sum(axis=1, keepdims=True) + 1e-8
    A_norm = A / deg
    return A_norm.astype(np.float32)


def build_feature_similarity_adjacency(
    df: pd.DataFrame,
    tickers: List[str],
    feature_cols: List[str],
    corr_threshold: float = 0.3,
    self_loops: bool = True,
) -> np.ndarray:
    """
    Build NxN adjacency from similarity of *selected features* over time.

    For each ticker:
      - take its time-series of `feature_cols`
      - z-score by feature
      - flatten to a single vector
      - compute corr between ticker vectors
      - threshold |corr| to create edges
    """
    tickers = [str(t).upper() for t in tickers]
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("build_feature_similarity_adjacency: feature_cols is empty")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["ticker"] = data["ticker"].astype(str).str.upper()
    data = data[data["ticker"].isin(tickers)]

    data = data.sort_values(["ticker", "date"])

    vectors = []
    for t in tickers:
        g = data[data["ticker"] == t]
        if g.empty:
            vectors.append(np.zeros(len(feature_cols), dtype=np.float32))
            continue

        M = g[feature_cols].to_numpy(dtype=np.float32)  # [T,F]
        mu = np.nanmean(M, axis=0, keepdims=True)
        sd = np.nanstd(M, axis=0, keepdims=True)
        sd[sd == 0.0] = 1.0
        Z = (M - mu) / sd
        Z = np.nan_to_num(Z, nan=0.0)
        v = Z.reshape(-1)
        if not np.any(np.isfinite(v)):
            v = np.zeros_like(v)
        vectors.append(v)

    if not vectors:
        # degenerate; identity
        return np.eye(len(tickers), dtype=np.float32)

    X = np.stack(vectors, axis=0)  # [N,D]

    if X.shape[1] == 0:
        A = np.eye(len(tickers), dtype=np.float32)
    else:
        C = np.corrcoef(X)
        if C.shape != (len(tickers), len(tickers)):
            C = np.zeros((len(tickers), len(tickers)), dtype=np.float32)
        C = np.nan_to_num(C, nan=0.0)
        A = (np.abs(C) >= corr_threshold).astype(np.float32)

    np.fill_diagonal(A, 0.0)

    if self_loops:
        A = A + np.eye(len(tickers), dtype=np.float32)

    A = np.maximum(A, A.T)

    deg = A.sum(axis=1, keepdims=True) + 1e-8
    A_norm = A / deg
    return A_norm.astype(np.float32)


def build_mixed_adjacency(
    df: pd.DataFrame,
    tickers: List[str],
    feature_cols: List[str],
    *,
    ret_col: str = "ret_1d",
    extra_prefixes: Tuple[str, ...] = (
        "sent_",
        "corr_",
        "graph_",
        "has_earnings",
        "days_to_earnings",
    ),
    base_corr_threshold: float = 0.3,
    extra_corr_threshold: float = 0.3,
    alpha: float = 0.5,          # blend weight for extra-feature graph
    self_loops: bool = True,
) -> np.ndarray:
    """
    Convenience:
      - Always build return-corr adjacency from `ret_col`.
      - If any feature_cols start with extra_prefixes, also build feature-similarity adjacency.
      - Blend: (1-alpha) * A_ret + alpha * A_feat, then row-normalize.

    If no extra features available, falls back to A_ret.
    """

    # ---- base: returns graph ----
    if ret_col not in df.columns:
        raise ValueError(f"build_mixed_adjacency: ret_col '{ret_col}' not in df")

    ret_long = (
        df[["date", "ticker", ret_col]]
        .rename(columns={ret_col: "ret"})
        .copy()
    )
    A_ret = build_corr_adjacency(
        ret_long,
        tickers,
        corr_threshold=base_corr_threshold,
        self_loops=self_loops,
    )

    # ---- extra: feature-sim graph (optional) ----
    extra_cols = [
        c
        for c in feature_cols
        if any(c.startswith(p) for p in extra_prefixes)
    ]

    if not extra_cols:
        return A_ret  # nothing extra; pure returns graph

    try:
        A_feat = build_feature_similarity_adjacency(
            df,
            tickers,
            extra_cols,
            corr_threshold=extra_corr_threshold,
            self_loops=self_loops,
        )
        # blend & renormalize
        alpha = float(alpha)
        alpha = max(0.0, min(1.0, alpha))
        A = (1.0 - alpha) * A_ret + alpha * A_feat
    except Exception:
        # if anything explodes, fall back safely
        return A_ret

    A = np.maximum(A, 0.0)
    deg = A.sum(axis=1, keepdims=True) + 1e-8
    A = A / deg
    return A.astype(np.float32)