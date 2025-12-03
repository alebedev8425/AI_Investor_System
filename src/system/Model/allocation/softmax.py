# src/system/Model/portfolio/softmax.py

from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
import pandas as pd


class SoftmaxAllocator:
    """
    Cross-sectional softmax allocator.

    For each date:
      1) Take model scores (e.g., pred_5d) across tickers.
      2) Optionally transform them cross-sectionally (zscore / rank / minmax).
      3) Optionally keep only top-K names by transformed score.
      4) Apply temperature-scaled softmax to get raw weights.
      5) Enforce:
           - optional long-only & per-name cap
           - either:
               * allow_cash=True : sum(weights) <= 1 (residual is cash)
               * allow_cash=False: rescale so sum(weights) = 1
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        weight_cap: float = 0.10,
        long_only: bool = True,
        transform: str = "zscore",  # 'zscore' | 'rank' | 'minmax' | 'none'
        top_k: Optional[int] = None,  # e.g., 2 or 3
        allow_cash: bool = True,
        eps: float = 1e-12,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0 < weight_cap <= 1):
            raise ValueError("weight_cap must be in (0,1]")
        if transform not in {"zscore", "rank", "minmax", "none"}:
            raise ValueError("transform must be one of {'zscore','rank','minmax','none'}")

        self.temperature = float(temperature)
        self.weight_cap = float(weight_cap)
        self.long_only = bool(long_only)
        self.transform = transform
        self.top_k = top_k
        self.allow_cash = allow_cash
        self.eps = float(eps)

    def _transform_scores(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            return a

        x = a.copy()
        x[~np.isfinite(x)] = 0.0

        if self.transform == "zscore":
            mu = np.nanmean(x)
            sd = np.nanstd(x)
            if not np.isfinite(sd) or sd < 1e-12:
                # all equal or degenerate -> no tilt
                return np.zeros_like(x)
            x = (x - mu) / (sd + 1e-12)

        elif self.transform == "rank":
            # average ranks starting at 1..N, then z-score them
            order = np.argsort(x, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
            mu = ranks.mean()
            sd = ranks.std()
            x = (ranks - mu) / (sd + 1e-12)

        elif self.transform == "minmax":
            lo = np.nanmin(x)
            hi = np.nanmax(x)
            if not np.isfinite(hi - lo) or (hi - lo) < 1e-12:
                return np.zeros_like(x)
            x = 2.0 * (x - lo) / (hi - lo) - 1.0  # map to [-1, 1]

        # 'none' returns raw scores (but cleaned for NaNs/inf)
        return x

    def allocate(
        self,
        preds: pd.DataFrame,
        *,
        date_col: str = "date",
        ticker_col: str = "ticker",
        score_col: str = "pred_5d",
        anchor_dates: Optional[Iterable[pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        if not {date_col, ticker_col, score_col}.issubset(preds.columns):
            raise ValueError(f"preds must contain {date_col},{ticker_col},{score_col}")

        df = preds.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        if anchor_dates is not None:
            anchors = set(pd.to_datetime(list(anchor_dates)))
            df = df[df[date_col].isin(anchors)]

        out_parts: list[pd.DataFrame] = []

        for d, g in df.groupby(date_col, sort=True):
            tickers = g[ticker_col].astype(str).to_numpy()
            raw = g[score_col].astype(float).to_numpy(dtype=np.float64)

            # Guard: replace non-finites
            raw[~np.isfinite(raw)] = 0.0

            # Optional cross-sectional transform (per date)
            s = self._transform_scores(raw)

            # Optional top-K mask (by transformed score)
            if self.top_k is not None and 0 < self.top_k < len(s):
                keep_idx = np.argsort(-s)[: self.top_k]
                mask = np.zeros_like(s, dtype=bool)
                mask[keep_idx] = True
            else:
                mask = np.ones_like(s, dtype=bool)

            # Temperature & stable softmax on transformed scores
            z = s / self.temperature
            z[~mask] = -1e9  # effectively -inf → exp→0
            z = z - np.nanmax(z)  # log-sum-exp stabilization
            z = np.clip(z, -50, 50)  # avoid under/overflow

            expz = np.exp(z)
            denom = expz.sum()
            if not np.isfinite(denom) or denom <= self.eps or mask.sum() == 0:
                # Fallback: equal-weight among valid names
                w = np.zeros_like(s)
                k = int(mask.sum())
                if k > 0:
                    w[mask] = 1.0 / k
            else:
                w = expz / denom

            # Long-only cap & normalization / cash logic
            if self.long_only:
                w = np.clip(w, 0.0, self.weight_cap)

            ssum = float(w.sum())

            if self.allow_cash:
                # If we allow cash, we only scale down if we exceed 100% gross.
                if ssum > 1.0 + self.eps:
                    w = w / ssum
                # If ssum <= 1, residual (1 - ssum) is cash.
            else:
                # No cash: force sum(w) = 1. If degenerate, equal-weight on active names.
                active = (w > 0.0) if self.long_only else mask
                k = int(active.sum())
                if ssum <= self.eps or k == 0:
                    w = np.zeros_like(w)
                    if k > 0:
                        w[active] = 1.0 / k
                    else:
                        # everything dead; equal-weight all tickers
                        w[:] = 1.0 / len(w)
                else:
                    w = w / ssum

            out_parts.append(pd.DataFrame({date_col: d, ticker_col: tickers, "weight": w}))

        if not out_parts:
            return pd.DataFrame(columns=[date_col, ticker_col, "weight"])

        return (
            pd.concat(out_parts, ignore_index=True)
            .sort_values([date_col, ticker_col])
            .reset_index(drop=True)
        )
