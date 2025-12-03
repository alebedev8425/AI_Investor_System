# src/system/Model/features/graph.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GraphFeatureConfig:
    """
    Configuration for graph-based features derived from correlations.

    We treat tickers as nodes, edges between them if |corr| >= corr_threshold.

    Assumes input:
        df with columns: ['date', 'ticker', 'ret']

    Output:
        df with columns:
            date, ticker,
            graph_degree,        # number of strong connections
            graph_strength,      # sum of |corr| over strong connections
            graph_avg_strength   # avg |corr| over strong connections (0 if none)
    """
    window: int = 60
    min_periods: int = 40
    corr_threshold: float = 0.6


class GraphFeatureBuilder:
    """
    Builds simple graph / network features from rolling correlation structure.

    This is intentionally:
        - self-contained (no networkx)
        - deterministic
        - aligned to (date, ticker) for easy merge with other features.

    Usage:
        builder = GraphFeatureBuilder()
        feats = builder.build(daily_returns_df)
    """

    def __init__(self, cfg: Optional[GraphFeatureConfig] = None):
        self.cfg = cfg or GraphFeatureConfig()

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: DataFrame with columns ['date', 'ticker', 'ret'].

        Returns:
            DataFrame with:
                ['date', 'ticker',
                 'graph_degree', 'graph_strength', 'graph_avg_strength']
        """
        required_cols = {"date", "ticker", "ret"}
        if df is None or df.empty or not required_cols.issubset(df.columns):
            return pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "graph_degree",
                    "graph_strength",
                    "graph_avg_strength",
                ]
            )

        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values(["date", "ticker"])

        mat = d.pivot(index="date", columns="ticker", values="ret").sort_index()
        tickers = list(mat.columns)

        if len(tickers) < 2:
            return pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "graph_degree",
                    "graph_strength",
                    "graph_avg_strength",
                ]
            )

        out_rows = []
        w = self.cfg.window
        minp = self.cfg.min_periods
        thr = float(self.cfg.corr_threshold)

        dates = mat.index.to_list()

        for i in range(len(dates)):
            if i + 1 < minp:
                continue
            start = max(0, i + 1 - w)
            window_slice = mat.iloc[start : i + 1]

            # drop all-nan assets in this window
            window_slice = window_slice.dropna(axis=1, how="all")
            if window_slice.shape[1] < 2:
                continue

            C = window_slice.corr()
            if C.empty:
                continue

            as_abs = C.abs()

            # for each node/ticker, compute simple graph stats over strong edges
            for t in window_slice.columns:
                peers = as_abs.loc[t].drop(labels=[t], errors="ignore")
                if peers.empty:
                    continue

                strong = peers[peers >= thr]
                degree = int(strong.size)
                strength = float(strong.sum()) if degree > 0 else 0.0
                avg_strength = float(strong.mean()) if degree > 0 else 0.0

                out_rows.append(
                    {
                        "date": dates[i],
                        "ticker": t,
                        "graph_degree": degree,
                        "graph_strength": strength,
                        "graph_avg_strength": avg_strength,
                    }
                )

        if not out_rows:
            return pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "graph_degree",
                    "graph_strength",
                    "graph_avg_strength",
                ]
            )

        out = pd.DataFrame(out_rows)
        out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
        return out