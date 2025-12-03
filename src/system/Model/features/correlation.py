from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


@dataclass
class CorrelationFeatureConfig:
    """
    Configuration for correlation-based features.

    Assumes input:
        df with columns: ['date', 'ticker', 'ret']

    Output:
        df with columns (per date, ticker):

            corr_mean_abs         : mean |corr| with all other names
            corr_max_abs          : max  |corr| with all other names
            corr_topk_mean_abs    : mean |corr| to top-K peers (local neighborhood)
            corr_deg_thresh       : # of peers with |corr| >= corr_threshold

        Optionally (if add_cs_zscore=True):

            corr_mean_abs_cs_z    : cross-sectional z-score of corr_mean_abs
            corr_deg_thresh_cs_z  : cross-sectional z-score of corr_deg_thresh

    These are closer to MASTER/MDGNN-style structural statistics:
    - centrality (mean_abs)
    - local cluster strength (topk_mean_abs)
    - degree / connectivity (deg_thresh)
    - normalized cross-sectional signal (cs_z)
    """

    # rolling window length for correlations
    window: int = 60

    # minimum # of observations in the window before computing correlations
    min_periods: int = 40

    # threshold for "strong" connections (degree feature)
    corr_threshold: float = 0.6

    # how many strongest peers to use for local neighborhood stats
    top_k: int = 3

    # whether to add daily cross-sectional z-scores for selected stats
    add_cs_zscore: bool = True


class CorrelationFeatureBuilder:
    """
    Builds rolling cross-sectional correlation features from per-ticker returns.

    Usage:
        builder = CorrelationFeatureBuilder()
        feats = builder.build(daily_returns_df)

    Integration notes:
    - Intended to be called from the main FeaturePipeline.
    - Returns a DataFrame keyed by (date, ticker) ready to merge with other features.
    - Output columns are all prefixed with "corr_", so they are picked up by
      TrainingManager._select_feature_cols().
    """

    def __init__(self, cfg: Optional[CorrelationFeatureConfig] = None):
        self.cfg = cfg or CorrelationFeatureConfig()

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: DataFrame with columns ['date', 'ticker', 'ret'].

        Returns:
            DataFrame with columns:

                ['date', 'ticker',
                 'corr_mean_abs',
                 'corr_max_abs',
                 'corr_topk_mean_abs',
                 'corr_deg_thresh',
                 (optional) 'corr_mean_abs_cs_z',
                 (optional) 'corr_deg_thresh_cs_z']

            Empty DataFrame if input is missing/invalid.
        """
        required_cols = {"date", "ticker", "ret"}
        if df is None or df.empty or not required_cols.issubset(df.columns):
            return self._empty_output()

        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values(["date", "ticker"])

        # pivot to date x ticker matrix of returns
        mat = d.pivot(index="date", columns="ticker", values="ret").sort_index()
        tickers = list(mat.columns)

        if len(tickers) < 2:
            # can't compute cross-sectional corr with <2 names
            return self._empty_output()

        out_rows: List[Dict] = []
        w = int(self.cfg.window)
        minp = int(self.cfg.min_periods)
        thr = float(self.cfg.corr_threshold)
        top_k = int(self.cfg.top_k)

        dates = mat.index.to_list()

        for i in range(len(dates)):
            # need at least minp observations to have a meaningful correlation estimate
            if i + 1 < minp:
                continue

            start = max(0, i + 1 - w)
            window_slice = mat.iloc[start : i + 1]

            # drop all-nan columns inside window
            window_slice = window_slice.dropna(axis=1, how="all")
            if window_slice.shape[1] < 2:
                continue

            # compute correlation matrix for this window
            C = window_slice.corr()
            if C.empty:
                continue

            as_abs = C.abs()

            current_date = dates[i]

            for t in window_slice.columns:
                # exclude self-corr
                peers = as_abs.loc[t].drop(labels=[t], errors="ignore")
                if peers.empty:
                    continue

                # 1) global centrality: mean |corr| across all peers
                mean_abs = float(peers.mean())

                # 2) global strongest connection
                max_abs = float(peers.max())

                # 3) local cluster: mean |corr| to top-K neighbors
                if top_k > 0:
                    k = min(top_k, len(peers))
                    topk_vals = peers.nlargest(k)
                    topk_mean_abs = float(topk_vals.mean())
                else:
                    topk_mean_abs = float("nan")

                # 4) connectivity / degree above threshold
                deg = int((peers >= thr).sum())

                out_rows.append(
                    {
                        "date": current_date,
                        "ticker": t,
                        "corr_mean_abs": mean_abs,
                        "corr_max_abs": max_abs,
                        "corr_topk_mean_abs": topk_mean_abs,
                        "corr_deg_thresh": deg,
                    }
                )

        if not out_rows:
            return self._empty_output()

        out = pd.DataFrame(out_rows)
        out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

        if self.cfg.add_cs_zscore:
            out = self._add_cross_sectional_zscores(
                out,
                cols_for_z=[
                    "corr_mean_abs",
                    "corr_deg_thresh",
                ],
            )

        return out

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_output() -> pd.DataFrame:
        cols = [
            "date",
            "ticker",
            "corr_mean_abs",
            "corr_max_abs",
            "corr_topk_mean_abs",
            "corr_deg_thresh",
        ]
        return pd.DataFrame(columns=cols)

    @staticmethod
    def _add_cross_sectional_zscores(
        feat_df: pd.DataFrame,
        cols_for_z: list[str],
    ) -> pd.DataFrame:
        """
        For each date, compute cross-sectional z-scores of selected columns.

        This is similar in spirit to how you do cross-sectional label views
        (e.g., target_5d_cs_z). It makes correlation stats more stationary
        and closer to what MASTER/MDGNN-style models want: *relative*
        connectivity instead of raw levels that drift by regime.
        """
        df = feat_df.copy()

        for col in cols_for_z:
            if col not in df.columns:
                continue
            zcol = f"{col}_cs_z"
            # group by date and compute z-score within that date
            grp = df.groupby("date")[col]
            mean = grp.transform("mean")
            std = grp.transform("std")

            # avoid divide-by-zero: if std == 0, the cross-section is flat => z = 0
            z = np.where(std > 0, (df[col] - mean) / std, 0.0)
            df[zcol] = z.astype(np.float32)

        return df
