# src/system/Model/evaluation/prediction_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class PredictionMetricsResult:
    metrics: Dict[str, Any]


class PredictionEvaluator:
    """
    Compute research-grade prediction metrics on the test set.

    Expects:
        preds_df:  ['date','ticker','pred_5d']
        labels_df: must contain ['date','ticker','target_5d']

    Outputs:
        - MAE / MSE / RMSE
        - Cross-sectional IC (Spearman) per date + mean IC and t-stat
        - Hit rate (sign agreement)
        - Simple decile spread: top-bucket minus bottom-bucket realized return
    """

    def __init__(self, *, horizon_col: str = "target_5d", pred_col: str = "pred_5d") -> None:
        self.horizon_col = horizon_col
        self.pred_col = pred_col

    def evaluate(self, preds_df: pd.DataFrame, labels_df: pd.DataFrame) -> PredictionMetricsResult:
        if preds_df.empty:
            return PredictionMetricsResult(metrics={})

        # --- align + clean ---
        preds = preds_df.copy()
        labels = labels_df.copy()

        preds["date"] = pd.to_datetime(preds["date"])
        labels["date"] = pd.to_datetime(labels["date"])
        preds["ticker"] = preds["ticker"].astype(str).str.upper()
        labels["ticker"] = labels["ticker"].astype(str).str.upper()

        if self.horizon_col not in labels.columns:
            raise ValueError(
                f"PredictionEvaluator: labels_df must contain '{self.horizon_col}'."
            )

        merged = preds.merge(
            labels[["date", "ticker", self.horizon_col]],
            on=["date", "ticker"],
            how="inner",
        ).rename(columns={self.horizon_col: "target"})

        merged = merged.dropna(subset=[self.pred_col, "target"])
        if merged.empty:
            return PredictionMetricsResult(metrics={})

        y_pred = merged[self.pred_col].astype(float).to_numpy()
        y_true = merged["target"].astype(float).to_numpy()

        # --- basic error metrics ---
        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))

        # --- hit rate (directional accuracy) ---
        s_pred = np.sign(y_pred)
        s_true = np.sign(y_true)
        mask_nz = s_true != 0
        if mask_nz.any():
            hit_rate = float((s_pred[mask_nz] == s_true[mask_nz]).mean())
        else:
            hit_rate = float("nan")

        # --- cross-sectional IC (Spearman) per date ---
        def _ic_for_day(g: pd.DataFrame) -> float:
            if g[self.pred_col].nunique() < 2 or g["target"].nunique() < 2:
                return np.nan
            return g[[self.pred_col, "target"]].corr(method="spearman").iloc[0, 1]

        ic_series = (
            merged
            .groupby("date")[ [self.pred_col, "target"] ]
            .apply(_ic_for_day)
        )
        ic_vals = ic_series.dropna().to_numpy()
        if ic_vals.size > 0:
            ic_mean = float(np.mean(ic_vals))
            ic_std = float(np.std(ic_vals, ddof=1)) if ic_vals.size > 1 else float("nan")
            ic_t = (
                float(ic_mean / (ic_std / np.sqrt(ic_vals.size)))
                if ic_vals.size > 1 and ic_std > 0
                else float("nan")
            )
        else:
            ic_mean = ic_std = ic_t = float("nan")

        # --- decile spread over predictions ---
        # For each date: bucket by prediction and take equal-weight realized target.
        def _assign_buckets(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            # up to 10 buckets, but not more than unique scores
            q = min(10, g[self.pred_col].nunique())
            if q < 2:
                g["bucket"] = np.nan
                return g
            g["bucket"] = pd.qcut(
                g[self.pred_col],
                q=q,
                labels=False,
                duplicates="drop",
            )
            return g
        
        buck = (
            merged
            .groupby("date", group_keys=True)[ [self.pred_col, "target"] ]
            .apply(_assign_buckets)
            .reset_index()  # keep date as a column from the index
        )

        buck = buck.drop(columns=["level_1"]) # remove extra index from groupby-apply
        if buck.empty:
            top_ret = bottom_ret = spread = float("nan")
        else:
            buck["bucket"] = buck["bucket"].astype(int)
            # daily bucket returns
            daily_bucket = (
                buck.groupby(["date", "bucket"])["target"]
                .mean()
                .reset_index()
            )
            # average across days
            avg_bucket = (
                daily_bucket.groupby("bucket")["target"]
                .mean()
                .sort_index()
            )

            bottom_ret = float(avg_bucket.iloc[0])
            top_ret = float(avg_bucket.iloc[-1])
            spread = float(top_ret - bottom_ret)

        metrics = {
            "n_obs": int(len(merged)),
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "hit_rate": hit_rate,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_tstat": ic_t,
            "decile_top_avg_return": top_ret,
            "decile_bottom_avg_return": bottom_ret,
            "decile_long_short_spread": spread,
            "horizon_col": self.horizon_col,
            "pred_col": self.pred_col,
        }

        return PredictionMetricsResult(metrics=metrics)