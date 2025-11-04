# system/Model/backtesting/backtester.py
from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import pandas as pd


class Backtester:
    """
    Phase-1, file-backed research backtester.

    This class does NOT import ExperimentConfig or BacktestConfig.
    It only takes the minimal parameters it needs, which are injected by RunManager.
    """

    def __init__(self, *, transaction_cost_bps: int, long_only: bool) -> None:
        self.tc_bps = transaction_cost_bps
        self.long_only = long_only

    def run(
        self,
        weights: pd.DataFrame,  # columns: ['date','ticker','weight']
        returns: pd.DataFrame,  # columns: ['date','ticker','ret']
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        # Basic sanitation
        w = weights.copy()
        r = returns.copy()
        w["date"] = pd.to_datetime(w["date"])
        r["date"] = pd.to_datetime(r["date"])

        if self.long_only:
            w["weight"] = w["weight"].clip(lower=0.0)

        # normalize per-date budget ≤ 1
        budget = w.groupby("date")["weight"].transform("sum")
        too_big = budget > 1.0
        if too_big.any():
            w.loc[too_big, "weight"] = w.loc[too_big, "weight"] / budget[too_big]

        # expand rebalance weights forward to daily using the returns' dates
        all_days = r["date"].drop_duplicates().sort_values()
        wide = w.set_index(["date", "ticker"])["weight"].unstack("ticker").reindex(all_days).ffill()
        expanded = (
            wide.stack().rename("weight").reset_index()  # back to ['date','ticker','weight']
        )

        # merge with returns, fill missing returns with 0
        merged = expanded.merge(r, on=["date", "ticker"], how="left").fillna({"ret": 0.0})

        # transaction costs based on turnover in weights (per ticker)
        merged = merged.sort_values(["ticker", "date"])
        merged["prev_weight"] = merged.groupby("ticker")["weight"].shift(1).fillna(0.0)
        merged["turnover"] = (merged["weight"] - merged["prev_weight"]).abs()
        merged["tc"] = merged["turnover"] * (self.tc_bps / 10000.0)

        # aggregate to daily portfolio P&L
        daily = (
            merged.assign(contrib=lambda df: df["weight"] * df["ret"] - df["tc"])
            .groupby("date", as_index=False)
            .agg(
                port_ret=("contrib", "sum"),
                gross_turnover=("turnover", "sum"),
                tc=("tc", "sum"),
            )
        )

        # simple metrics
        if daily.empty:
            metrics = {
                "cumulative_return": 0.0,
                "sharpe_like": 0.0,
                "max_drawdown": 0.0,
                "avg_daily_turnover": 0.0,
                "transaction_cost_bps": float(self.tc_bps),
                "long_only": bool(self.long_only),
                "n_days": 0,
            }
            return daily, metrics

        equity = (1.0 + daily["port_ret"]).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        std = float(daily["port_ret"].std())
        sharpe_like = (
            float(daily["port_ret"].mean() / (std + 1e-12) * np.sqrt(252)) if std > 0 else 0.0
        )

        metrics = {
            "cumulative_return": float(equity.iloc[-1] - 1.0),
            "sharpe_like": sharpe_like,
            "max_drawdown": float(drawdown.min()),
            "avg_daily_turnover": float(daily["gross_turnover"].mean()),
            "transaction_cost_bps": float(self.tc_bps),
            "long_only": bool(self.long_only),
            "n_days": int(len(daily)),
        }
        return daily, metrics
