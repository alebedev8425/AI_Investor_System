# system/Model/backtesting/backtester.py
from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import pandas as pd


class Backtester:
    """
    Simple backtester for evaluating portfolio weights against returns with transaction costs.
    """

    def __init__(self, *, transaction_cost_bps: int, long_only: bool) -> None:
        self.tc_bps = float(transaction_cost_bps)
        self.long_only = bool(long_only)

    def run(
        self,
        weights: pd.DataFrame,  # columns: ['date','ticker','weight']
        returns: pd.DataFrame,  # columns: ['date','ticker','ret']
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        # ---------- basic sanitation ----------
        w = weights.copy()
        r = returns.copy()

        w["date"] = pd.to_datetime(w["date"])
        r["date"] = pd.to_datetime(r["date"])
        w["ticker"] = w["ticker"].astype(str)
        r["ticker"] = r["ticker"].astype(str)

        # drop obvious bad rows
        w = w.dropna(subset=["date", "ticker", "weight"])
        r = r.dropna(subset=["date", "ticker", "ret"])

        if self.long_only:
            w["weight"] = w["weight"].clip(lower=0.0)

        # ---------- normalize per-date budget ≤ 1 ----------
        budget = w.groupby("date")["weight"].transform("sum")
        too_big = budget > 1.0
        if too_big.any():
            w.loc[too_big, "weight"] = w.loc[too_big, "weight"] / budget[too_big]

        # ---------- expand rebalance weights to daily using returns' dates ----------
        all_days = r["date"].drop_duplicates().sort_values()

        wide = w.set_index(["date", "ticker"])["weight"].unstack("ticker").reindex(all_days).ffill()

        # if there were days before the first rebalance, treat weights as 0 there
        wide = wide.fillna(0.0)

        expanded = (
            wide.stack().rename("weight").reset_index()  # back to ['date','ticker','weight']
        )

        # ---------- merge with returns, fill missing returns with 0 ----------
        merged = expanded.merge(r, on=["date", "ticker"], how="left").fillna({"ret": 0.0})

        # ---------- transaction costs based on turnover in weights (per ticker) ----------
        merged = merged.sort_values(["ticker", "date"])
        merged["prev_weight"] = merged.groupby("ticker")["weight"].shift(1).fillna(0.0)
        merged["turnover"] = (merged["weight"] - merged["prev_weight"]).abs()
        merged["tc"] = merged["turnover"] * (self.tc_bps / 10000.0)

        # ---------- aggregate to daily portfolio P&L ----------
        daily = (
            merged.assign(contrib=lambda df: df["weight"] * df["ret"] - df["tc"])
            .groupby("date", as_index=False)
            .agg(
                port_ret=("contrib", "sum"),
                gross_turnover=("turnover", "sum"),
                tc=("tc", "sum"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

        # ---------- simple metrics ----------
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

        # ensure no NaNs/inf in daily returns before computing stats
        daily["port_ret"] = daily["port_ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # equity curve
        equity = (1.0 + daily["port_ret"]).cumprod()

        equity = (
            equity.replace([np.inf, -np.inf], np.nan)  # avoid infs
            .ffill()  # fill forward NaNs
            .fillna(1.0)  # initial equity if starting with NaN
        )
        # drawdowns
        roll_max = equity.cummax()  # running max
        drawdown = equity / roll_max - 1.0  # drawdown series

        # Sharpe-like (annualized) – same for every model, since it's portfolio-level
        ann_factor = 252.0  # trading days per year
        mean_daily = float(daily["port_ret"].mean())
        std_daily = float(daily["port_ret"].std(ddof=1))
        if std_daily > 0 and np.isfinite(std_daily):
            sharpe_like = mean_daily / std_daily * np.sqrt(ann_factor)
        else:
            sharpe_like = 0.0

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

    def _compute_returns_for_backtest(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build tidy daily returns for backtest.
        Uses 'adj_close' if present, else 'close'. Expects columns: ['date','ticker',...].
        Output columns: ['date','ticker','ret'] with date as Timestamp (normalized to midnight).
        """
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"])
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        if price_col not in df.columns:
            raise ValueError("Prices missing both 'adj_close' and 'close' columns.")

        df = df.sort_values(["ticker", "date"])
        df["ret"] = df.groupby("ticker")[price_col].pct_change().fillna(0.0)
        return df[["date", "ticker", "ret"]].reset_index(drop=True)
