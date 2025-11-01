# src/system/model/calendar.py
from __future__ import annotations
from typing import Iterable, Optional
import pandas as pd

class TradingCalendar:
    """
    Phase 1: market days = Monday..Friday.
    Future: later, inject a holiday provider.
    """

    def __init__(self, market: str = "US", holiday_provider: Optional[object] = None) -> None:
        self.market = market
        self._holiday_provider = holiday_provider  # reserved for later

    # ---------- core predicate ----------
    def is_trading_day(self, ts: pd.Timestamp) -> bool:
        """Phase 1: weekend filter only. (Mon=0 ... Sun=6)"""
        return ts.dayofweek < 5

    # ---------- alignment utilities ----------
    def align(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        Keep calendar scope tight:
        - Convert to Timestamp
        - Filter to trading days
        - Sort by (date, ticker) if ticker exists
        NOTE: Do NOT drop duplicates or fix schema here (validator’s job).
        """
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        mask = out[date_col].apply(self.is_trading_day)
        out = out.loc[mask]
        sort_cols: list[str] = [date_col] + (["ticker"] if "ticker" in out.columns else [])
        return out.sort_values(sort_cols).reset_index(drop=True)

    # ---------- date-range helpers ----------
    def trading_days_between(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Inclusive trading-day index between start and end (phase 1 = Mon–Fri)."""
        rng = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", inclusive="both")
        return pd.DatetimeIndex([d for d in rng if self.is_trading_day(pd.Timestamp(d))])

    def next_trading_day(self, ts: pd.Timestamp) -> pd.Timestamp:
        """First trading day >= ts."""
        ts = pd.Timestamp(ts)
        while not self.is_trading_day(ts):
            ts += pd.Timedelta(days=1)
        return ts

    def previous_trading_day(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Last trading day <= ts."""
        ts = pd.Timestamp(ts)
        while not self.is_trading_day(ts):
            ts -= pd.Timedelta(days=1)
        return ts

    def weekly_rebalance_anchors(self, dates: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
        """
        Phase 1 simple rule: pick *last* trading day of each (year, week).
        Later you can switch to 'Friday close' specifically, or exchange schedules.
        """
        idx = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
        idx = pd.DatetimeIndex([d for d in idx if self.is_trading_day(d)])
        if idx.empty:
            return []
        # group by ISO week; take the last date in each group
        df = pd.DataFrame({"dt": idx})
        df["year"] = df["dt"].dt.isocalendar().year
        df["week"] = df["dt"].dt.isocalendar().week
        anchors = (df.groupby(["year", "week"])["dt"].max().sort_values().tolist())
        return anchors