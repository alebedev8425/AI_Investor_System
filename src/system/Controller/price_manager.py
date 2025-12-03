# src/system/Controller/price_manager.py
from __future__ import annotations

from datetime import date
from typing import Sequence
import logging
import pandas as pd

from system.Model.data.data_sources.prices import PriceDataSource
from system.Controller.cache_manager import CacheManager 
from system.Model.data.trading_calendar import TradingCalendar
from system.Model.data.data_validator import DataValidator


class PriceManager:
    """
    Controller for 'Ingest Price Data' (Phase-1).

    Flow:
      * compute a shared-cache key from (tickers, start, end, adjust, source)
      * use CacheManager.ensure_prices(...) to hydrate into this run
      * (for fresh fetch) align to calendar + validate BEFORE writing shared cache
      * slice defensively to the exact universe/window

    This keeps runs reproducible (same inputs -> same cached CSV) and avoids recomputation.
    """

    def __init__(
        self,
        *,
        source: PriceDataSource,
        cache: CacheManager,
        calendar: TradingCalendar,
        validator: DataValidator,
        logger: logging.Logger | None = None,
    ) -> None:
        self._src = source
        self._cache = cache
        self._cal = calendar
        self._val = validator
        self._log = logger or logging.getLogger(__name__)

    def ingest_prices(
        self,
        tickers: Sequence[str],
        start: date,
        end: date,
        *,
        force_refresh: bool = False,
        adjust: bool = True,
    ) -> pd.DataFrame:
        tickers = [t.upper() for t in tickers]

        # Wrap the data-source fetch so that any FRESH download is aligned+validated
        def _fetch_aligned(*, tickers, start, end, adjust):
            df = self._src.fetch(tickers=tickers, start=start, end=end, adjust=adjust)
            self._log.info("Fetched prices: %d rows", len(df))
            df = self._cal.align(df)
            df = self._val.validate_prices(df)
            return df

        # Hydrate (shared cache -> this run) or fetch fresh then store shared+run
        df = self._cache.ensure_prices(
            tickers=tickers,
            start=start,
            end=end,
            adjust=adjust,
            fetch_fn=_fetch_aligned,
            force_refresh=force_refresh,
        )

        # Defensive guard: enforce the exact universe/window even if upstream broadens later
        df = self._slice(df, tickers, start, end)
        self._log.info("Saved prices to cache")
        return df

    # ----- helpers -----
    @staticmethod
    def _slice(df: pd.DataFrame, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
        tickers = [t.upper() for t in tickers]
        mask = (
            (out["date"].dt.date >= start)
            & (out["date"].dt.date <= end)
            & (out["ticker"].str.upper().isin(tickers))
        )
        out = out.loc[mask]
        cols_order = [c for c in ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"] if c in out.columns]
        return out[cols_order].sort_values(["date", "ticker"]).reset_index(drop=True)