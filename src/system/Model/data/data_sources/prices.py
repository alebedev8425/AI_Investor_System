# src/system/data/prices.py
from __future__ import annotations

from typing import Iterable
import pandas as pd
from datetime import date


class PriceDataSource:
    """
    Phase-1: yfinance-backed implementation.
    Returns a long-form DataFrame with columns:
      ['date','ticker','open','high','low','close','adj_close','volume']
    """

    def __init__(self) -> None:
        try:
            import yfinance as yf  # local import keeps dependency optional
        except Exception as e:
            raise RuntimeError("yfinance is required. Install it first.") from e
        self._yf = yf

    def fetch(self, tickers: Iterable[str], start, end, adjust: bool = True) -> pd.DataFrame:
        tickers = list(tickers)
        if not tickers:
            raise ValueError("No tickers provided.")

        df = self._yf.download(
            tickers=" ".join(tickers),
            start=start,
            end=end,
            auto_adjust=False,  # we keep raw + adj close
            progress=False,
            group_by="ticker",
            threads=True,
        )

        frames: list[pd.DataFrame] = []

        if isinstance(df.columns, pd.MultiIndex):
            # Multiple tickers: columns like ('AAPL','Open'), index is DatetimeIndex named 'Date'
            for t in tickers:
                if t not in df.columns.get_level_values(0):
                    continue
                sub = df[t].copy()  # columns: Open, High, Low, Close, Adj Close, Volume
                sub = sub.reset_index()  # brings index out as a column named 'Date'
                sub.rename(columns={"Date": "date"}, inplace=True)
                # now lowercase all columns to normalize
                sub.columns = [c.lower() for c in sub.columns]
                sub["ticker"] = t
                # ensure consistent 'adj_close' name
                sub.rename(columns={"adj close": "adj_close"}, inplace=True)
                frames.append(
                    sub[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
                )

            if not frames:
                raise ValueError("No data returned from yfinance for requested tickers/dates.")
            out = pd.concat(frames, ignore_index=True)

        else:
            # Single ticker: columns are Open, High, Low, Close, Adj Close, Volume
            sub = df.copy().reset_index()  # 'Date' column created
            sub.rename(columns={"Date": "date"}, inplace=True)
            sub.columns = [c.lower() for c in sub.columns]
            sub["ticker"] = tickers[0]
            sub.rename(columns={"adj close": "adj_close"}, inplace=True)
            out = sub[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]

        # final tidy + types
        out["date"] = pd.to_datetime(
            out["date"]
        )  # keep as Timestamp (RunManager later handles .dt)
        out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

        # quick sanity check to fail fast if yfinance changed schema
        required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
        missing = required - set(out.columns)
        if missing:
            raise ValueError(
                f"PriceDataSource.fetch: missing columns after normalize: {sorted(missing)}"
            )

        return out
