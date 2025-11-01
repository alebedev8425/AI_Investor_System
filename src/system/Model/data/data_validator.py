# ai_investor/model/validation.py
from __future__ import annotations

import numpy as np
import pandas as pd


class DataValidator:
    """Stateless checks for various data so downstream code can trust inputs."""

    # checks that the input price DataFrame has required columns and types
    REQUIRED_PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]

    # function to validate price data
    def validate_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        # Columns
        missing = [c for c in self.REQUIRED_PRICE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Price data missing required columns: {missing}")

        df = df.copy()

        # Types
        if not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)

        if df["ticker"].isna().any():
            raise ValueError("Price data has null tickers.")
        if (df["volume"] < 0).any():
            raise ValueError("Price data has negative volume.")

        # no duplicates by (date, ticker)
        df = df.drop_duplicates(subset=["date", "ticker"])

        # sort per ticker (optional but helpful)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        return df
