# src/system/Model/features/technical_features.py
from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalFeatureBuilder:
    """
    Phase-1: minimal technical features
      - daily returns (pct + log1p)
      - simple moving averages and return vol
      - 5-day forward return as target
    """

    def __init__(self, use_adj: bool = True, ma_windows: tuple[int, ...] = (5, 10, 20)) -> None:
        self.use_adj = use_adj
        self.ma_windows = ma_windows

    def build(self, prices: pd.DataFrame) -> pd.DataFrame:
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        price_col = "adj_close" if (self.use_adj and "adj_close" in df.columns) else "close"
        if price_col not in df.columns:
            raise ValueError(f"Prices missing '{price_col}' column.")

        # 1d pct + log return
        df["ret_1d"] = df.groupby("ticker")[price_col].transform(lambda s: s.pct_change()).fillna(0.0)
        df["log_ret_1d"] = np.log1p(df["ret_1d"])

        # MAs & rolling vol (full windows only)
        for w in self.ma_windows:
            df[f"ma_{w}"]  = df.groupby("ticker")[price_col].transform(lambda s: s.rolling(w, min_periods=w).mean())
            df[f"vol_{w}"] = df.groupby("ticker")["ret_1d"].transform(lambda s: s.rolling(w, min_periods=w).std())

        # 5-day forward return target
        df["target_5d"] = df.groupby("ticker")[price_col].transform(lambda s: (s.shift(-5) - s) / s)

        # *** Drop any rows with NaN across engineered features + target ***
        eng_cols = ["ret_1d", "log_ret_1d"] \
                 + [f"ma_{w}" for w in self.ma_windows] \
                 + [f"vol_{w}" for w in self.ma_windows] \
                 + ["target_5d"]
        present = [c for c in eng_cols if c in df.columns]
        df = df.dropna(subset=present).reset_index(drop=True)

        ordered_cols = [c for c in [
            "date","ticker","open","high","low","close","adj_close","volume",
            "ret_1d","log_ret_1d",
            *[f"ma_{w}" for w in self.ma_windows],
            *[f"vol_{w}" for w in self.ma_windows],
            "target_5d"
        ] if c in df.columns]
        return df[ordered_cols]
