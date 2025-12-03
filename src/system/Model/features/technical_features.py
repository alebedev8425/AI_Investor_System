# src/system/Model/features/technical_features.py
from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalFeatureBuilder:
    """
    - Uses adj_close when available.
    - BACKWARD-looking features only (no leakage):
        * ret_1d, log_ret_1d
        * rolling means / vol of price + returns
    - FORWARD-looking labels:
        * y_1d_raw, y_5d_raw, y_20d_raw
        * target_5d = y_5d_raw  (for backward compatibility)

    All labels are defined as return from t+1 to t+H for row at date t:

        y_H_raw(t) = (P_{t+H} - P_{t+1}) / P_{t+1}
                   = P_{t+H} / P_{t+1} - 1
    """

    def __init__(
        self,
        use_adj: bool = True,
        ma_windows: tuple[int, ...] = (5, 10, 20),
        label_horizons: tuple[int, ...] = (1, 5, 20),
    ) -> None:
        self.use_adj = use_adj
        self.ma_windows = tuple(ma_windows)
        self.label_horizons = tuple(label_horizons)

    def build(self, prices: pd.DataFrame) -> pd.DataFrame:
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        price_col = "adj_close" if (self.use_adj and "adj_close" in df.columns) else "close"
        if price_col not in df.columns:
            raise ValueError(f"Prices missing '{price_col}' column.")

        px = df[price_col].astype(float)

        # 1d pct + log return (BACKWARD looking)
        df["ret_1d"] = df.groupby("ticker")[px.name].transform(lambda s: s.pct_change())
        df["ret_1d"] = df["ret_1d"].fillna(0.0)
        df["log_ret_1d"] = np.log1p(df["ret_1d"])

        # MAs & rolling vol (BACKWARD looking)
        g_by = df.groupby("ticker")
        for w in self.ma_windows:
            df[f"ma_{w}"] = g_by[price_col].transform(
                lambda s, w=w: s.rolling(w, min_periods=w).mean()
            )
            df[f"vol_{w}"] = g_by["log_ret_1d"].transform(
                lambda s, w=w: s.rolling(w, min_periods=w).std()
            )

        # FORWARD-looking raw labels (no normalization here!)
        # y_H_raw(t) = (P_{t+H} - P_{t+1}) / P_{t+1}
        for H in self.label_horizons:
            col = f"y_{H}d_raw"
            df[col] = g_by[price_col].transform(
                lambda s, H=H: s.shift(-H) / s - 1.0
            )

        # Backwards-compatibility: keep target_5d
        if 5 in self.label_horizons:
            df["target_5d"] = df["y_5d_raw"]
        else:
            df["target_5d"] = g_by[price_col].transform(
                lambda s: s.shift(-5) / s - 1.0
            )

        # Drop rows where we don't have the 5d label (end of series).
        df = df.dropna(subset=["target_5d"]).reset_index(drop=True)

        ordered_cols = [
            c
            for c in [
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "ret_1d",
                "log_ret_1d",
                *[f"ma_{w}" for w in self.ma_windows],
                *[f"vol_{w}" for w in self.ma_windows],
                *[f"y_{H}d_raw" for H in self.label_horizons],
                "target_5d",
            ]
            if c in df.columns
        ]
        return df[ordered_cols]
