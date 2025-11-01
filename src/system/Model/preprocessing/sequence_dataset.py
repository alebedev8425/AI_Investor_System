from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Supports single- or multi-feature sequences.
    __getitem__ returns only (X[T,F], y) so DataLoader can collate cleanly.
    Dates/tickers are kept as parallel arrays for downstream use.
    """

    def __init__(self, df: pd.DataFrame, window: int) -> None:
        # Legacy constructor: expects ['date','ticker','ret_1d','y_5d']
        if not {"date", "ticker", "ret_1d", "y_5d"}.issubset(df.columns):
            raise ValueError("Expected columns: date, ticker, ret_1d, y_5d for legacy constructor.")

        self.window = int(window)
        self._samples: List[Tuple[np.ndarray, float]] = []
        self._dates: List[pd.Timestamp] = []
        self._tickers: List[str] = []

        for tkr, g in df.groupby("ticker"):
            g = g.sort_values("date")
            x = g["ret_1d"].to_numpy(dtype=np.float32)  # [N]
            y = g["y_5d"].to_numpy(dtype=np.float32)  # [N]
            dates = pd.to_datetime(g["date"]).to_numpy()  # [N]
            if len(g) <= self.window:
                continue
            for i in range(self.window, len(g)):
                seq_x = x[i - self.window : i]  # [W]
                target = y[i]
                self._samples.append((seq_x, target))
                self._dates.append(pd.Timestamp(dates[i]))
                self._tickers.append(tkr)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        window: int,
    ) -> "SequenceDataset":
        need = {"date", "ticker", *feature_cols, target_col}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"SequenceDataset.from_dataframe: missing columns: {missing}")

        self = cls.__new__(cls)  # bypass legacy __init__
        self.window = int(window)
        self._samples = []
        self._dates = []
        self._tickers = []

        gdf = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        for tkr, g in gdf.groupby("ticker", sort=False):
            # Drop any lingering NaNs in required columns
            g = g.dropna(subset=list(need)).copy()
            if len(g) <= window:
                continue

            X = g[feature_cols].to_numpy(dtype=np.float32)  # [N,F]
            y = g[target_col].to_numpy(dtype=np.float32)  # [N]
            dates = pd.to_datetime(g["date"]).to_numpy()  # [N]

            for i in range(window, len(g)):
                seq_x = X[i - window : i, :]  # [W,F]
                target = y[i]  # scalar
                self._samples.append((seq_x, target))
                self._dates.append(pd.Timestamp(dates[i]))
                self._tickers.append(str(tkr))

        return self

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        seq_x, target = self._samples[idx]
        x = torch.from_numpy(seq_x)
        if x.ndim == 1:
            x = x.unsqueeze(-1)  # legacy single-feature path -> [W,1]
        y = torch.tensor(target, dtype=torch.float32)
        return x, y

    @property
    def dates(self) -> List[pd.Timestamp]:
        return self._dates

    @property
    def tickers(self) -> List[str]:
        return self._tickers
