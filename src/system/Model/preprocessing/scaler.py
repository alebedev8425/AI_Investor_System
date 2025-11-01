from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Optional, List
import numpy as np
import pandas as pd

@dataclass
class StandardScaler:
    cols: Optional[List[str]] = None
    mean_: Optional[np.ndarray] = field(default=None, init=False)
    std_: Optional[np.ndarray] = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> None:
        cols = list(feature_cols) if feature_cols is not None else (self.cols or [])
        if not cols:
            raise ValueError("StandardScaler.fit: feature_cols must be provided at first fit() or preset in ctor.")

        X = df[cols].astype(float).to_numpy()
        # NaN-safe stats
        mean = np.nanmean(X, axis=0)
        std  = np.nanstd(X, axis=0)
        std[std == 0.0] = 1.0

        self.cols_ = cols
        self.mean_ = mean
        self.std_  = std
        self.fitted_ = True

    def transform(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("StandardScaler.transform called before fit().")
        cols = list(feature_cols) if feature_cols is not None else self.cols_
        if cols is None:
            raise RuntimeError("StandardScaler has no fitted columns.")

        out = df.copy()
        X = out[cols].astype(float).to_numpy()
        X = (X - self.mean_) / self.std_
        # sanitize any remaining bad values
        X = np.where(np.isfinite(X), X, 0.0)
        out[cols] = X
        return out