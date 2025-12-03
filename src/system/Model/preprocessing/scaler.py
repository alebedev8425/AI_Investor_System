# src/system/Model/preprocessing/scaler.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Optional, List, Tuple
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
    

@dataclass
class LabelNormalizer:
    """
    Train-only label standardization with tail clipping.

    Typical use:
        ln = LabelNormalizer(col="target_5d")
        ln.fit(train_df)
        train_df = ln.transform(train_df)
        val_df   = ln.transform(val_df)
        test_df  = ln.transform(test_df)

    After training, model outputs predictions in the normalized space.
    You then map them back to real returns via inverse_array().

    Notes:
      - Clipping is *not* invertible, so inverse_array() uses only the
        mean/std of the clipped distribution. This is exactly what we want:
        the model trains on a “robust” target, and predictions are interpreted
        as coming from that stabilized distribution.
    """
    col: str = "target_5d"
    clip_low: float = 0.005   # lower quantile for clipping (0.5%)
    clip_high: float = 0.995  # upper quantile for clipping (99.5%)

    mean_: Optional[float] = field(default=None, init=False)
    std_: Optional[float] = field(default=None, init=False)
    q_low_: Optional[float] = field(default=None, init=False)
    q_high_: Optional[float] = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, df: pd.DataFrame) -> None:
        if self.col not in df.columns:
            raise ValueError(f"LabelNormalizer.fit: column '{self.col}' not in DataFrame.")

        y = df[self.col].astype(float).to_numpy()
        y = y[np.isfinite(y)]
        if y.size == 0:
            raise ValueError("LabelNormalizer.fit: no finite label values.")

        # quantile-based clipping bounds
        q_low = float(np.nanquantile(y, self.clip_low))
        q_high = float(np.nanquantile(y, self.clip_high))
        if q_low > q_high:  # sanity guard
            q_low, q_high = q_high, q_low

        y_clipped = np.clip(y, q_low, q_high)
        mean = float(np.nanmean(y_clipped))
        std = float(np.nanstd(y_clipped))
        if std <= 0.0 or not np.isfinite(std):
            std = 1.0  # avoid degenerate scaling

        self.q_low_ = q_low
        self.q_high_ = q_high
        self.mean_ = mean
        self.std_ = std
        self.fitted_ = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a '{col}_norm' column with clipped + standardized labels.
        Original label column is left untouched.
        """
        if not self.fitted_:
            raise RuntimeError("LabelNormalizer.transform called before fit().")
        if self.col not in df.columns:
            raise ValueError(f"LabelNormalizer.transform: column '{self.col}' not in DataFrame.")

        out = df.copy()
        y = out[self.col].astype(float).to_numpy()
        y = np.clip(y, self.q_low_, self.q_high_)
        y_norm = (y - self.mean_) / self.std_
        y_norm = np.where(np.isfinite(y_norm), y_norm, 0.0)

        out[f"{self.col}_norm"] = y_norm.astype(np.float32)
        return out
    
    def inverse_array(self, y_norm: np.ndarray) -> np.ndarray:
        """
        Map normalized predictions back to raw return space.

        This is the exact inverse of the standardization step:
            y = y_norm * std + mean
        """
        if not self.fitted_:
            raise RuntimeError("LabelNormalizer.inverse_array called before fit().")

        y_norm = np.asarray(y_norm, dtype=np.float32)
        y = y_norm * self.std_ + self.mean_
        return y.astype(np.float32)

    def inverse_series(self, s: pd.Series) -> pd.Series:
        """Convenience wrapper for pandas Series."""
        y = self.inverse_array(s.to_numpy(dtype=np.float32))
        return pd.Series(y, index=s.index, name=self.col)
    


def add_cross_sectional_label_views(
    df: pd.DataFrame,
    label_col: str = "target_5d",
) -> pd.DataFrame:
    """
    Adds per-date cross-sectional variants of the label:

        label_cs_z    : z-score across tickers on each date
        label_cs_rank : percentile rank in [0,1]

    No look-ahead: uses only same-date cross section.
    """
    if df.empty or label_col not in df.columns:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    grp = out.groupby("date")[label_col]

    # cross-sectional mean/std
    cs_mean = grp.transform("mean")
    cs_std = grp.transform("std").replace(0.0, np.nan)

    cs_z = (out[label_col] - cs_mean) / cs_std
    cs_z = cs_z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out[f"{label_col}_cs_z"] = cs_z.astype(np.float32)

    # percentile rank in [0,1]
    cs_rank = grp.rank(method="average", pct=True)
    out[f"{label_col}_cs_rank"] = cs_rank.astype(np.float32)

    return out