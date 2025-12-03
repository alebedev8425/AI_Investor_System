from __future__ import annotations
from typing import Iterable, Dict, Any
import numpy as np
import pandas as pd
import logging

class DataValidator:
    """Stateless checks for various data so downstream code can trust inputs."""
    
    REQUIRED_PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]
    
    def __init__(self, logger: logging.Logger | None = None):
        self._log = logger or logging.getLogger(__name__)

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
        
        df = self._validate_price_sanity(df)
        return df

    def _validate_price_sanity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for obvious data quality issues in prices"""
        initial_rows = len(df)
        
        # 1. Remove rows with zero or negative prices
        price_cols = [c for c in ['open', 'high', 'low', 'close', 'adj_close'] if c in df.columns]
        for col in price_cols:
            df = df[df[col] > 0]
        
        # 2. Remove rows where high < low or high < open or high < close
        if all(c in df.columns for c in ['high', 'low']):
            df = df[df['high'] >= df['low']]
        if all(c in df.columns for c in ['high', 'open']):
            df = df[df['high'] >= df['open']]
        if all(c in df.columns for c in ['high', 'close']):
            df = df[df['high'] >= df['close']]
            
        # 3. Remove extreme price moves (>50% in one day)
        if 'close' in df.columns and 'open' in df.columns:
            daily_move = (df['close'] - df['open']).abs() / df['open']
            df = df[daily_move <= 0.5]  # Remove moves > 50%
        
        removed = initial_rows - len(df)
        if removed > 0:
            self._log.warning(f"Removed {removed} rows due to price sanity checks")
            
        return df

    def validate_features(
        self,
        df: pd.DataFrame,
        target_col: str = "target_5d",
        structural_only: bool = False,
    ) -> pd.DataFrame:
        """
        Comprehensive feature validation for training readiness.

        If structural_only=True, only structural checks are applied
        (constant cols, tiny tickers, date coverage). Target clipping
        is left to train/val/test-specific preprocessors.
        """
        if df.empty:
            return df

        df = df.copy()
        initial_shape = df.shape

        # 1. Remove constant numerical columns (safe even in structural_only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
        if constant_cols:
            self._log.warning(f"Removing constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)

        # 2. Target outlier handling (skip when structural_only)
        if (not structural_only) and (target_col in df.columns):
            q_999 = df[target_col].quantile(0.999)
            q_001 = df[target_col].quantile(0.001)
            outliers = ((df[target_col] > q_999) | (df[target_col] < q_001)).sum()
            if outliers > 0:
                self._log.warning(f"Clipping {outliers} extreme outliers in {target_col}")
                df[target_col] = df[target_col].clip(lower=q_001, upper=q_999)

        # 3. Check for sufficient data per ticker
        if "ticker" in df.columns:
            ticker_counts = df["ticker"].value_counts()
            insufficient_tickers = ticker_counts[ticker_counts < 10].index.tolist()
            if insufficient_tickers:
                self._log.warning(
                    f"Removing tickers with insufficient data: {insufficient_tickers}"
                )
                df = df[~df["ticker"].isin(insufficient_tickers)]

        # 4. Validate date continuity (informational)
        date_coverage = self._check_date_continuity(df)
        if date_coverage < 0.8:
            self._log.warning(f"Low date coverage: {date_coverage:.1%}")

        final_shape = df.shape
        self._log.info(f"Feature validation: {initial_shape} -> {final_shape}")

        return df

    def _check_date_continuity(self, df: pd.DataFrame) -> float:
        """Check if we have reasonably continuous date coverage"""
        if 'date' not in df.columns:
            return 1.0
            
        dates = pd.to_datetime(df['date']).dt.date
        unique_dates = sorted(dates.unique())
        
        if len(unique_dates) < 2:
            return 0.0
            
        date_range = (max(unique_dates) - min(unique_dates)).days
        expected_business_days = date_range * 5 / 7  # Rough estimate
        coverage = len(unique_dates) / expected_business_days
        
        return min(coverage, 1.0)  # Cap at 100%

    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a data quality report for debugging"""
        report = {
            "total_rows": len(df),
            "total_tickers": df['ticker'].nunique(),
            "date_range": {
                "start": df['date'].min(),
                "end": df['date'].max(),
                "days": (pd.to_datetime(df['date'].max()) - pd.to_datetime(df['date'].min())).days
            },
            "missing_values": df.isnull().sum().to_dict(),
            "target_stats": {}
        }
        
        if 'target_5d' in df.columns:
            report["target_stats"] = {
                "mean": float(df['target_5d'].mean()),
                "std": float(df['target_5d'].std()),
                "min": float(df['target_5d'].min()),
                "max": float(df['target_5d'].max()),
                "outliers_5sigma": len(df[np.abs(df['target_5d'] - df['target_5d'].mean()) > 5 * df['target_5d'].std()])
            }
        
        return report
    
    @staticmethod
    def normalize_block(df: pd.DataFrame | None, required: Iterable[str] = ("date","ticker")) -> pd.DataFrame:
        """Ensure presence/order of keys, normalize dtypes; return empty with required cols if input empty."""
        req = tuple(required)
        if df is None or df.empty:
            return pd.DataFrame(columns=list(req))

        out = df.copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        if "ticker" in out.columns:
            out["ticker"] = out["ticker"].astype(str).str.upper()

        # If required keys missing, return empty shell with required headers
        if any(c not in out.columns for c in req):
            return pd.DataFrame(columns=list(req))

        cols = list(req) + [c for c in out.columns if c not in req]
        out = out[cols]
        return out
