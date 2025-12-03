# src/system/Model/allocation/meanvar.py
from __future__ import annotations

import numpy as np
import pandas as pd


class MeanVarianceAllocator:
    """
    Markowitz-style allocator with practical stabilizations.

    For each rebalance date d:
      - Use trailing realized returns up to d-1 to estimate covariance Σ_d.
      - Use model predictions at d as cross-sectional expected returns μ_d.
      - Build two portfolios:
          * w_mv : minimum-variance (risk-only)
          * w_mu : mean-variance tangency-like (return-tilted)
      - Mix them using a convex combination controlled by `risk_aversion`:
            λ = 1 / (1 + risk_aversion)   in (0,1]
            w* = (1 - λ) * w_mv + λ * w_mu
      - Enforce:
          * long-only w_i ≥ 0
          * optional per-asset cap w_i ≤ weight_cap
          * fully invested sum(w) = 1 (no cash here; cash can be modeled upstream).

    This is close to how many production systems approximate Markowitz:
      - Shrinkage Σ for robustness
      - Convex mix between min-var and return-tilted portfolio
      - Simple box constraints via clipping + renormalization.
    """

    def __init__(
        self,
        risk_aversion: float = 5.0,
        cov_lookback: int = 60,
        weight_cap: float | None = None,
        shrinkage: float = 0.1,
        jitter: float = 1e-6,
    ) -> None:
        if cov_lookback <= 0:
            raise ValueError("cov_lookback must be positive")

        if shrinkage < 0.0 or shrinkage >= 1.0:
            raise ValueError("shrinkage must be in [0,1)")

        self.risk_aversion = float(risk_aversion)
        self.cov_lookback = int(cov_lookback)
        self.weight_cap = float(weight_cap) if weight_cap is not None else None
        self.shrinkage = float(shrinkage)
        self.jitter = float(jitter)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _min_var_weights(Sigma: np.ndarray) -> np.ndarray:
        """
        Compute unconstrained minimum-variance portfolio:
            w_mv ∝ Σ^{-1} 1

        Returns unnormalized vector; caller handles normalization and constraints.
        """
        n = Sigma.shape[0]
        ones = np.ones(n, dtype=np.float64)
        try:
            inv_S = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            inv_S = np.linalg.pinv(Sigma)
        w_mv = inv_S @ ones  # might have negatives; we'll fix later
        return w_mv

    @staticmethod
    def _return_tilted_weights(Sigma: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Compute Markowitz-style return-tilted direction:
            w_mu ∝ Σ^{-1} μ

        Returns unnormalized vector; caller handles normalization and constraints.
        """
        try:
            inv_S = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            inv_S = np.linalg.pinv(Sigma)
        w_mu = inv_S @ mu
        return w_mu

    @staticmethod
    def _normalize_long_only(
        w: np.ndarray,
        weight_cap: float | None,
    ) -> np.ndarray:
        """
        Project weights to a simple long-only box:
            w_i >= 0, optionally w_i <= cap, and sum(w) = 1.

        This is NOT a full QP, but a robust, interpretable projection used widely
        as a practical approximation in industry systems.
        """
        w = np.asarray(w, dtype=np.float64)

        # Enforce non-negativity
        w = np.maximum(w, 0.0)

        if weight_cap is not None and weight_cap > 0.0:
            cap = float(weight_cap)
            w = np.minimum(w, cap)

        s = w.sum()
        if s <= 0.0:
            # Fallback: equal-weight
            n = w.shape[0]
            return np.full(n, 1.0 / n, dtype=np.float64)

        return w / s

    def allocate(self, preds: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        preds : DataFrame
            Columns ['date','ticker','pred_5d'] – model's horizon-5d predictions.
        rets : DataFrame
            Columns ['date','ticker','ret'] – realized daily returns.

        Returns
        -------
        DataFrame
            ['date','ticker','weight'] – fully invested, long-only weights per date.
        """
        if preds.empty:
            return pd.DataFrame(columns=["date", "ticker", "weight"])

        preds = preds.copy()
        rets = rets.copy()

        preds["date"] = pd.to_datetime(preds["date"])
        rets["date"] = pd.to_datetime(rets["date"])
        preds["ticker"] = preds["ticker"].astype(str).str.upper()
        rets["ticker"] = rets["ticker"].astype(str).str.upper()

        # We only ever allocate among tickers we have signals for.
        # Covariance is always built on this moving universe.
        all_tickers = sorted(preds["ticker"].unique())

        # Pivot realized returns [T, N_all], then we will subset columns per date.
        ret_piv = (
            rets.pivot(index="date", columns="ticker", values="ret")
            .reindex(columns=all_tickers)
            .sort_index()
        )

        out_rows: list[dict] = []

        # Loop over prediction dates in chronological order
        for d, grp in preds.groupby("date", sort=True):
            if d not in ret_piv.index:
                # Prediction date beyond last realized date; cannot compute Σ
                continue

            # Universe for this date = tickers that have predictions at d
            active_tickers = sorted(grp["ticker"].unique())
            if len(active_tickers) == 0:
                continue

            # Trailing window up to d-1 for active tickers only
            idx = ret_piv.index.get_loc(d)
            if isinstance(idx, slice):
                idx = idx.start

            start = max(0, idx - self.cov_lookback)
            hist = ret_piv.iloc[start:idx][active_tickers]

            # Require some reasonable amount of history
            min_hist = max(10, self.cov_lookback // 3)
            if len(hist) < min_hist:
                # Not enough data to estimate covariance robustly
                continue

            # Drop columns with extremely sparse data
            non_null_counts = hist.notna().sum(axis=0)
            keep_cols = non_null_counts[non_null_counts >= 5].index.tolist()
            if len(keep_cols) < 2:
                continue

            hist = hist[keep_cols]
            n = hist.shape[1]

            # Sample covariance, then diagonal shrinkage for robustness
            Sigma = hist.cov().to_numpy(dtype=np.float64)
            Sigma = np.nan_to_num(Sigma, nan=0.0)

            # Basic check: if all-zero, fallback to identity
            if not np.isfinite(Sigma).all() or np.allclose(Sigma, 0.0):
                Sigma = np.eye(n, dtype=np.float64)

            # Diagonal shrinkage: Σ_shrunk = (1-α) Σ + α * diag(Σ)
            if self.shrinkage > 0.0:
                diag = np.diag(np.diag(Sigma))
                alpha = self.shrinkage
                Sigma = (1.0 - alpha) * Sigma + alpha * diag

            # Add jitter to ensure positive definiteness for inversion
            Sigma = Sigma + self.jitter * np.eye(n, dtype=np.float64)

            # Expected returns vector μ from preds (aligned to keep_cols)
            mu = (
                grp.set_index("ticker")
                .reindex(keep_cols)["pred_5d"]
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )

            # Degenerate case: if μ is all zero, just go min-var
            if not np.isfinite(mu).any() or np.allclose(mu, 0.0):
                mu = np.zeros_like(mu)

            # Compute minimum-variance and return-tilted directions
            w_mv = self._min_var_weights(Sigma)
            w_mu = self._return_tilted_weights(Sigma, mu)

            # Normalize both directions to sum to 1 (before constraints)
            def _safe_normalize(v: np.ndarray) -> np.ndarray:
                v = np.asarray(v, dtype=np.float64)
                s = v.sum()
                if not np.isfinite(s) or abs(s) < 1e-12:
                    return np.full_like(v, 1.0 / len(v))
                return v / s

            w_mv = _safe_normalize(w_mv)
            w_mu = _safe_normalize(w_mu)

            # Mix min-var and return-tilted using a convex combination
            # λ = 1 / (1 + RA) ∈ (0,1]; RA large -> closer to min-var.
            ra = max(self.risk_aversion, 0.0)
            lam = 1.0 / (1.0 + ra)  # 0 -> fully return-tilted, ∞ -> fully min-var
            w_mix = (1.0 - lam) * w_mv + lam * w_mu

            # Project to long-only + optional cap + sum-to-1
            w_final = self._normalize_long_only(w_mix, self.weight_cap)

            # Emit rows only for tickers we actually used (keep_cols)
            for t, wt in zip(keep_cols, w_final):
                if wt <= 0.0:
                    continue
                out_rows.append({"date": d.date(), "ticker": str(t), "weight": float(wt)})

        if not out_rows:
            return pd.DataFrame(columns=["date", "ticker", "weight"])

        out = (
            pd.DataFrame(out_rows)
            .groupby(["date", "ticker"], as_index=False)["weight"]
            .sum()
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )
        return out
