from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd


@dataclass
class StatVolFeatureConfig:
    """
    Approximate AR(1) + GARCH(1,1)-style priors using only past data.

    Inputs (per (date, ticker)):
      - ret_1d  (backward-looking daily return)

    Outputs:
      - stat_mu             : AR-type conditional mean (rolling mean proxy)
      - stat_eps            : residual r_t - stat_mu_t
      - stat_sigma2_ewm     : EWMA of residual^2 (GARCH-like vol)
      - stat_realized_var_5d: future 5-day realized variance (label-like)
    """

    ar_window: int = 20  # rolling mean window
    garch_alpha: float = 0.05  # α in GARCH(1,1)
    garch_beta: float = 0.90  # β in GARCH(1,1)
    garch_omega: float = 1e-6  # ω (floor term)
    horizon: int = 5  # future H for realized variance


class StatVolFeatureBuilder:
    def __init__(self, cfg: Optional[StatVolFeatureConfig] = None) -> None:
        self.cfg = cfg or StatVolFeatureConfig()

    def build(self, tech_df: pd.DataFrame) -> pd.DataFrame:
        """
        tech_df: must contain ['date','ticker','ret_1d'] (from TechnicalFeatureBuilder).

        Returns a DataFrame keyed by (date, ticker) with:
            ['date','ticker',
             'stat_mu',
             'stat_eps',
             'stat_sigma2_ewm',
             'stat_realized_var_5d']
        """
        required = {"date", "ticker", "ret_1d"}
        if tech_df is None or tech_df.empty or not required.issubset(tech_df.columns):
            return pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "stat_mu",
                    "stat_eps",
                    "stat_sigma2_ewm",
                    "stat_realized_var_5d",
                ]
            )

        df = tech_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        rows: List[dict] = []
        H = int(self.cfg.horizon)
        w_ar = int(self.cfg.ar_window)

        alpha = float(self.cfg.garch_alpha)
        beta = float(self.cfg.garch_beta)
        omega = float(self.cfg.garch_omega)

        for tkr, g in df.groupby("ticker", sort=False):
            g = g.sort_values("date").copy()
            r = g["ret_1d"].astype(float).to_numpy()

            if r.size < max(w_ar, H + 5):
                continue

            # 1) AR-like conditional mean: rolling mean of past returns
            #    μ_t = mean(r_{t-w_ar+1 ... t})
            mu = pd.Series(r).rolling(w_ar, min_periods=w_ar).mean().to_numpy()

            # 2) residuals
            eps = r - np.nan_to_num(mu, nan=0.0)

            # 3) GARCH(1,1)-style conditional variance sigma2_t
            # -----------------------------------------------
            # clamp to sane region: alpha >= 0, beta >= 0, alpha+beta < 1
            beta = min(max(beta, 0.0), 0.999)
            alpha = min(max(alpha, 0.0), 0.999)
            if alpha + beta >= 0.999:
                scale = 0.999 / max(alpha + beta, 1e-6)
                alpha *= scale
                beta *= scale

            sigma2 = np.zeros_like(eps)

            # initial variance from first window of residuals
            if len(eps) >= w_ar:
                s0 = float(np.nanvar(eps[:w_ar]))
            else:
                s0 = float(np.nanvar(eps)) if len(eps) > 1 else 1e-6

            s_prev = max(s0, 1e-8)
            sigma2[0] = s_prev

            for i in range(1, len(eps)):
                # GARCH(1,1): sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}
                e2_prev = eps[i - 1] ** 2
                s_prev = omega + alpha * e2_prev + beta * s_prev
                s_prev = max(s_prev, 1e-12)  # avoid negative / zero
                sigma2[i] = s_prev

            # 4) realized variance over future H days (unchanged)
            r_series = pd.Series(r)
            rv_forward = r_series.rolling(H, min_periods=H).var().shift(-H).to_numpy()

            for i, (dt, mu_i, eps_i, s2_i, rv_i) in enumerate(
                zip(g["date"].to_numpy(), mu, eps, sigma2, rv_forward)
            ):
                if np.isnan(mu_i) or np.isnan(rv_i):
                    continue
                rows.append(
                    {
                        "date": pd.Timestamp(dt),
                        "ticker": tkr,
                        "stat_mu": float(mu_i),
                        "stat_eps": float(eps_i),
                        "stat_sigma2_ewm": float(s2_i),
                        "stat_realized_var_5d": float(max(rv_i, 0.0)),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "stat_mu",
                    "stat_eps",
                    "stat_sigma2_ewm",
                    "stat_realized_var_5d",
                ]
            )

        out = pd.DataFrame(rows)
        out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
        return out
