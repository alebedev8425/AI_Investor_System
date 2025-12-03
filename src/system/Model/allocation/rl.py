# src/system/Model/allocation/rl_allocator.py
from __future__ import annotations

import numpy as np
import pandas as pd


class RLAllocator:
    """
    Minimal offline RL-style allocator with turnover costs.

    Policy (per date d):
        scores_i(d) = theta * pred_i(d)
        w(d) = softmax(scores(d))  (optionally capped long-only)

    Training:
        - We treat each date as one REINFORCE "episode step":
            reward(d) = w(d) · target_5d(d)  -  trade_cost * turnover(d)
        - turnover(d) ≈ 0.5 * || w(d) - w(d-1) ||_1  (fraction of portfolio traded)
        - Gradient estimate:
            grad_theta ≈ E_d [ (reward(d) - baseline) * Σ_i w_i(d)*(pred_i(d) - E_w[pred(d)]) ]
    """

    def __init__(
        self,
        lr: float = 1e-2,
        epochs: int = 50,
        weight_cap: float | None = None,
        trade_cost_bps: float = 0.0,
    ) -> None:
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.weight_cap = float(weight_cap) if weight_cap is not None else None
        self.trade_cost = float(trade_cost_bps) / 1e4  # bps -> fraction of notional
        self.theta = 1.0  # single scalar scale parameter for the signal

    # ---------- internals ----------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = x - np.max(x)  # numerical stability
        x = np.clip(x, -50.0, 50.0)
        e = np.exp(x)
        s = e.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full_like(x, 1.0 / len(x))
        return e / s

    def _compute_weights(self, preds_vec: np.ndarray) -> np.ndarray:
        """
        Convert prediction vector into a normalized weight vector
        using the current theta, with optional per-asset cap.
        """
        preds_vec = np.asarray(preds_vec, dtype=np.float64)
        preds_vec[~np.isfinite(preds_vec)] = 0.0

        scores = self.theta * preds_vec
        w = self._softmax(scores)

        if self.weight_cap is not None and self.weight_cap > 0.0:
            cap = float(self.weight_cap)
            w = np.minimum(w, cap)
            s = w.sum()
            if s <= 0.0:
                w[:] = 1.0 / len(w)
            else:
                w /= s

        return w

    def _train(self, steps: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Offline REINFORCE training over a sequence of (pred_vec, realized_vec) steps.
        Each step corresponds to one rebalance date in chronological order.
        """
        if not steps:
            return

        for _ in range(self.epochs):
            grad = 0.0
            rewards: list[float] = []

            # First pass: compute rewards with turnover cost for this epoch's theta
            prev_w = None
            precomputed: list[tuple[np.ndarray, float, float]] = []  # (preds_vec, reward, p_bar)

            for preds_vec, realized_vec in steps:
                preds_vec = np.asarray(preds_vec, dtype=np.float64)
                realized_vec = np.asarray(realized_vec, dtype=np.float64)

                w = self._compute_weights(preds_vec)

                # Turnover = fraction of portfolio traded since previous rebalance
                if prev_w is None:
                    turnover = np.sum(np.abs(w)) * 0.5  # from all-cash / zero weights
                else:
                    turnover = 0.5 * np.sum(np.abs(w - prev_w))

                prev_w = w

                gross_ret = float(np.dot(w, realized_vec))
                cost = self.trade_cost * turnover
                reward = gross_ret - cost

                # For REINFORCE: we need E_w[preds] under current policy
                p_bar = float(np.dot(w, preds_vec))

                rewards.append(reward)
                precomputed.append((preds_vec, reward, p_bar))

            # Baseline: mean reward over steps (variance reduction)
            baseline = float(np.mean(rewards)) if rewards else 0.0

            # Second pass: accumulate gradient using baseline
            for (preds_vec, reward, p_bar), (w_step, _, _) in zip(precomputed, precomputed):
                # Recompute w for consistency with current theta
                w = self._compute_weights(preds_vec)

                # ∂logπ_i/∂θ = preds_i - E_w[preds]
                dlogpi = preds_vec - p_bar
                # Expected gradient contribution is E[ (R - b) * Σ_i w_i dlogπ_i ]
                grad += (reward - baseline) * float(np.dot(w, dlogpi))

            grad /= len(steps)

            # Gradient ascent on theta (maximize expected reward)
            if np.isfinite(grad):
                self.theta += self.lr * grad
            # Optional: clip theta to a reasonable range to avoid blow-ups
            self.theta = float(np.clip(self.theta, -50.0, 50.0))

    # ---------- public ----------

    def allocate(self, preds_df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
        """
        preds_df:   ['date','ticker','pred_5d']
        targets_df: ['date','ticker','target_5d']  (realized 5d returns)

        Returns:
            DataFrame['date','ticker','weight']

        NOTE:
            This method *fits* theta offline on the entire joined (pred, target)
            panel and then emits weights for ALL dates in preds_df using that
            fitted theta. For strict out-of-sample testing, split your data and
            reuse the same RLAllocator instance across fits and allocates.
        """
        if preds_df.empty:
            return pd.DataFrame(columns=["date", "ticker", "weight"])

        preds = preds_df.copy()
        tdf = targets_df.copy()

        preds["date"] = pd.to_datetime(preds["date"])
        tdf["date"] = pd.to_datetime(tdf["date"])
        preds["ticker"] = preds["ticker"].astype(str).str.upper()
        tdf["ticker"] = tdf["ticker"].astype(str).str.upper()

        tickers = sorted(preds["ticker"].unique())

        # join predictions with realized target_5d (reward proxy)
        joined = preds.merge(
            tdf[["date", "ticker", "target_5d"]],
            on=["date", "ticker"],
            how="inner",
        ).dropna(subset=["target_5d"])

        if joined.empty:
            # fallback: equal weights for all dates
            rows = []
            for d, grp in preds.groupby("date"):
                n = len(tickers)
                w = 1.0 / max(1, n)
                for t in tickers:
                    rows.append({"date": d.date(), "ticker": t, "weight": w})
            return pd.DataFrame(rows)

        # Build training steps by date in chronological order
        steps: list[tuple[np.ndarray, np.ndarray]] = []
        for d, grp in joined.sort_values("date").groupby("date", sort=True):
            v = grp.set_index("ticker").reindex(tickers)
            p_vec = v["pred_5d"].fillna(0.0).to_numpy(dtype=float)
            r_vec = v["target_5d"].fillna(0.0).to_numpy(dtype=float)
            steps.append((p_vec, r_vec))

        # Train theta offline
        self._train(steps)

        # Now generate weights for ALL prediction dates using learned theta
        out_rows: list[dict] = []
        for d, grp in preds.groupby("date", sort=True):
            v = grp.set_index("ticker").reindex(tickers)
            p_vec = v["pred_5d"].fillna(0.0).to_numpy(dtype=float)
            w = self._compute_weights(p_vec)

            for t, wt in zip(tickers, w):
                out_rows.append({"date": d.date(), "ticker": t, "weight": float(wt)})

        return (
            pd.DataFrame(out_rows)
              .sort_values(["date", "ticker"])
              .reset_index(drop=True)
        )