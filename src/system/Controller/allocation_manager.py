# src/system/Controller/allocation_manager.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from system.Model.artifact_store import ArtifactStore
from system.Model.experiment_config import ExperimentConfig

from system.Model.allocation.softmax import SoftmaxAllocator
from system.Model.allocation.meanvar import MeanVarianceAllocator
from system.Model.allocation.rl import RLAllocator


class AllocationManager:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def run(self, cfg: ExperimentConfig, preds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Dispatch to the appropriate allocator based on cfg.allocator.type.

        preds_df must be ['date','ticker','pred_5d'] as produced by TrainingManager.
        """
        a = cfg.allocator
        atype = a.type.lower()

        if preds_df.empty:
            # still write an empty weights file for consistency
            empty = pd.DataFrame(columns=["date", "ticker", "weight"])
            self._store.save_csv(empty, self._store.weights_path(), index=False)
            return empty

        # ---- softmax ----
        if atype == "softmax":
            alloc = SoftmaxAllocator(
                temperature=float(a.temperature),
                weight_cap=float(a.weight_cap),
            )
            weights = alloc.allocate(preds_df)

        # ---- mean-variance ----
        elif atype in ("meanvar", "mean-variance"):
            # need realized daily returns; pull from technical features
            feats_path: Path = self._store.technical_features_path()
            feats = self._store.load_csv(feats_path)

            if "ret_1d" not in feats.columns:
                raise ValueError("Mean-variance allocator requires 'ret_1d' in features CSV.")

            rets = (
                feats[["date", "ticker", "ret_1d"]]
                .rename(columns={"ret_1d": "ret"})
            )

            alloc = MeanVarianceAllocator(
                risk_aversion=float(a.risk_aversion or 5.0),
                cov_lookback=int(a.cov_lookback or 60),
                weight_cap=a.weight_cap,
            )
            weights = alloc.allocate(preds_df, rets)

        # ---- RL ----
        elif atype == "rl":
            # need realized target_5d as reward
            feats_path: Path = self._store.technical_features_path()
            feats = self._store.load_csv(feats_path)

            if "target_5d" not in feats.columns:
                raise ValueError("RL allocator requires 'target_5d' in features CSV.")

            targets = feats[["date", "ticker", "target_5d"]]

            alloc = RLAllocator(
                lr=float(a.rl_lr or 1e-2),
                epochs=int(a.rl_epochs or 50),
                weight_cap=a.weight_cap,
                trade_cost_bps=float(a.rl_trade_cost_bps or 0.0),
            )
            weights = alloc.allocate(preds_df, targets)

        else:
            raise ValueError(f"Unknown allocator type: {atype}")

        # persist
        self._store.save_csv(weights, self._store.weights_path(), index=False)
        return weights