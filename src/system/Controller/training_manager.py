from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from system.Model.artifact_store import ArtifactStore
from system.Model.experiment_config import ExperimentConfig

# Your existing preprocessing utilities (you already planned/created these)
from system.Model.preprocessing.splitter import DataSplitter  # chronological split
from system.Model.preprocessing.scaler import StandardScaler  # fit on train, transform splits
from system.Model.preprocessing.sequence_dataset import SequenceDataset  # builds [T,F] windows

# Model + Trainer (domain)
from system.Model.trainingModels.lstm import LstmModel  # the torch.nn.Module
from system.Model.training.lstm_trainer import LstmModelTrainer  # pure trainer (no DS import)

# Utility for seeding DataLoader workers
from system.Model.utils.repro import _seed_worker



class TrainingManager:
    """
    Controller that orchestrates Phase-1 training.
      - loads engineered features
      - splits chronologically
      - scales using train stats
      - builds SequenceDataset windows
      - instantiates the model
      - calls the trainer with dataloaders
      - persists predictions/checkpoint via ArtifactStore
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def run(self, cfg: ExperimentConfig) -> pd.DataFrame:
        if cfg.model.type.lower() != "lstm":
            raise NotImplementedError("Phase-1 supports only 'lstm'")

        features_path: Path = self._store.technical_features_path()
        if not features_path.exists():
            raise FileNotFoundError(
                f"Missing features: {features_path}. Run technical feature engineering first."
            )

        df = self._store.load_csv(features_path)
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str)

        # Identify numeric feature columns (exclude non-features)
        engineered_prefixes = ("ret_", "log_ret_", "ma_", "vol_")
        feature_cols: list[str] = [
            c for c in df.columns if any(c.startswith(p) for p in engineered_prefixes)
        ]
        if not feature_cols:
            raise ValueError("No engineered feature columns found for LSTM input.")

        # --- 1) Chronological split (70/15/15) ---
        splitter = DataSplitter()
        train_df, val_df, test_df = splitter.chronological_split(
            df, ratios=(0.7, 0.15, 0.15), by="date"
        )

        # --- 2) Standardize using train stats only ---
        scaler = StandardScaler()
        scaler.fit(train_df, feature_cols=feature_cols)
        train_df = scaler.transform(train_df, feature_cols=feature_cols)
        val_df = scaler.transform(val_df, feature_cols=feature_cols)
        test_df = scaler.transform(test_df, feature_cols=feature_cols)

        # --- 3) Build rolling sequence datasets (your class) ---
        # Assumes your SequenceDataset has a classmethod like:
        #   from_dataframe(df, feature_cols, target_col, window) -> SequenceDataset
        window = cfg.model.window
        target_col = "target_5d"

        train_ds = SequenceDataset.from_dataframe(train_df, feature_cols, target_col, window)
        val_ds = SequenceDataset.from_dataframe(val_df, feature_cols, target_col, window)
        test_ds = SequenceDataset.from_dataframe(test_df, feature_cols, target_col, window)

        # --- 4) DataLoaders (deterministic shuffle) ---
        # You can choose multi-worker via env; default 0 for simplicity.
        num_workers = int(os.getenv("AI_INV_WORKERS", "0"))
        persistent_workers = num_workers > 0
        # Seeded generator guarantees deterministic batch order with shuffle=True
        g = torch.Generator()
        g.manual_seed(cfg.seeds.torch)

        train_loader = DataLoader(
            train_ds,
            batch_size=64,
            shuffle=True,
            generator=g,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=256,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )

        # --- 5) Instantiate the model (domain object), not the trainer ---
        input_size = len(feature_cols)
        model = LstmModel(input_size=input_size, hidden=cfg.model.hidden)

        # --- 6) Trainer only knows model + loaders; it does not reach into the filesystem ---
        trainer = LstmModelTrainer(
            model=model,
            epochs=cfg.model.epochs,
            lr=cfg.model.lr,
            device=self._pick_device(),
        )

        # Train & evaluate
        trainer.fit(train_loader, val_loader)
        preds_df = trainer.predict(test_loader, dates=test_ds.dates, tickers=test_ds.tickers)
        preds_df = preds_df.rename(columns={"y_pred": "pred_5d"})

        # --- 7) Persist outputs via the controller (ArtifactStore) ---
        self._store.save_csv(preds_df, self._store.predictions_path(), index=False)

        # Save checkpoint: model weights + scaler stats + metadata
        checkpoint = {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "features": feature_cols,
            "window": window,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_std": scaler.std_.tolist(),
        }
        try:
            torch.save(checkpoint, self._store.model_checkpoint_path())
        except Exception as e:
            print(f"[TrainingManager] Warning: could not save checkpoint: {e}")

        return preds_df

    def _pick_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
