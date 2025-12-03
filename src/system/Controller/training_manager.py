# src/system/Controller/training_manager.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from system.Model.artifact_store import ArtifactStore
from system.Model.experiment_config import (
    ExperimentConfig,
    LstmModelConfig,
    TransformerModelConfig,
    GnnModelConfig,
    BaselineModelConfig,
)

from system.Model.preprocessing.splitter import DataSplitter
from system.Model.preprocessing.scaler import (
    StandardScaler,
    LabelNormalizer,
    add_cross_sectional_label_views,
)
from system.Model.preprocessing.sequence_dataset import (
    SequenceDataset,
    VolSequenceDataset,
)
from system.Model.preprocessing.graph_dataset import (
    GraphSequenceDataset,
    GraphSequenceConfig,
    build_mixed_adjacency,  # <<— use mixed adjacency
)

from system.Model.trainingModels.lstm import LstmModel
from system.Model.training.lstm_trainer import LstmModelTrainer

from system.Model.trainingModels.transformer import TransformerModel
from system.Model.training.transformer_trainer import TransformerModelTrainer

from system.Model.trainingModels.gnn import CrossAssetGNNModel
from system.Model.training.gnn_trainer import GnnModelTrainer

from system.Model.utils.repro import _seed_worker


class TrainingManager:
    """
    Orchestrates training for:
      - LSTM (per-asset sequence)
      - Transformer (per-asset sequence)
      - GNN (cross-asset temporal graph)

    Steps:
      - load merged engineered features
      - select feature columns
      - chronological split
      - scale features
      - build datasets + loaders
      - instantiate model + trainer
      - train and generate predictions
      - persist preds + checkpoint via ArtifactStore
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    # ---------- public ----------

    def run(self, cfg: ExperimentConfig) -> pd.DataFrame:
        model_cfg = cfg.model

        features_path: Path = self._store.technical_features_path()
        if not features_path.exists():
            raise FileNotFoundError(
                f"Missing features: {features_path}. Run feature engineering first."
            )

        df = self._store.load_csv(features_path)
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str)

        feature_cols = self._select_feature_cols(df)

        if isinstance(model_cfg, GnnModelConfig):
            preds = self._run_gnn(cfg, df, feature_cols)
        elif isinstance(model_cfg, TransformerModelConfig):
            preds = self._run_transformer(cfg, df, feature_cols)
        elif isinstance(model_cfg, LstmModelConfig):
            preds = self._run_lstm(cfg, df, feature_cols)
        elif isinstance(model_cfg, BaselineModelConfig):
            preds = self._run_baseline(cfg, df)
        else:
            raise ValueError(f"Unsupported model config type: {type(model_cfg)}")

        # preds already ['date','ticker','pred_5d']
        self._store.save_csv(preds, self._store.predictions_path(), index=False)
        return preds

    # ---------- shared helpers ----------

    def _select_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """
        Select engineered feature columns shared across models.
        """
        prefixes = (
            "ret_",
            "log_ret_",
            "ma_",
            "vol_",
            "sent_",
            "has_earnings",
            "days_to_earnings",
            "corr_",
            "graph_",
            "stat_",
        )
        cand = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]

        n = len(df)
        min_non_null_frac = 0.05
        feat_cols: list[str] = []

        for c in cand:
            frac = df[c].notna().sum() / n
            if frac >= min_non_null_frac:
                feat_cols.append(c)

        if not feat_cols:
            raise ValueError("No engineered feature columns found after filtering.")
        return feat_cols

    def _make_loaders_seq(
        self,
        train_ds: SequenceDataset,
        val_ds: SequenceDataset,
        test_ds: SequenceDataset,
        batch_sizes=(128, 512, 512),
    ):
        num_workers = int(os.getenv("AI_INV_WORKERS", "0"))
        persistent_workers = num_workers > 0

        g = torch.Generator()

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_sizes[0],
            shuffle=True,
            generator=g,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_sizes[1],
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_sizes[2],
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )
        return train_loader, val_loader, test_loader

    def _pick_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ---------- LSTM path ----------

    def _run_lstm(
        self,
        cfg: ExperimentConfig,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        m: LstmModelConfig = cfg.model  # type: ignore[assignment]

        # Build “vol inputs” subset: priors + existing vol features
        # we only use technical and prior features for LStM prediction of returns + vol
        vol_input_cols = [
            c
            for c in feature_cols
            if (
                c.startswith("stat_")
                or c.startswith("vol_")
                or c.startswith("ret_")
                or c.startswith("log_ret_")
            )
        ]
        if not vol_input_cols:
            raise ValueError(
                "No suitable volatility-related feature columns found "
                "for LSTM input. Check StatVolFeatureBuilder / TechnicalFeatureBuilder."
            )

        # ---------- 1) Construct vol labels in log-space ----------
        # To stabilize MSE on variance, work in log(1 + var) space.
        eps = 1e-8
        df = df.copy()
        df["vol_real_5d"] = np.log1p(df["stat_realized_var_5d"].clip(lower=0.0) + eps)
        df["vol_garch_5d"] = np.log1p(df["stat_sigma2_ewm"].clip(lower=0.0) + eps)

        # 1) Chronological split (no lookahead)
        splitter = DataSplitter()
        train_df, val_df, test_df = splitter.chronological_split(df, by="date")

        # This helper compute things like target_5d_cs_z, target_5d_cs_rank
        # without changing target_5d itself.
        train_df = add_cross_sectional_label_views(train_df, label_col="target_5d")
        val_df = add_cross_sectional_label_views(val_df, label_col="target_5d")
        test_df = add_cross_sectional_label_views(test_df, label_col="target_5d")

        # 3) LABEL NORMALIZATION (train-only fit -> transform all splits)
        label_norm = LabelNormalizer(col="target_5d")
        label_norm.fit(train_df)  # uses ONLY training data stats
        train_df = label_norm.transform(train_df)
        val_df = label_norm.transform(val_df)
        test_df = label_norm.transform(test_df)

        # From here on, we train on the normalized target column:
        ret_target_col = "target_5d_norm"

        # ---------- 4) NORMALIZE vol labels (train-only stats) ----------
        # reuse StandardScaler for vol columns
        vol_scaler = StandardScaler(cols=["vol_real_5d", "vol_garch_5d"])
        vol_scaler.fit(train_df)
        train_df = vol_scaler.transform(train_df)
        val_df = vol_scaler.transform(val_df)
        test_df = vol_scaler.transform(test_df)

        vol_real_col = "vol_real_5d"
        vol_garch_col = "vol_garch_5d"

        # ---------- 5) FEATURE SCALING for inputs ----------
        scaler = StandardScaler()
        scaler.fit(train_df, feature_cols=vol_input_cols)
        train_df = scaler.transform(train_df, feature_cols=vol_input_cols)
        val_df = scaler.transform(val_df, feature_cols=vol_input_cols)
        test_df = scaler.transform(test_df, feature_cols=vol_input_cols)

        # ---------- 6) Build GINN-style sequence datasets ----------
        window = m.window

        train_ds = VolSequenceDataset(
            train_df,
            feature_cols=vol_input_cols,
            ret_target_col=ret_target_col,
            vol_real_col=vol_real_col,
            vol_garch_col=vol_garch_col,
            window=window,
        )
        val_ds = VolSequenceDataset(
            val_df,
            feature_cols=vol_input_cols,
            ret_target_col=ret_target_col,
            vol_real_col=vol_real_col,
            vol_garch_col=vol_garch_col,
            window=window,
        )
        test_ds = VolSequenceDataset(
            test_df,
            feature_cols=vol_input_cols,
            ret_target_col=ret_target_col,
            vol_real_col=vol_real_col,
            vol_garch_col=vol_garch_col,
            window=window,
        )

        seq_len = getattr(cfg.model, "seq_len", window)
        horizon = getattr(cfg.model, "horizon", 5)

        n_train = len(train_ds)
        n_val = len(val_ds)
        n_test = len(test_ds)

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            raise ValueError(
                "INSUFFICIENT_DATA: "
                f"seq_len={seq_len}, horizon={horizon}, "
                f"split_sizes(train/val/test)={n_train}/{n_val}/{n_test}. "
                "Increase the date range to ≥120 business days (~6 months); "
                "~252 business days (~1 year) recommended."
            )

        train_loader, val_loader, test_loader = self._make_loaders_seq(train_ds, val_ds, test_ds)

        device = self._pick_device()
        model = LstmModel(input_size=len(vol_input_cols), hidden=m.hidden)

        trainer = LstmModelTrainer(
            model=model,
            epochs=m.epochs,
            lr=m.lr,
            device=device,
            lambda_vol=m.lambda_vol,
            w_vol=m.w_vol,
        )

        # 6) Train in normalized space
        trainer.fit(train_loader, val_loader)

        # 7) Predict in normalized space, then map back to raw returns
        preds_df = trainer.predict(
            test_loader,
            dates=test_ds.dates,
            tickers=test_ds.tickers,
        )
        # At this point, the network's output is on the same scale as target_5d_norm.
        # We rename and then invert:
        preds_df.rename(columns={"pred_5d": "pred_5d_norm"}, inplace=True)
        preds_df["pred_5d"] = label_norm.inverse_array(
            preds_df["pred_5d_norm"].to_numpy(dtype=np.float32)
        )

        # 8) checkpoint (include label normalizer for reproducibility)
        ckpt = {
            "model_type": "lstm",
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "features": feature_cols,
            "window": window,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_std": scaler.std_.tolist(),
            "label_mean": float(label_norm.mean_),
            "label_std": float(label_norm.std_),
            "label_clip": (
                float(label_norm.q_low_),
                float(label_norm.q_high_),
            ),
            "lambda_vol": float(m.lambda_vol),
            "w_vol": float(m.w_vol),
        }
        try:
            torch.save(ckpt, self._store.model_checkpoint_path())
        except Exception as e:
            print(f"[TrainingManager] Warning: could not save LSTM checkpoint: {e}")

        return preds_df

    # ---------- Transformer path ----------

    def _run_transformer(
        self,
        cfg: ExperimentConfig,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        m: TransformerModelConfig = cfg.model  # type: ignore[assignment]

        # 1) Chronological split (no lookahead)
        splitter = DataSplitter()
        train_df, val_df, test_df = splitter.chronological_split(df, by="date")

        # This helper compute things like target_5d_cs_z, target_5d_cs_rank
        # without changing target_5d itself.
        train_df = add_cross_sectional_label_views(train_df, label_col="target_5d")
        val_df = add_cross_sectional_label_views(val_df, label_col="target_5d")
        test_df = add_cross_sectional_label_views(test_df, label_col="target_5d")

        # 3) LABEL NORMALIZATION (train-only fit -> transform all splits)
        label_norm = LabelNormalizer(col="target_5d")
        label_norm.fit(train_df)  # uses ONLY training data stats
        train_df = label_norm.transform(train_df)
        val_df = label_norm.transform(val_df)
        test_df = label_norm.transform(test_df)

        # From here on, we train on the normalized target column:
        target_col = "target_5d_norm"

        # 4) MASTER feature selection (NO stat_*/GARCH features here)
        #    We want technical + corr (+ graph/sent/events if present).
        master_input_cols: list[str] = [
            c
            for c in feature_cols
            if (
                c.startswith("ret_")
                or c.startswith("log_ret_")
                or c.startswith("ma_")
                or c.startswith("vol_")
                or c.startswith("corr_")
            )
            and not c.startswith("stat_")
        ]

        if not master_input_cols:
            raise ValueError(
                "No suitable MASTER Transformer feature columns found. "
                "Ensure technical and correlation features are enabled, "
                "and that feature engineering has run."
            )

        # 5) FEATURE SCALING (train-only fit -> transform all splits)
        scaler = StandardScaler()
        scaler.fit(train_df, feature_cols=master_input_cols)
        train_df = scaler.transform(train_df, feature_cols=master_input_cols)
        val_df = scaler.transform(val_df, feature_cols=master_input_cols)
        test_df = scaler.transform(test_df, feature_cols=master_input_cols)

        # 6) Build sequence datasets
        window = m.window

        train_ds = SequenceDataset.from_dataframe(train_df, master_input_cols, target_col, window)
        val_ds = SequenceDataset.from_dataframe(val_df, master_input_cols, target_col, window)
        test_ds = SequenceDataset.from_dataframe(test_df, master_input_cols, target_col, window)

        seq_len = getattr(cfg.model, "seq_len", window)
        horizon = getattr(cfg.model, "horizon", 5)

        n_train = len(train_ds)
        n_val = len(val_ds)
        n_test = len(test_ds)

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            raise ValueError(
                "INSUFFICIENT_DATA: "
                f"seq_len={seq_len}, horizon={horizon}, "
                f"split_sizes(train/val/test)={n_train}/{n_val}/{n_test}. "
                "Increase the date range to ≥120 business days (~6 months); "
                "~252 business days (~1 year) recommended."
            )

        train_loader, val_loader, test_loader = self._make_loaders_seq(
            train_ds, val_ds, test_ds, batch_sizes=(256, 512, 512)
        )

        # 7) Build MASTER-style Transformer model + trainer
        device = self._pick_device()

        from system.Model.trainingModels.transformer import TransformerConfig

        model_cfg = TransformerConfig(
            d_model=m.d_model,
            nhead=m.nhead,
            num_layers=m.num_layers,
            dim_feedforward=m.dim_feedforward,
            dropout=m.dropout,
            pooling=m.pooling,
        )

        model = TransformerModel(
            input_size=len(master_input_cols),
            cfg=model_cfg,
        )

        trainer = TransformerModelTrainer(
            model=model,
            epochs=m.epochs,
            lr=m.lr,
            device=device,
            patience=m.patience,
            weight_decay=m.weight_decay,
            max_grad_norm=m.max_grad_norm,
        )

        # 8) Train in normalized space
        trainer.fit(train_loader, val_loader)

        # 9) Predict in normalized space, then map back to raw returns
        preds_df = trainer.predict(
            test_loader,
            dates=test_ds.dates,
            tickers=test_ds.tickers,
        )
        preds_df.rename(columns={"pred_5d": "pred_5d_norm"}, inplace=True)
        preds_df["pred_5d"] = label_norm.inverse_array(
            preds_df["pred_5d_norm"].to_numpy(dtype=np.float32)
        )

        # 10) Checkpoint MASTER-style Transformer
        ckpt = {
            "model_type": "transformer",
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "features": master_input_cols,
            "window": window,
            "d_model": m.d_model,
            "nhead": m.nhead,
            "num_layers": m.num_layers,
            "dim_feedforward": m.dim_feedforward,
            "dropout": m.dropout,
            "pooling": m.pooling,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_std": scaler.std_.tolist(),
            "label_mean": float(label_norm.mean_),
            "label_std": float(label_norm.std_),
            "label_clip": (
                float(label_norm.q_low_),
                float(label_norm.q_high_),
            ),
        }
        try:
            torch.save(ckpt, self._store.model_checkpoint_path())
        except Exception as e:
            print(f"[TrainingManager] Warning: could not save Transformer checkpoint: {e}")

        return preds_df

    # ---------- GNN path (cross-asset) ----------

    def _run_gnn(
        self,
        cfg: ExperimentConfig,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        m: GnnModelConfig = cfg.model  # type: ignore[assignment]

        splitter = DataSplitter()
        train_df, val_df, test_df = splitter.chronological_split(df, by="date")

        train_df = add_cross_sectional_label_views(train_df, label_col="target_5d")
        val_df = add_cross_sectional_label_views(val_df, label_col="target_5d")
        test_df = add_cross_sectional_label_views(test_df, label_col="target_5d")

        # LABEL NORMALIZATION
        label_norm = LabelNormalizer(col="target_5d")
        label_norm.fit(train_df)
        train_df = label_norm.transform(train_df)
        val_df = label_norm.transform(val_df)
        test_df = label_norm.transform(test_df)

        target_col = "target_5d_norm"

        scaler = StandardScaler()
        scaler.fit(train_df, feature_cols=feature_cols)
        train_df = scaler.transform(train_df, feature_cols=feature_cols)
        val_df = scaler.transform(val_df, feature_cols=feature_cols)
        test_df = scaler.transform(test_df, feature_cols=feature_cols)

        window = m.window
        universe = [t.upper() for t in cfg.tickers]

        gcfg = GraphSequenceConfig(window=window, target_col=target_col, min_coverage=0.5)

        train_ds = GraphSequenceDataset(train_df, universe, feature_cols, gcfg)
        val_ds = GraphSequenceDataset(val_df, universe, feature_cols, gcfg)
        test_ds = GraphSequenceDataset(test_df, universe, feature_cols, gcfg)

        seq_len = getattr(cfg.model, "seq_len", 60)
        horizon = getattr(cfg.model, "horizon", 5)

        n_train = len(train_ds)
        n_val = len(val_ds)
        n_test = len(test_ds)

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            raise ValueError(
                "INSUFFICIENT_DATA: "
                f"seq_len={seq_len}, horizon={horizon}, "
                f"split_sizes(train/val/test)={n_train}/{n_val}/{n_test}. "
                "Increase the date range to ≥120 business days (~6 months); "
                "~252 business days (~1 year) recommended."
            )

        # ---- custom collate: keep dates as python objects ----
        def _graph_collate(batch):
            xs, ys, masks, dates = zip(*batch)
            x = torch.stack(xs, dim=0)  # [B,N,W,F]
            y = torch.stack(ys, dim=0)  # [B,N]
            m = torch.stack(masks, dim=0)  # [B,N]
            return x, y, m, list(dates)

        num_workers = int(os.getenv("AI_INV_WORKERS", "0"))
        persistent_workers = num_workers > 0

        train_loader = DataLoader(
            train_ds,
            batch_size=8,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_graph_collate,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=8,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_graph_collate,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=8,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_graph_collate,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )

        # ---- adjacency: use ONLY information available up to end of training period ----
        if "ret_1d" not in df.columns:
            raise ValueError("GNN model requires 'ret_1d' column to build adjacency.")

        last_train_date = train_df["date"].max()
        adj_source = df[df["date"] <= last_train_date].copy()

        A_np = build_mixed_adjacency(
            adj_source,
            universe,
            feature_cols,
            ret_col="ret_1d",
            extra_prefixes=(
                "sent_",
                "corr_",
                "graph_",
                "has_earnings",
                "days_to_earnings",
            ),
            base_corr_threshold=0.3,
            extra_corr_threshold=0.3,
            alpha=0.5,
            self_loops=True,
        )
        A = torch.from_numpy(A_np.astype(np.float32))

        device = self._pick_device()

        model = CrossAssetGNNModel(
            num_nodes=len(universe),
            feature_dim=len(feature_cols),
            window=window,
            adj=A,
            temp_hidden=m.hidden,
            gnn_hidden=m.hidden,
            gnn_layers=m.layers,
        )

        trainer = GnnModelTrainer(
            model=model,
            epochs=m.epochs,
            lr=m.lr,
            device=device,
        )

        trainer.fit(train_loader, val_loader)
        preds_df = trainer.predict(
            test_loader,
            tickers=universe,
        )

        # network output is in normalized units
        preds_df.rename(columns={"pred_5d": "pred_5d_norm"}, inplace=True)
        preds_df["pred_5d"] = label_norm.inverse_array(
            preds_df["pred_5d_norm"].to_numpy(dtype=np.float32)
        )

        ckpt = {
            "model_type": "gnn",
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "features": feature_cols,
            "window": window,
            "hidden": m.hidden,
            "layers": m.layers,
            "adjacency": A_np.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_std": scaler.std_.tolist(),
        }
        try:
            torch.save(ckpt, self._store.model_checkpoint_path())
        except Exception as e:
            print(f"[TrainingManager] Warning: could not save GNN checkpoint: {e}")

        return preds_df

    # ---------- Baseline model path ----------
    def _run_baseline(
        self,
        cfg: ExperimentConfig,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simple baseline forecaster.

        - Splits data train/val/test by date (same as other models).
        - Computes a purely backward-looking 5-day momentum per ticker:
              past5d_ret_t = sum_{i=t-5..t-1} ret_1d_i
        - Uses this as pred_5d on the TEST set dates only.
        """
        m: BaselineModelConfig = cfg.model  # type: ignore[assignment]

        # returns required for baseline
        if "ret_1d" not in df.columns:
            raise ValueError(
                "BaselineModelConfig requires 'ret_1d' in the feature table "
                "(from TechnicalFeatureBuilder)."
            )
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df = df.sort_values(["ticker", "date"])

        # horizon, using 5 days of past returns as momentum proxy
        H = int(m.horizon)

        # create past 5 returns to use as the pred_5d for the baseline model
        df["past_5d_ret"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(window=H, min_periods=H)
            .sum()
            .shift(1)  # <-- ensures we only use returns up to t-1
            .reset_index(level=0, drop=True)
        )

        # 2) Chronological split (only to define test window)
        splitter = DataSplitter()
        train_df, val_df, test_df = splitter.chronological_split(df, by="date")

        # 3) Choose baseline rule
        if m.baseline_type == "zero":
            test_df["pred_5d"] = 0.0
        elif m.baseline_type == "past5d_mom":
            test_df["pred_5d"] = test_df["past_5d_ret"].fillna(0.0)
        else:
            raise ValueError(f"Unknown baseline_type: {m.baseline_type}")

        # 4) Build preds_df only for TEST set, as expected by RunManager/Allocator
        preds_df = test_df[["date", "ticker", "pred_5d"]].copy()
        preds_df = preds_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        return preds_df
