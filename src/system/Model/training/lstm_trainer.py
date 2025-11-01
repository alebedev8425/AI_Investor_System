# src/system/Model/training/lstm_trainer.py
from __future__ import annotations

from typing import Optional, Sequence
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


class LstmModelTrainer:
    """
    Phase-1 trainer:
      - trains a provided model with (train,val) loaders
      - early-stops on validation loss (patience)
      - returns best state_dict from fit()
      - produces a predictions DataFrame in predict()

    Notes:
      * No filesystem here. Saving is the controller's job (TrainingManager).
      * No dataset construction here. DataLoaders are injected by the controller.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        epochs: int,
        lr: float,
        device: torch.device,
        patience: int = 5,
    ) -> None:
        self.model = model.to(device)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = device
        self.patience = int(patience)
        self._best_state: Optional[dict] = None

    def _loss(self) -> nn.Module:
        return nn.MSELoss()

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> dict:
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        best_val = float("inf")
        bad = 0

        for _ in range(self.epochs):
            # ---- train ----
            self.model.train()
            for batch in train_loader:
                # expect (x, y, ...) from SequenceDataset
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("Expected DataLoader to yield (x, y, ...)")
                x, y = batch[0].to(self.device), batch[1].to(self.device)

                opt.zero_grad()
                pred = self.model(x).squeeze(-1)  # [B]
                loss = self._loss()(pred, y)
                loss.backward()
                opt.step()

            # ---- validate ----
            if val_loader is None:
                # Still capture a state so TrainingManager can save it
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                continue

            self.model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for batch in val_loader:
                    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                        raise ValueError("Expected DataLoader to yield (x, y, ...)")
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    pred = self.model(x).squeeze(-1)  # ensure [B]
                    val_loss += self._loss()(pred, y).item()
                    n += 1
            val_loss /= max(1, n)

            improved = val_loss < best_val - 1e-9
            if improved:
                best_val = val_loss
                bad = 0
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                bad += 1
                if bad >= self.patience:
                    break

        # ensure we return something
        if self._best_state is None:
            self._best_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }

        # load the best weights back into the live model
        self.model.load_state_dict(self._best_state)
        return self._best_state

    def predict(
        self,
        loader: DataLoader,
        *,
        dates: Sequence[pd.Timestamp],
        tickers: Sequence[str],
    ) -> pd.DataFrame:
        """
        Produces a long-form DataFrame with columns: ['date','ticker','pred_5d'].

        We rely on the controller to pass the aligned `dates` and `tickers`
        from the underlying SequenceDataset (and loader must be shuffle=False).
        """
        self.model.eval()
        preds: list[float] = []

        with torch.no_grad():
            for batch in loader:
                # accept (x, ...) (y may or may not be present for test loader)
                if not isinstance(batch, (tuple, list)) or len(batch) < 1:
                    raise ValueError("Expected DataLoader to yield (x, ...)")
                x = batch[0].to(self.device)
                yhat = self.model(x).squeeze(-1).detach().cpu()  # [B]
                preds.extend(yhat.tolist())

        if not (len(preds) == len(dates) == len(tickers)):
            raise ValueError(
                f"Prediction length mismatch: preds={len(preds)}, "
                f"dates={len(dates)}, tickers={len(tickers)}"
            )

        rows = (
            {"date": pd.Timestamp(d).date(), "ticker": str(t), "pred_5d": float(p)}
            for p, d, t in zip(preds, dates, tickers)
        )
        return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)

    @property
    def best_state_dict(self) -> Optional[dict]:
        return self._best_state