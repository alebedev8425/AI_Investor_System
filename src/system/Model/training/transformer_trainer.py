from __future__ import annotations

from typing import Optional, Sequence
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


# ---------------- Trainer (builder + optimizer logic) ----------------

class TransformerModelTrainer:
    """
    MASTER-style Transformer trainer.

    - Optimizes the provided TransformerModel on (train, val) loaders.
    - Early-stops on validation loss (patience).
    - Clips gradients (max_grad_norm).
    - Provides predict() -> DataFrame(['date','ticker','pred_5d']).
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        epochs: int,
        lr: float,
        device: torch.device,
        patience: int = 3,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = device
        self.patience = int(patience)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)

        self._best_state: Optional[dict] = None

    def _loss(self) -> nn.Module:
        # MSE on normalized target_5d_norm
        return nn.MSELoss()

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> dict:
        opt = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        best_val = float("inf")
        bad = 0

        for _ in range(self.epochs):
            # ---- train ----
            self.model.train()
            for batch in train_loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("Expected DataLoader to yield (x, y, ...)")
                x, y = batch[0].to(self.device), batch[1].to(self.device)

                opt.zero_grad()
                pred = self.model(x).squeeze(-1)
                loss = self._loss()(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                opt.step()

            # ---- validate ----
            if val_loader is None:
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
                    pred = self.model(x).squeeze(-1)
                    val_loss += self._loss()(pred, y).item()
                    n += 1
            val_loss /= max(1, n)

            if val_loss < best_val - 1e-9:
                best_val = val_loss
                bad = 0
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if self._best_state is None:
            self._best_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }

        self.model.load_state_dict(self._best_state)
        return self._best_state

    def predict(
        self,
        loader: DataLoader,
        *,
        dates: Sequence[pd.Timestamp],
        tickers: Sequence[str],
    ) -> pd.DataFrame:
        self.model.eval()
        preds: list[float] = []

    
        with torch.no_grad(): 
            for batch in loader: 
                if not isinstance(batch, (tuple, list)) or len(batch) < 1: 
                    raise ValueError("Expected DataLoader to yield (x, ...)") 
                x = batch[0].to(self.device)
                yhat = self.model(x).squeeze(-1).detach().cpu()
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