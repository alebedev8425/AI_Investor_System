from __future__ import annotations

from typing import Optional, Sequence, List
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim


class GnnModelTrainer:
    """
    Trainer for CrossAssetGNNModel (MDGNN-style variant).

    Expects DataLoader yielding:
        (x, y, mask, date)
        x:    [B, N, W, F]  node features (already include tech/sent/etc.)
        y:    [B, N]        targets (e.g., 5d-ahead returns in normalized space)
        mask: [B, N]        bool, True where target is valid
        date: length-B sequence of pd.Timestamp

    Optimizes a single scalar loss:
        masked MSE(pred, y) over valid nodes.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        epochs: int,
        lr: float,
        device: torch.device,
        patience: int = 5,
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
    ) -> None:
        self.model = model.to(device)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = device
        self.patience = int(patience)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = max_grad_norm
        self._best_state: dict | None = None

    def _loss(self, pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred, y: [B, N]
        mask:   [B, N] (bool)
        """
        # ensure bool mask
        valid = mask.bool() & torch.isfinite(y)
        if not valid.any():
            # if no valid labels in this batch, return zero-loss
            return torch.tensor(0.0, device=self.device)

        diff = pred[valid] - y[valid]
        return (diff * diff).mean()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict:
        """
        Train with early stopping on validation loss.

        Returns:
            best_state_dict (copied to CPU) of the model.
        """
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
            for x, y, mask, _ in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)

                opt.zero_grad()
                pred = self.model(x)  # [B, N]
                loss = self._loss(pred, y, mask)
                loss.backward()

                # optional gradient clipping for stability
                if self.max_grad_norm is not None and self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm,
                    )

                opt.step()

            # ---- validate ----
            if val_loader is None:
                # no validation loader: just keep the last epoch weights
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                continue

            self.model.eval()
            val_loss = 0.0
            n_batches = 0

            with torch.no_grad():
                for x, y, mask, _ in val_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    mask = mask.to(self.device)
                    pred = self.model(x)
                    l = self._loss(pred, y, mask)
                    val_loss += float(l.item())
                    n_batches += 1

            val_loss /= max(1, n_batches)

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
            # fallback: store final model state
            self._best_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }

        self.model.load_state_dict(self._best_state)
        return self._best_state

    def predict(
        self,
        loader: DataLoader,
        *,
        tickers: Sequence[str],
    ) -> pd.DataFrame:
        """
        Returns long-form ['date','ticker','pred_5d'] from graph batches.

        Assumes model outputs are on the same normalized scale as the label
        used in TrainingManager (e.g., 'target_5d_norm'), and that the caller
        will handle inverse-transform if needed.
        """
        self.model.eval()
        rows: List[dict] = []

        with torch.no_grad():
            for x, _, mask, dates in loader:
                x = x.to(self.device)
                mask = mask.to(self.device)

                pred = self.model(x).cpu()  # [B, N]
                mask_np = mask.cpu().numpy()
                pred_np = pred.numpy()

                for b, d in enumerate(dates):
                    d_ts = pd.to_datetime(d)
                    for j, t in enumerate(tickers):
                        if not mask_np[b, j]:
                            continue
                        rows.append(
                            {
                                "date": d_ts.date(),
                                "ticker": str(t),
                                "pred_5d": float(pred_np[b, j]),
                            }
                        )

        if not rows:
            return pd.DataFrame(columns=["date", "ticker", "pred_5d"])

        return (
            pd.DataFrame(rows)
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )

    @property
    def best_state_dict(self) -> Optional[dict]:
        return self._best_state