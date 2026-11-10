"""
Training loop and trainer class for RiskFormer.

Responsibilities:
  - Run training for multiple epochs
  - Validation after each epoch
  - Gradient computation and backprop (with gradient clipping)
  - Linear warmup + cosine learning-rate schedule
  - Checkpoint saving
  - Early stopping (restores the best weights at the end)
  - Logging

The loss function is called as ``loss_fn(outputs, targets)`` and may return
either a scalar tensor or ``(scalar_tensor, {component_name: float})``.
"""

import copy
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
    """LambdaLR: linear warmup, then cosine decay to ``min_lr_ratio`` of the base LR."""

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Trainer for RiskFormer model.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device="cpu",
        early_stopping_patience=5,
        grad_clip=1.0,
        scheduler=None,
        log_every=0,
    ):
        """
        Initialize trainer.

        Args:
            model: nn.Module (RiskFormer model)
            optimizer: torch.optim optimizer
            loss_fn: loss function, loss_fn(outputs, targets)
            device: "cpu" or "cuda"
            early_stopping_patience: epochs to wait before stopping
            grad_clip: max gradient norm (None/0 disables clipping)
            scheduler: optional LR scheduler stepped once per batch
            log_every: log every N epochs (0 = silent)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.log_every = log_every
        self.best_val_loss = float("inf")
        self.best_state = None
        self.patience_counter = 0
        self.history = []

    # ------------------------------------------------------------------ utils
    def _compute_loss(self, X, y):
        out = self.model(X)
        result = self.loss_fn(out, y)
        if isinstance(result, tuple):
            return result
        return result, {}

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total, n = 0.0, 0
        parts_sum = {}
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                loss, parts = self._compute_loss(X, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                bs = X.shape[0]
                total += loss.item() * bs
                n += bs
                for k, v in parts.items():
                    parts_sum[k] = parts_sum.get(k, 0.0) + v * bs

        if n == 0:
            raise ValueError("Empty DataLoader")
        return total / n, {k: v / n for k, v in parts_sum.items()}

    # ------------------------------------------------------------------ API
    def train_epoch(self, train_loader):
        """
        Run one training epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            average training loss
        """
        loss, self._last_train_parts = self._run_epoch(train_loader, train=True)
        return loss

    def val_epoch(self, val_loader):
        """
        Run validation.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            average validation loss
        """
        loss, self._last_val_parts = self._run_epoch(val_loader, train=False)
        return loss

    def train(self, train_loader, val_loader, max_epochs=50):
        """
        Full training loop with early stopping.

        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation (None = no early stopping)
            max_epochs: maximum number of epochs

        Returns:
            training history (list of dicts with losses)
        """
        self.history = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_state = None

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in self._last_train_parts.items()},
            }

            if val_loader is not None:
                val_loss = self.val_epoch(val_loader)
                record["val_loss"] = val_loss
                record.update({f"val_{k}": v for k, v in self._last_val_parts.items()})

                if val_loss < self.best_val_loss - 1e-6:
                    self.best_val_loss = val_loss
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

            record["seconds"] = time.time() - t0
            self.history.append(record)

            if self.log_every and epoch % self.log_every == 0:
                msg = f"epoch {epoch:3d}  train {train_loss:.4f}"
                if "val_loss" in record:
                    msg += f"  val {record['val_loss']:.4f}  (best {self.best_val_loss:.4f})"
                logger.info(msg)

            if val_loader is not None and self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    def save_checkpoint(self, path, **extra):
        """Save model and optimizer state (plus any extra metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
                **extra,
            },
            path,
        )

    def load_checkpoint(self, path):
        """Load model and optimizer state. Returns the full checkpoint dict."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.history = ckpt.get("history", [])
        return ckpt
