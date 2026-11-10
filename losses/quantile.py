"""
Pinball loss for quantile regression.

Used for predicting specific quantiles (e.g., 0.05, 0.50, 0.95) of the return distribution.

Loss (for quantile τ):
  L_τ(y, q) = max(τ(y - q), (τ - 1)(y - q))

Interpretation:
  - If y > q (actual is above quantile): loss = τ(y - q)
    Higher τ → steeper penalty when the quantile sits too low
  - If y < q (actual is below quantile): loss = (1-τ)(q - y)
    Higher τ → softer penalty when the quantile sits too high

Example: τ=0.95 (95th percentile)
  - We want actual to be below q 95% of the time
  - Actual above q is penalized heavily: 0.95 * error
  - Actual below q is penalized lightly: 0.05 * error
"""

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    """

    def __init__(self, quantiles=(0.05, 0.5, 0.95), reduction="mean"):
        """
        Initialize quantile loss.

        Args:
            quantiles: sequence of quantile levels in (0, 1)
            reduction: "mean", "sum" or "none"
        """
        super().__init__()
        quantiles = list(quantiles)
        if any(not 0.0 < q < 1.0 for q in quantiles):
            raise ValueError(f"Quantiles must be in (0, 1): {quantiles}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction: {reduction}")
        # Buffer so it follows the module across devices
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.reduction = reduction

    def forward(self, q_pred, y):
        """
        Compute quantile loss.

        Args:
            q_pred: (batch_size, n_assets, n_quantiles) predicted quantiles
            y: (batch_size, n_assets) target values

        Returns:
            scalar loss (or per-element loss if reduction="none")
        """
        if q_pred.shape[-1] != self.quantiles.numel():
            raise ValueError(
                f"q_pred has {q_pred.shape[-1]} quantiles, expected {self.quantiles.numel()}"
            )
        err = y.unsqueeze(-1) - q_pred  # (batch, assets, n_quantiles)
        tau = self.quantiles.to(q_pred.dtype)
        loss = torch.maximum(tau * err, (tau - 1.0) * err)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
