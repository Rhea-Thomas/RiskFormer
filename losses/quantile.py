"""
Pinball loss for quantile regression.

Used for predicting specific quantiles (e.g., 0.05, 0.50, 0.95) of the return distribution.

Loss (for quantile τ):
  L_τ(y, q) = max(τ(y - q), (τ - 1)(y - q))

Interpretation:
  - If y > q (actual is above quantile): loss = τ(y - q)
    Higher τ → steeper penalty for over-predictions
  - If y < q (actual is below quantile): loss = (1-τ)(q - y)
    Higher τ → softer penalty for under-predictions

Example: τ=0.95 (95th percentile)
  - We want actual to be below q 95% of the time
  - Over-prediction (y > q) is penalized heavily: 0.95 * error
  - Under-prediction (y < q) is penalized lightly: 0.05 * error
"""

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    """

    def __init__(self, quantiles=[0.05, 0.5, 0.95], reduction="mean"):
        """
        Initialize quantile loss.

        Args:
            quantiles: list of quantile levels
            reduction: "mean" or "sum"
        """
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, q_pred, y):
        """
        Compute quantile loss.

        Args:
            q_pred: (batch_size, n_assets, n_quantiles) predicted quantiles
            y: (batch_size, n_assets) target values

        Returns:
            scalar loss
        """
        pass
