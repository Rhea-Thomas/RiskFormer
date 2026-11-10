"""
Gaussian negative log-likelihood loss.

Used for predicting both mean (μ) and uncertainty (σ) of return/volatility.

Loss:
  NLL = 0.5 * log(σ²) + (y - μ)² / (2σ²)

This is equivalent to the negative log-likelihood of a Gaussian distribution
(dropping the constant 0.5 * log(2π)). The model learns both a point estimate
(μ) and confidence (σ).

Key insight: High uncertainty → loss goes up from the log(σ²) term only
            Low uncertainty → loss goes up sharply when the error is large
            This balances overconfidence vs underfitting
"""

import torch
import torch.nn as nn


class GaussianNLL(nn.Module):
    """
    Gaussian negative log-likelihood loss.

    Model outputs mean (mu) and log-std (log_sigma).
    """

    def __init__(self, reduction="mean", min_log_sigma=-7.0, max_log_sigma=7.0):
        """
        Initialize Gaussian NLL.

        Args:
            reduction: "mean", "sum" or "none"
            min_log_sigma: lower clamp on log σ (prevents division blow-ups)
            max_log_sigma: upper clamp on log σ
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction: {reduction}")
        self.reduction = reduction
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

    def forward(self, mu, log_sigma, y):
        """
        Compute Gaussian NLL.

        Args:
            mu: (batch_size, n_outputs) predicted mean
            log_sigma: (batch_size, n_outputs) predicted log-std
            y: (batch_size, n_outputs) target values

        Returns:
            scalar loss (or per-element loss if reduction="none")
        """
        log_sigma = log_sigma.clamp(self.min_log_sigma, self.max_log_sigma)
        # 0.5*log(σ²) = log σ ;  (y-μ)²/(2σ²) = 0.5 * ((y-μ)·e^{-log σ})²
        z = (y - mu) * torch.exp(-log_sigma)
        nll = log_sigma + 0.5 * z.pow(2)

        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll
