"""
Gaussian negative log-likelihood loss.

Used for predicting both mean (μ) and uncertainty (σ) of return/volatility.

Loss:
  NLL = 0.5 * log(σ²) + (y - μ)² / (2σ²)

This is equivalent to the negative log-likelihood of a Gaussian distribution.
The model learns both a point estimate (μ) and confidence (σ).

Key insight: High uncertainty → loss goes up from second term only
            Low uncertainty → loss goes up from both terms
            This balances overconfidence vs underfitting
"""

import torch
import torch.nn as nn


class GaussianNLL(nn.Module):
    """
    Gaussian negative log-likelihood loss.

    Model outputs mean (mu) and log-std (log_sigma).
    """

    def __init__(self, reduction="mean"):
        """
        Initialize Gaussian NLL.

        Args:
            reduction: "mean" or "sum"
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, mu, log_sigma, y):
        """
        Compute Gaussian NLL.

        Args:
            mu: (batch_size, n_outputs) predicted mean
            log_sigma: (batch_size, n_outputs) predicted log-std
            y: (batch_size, n_outputs) target values

        Returns:
            scalar loss
        """
        pass
