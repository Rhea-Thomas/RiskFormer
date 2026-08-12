"""
Prediction heads for multi-task learning.

Each head produces a different prediction task:
  - Return head: Expected future return
  - Volatility head: Expected future volatility/risk
  - Quantile head: Quantiles of return distribution (q_0.05, q_0.50, q_0.95)
  - Regime head: Probability of each market regime

All heads receive the same Transformer backbone output and produce
task-specific predictions.
"""

import torch
import torch.nn as nn


class ReturnHead(nn.Module):
    """
    Predict expected future return (1-step ahead or h-step ahead).

    Output: (batch_size, n_assets) or (batch_size, n_assets, 1)
    """

    def __init__(self, d_model, n_assets):
        """
        Initialize return prediction head.

        Args:
            d_model: Transformer output dimension
            n_assets: number of assets
        """
        super().__init__()
        self.fc = nn.Linear(d_model, n_assets)

    def forward(self, x):
        """
        Predict returns.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, n_assets) predicted returns
        """
        pass


class VolatilityHead(nn.Module):
    """
    Predict future volatility / risk.

    Output: (batch_size, n_assets)
    Must be positive (e.g., via softplus or exp).
    """

    def __init__(self, d_model, n_assets):
        """
        Initialize volatility prediction head.

        Args:
            d_model: Transformer output dimension
            n_assets: number of assets
        """
        super().__init__()
        self.fc = nn.Linear(d_model, n_assets)

    def forward(self, x):
        """
        Predict volatility.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, n_assets) predicted volatility (non-negative)
        """
        pass


class QuantileHead(nn.Module):
    """
    Predict return quantiles.

    For each asset, predict multiple quantiles (e.g., 0.05, 0.50, 0.95).
    Used for probabilistic forecasting and risk assessment.

    Output: (batch_size, n_assets, n_quantiles)
    """

    def __init__(self, d_model, n_assets, quantiles=[0.05, 0.5, 0.95]):
        """
        Initialize quantile head.

        Args:
            d_model: Transformer output dimension
            n_assets: number of assets
            quantiles: list of quantile levels
        """
        super().__init__()
        self.quantiles = quantiles
        self.fc = nn.Linear(d_model, n_assets * len(quantiles))

    def forward(self, x):
        """
        Predict quantiles.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, n_assets, n_quantiles) predicted quantiles
        """
        pass


class RegimeHead(nn.Module):
    """
    Classify market regime (e.g., bullish, bearish, high-vol, low-vol).

    Outputs probability distribution over regimes.

    Output: (batch_size, n_regimes)
    """

    def __init__(self, d_model, n_regimes=4):
        """
        Initialize regime classification head.

        Args:
            d_model: Transformer output dimension
            n_regimes: number of market regimes
        """
        super().__init__()
        self.fc = nn.Linear(d_model, n_regimes)

    def forward(self, x):
        """
        Classify market regime.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, n_regimes) regime logits (apply softmax for probabilities)
        """
        pass
