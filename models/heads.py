"""
Prediction heads for RiskFormer.

Multi-task output layers for forecasting:
  - Price: predict returns or price changes
  - Volatility: predict realized volatility
  - Direction: predict up/down classification

Tensor flow:
  Backbone output: (batch_size, seq_len, d_model)
    v
  Each head processes independently
    v
  Price output: (batch_size, seq_len, n_price_outputs)
  Volatility output: (batch_size, seq_len, 1)
  Direction output: (batch_size, seq_len, 1) in [0,1]

Multi-task learning: joint optimization with weighted loss
  Loss = w_price * L_price + w_vol * L_vol + w_dir * L_dir
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PriceHead(nn.Module):
    """
    Price prediction head.

    Predicts next-period price changes or returns.
    Can output single value (point estimate) or quantiles (risk bounds).
    """

    def __init__(self, d_model: int, n_outputs: int = 1, hidden_dim: int = None, dropout: float = 0.1):
        """
        Initialize price head.

        Args:
            d_model: input dimension (from backbone)
            n_outputs: output dimension (1 for point, >1 for quantiles)
            hidden_dim: hidden dimension (default: 2*d_model)
            dropout: dropout probability
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * d_model

        self.d_model = d_model
        self.n_outputs = n_outputs

        # Simple MLP: d_model -> hidden -> n_outputs
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_outputs)
        self.dropout = nn.Dropout(dropout)

        logger.info(f"PriceHead: {d_model} -> {hidden_dim} -> {n_outputs}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict prices/returns.

        Args:
            x: (batch_size, seq_len, d_model) backbone output

        Returns:
            (batch_size, seq_len, n_outputs) price predictions
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class VolatilityHead(nn.Module):
    """
    Volatility prediction head.

    Predicts next-period realized volatility.
    Output is always positive (squared-returns or log-variance).
    """

    def __init__(self, d_model: int, hidden_dim: int = None, dropout: float = 0.1):
        """
        Initialize volatility head.

        Args:
            d_model: input dimension (from backbone)
            hidden_dim: hidden dimension (default: 2*d_model)
            dropout: dropout probability
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * d_model

        self.d_model = d_model

        # Simple MLP: d_model -> hidden -> 1
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        logger.info(f"VolatilityHead: {d_model} -> {hidden_dim} -> 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict volatility.

        Args:
            x: (batch_size, seq_len, d_model) backbone output

        Returns:
            (batch_size, seq_len, 1) volatility predictions (positive)
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # Softplus ensures positive output
        x = F.softplus(x)
        return x


class DirectionHead(nn.Module):
    """
    Direction prediction head.

    Classifies next-period movement: up or down.
    Output is probability in [0, 1].
    """

    def __init__(self, d_model: int, hidden_dim: int = None, dropout: float = 0.1):
        """
        Initialize direction head.

        Args:
            d_model: input dimension (from backbone)
            hidden_dim: hidden dimension (default: 2*d_model)
            dropout: dropout probability
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * d_model

        self.d_model = d_model

        # Simple MLP: d_model -> hidden -> 1 (logit)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        logger.info(f"DirectionHead: {d_model} -> {hidden_dim} -> 1 (sigmoid)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict direction.

        Args:
            x: (batch_size, seq_len, d_model) backbone output

        Returns:
            (batch_size, seq_len, 1) direction probability in [0, 1]
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # Sigmoid squashes to [0, 1]
        x = torch.sigmoid(x)
        return x


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head.

    Combines price, volatility, and direction heads.
    Enables joint training with weighted loss.
    """

    def __init__(
        self,
        d_model: int,
        n_price_outputs: int = 1,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-task head.

        Args:
            d_model: input dimension (from backbone)
            n_price_outputs: output dimension for price head
            hidden_dim: hidden dimension in each head
            dropout: dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.n_price_outputs = n_price_outputs

        # Individual heads
        self.price_head = PriceHead(d_model, n_price_outputs, hidden_dim, dropout)
        self.vol_head = VolatilityHead(d_model, hidden_dim, dropout)
        self.dir_head = DirectionHead(d_model, hidden_dim, dropout)

        logger.info(
            f"MultiTaskHead: price={n_price_outputs}, vol=1, dir=1 outputs"
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Predict all tasks.

        Args:
            x: (batch_size, seq_len, d_model) backbone output

        Returns:
            dict with keys:
              - 'price': (batch_size, seq_len, n_price_outputs)
              - 'volatility': (batch_size, seq_len, 1)
              - 'direction': (batch_size, seq_len, 1) in [0, 1]
        """
        return {
            'price': self.price_head(x),
            'volatility': self.vol_head(x),
            'direction': self.dir_head(x),
        }
