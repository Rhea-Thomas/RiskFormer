"""
Multi-task loss combining multiple objectives.

RiskFormer predicts multiple targets (return, volatility, quantiles, regime).
This loss combines them with learned or fixed weights.

Total loss:
  L = w_return * L_return
    + w_volatility * L_volatility
    + w_quantile * L_quantile
    + w_regime * L_regime

Weights can be fixed (in config) or learned dynamically.
"""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    Weighted combination of multiple task losses.
    """

    def __init__(
        self,
        return_weight=1.0,
        volatility_weight=0.5,
        quantile_weight=0.5,
        regime_weight=0.0,
        learn_weights=False,
    ):
        """
        Initialize multi-task loss.

        Args:
            return_weight: weight for return prediction loss
            volatility_weight: weight for volatility prediction loss
            quantile_weight: weight for quantile regression loss
            regime_weight: weight for regime classification loss
            learn_weights: whether to learn weights dynamically
        """
        super().__init__()
        self.weights = {
            "return": return_weight,
            "volatility": volatility_weight,
            "quantile": quantile_weight,
            "regime": regime_weight,
        }
        self.learn_weights = learn_weights

    def forward(self, losses_dict):
        """
        Combine task losses.

        Args:
            losses_dict: dict with keys ["return", "volatility", "quantile", "regime"]
                        values are scalar tensors

        Returns:
            scalar total loss
        """
        pass
