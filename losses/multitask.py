"""
Multi-task loss combining multiple objectives.

RiskFormer predicts multiple targets (return, volatility, quantiles, direction).
This loss combines them with fixed or learned weights.

Total loss (fixed weights):
  L = w_return * L_return
    + w_volatility * L_volatility
    + w_quantile * L_quantile
    + w_direction * L_direction
    + w_regime * L_regime

Learned weights (Kendall, Gal & Cipolla, 2018 — homoscedastic uncertainty):
  L = Σ_k  exp(-s_k) * w_k * L_k + s_k
where s_k = log σ_k² is a learnable scalar per task. Tasks the model finds
noisy get down-weighted automatically; the + s_k term stops all weights
collapsing to zero.

`RiskFormerLoss` wires the individual losses to the model's output dict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.gaussian_nll import GaussianNLL
from losses.quantile import QuantileLoss

TASKS = ("return", "volatility", "quantile", "direction", "regime")


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
        direction_weight=0.0,
        learn_weights=False,
    ):
        """
        Initialize multi-task loss.

        Args:
            return_weight: weight for return prediction loss
            volatility_weight: weight for volatility prediction loss
            quantile_weight: weight for quantile regression loss
            regime_weight: weight for regime classification loss
            direction_weight: weight for up/down classification loss
            learn_weights: whether to learn weights dynamically
        """
        super().__init__()
        self.weights = {
            "return": return_weight,
            "volatility": volatility_weight,
            "quantile": quantile_weight,
            "direction": direction_weight,
            "regime": regime_weight,
        }
        self.learn_weights = learn_weights
        if learn_weights:
            # One log-variance per task, initialised to 0 (=> weight 1)
            self.log_vars = nn.ParameterDict(
                {k: nn.Parameter(torch.zeros(())) for k in TASKS}
            )

    def forward(self, losses_dict):
        """
        Combine task losses.

        Args:
            losses_dict: dict with any subset of keys
                         ["return", "volatility", "quantile", "direction", "regime"];
                         values are scalar tensors. Missing tasks contribute 0.

        Returns:
            scalar total loss
        """
        unknown = set(losses_dict) - set(TASKS)
        if unknown:
            raise KeyError(f"Unknown task(s) in losses_dict: {sorted(unknown)}")

        total = None
        for task, loss in losses_dict.items():
            w = self.weights[task]
            if w == 0.0:
                continue
            if self.learn_weights:
                s = self.log_vars[task]
                term = torch.exp(-s) * w * loss + s
            else:
                term = w * loss
            total = term if total is None else total + term

        if total is None:
            # Nothing weighted: return a zero that still lives on the right device
            any_loss = next(iter(losses_dict.values()))
            total = any_loss * 0.0
        return total


class RiskFormerLoss(nn.Module):
    """
    Maps RiskFormer outputs + targets to the individual task losses.

    Targets tensor layout (last dim):
      0: next-day return (percent)
      1: forward realized volatility (percent, daily units)
      2: direction label (1 if next-day return > 0 else 0)

    Model outputs (dict):
      'mu'        (B, A)        mean return
      'sigma'     (B, A)        predicted daily volatility (> 0)
      'quantiles' (B, A, n_q)   return quantiles
      'direction_logit' (B, A)  logit of P(up)
    """

    def __init__(self, quantiles=(0.05, 0.5, 0.95), **weights):
        super().__init__()
        self.nll = GaussianNLL()
        self.pinball = QuantileLoss(quantiles)
        self.combiner = MultiTaskLoss(**weights)

    def components(self, out, y):
        y_ret, y_vol, y_dir = y[..., 0], y[..., 1], y[..., 2]
        sigma = out["sigma"].clamp_min(1e-4)
        return {
            # Return distribution: Gaussian NLL ties μ and σ together
            "return": self.nll(out["mu"], torch.log(sigma), y_ret),
            # Volatility: match realized vol in log space (scale-free, robust)
            "volatility": F.mse_loss(torch.log(sigma), torch.log(y_vol.clamp_min(1e-4))),
            "quantile": self.pinball(out["quantiles"], y_ret),
            "direction": F.binary_cross_entropy_with_logits(out["direction_logit"], y_dir),
        }

    def forward(self, out, y):
        parts = self.components(out, y)
        total = self.combiner(parts)
        return total, {k: v.detach().item() for k, v in parts.items()}
