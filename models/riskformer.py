"""
RiskFormer: the full model.

Wires together the components built in Phases 5–8:

  X (B, L, A, F)
    │  EmbeddingCombiner   feature projection + sinusoidal time + asset embedding
    ▼
  (B, L, A, d_model) ──► fold assets into batch ──► (B·A, L, d_model)
    │  TransformerBackbone  causal self-attention (no peeking at future days)
    ▼
  last time step  (B·A, d_model) ──► (B, A, d_model)
    │  heads
    ▼
  mu (B, A)               expected next-day return
  sigma (B, A)            predicted daily volatility (softplus, > 0)
  quantiles (B, A, n_q)   monotone return quantiles (no crossing by construction)
  direction_logit (B, A)  logit of P(next return > 0)

Each asset is processed as its own sequence (shared weights, asset identity
from the asset embedding). Folding assets into the batch keeps the causal mask
purely temporal.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import TransformerBackbone
from models.embeddings import EmbeddingCombiner
from models.heads import DirectionHead, PriceHead, VolatilityHead

logger = logging.getLogger(__name__)


class RiskFormer(nn.Module):
    """End-to-end probabilistic multi-asset forecaster."""

    def __init__(
        self,
        n_features: int,
        n_assets: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = None,
        dropout: float = 0.1,
        quantiles=(0.05, 0.5, 0.95),
        max_lookback: int = 512,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_assets = n_assets
        self.d_model = d_model
        self.quantiles = tuple(quantiles)

        # Phase 5 embeddings. EmbeddingCombiner is a plain class, so register its
        # learnable layers here to make them part of this module's parameters.
        self.embedder = EmbeddingCombiner(n_features, n_assets, d_model, max_lookback)
        self.feature_proj = self.embedder.feature_emb.linear
        self.asset_table = self.embedder.asset_emb.embedding
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Phases 6–7 backbone
        self.backbone = TransformerBackbone(d_model, n_heads, n_layers, dropout, d_ff)
        self.final_norm = nn.LayerNorm(d_model)

        # Phase 8 heads
        n_q = len(self.quantiles)
        self.return_head = PriceHead(d_model, 1, dropout=dropout)
        self.vol_head = VolatilityHead(d_model, dropout=dropout)
        self.quantile_head = PriceHead(d_model, n_q, dropout=dropout)
        self.direction_head = DirectionHead(d_model, dropout=dropout)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """(B, L, A, F) -> (B, A, d_model) representation at the last time step."""
        if X.dim() != 4:
            raise ValueError(f"Expected (batch, lookback, assets, features), got {tuple(X.shape)}")
        B, L, A, _ = X.shape
        h = self.embedder(X)  # (B, L, A, d)
        h = self.input_dropout(self.input_norm(h))
        h = h.permute(0, 2, 1, 3).reshape(B * A, L, self.d_model)
        h = self.backbone(h)
        h = self.final_norm(h[:, -1, :])
        return h.reshape(B, A, self.d_model)

    def forward(self, X: torch.Tensor) -> dict:
        h = self.encode(X)

        mu = self.return_head(h).squeeze(-1)
        sigma = self.vol_head(h).squeeze(-1) + 1e-3

        # Monotone quantiles: anchor at the first quantile, add positive gaps
        raw_q = self.quantile_head(h)
        gaps = F.softplus(raw_q[..., 1:])
        quantiles = torch.cat([raw_q[..., :1], raw_q[..., :1] + torch.cumsum(gaps, -1)], -1)

        # DirectionHead ends with a sigmoid; recover a stable logit for BCE
        p_up = self.direction_head(h).squeeze(-1).clamp(1e-6, 1 - 1e-6)
        direction_logit = torch.log(p_up) - torch.log1p(-p_up)

        return {
            "mu": mu,
            "sigma": sigma,
            "quantiles": quantiles,
            "direction_logit": direction_logit,
            "p_up": p_up,
        }

    @classmethod
    def from_config(cls, config: dict, n_features: int, n_assets: int) -> "RiskFormer":
        t = config["transformer"]
        return cls(
            n_features=n_features,
            n_assets=n_assets,
            d_model=t["d_model"],
            n_heads=t["n_heads"],
            n_layers=t["n_layers"],
            d_ff=t.get("d_ff"),
            dropout=t.get("dropout", 0.1),
            quantiles=config.get("heads", {}).get("quantiles", (0.05, 0.5, 0.95)),
            max_lookback=max(512, config["data"]["lookback_window"]),
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
