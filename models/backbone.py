"""
Transformer backbone for RiskFormer.

Stacks causal self-attention blocks with feed-forward networks.

Architecture:
  N blocks of [LayerNorm -> Attention -> Residual] + [LayerNorm -> FFN -> Residual]

Each block:
  1. Layer normalization (pre-norm for stable training)
  2. Causal self-attention (learns what to attend to)
  3. Residual connection (skip connection preserves gradient flow)
  4. Layer normalization
  5. Feed-forward network (position-wise MLP)
  6. Residual connection

Tensor flow:
  Input: (batch_size, seq_len, d_model)
    v
  Block 1: Attention + Residual + FFN + Residual
    v
  Block 2: Attention + Residual + FFN + Residual
    v
  ... N blocks ...
    v
  Output: (batch_size, seq_len, d_model)

Key invariant: residual connections preserve gradient flow for deep networks.
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

if HAS_TORCH:
    from models.attention import CausalSelfAttention


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network.

    Applied independently at each time step (same weights across time).
    Typically: d_model -> hidden_dim (4*d_model) -> d_model with ReLU activation.
    """

    def __init__(self, d_model: int, hidden_dim: int = None, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            d_model: input/output dimension
            hidden_dim: hidden layer dimension (default: 4*d_model)
            dropout: dropout probability
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model

        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

        logger.info(f"FeedForwardNetwork: {d_model} -> {hidden_dim} -> {d_model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        # Linear -> Activation -> Dropout -> Linear
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer block.

    Combines causal self-attention with feed-forward network.
    Uses layer normalization (pre-norm) and residual connections.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, hidden_dim: int = None):
        """
        Initialize transformer block.

        Args:
            d_model: model dimension
            n_heads: number of attention heads
            dropout: dropout probability
            hidden_dim: hidden dimension in FFN (default: 4*d_model)
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Attention block: LN -> Attention -> Residual
        self.ln_attn = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)

        # Feed-forward block: LN -> FFN -> Residual
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, hidden_dim, dropout)

        logger.info(f"TransformerBlock: d_model={d_model}, n_heads={n_heads}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        # Attention with residual
        x_norm = self.ln_attn(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out  # Residual connection

        # Feed-forward with residual
        x_norm = self.ln_ffn(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out  # Residual connection

        return x


class TransformerBackbone(nn.Module):
    """
    Stack of Transformer blocks.

    Processes embedded sequences through multiple layers of attention
    and feed-forward networks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        hidden_dim: int = None,
    ):
        """
        Initialize transformer backbone.

        Args:
            d_model: model dimension
            n_heads: number of attention heads per block
            n_layers: number of transformer blocks
            dropout: dropout probability
            hidden_dim: hidden dimension in FFN (default: 4*d_model)
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, dropout, hidden_dim)
                for _ in range(n_layers)
            ]
        )

        logger.info(
            f"TransformerBackbone: {n_layers} blocks x "
            f"({d_model}D, {n_heads} heads)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer backbone.

        Args:
            x: (batch_size, seq_len, d_model) embedded sequence

        Returns:
            (batch_size, seq_len, d_model) refined representations
        """
        # Pass through each block
        for block in self.blocks:
            x = block(x)

        return x
