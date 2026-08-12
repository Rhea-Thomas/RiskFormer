"""
Transformer backbone for RiskFormer.

Architecture:
  - Stacked Transformer blocks
  - Each block contains:
    * Multi-head causal self-attention
    * Feed-forward network
    * Residual connections
    * Layer normalization
    * Dropout

Tensor flow:
  Input (batch_size, seq_len, d_model)
    → Attention
    → Add & Norm
    → Feed-forward
    → Add & Norm
    → Output
"""

import torch
import torch.nn as nn
from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    Feed-forward network (position-wise fully connected).

    Applied identically to each position:
      FFN(x) = max(0, xW1 + b1) W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize FFN.

        Args:
            d_model: input/output dimension
            d_ff: hidden dimension (typically 2-4x d_model)
            dropout: dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply FFN.

        Args:
            x: (..., d_model)

        Returns:
            (..., d_model)
        """
        pass


class TransformerBlock(nn.Module):
    """
    Single Transformer block with attention and FFN.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Initialize Transformer block.

        Args:
            d_model: model dimension
            n_heads: number of attention heads
            d_ff: feed-forward hidden dimension
            dropout: dropout probability
        """
        super().__init__()
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply Transformer block.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        pass


class TransformerBackbone(nn.Module):
    """
    Stacked Transformer blocks.

    Main feature extraction component.
    """

    def __init__(
        self, d_model, n_heads, n_layers, d_ff, dropout=0.1
    ):
        """
        Initialize Transformer backbone.

        Args:
            d_model: model dimension
            n_heads: number of attention heads
            n_layers: number of Transformer blocks
            d_ff: feed-forward hidden dimension
            dropout: dropout probability
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Apply Transformer backbone.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
