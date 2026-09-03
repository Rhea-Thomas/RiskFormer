"""
Causal self-attention mechanism for RiskFormer.

Core equation:
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Where:
  - Q (queries): what the model wants to look for
  - K (keys): where information is located
  - V (values): the actual information
  - sqrt(d_k) scaling: prevents attention logits from blowing up

For financial time series, we use CAUSAL masking:
  - At time t, the model can only attend to times [1, 2, ..., t]
  - Cannot attend to future information
  - Essential for preventing look-ahead bias

Tensor shapes (batch_size, seq_len, d_model):
  Input X (seq_len=L): [B, L, d_model]
  Q, K, V: [B, n_heads, L, d_k] where d_k = d_model / n_heads
  Attention scores: [B, n_heads, L, L]
  Output: [B, L, d_model]
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


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
    Compute scaled dot-product attention.

    Args:
        q: (batch, n_heads, seq_len, d_k) queries
        k: (batch, n_heads, seq_len, d_k) keys
        v: (batch, n_heads, seq_len, d_k) values
        mask: (seq_len, seq_len) or (batch, 1, seq_len, seq_len) boolean mask
              True = attend, False = mask out
        dropout: dropout layer (optional)

    Returns:
        output: (batch, n_heads, seq_len, d_k) attention output
        weights: (batch, n_heads, seq_len, seq_len) attention weights
    """
    d_k = q.shape[-1]

    # Compute attention scores: QK^T / sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)

    # Apply mask: set future positions to -inf so softmax makes them 0
    if mask is not None:
        # Convert boolean mask (True=attend, False=mask) to additive mask
        # -inf where we want to mask, 0 where we want to attend
        if mask.dim() == 2:
            # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(0)

        # mask is True where we attend, False where we mask
        # Convert: True->0, False->-inf
        additive_mask = torch.where(mask, torch.tensor(0.0), torch.tensor(float('-inf')))
        scores = scores + additive_mask

    # Softmax to get attention weights
    weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout is not None:
        weights = dropout(weights)

    # Multiply by values
    output = torch.matmul(weights, v)

    return output, weights


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with causal masking.

    Prevents attending to future time steps.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize causal self-attention.

        Args:
            d_model: model dimension (must be divisible by n_heads)
            n_heads: number of attention heads
            dropout: dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V, output
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout_layer = nn.Dropout(dropout)

        logger.info(
            f"CausalSelfAttention: d_model={d_model}, n_heads={n_heads}, d_k={self.d_k}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal self-attention.

        Args:
            x: (batch_size, seq_len, d_model) input tensor

        Returns:
            (batch_size, seq_len, d_model) attended output
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.linear_q(x)  # (batch, seq_len, d_model)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # Reshape for multi-head: (batch, seq_len, d_model)
        # -> (batch, seq_len, n_heads, d_k)
        # -> (batch, n_heads, seq_len, d_k)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Create causal mask
        causal_mask = self._causal_mask(seq_len, x.device)

        # Apply scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=causal_mask, dropout=self.dropout_layer
        )

        # Concatenate heads: (batch, n_heads, seq_len, d_k)
        # -> (batch, seq_len, n_heads, d_k)
        # -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, d_model
        )

        # Final output projection
        output = self.linear_out(attn_output)

        return output

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """
        Create causal mask (lower triangular).

        At position t, allow attention to positions [0, 1, ..., t].
        Mask out positions [t+1, t+2, ..., seq_len-1].

        Args:
            seq_len: sequence length
            device: torch device

        Returns:
            (seq_len, seq_len) boolean mask
            True = attend (lower triangular + diagonal)
            False = mask out (upper triangular)
        """
        # Lower triangular matrix: position i can attend to positions 0..i
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask
