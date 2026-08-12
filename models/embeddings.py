"""
Embeddings for RiskFormer.

Types of embeddings:
  - Temporal: Encodes position in time (similar to positional encoding)
  - Asset: Learned representations of different assets
  - Feature: Projects raw features into dense space

The goal is to lift raw inputs into a common learned representation
that the Transformer backbone can process.

Embedding dimension is shared across all embeddings (d_model).
"""

import torch
import torch.nn as nn


class TemporalEmbedding(nn.Module):
    """
    Temporal positional encoding.

    Maps time step index to a dense vector.
    Allows the model to understand temporal order.
    """

    def __init__(self, d_model, max_len=1000):
        """
        Initialize temporal embedding.

        Args:
            d_model: embedding dimension
            max_len: maximum sequence length
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, t):
        """
        Compute temporal embedding for time steps.

        Args:
            t: (batch_size, seq_len) long tensor of time indices

        Returns:
            (batch_size, seq_len, d_model) embedding
        """
        pass


class AssetEmbedding(nn.Module):
    """
    Learned asset embeddings.

    Maps each asset (stock, index, etc.) to a dense representation.
    Allows the model to learn asset-specific patterns.
    """

    def __init__(self, n_assets, d_model):
        """
        Initialize asset embedding.

        Args:
            n_assets: number of assets (stocks, etc.)
            d_model: embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(n_assets, d_model)

    def forward(self, asset_indices):
        """
        Compute asset embeddings.

        Args:
            asset_indices: (batch_size, n_assets) or (n_assets,)

        Returns:
            (batch_size, n_assets, d_model) or (n_assets, d_model)
        """
        pass


class FeatureEmbedding(nn.Module):
    """
    Feature embedding / projection.

    Projects raw features into d_model-dimensional space.
    """

    def __init__(self, n_features, d_model):
        """
        Initialize feature embedding.

        Args:
            n_features: number of input features
            d_model: embedding dimension
        """
        super().__init__()
        self.linear = nn.Linear(n_features, d_model)

    def forward(self, features):
        """
        Project features to embedding space.

        Args:
            features: (..., n_features)

        Returns:
            (..., d_model)
        """
        pass
