"""
Embeddings for RiskFormer.

Three types of embeddings combine to create rich input representations:

1. Feature Embedding: Projects raw features (OHLCV, returns, vol, etc.)
   into d_model dimensional space. Learned via backprop.

2. Temporal Embedding: Encodes position in the lookback window
   using sinusoidal positional encoding. No learning needed.

3. Asset Embedding: Learned representation for each asset (SPY vs QQQ, etc.).
   Allows model to learn asset-specific patterns.

Tensor flow:
  Input: (lookback, n_assets, n_features)
    ↓
  For each position t, asset a:
    - Project features: n_features → d_model
    - Add temporal encoding: + d_model
    - Add asset embedding: + d_model
    ↓
  Output: (lookback, n_assets, d_model)
    ↓
  Into Transformer backbone

All embeddings have dimension d_model (unified representation).
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TemporalEmbedding:
    """
    Sinusoidal positional encoding (no learning).

    Encodes position in sequence using sin/cos functions.
    Formula:
      PE(t, 2i)   = sin(t / 10000^(2i/d_model))
      PE(t, 2i+1) = cos(t / 10000^(2i/d_model))

    This gives the model information about relative positions without
    requiring any learnable parameters.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        """
        Initialize temporal embedding.

        Args:
            d_model: embedding dimension
            max_len: maximum sequence length to pre-compute
        """
        self.d_model = d_model
        self.max_len = max_len

        # Pre-compute positional encodings
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe

        logger.info(f"TemporalEmbedding: d_model={d_model}, max_len={max_len}")

    def __call__(self, seq_len: int, batch_size: int = 1):
        """
        Get positional encodings for a sequence.

        Args:
            seq_len: sequence length
            batch_size: batch size (for repeating across batch)

        Returns:
            (seq_len, d_model) or (batch_size, seq_len, d_model) numpy array
        """
        if seq_len > self.max_len:
            raise ValueError(f"seq_len {seq_len} > max_len {self.max_len}")

        pe = self.pe[:seq_len]  # (seq_len, d_model)

        if HAS_TORCH:
            pe = torch.from_numpy(pe).float()
            if batch_size > 1:
                pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if batch_size > 1:
                pe = np.expand_dims(pe, 0)
                pe = np.repeat(pe, batch_size, axis=0)

        return pe


class AssetEmbedding:
    """
    Learned asset embeddings.

    Each asset gets a unique learnable embedding vector.
    Allows model to learn asset-specific patterns (e.g., SPY vs QQQ volatility).
    """

    def __init__(self, n_assets: int, d_model: int):
        """
        Initialize asset embedding.

        Args:
            n_assets: number of assets
            d_model: embedding dimension
        """
        self.n_assets = n_assets
        self.d_model = d_model

        if HAS_TORCH:
            self.embedding = nn.Embedding(n_assets, d_model)
        else:
            # Initialize random embeddings for numpy
            self.embeddings = np.random.randn(n_assets, d_model) / np.sqrt(d_model)

        logger.info(f"AssetEmbedding: n_assets={n_assets}, d_model={d_model}")

    def __call__(self, asset_ids):
        """
        Get asset embeddings.

        Args:
            asset_ids: asset indices (0 to n_assets-1)
                      Can be scalar, 1D, or 2D array/tensor

        Returns:
            embeddings with same shape as input, last dim = d_model
        """
        if HAS_TORCH:
            if isinstance(asset_ids, np.ndarray):
                asset_ids = torch.from_numpy(asset_ids).long()
            return self.embedding(asset_ids)
        else:
            # Numpy version
            if np.isscalar(asset_ids):
                return self.embeddings[asset_ids]
            else:
                asset_ids = np.asarray(asset_ids)
                return self.embeddings[asset_ids]


class FeatureEmbedding:
    """
    Feature projection / embedding.

    Projects raw features (close, returns, volatility, etc.)
    into d_model dimensional space. Learned via backprop.
    """

    def __init__(self, n_features: int, d_model: int):
        """
        Initialize feature embedding.

        Args:
            n_features: number of input features
            d_model: embedding dimension
        """
        self.n_features = n_features
        self.d_model = d_model

        if HAS_TORCH:
            self.linear = nn.Linear(n_features, d_model)
        else:
            # Initialize random projection for numpy
            self.weight = np.random.randn(n_features, d_model) / np.sqrt(n_features)
            self.bias = np.zeros(d_model)

        logger.info(f"FeatureEmbedding: n_features={n_features}, d_model={d_model}")

    def __call__(self, features):
        """
        Project features to embedding space.

        Args:
            features: (..., n_features) array/tensor

        Returns:
            (..., d_model) array/tensor
        """
        if HAS_TORCH:
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            return self.linear(features)
        else:
            # Numpy version: features @ weight + bias
            return np.dot(features, self.weight) + self.bias


class EmbeddingCombiner:
    """
    Combines temporal, asset, and feature embeddings.

    For each position and asset, combines three d_model vectors:
      output = feature_emb + temporal_emb + asset_emb

    This creates a unified representation for the Transformer.
    """

    def __init__(
        self,
        n_features: int,
        n_assets: int,
        d_model: int,
        max_lookback: int = 1000,
    ):
        """
        Initialize embedding combiner.

        Args:
            n_features: number of input features per asset
            n_assets: number of assets
            d_model: embedding dimension (shared across all embeddings)
            max_lookback: maximum lookback window size
        """
        self.n_features = n_features
        self.n_assets = n_assets
        self.d_model = d_model

        self.feature_emb = FeatureEmbedding(n_features, d_model)
        self.temporal_emb = TemporalEmbedding(d_model, max_lookback)
        self.asset_emb = AssetEmbedding(n_assets, d_model)

        logger.info(
            f"EmbeddingCombiner: {n_features} features x {n_assets} assets -> {d_model}D"
        )

    def __call__(self, X):
        """
        Combine embeddings.

        Args:
            X: (lookback, n_assets, n_features) or (batch_size, lookback, n_assets, n_features)

        Returns:
            (lookback, n_assets, d_model) or (batch_size, lookback, n_assets, d_model)
        """
        is_batch = len(X.shape) == 4
        original_shape = X.shape

        if is_batch:
            batch_size, lookback, n_assets_x, n_features_x = X.shape
            # Reshape to (batch*lookback*n_assets, n_features) for processing
            X_flat = X.reshape(-1, n_features_x)
        else:
            lookback, n_assets_x, n_features_x = X.shape
            batch_size = 1
            # Reshape to (lookback*n_assets, n_features) for processing
            X_flat = X.reshape(-1, n_features_x)

        # 1. Project raw features
        feature_emb = self.feature_emb(X_flat)  # (flat_size, d_model)

        # 2. Temporal embeddings: create (lookback, d_model) and repeat for batch/assets
        temporal_raw = self.temporal_emb(lookback, 1)  # (1, lookback, d_model) or (lookback, d_model)
        if HAS_TORCH and temporal_raw.dim() == 3:
            temporal_raw = temporal_raw[0]  # Remove batch dim: (lookback, d_model)

        # Expand temporal to (batch*lookback*n_assets, d_model)
        if HAS_TORCH:
            temporal_exp = temporal_raw.repeat_interleave(n_assets_x, dim=0)  # (lookback*n_assets, d_model)
            if is_batch:
                temporal_exp = temporal_exp.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, lookback*n_assets, d_model)
                temporal_exp = temporal_exp.reshape(-1, self.d_model)  # (batch*lookback*n_assets, d_model)
        else:
            temporal_exp = np.repeat(temporal_raw, n_assets_x, axis=0)  # (lookback*n_assets, d_model)
            if is_batch:
                temporal_exp = np.tile(temporal_exp, (batch_size, 1))  # (batch*lookback*n_assets, d_model)

        # 3. Asset embeddings: create (n_assets, d_model) and repeat for batch/lookback
        asset_ids = np.arange(n_assets_x)
        asset_raw = self.asset_emb(asset_ids)  # (n_assets, d_model)

        # Expand asset to (batch*lookback*n_assets, d_model)
        if HAS_TORCH:
            asset_exp = asset_raw.unsqueeze(0).expand(lookback, -1, -1)  # (lookback, n_assets, d_model)
            asset_exp = asset_exp.reshape(-1, self.d_model)  # (lookback*n_assets, d_model)
            if is_batch:
                asset_exp = asset_exp.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, lookback*n_assets, d_model)
                asset_exp = asset_exp.reshape(-1, self.d_model)  # (batch*lookback*n_assets, d_model)
        else:
            asset_exp = np.expand_dims(asset_raw, 0)
            asset_exp = np.repeat(asset_exp, lookback, axis=0)  # (lookback, n_assets, d_model)
            asset_exp = asset_exp.reshape(-1, self.d_model)  # (lookback*n_assets, d_model)
            if is_batch:
                asset_exp = np.tile(asset_exp, (batch_size, 1))  # (batch*lookback*n_assets, d_model)

        # Combine all embeddings
        combined = feature_emb + temporal_exp + asset_exp

        # Reshape back to original structure
        if is_batch:
            combined = combined.reshape(*original_shape[:-1], self.d_model)
        else:
            combined = combined.reshape(lookback, n_assets_x, self.d_model)

        return combined
