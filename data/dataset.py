"""
PyTorch Dataset and DataLoader for financial time series.

Creates rolling windows of historical data with proper temporal ordering.
Each sample is a sequence: [t-L, t-L+1, ..., t-1, t] with target y_(t+h).

Key responsibilities:
  - Construct rolling windows respecting chronological order (NO SHUFFLING)
  - Align multiple assets and features
  - Handle MultiIndex columns (multiple assets)
  - Reshape into 4D tensors: (n_samples, lookback, n_assets, n_features)
  - Return torch.Tensors for PyTorch training
  - Prevent temporal leakage

Temporal constraint:
  At sample i (time t = i + lookback):
    X[i] uses data[i:i+lookback]           ← only past data
    y[i] uses data[i+lookback+horizon-1]   ← future target
    Never: X contains future data
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create dummy classes for when torch isn't available
    class Dataset:
        pass
    class DataLoader:
        pass


class FinancialTimeSeriesDataset(Dataset):
    """
    Rolling window dataset for financial time series.

    Converts 2D time series (T, features) into rolling windows (n_samples, L, assets, features).

    Each sample is a tuple (X, y):
      - X: historical window (lookback_window, n_assets, n_features)
      - y: future target (n_assets, n_targets) or scalar

    Attributes:
        X: (n_samples, lookback_window, n_assets, n_features)
        y: (n_samples, n_assets, n_targets) or (n_samples,)
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        lookback_window: int = 60,
        horizon: int = 1,
        n_assets: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            features_df: pd.DataFrame with MultiIndex columns (asset, feature)
                        Shape: (T, n_assets * n_features)
                        Example: columns = [('SPY', 'Close'), ('SPY', 'returns'), ...]

            targets_df: pd.DataFrame with target values
                       Shape: (T, n_targets) or (T, n_assets, n_targets)
                       Example: next-day return for each asset

            lookback_window: historical context length (days)
                            At time t, use data[t-lookback:t]

            horizon: forecast horizon (days ahead)
                    At time t, predict y[t+horizon]

            n_assets: number of assets (inferred from features_df if None)

        Example:
            lookback=60, horizon=1 means:
              X[i] = features[i:i+60]  (60 days of history)
              y[i] = targets[i+60]      (1 day ahead)
        """
        self.features_df = features_df
        self.targets_df = targets_df
        self.lookback_window = lookback_window
        self.horizon = horizon

        # Infer n_assets from MultiIndex columns
        if isinstance(features_df.columns, pd.MultiIndex):
            self.n_assets = len(features_df.columns.get_level_values(0).unique())
            self.asset_names = features_df.columns.get_level_values(0).unique().tolist()
        else:
            # Single-asset case (regular columns)
            self.n_assets = n_assets or 1
            self.asset_names = None

        # Infer n_features per asset
        if isinstance(features_df.columns, pd.MultiIndex):
            self.n_features = len(features_df.columns.get_level_values(1).unique())
        else:
            self.n_features = len(features_df.columns)

        # Number of valid samples
        # We need: lookback + horizon data to form one sample
        # Last valid idx for X: T - lookback - horizon + 1
        self.n_samples = len(features_df) - lookback_window - horizon + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough data: need {lookback_window + horizon} rows, got {len(features_df)}"
            )

        logger.info(
            f"Dataset: {self.n_samples} samples, {lookback_window} lookback, "
            f"{self.n_assets} assets, {self.n_features} features/asset"
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, "torch.Tensor"], Union[np.ndarray, "torch.Tensor"]]:
        """
        Get a single sample (X, y).

        Args:
            idx: sample index (0 to n_samples-1)

        Returns:
            (X, y) where:
              - X: torch.Tensor or np.ndarray (lookback_window, n_assets, n_features)
              - y: torch.Tensor or np.ndarray (n_assets, n_targets) or scalar
        """
        # At sample idx:
        # - X uses data from [idx, idx+lookback]
        # - y uses data at [idx+lookback+horizon-1]
        start_idx = idx
        end_idx = idx + self.lookback_window
        target_idx = idx + self.lookback_window + self.horizon - 1

        # Extract features window
        features_window = self.features_df.iloc[start_idx:end_idx]

        # Reshape into (lookback, n_assets, n_features)
        X = self._reshape_features(features_window)

        # Extract target
        target_row = self.targets_df.iloc[target_idx]
        y_numpy = target_row.to_numpy()

        if HAS_TORCH:
            y = torch.from_numpy(y_numpy).float()
        else:
            y = y_numpy.astype(np.float32)

        return X, y

    def _reshape_features(self, features_window: pd.DataFrame) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Reshape features from (lookback, n_assets*n_features) to
        (lookback, n_assets, n_features).

        Args:
            features_window: pd.DataFrame (lookback, n_assets*n_features)
                           with MultiIndex columns if multi-asset

        Returns:
            torch.Tensor or np.ndarray (lookback, n_assets, n_features)
        """
        if isinstance(features_window.columns, pd.MultiIndex):
            # Multi-asset case: reshape from wide to 3D
            lookback = len(features_window)
            features_3d = np.zeros((lookback, self.n_assets, self.n_features))

            for asset_idx, asset_name in enumerate(self.asset_names):
                asset_data = features_window[asset_name].to_numpy()  # (lookback, n_features)
                features_3d[:, asset_idx, :] = asset_data

        else:
            # Single-asset case: reshape from 2D to 3D
            lookback = len(features_window)
            features_2d = features_window.to_numpy()  # (lookback, n_features)
            features_3d = features_2d[:, np.newaxis, :]  # (lookback, 1, n_features)

        # Convert to torch if available
        if HAS_TORCH:
            X = torch.from_numpy(features_3d).float()
        else:
            X = features_3d.astype(np.float32)

        return X


def create_rolling_window_dataset(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    lookback_window: int = 60,
    horizon: int = 1,
) -> FinancialTimeSeriesDataset:
    """
    Create a rolling window dataset.

    Convenience function to create a dataset from features and targets.

    Args:
        features_df: DataFrame with features (T, n_assets*n_features)
        targets_df: DataFrame with targets (T, n_targets)
        lookback_window: historical context window
        horizon: forecast horizon

    Returns:
        FinancialTimeSeriesDataset
    """
    return FinancialTimeSeriesDataset(
        features_df, targets_df, lookback_window=lookback_window, horizon=horizon
    )


def create_dataloader(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    lookback_window: int = 60,
    horizon: int = 1,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for time series data.

    Args:
        features_df: feature data
        targets_df: target data
        lookback_window: historical context window
        horizon: forecast horizon
        batch_size: batch size
        shuffle: whether to shuffle (should be False for time series)
        num_workers: parallel data loading workers

    Returns:
        torch.utils.data.DataLoader

    Example:
        >>> loader = create_dataloader(features, targets, lookback_window=60, batch_size=32)
        >>> for X_batch, y_batch in loader:
        ...     print(X_batch.shape)  # (32, 60, n_assets, n_features)
        ...     print(y_batch.shape)   # (32, n_assets, n_targets)
    """
    dataset = FinancialTimeSeriesDataset(
        features_df, targets_df, lookback_window=lookback_window, horizon=horizon
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    logger.info(
        f"DataLoader created: {len(dataset)} samples, batch_size={batch_size}, "
        f"{len(loader)} batches"
    )

    return loader
