"""
PyTorch Dataset and DataLoader for financial time series.

Creates rolling windows of historical data with proper temporal ordering.
Each sample is a sequence: [t-L, t-L+1, ..., t-1, t] with target y_(t+h).

Key responsibilities:
  - Construct rolling windows respecting chronological order
  - Align multiple assets and features
  - Handle batch construction
  - Return torch.Tensors for PyTorch training
"""

import torch
from torch.utils.data import Dataset, DataLoader


class FinancialTimeSeriesDataset(Dataset):
    """
    Rolling window dataset for financial time series.

    Each sample is a window of historical data and a target (future return/vol/etc).

    Attributes:
        X: (n_samples, lookback_window, n_assets, n_features)
        y: (n_samples, n_assets, n_targets)
    """

    def __init__(self, features_df, targets_df, lookback_window=60):
        """
        Initialize dataset.

        Args:
            features_df: pd.DataFrame (T x (n_assets * n_features))
            targets_df: pd.DataFrame (T x n_targets)
            lookback_window: historical context length
        """
        self.features_df = features_df
        self.targets_df = targets_df
        self.lookback_window = lookback_window
        self.n_samples = len(features_df) - lookback_window

    def __len__(self):
        """Return number of samples."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a single sample.

        Args:
            idx: sample index

        Returns:
            (X, y) where X is historical window and y is target
        """
        pass


def create_dataloader(
    features_df,
    targets_df,
    lookback_window=60,
    batch_size=32,
    shuffle=False,
    num_workers=0,
):
    """
    Create a PyTorch DataLoader.

    Args:
        features_df: feature data
        targets_df: target data
        lookback_window: historical context window
        batch_size: batch size
        shuffle: whether to shuffle (should be False for time series)
        num_workers: parallel data loading workers

    Returns:
        torch.utils.data.DataLoader
    """
    pass
