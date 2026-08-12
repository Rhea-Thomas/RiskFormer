"""
Tests for data module.

Focus:
  - Feature engineering correctness
  - No temporal leakage in rolling windows
  - Dataset shape and alignment
  - Normalization applied correctly (train stats only)
"""

import pytest


def test_no_future_leakage_in_rolling_window():
    """Verify rolling windows don't include future information."""
    pass


def test_normalization_fit_on_train_only():
    """Verify normalization parameters come from train set only."""
    pass


def test_dataset_shapes():
    """Verify dataset produces correct tensor shapes."""
    pass
