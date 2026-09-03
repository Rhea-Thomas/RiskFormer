"""
Tests for data module.

Focus:
  - Feature engineering correctness
  - No temporal leakage in rolling windows
  - Dataset shape and alignment
  - Normalization applied correctly (train stats only)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from data.ingestion import (
    validate_market_data,
    align_assets,
)


class TestDataIngestion:
    """Test market data ingestion."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        data = {
            "SPY": pd.DataFrame(
                {
                    "Open": np.random.uniform(400, 420, 20),
                    "High": np.random.uniform(420, 440, 20),
                    "Low": np.random.uniform(380, 400, 20),
                    "Close": np.random.uniform(400, 420, 20),
                    "Volume": np.random.uniform(50e6, 100e6, 20),
                    "Adj Close": np.random.uniform(400, 420, 20),
                },
                index=dates,
            ),
            "QQQ": pd.DataFrame(
                {
                    "Open": np.random.uniform(300, 320, 20),
                    "High": np.random.uniform(320, 340, 20),
                    "Low": np.random.uniform(280, 300, 20),
                    "Close": np.random.uniform(300, 320, 20),
                    "Volume": np.random.uniform(30e6, 60e6, 20),
                    "Adj Close": np.random.uniform(300, 320, 20),
                },
                index=dates,
            ),
        }

        # Ensure High >= Close >= Low
        for asset in data:
            data[asset]["High"] = data[asset][["Open", "Close", "High"]].max(axis=1) + 5
            data[asset]["Low"] = data[asset][["Open", "Close", "Low"]].min(axis=1) - 5

        return data

    def test_validate_good_data(self, sample_data):
        """Test validation passes for clean data."""
        is_valid, report = validate_market_data(sample_data)
        assert is_valid, f"Validation failed: {report}"

    def test_validate_detects_nan(self, sample_data):
        """Test validation detects NaN values."""
        sample_data["SPY"].loc[sample_data["SPY"].index[0], "Close"] = np.nan
        is_valid, report = validate_market_data(sample_data)
        assert not is_valid
        assert "NaN" in report["SPY"][0]

    def test_validate_detects_price_inversion(self, sample_data):
        """Test validation detects High < Low."""
        sample_data["SPY"].loc[sample_data["SPY"].index[0], "High"] = 100
        sample_data["SPY"].loc[sample_data["SPY"].index[0], "Low"] = 200
        is_valid, report = validate_market_data(sample_data)
        assert not is_valid

    def test_align_assets(self, sample_data):
        """Test that assets are aligned to common dates."""
        # Remove one date from QQQ to test intersection
        sample_data["QQQ"] = sample_data["QQQ"].iloc[1:]

        aligned = align_assets(sample_data)

        # Should have 19 dates (intersection)
        assert len(aligned) == 19

        # Should have columns for both assets
        assert "SPY" in aligned.columns.get_level_values(0)
        assert "QQQ" in aligned.columns.get_level_values(0)

    def test_align_assets_no_nan_after_alignment(self, sample_data):
        """Test no NaN after alignment (intersection only)."""
        aligned = align_assets(sample_data)
        assert not aligned.isnull().any().any(), "Alignment introduced NaN values"

    def test_validate_negative_volume(self, sample_data):
        """Test validation detects negative volume."""
        sample_data["SPY"].loc[sample_data["SPY"].index[0], "Volume"] = -1000
        is_valid, report = validate_market_data(sample_data)
        assert not is_valid


def test_no_future_leakage_in_rolling_window():
    """Verify rolling windows don't include future information."""
    # This will be tested in Phase 4 (dataset.py)
    pass


def test_normalization_fit_on_train_only():
    """Verify normalization parameters come from train set only."""
    # This will be tested in Phase 3 (normalization.py)
    pass


def test_dataset_shapes():
    """Verify dataset produces correct tensor shapes."""
    # This will be tested in Phase 4 (dataset.py)
    pass
