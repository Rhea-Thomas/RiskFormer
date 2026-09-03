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

    def test_validate_multi_asset_dict(self, sample_data):
        """
        Regression test: Ensure validation works with multiple assets in dict.

        This catches the bug where yfinance multi-ticker returns MultiIndex columns,
        causing ambiguous truth values on Series/DataFrame comparisons.
        """
        # Validate all assets at once (as a dict, like yfinance.download returns)
        is_valid, report = validate_market_data(sample_data)
        assert is_valid
        assert all(v == "OK" for v in report.values())


class TestCleaning:
    """Test data cleaning module."""

    @pytest.fixture
    def data_with_missing(self):
        """Create data with missing values."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 110, 20),
                "Close": np.random.uniform(100, 110, 20),
                "Volume": np.random.uniform(1e6, 2e6, 20),
            },
            index=dates,
        )
        # Introduce missing values
        data.loc[data.index[5], "Close"] = np.nan
        data.loc[data.index[10], "Volume"] = np.nan
        return data

    def test_forward_fill(self, data_with_missing):
        """Test forward fill handles missing values."""
        from data.cleaning import handle_missing_values

        cleaned = handle_missing_values(data_with_missing, method="forward_fill")
        assert not cleaned.isnull().any().any()

    def test_detect_outliers(self):
        """Test outlier detection."""
        from data.cleaning import detect_outliers

        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        prices = np.ones(20) * 100
        prices[5] = 150  # 50% jump
        data = pd.DataFrame(
            {
                "Close": prices,
                "Volume": np.ones(20) * 1e6,
            },
            index=dates,
        )

        anomalies = detect_outliers(data, price_change_threshold=0.20)
        assert len(anomalies) > 0
        assert "Extreme price move" in anomalies.iloc[0]["Reason"]


class TestFeatures:
    """Test feature engineering module."""

    @pytest.fixture
    def simple_price_series(self):
        """Create simple price series for testing."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        # Linear increase: 100, 101, 102, ...
        prices = pd.Series(100 + np.arange(50), index=dates)
        return prices

    def test_returns_computation(self, simple_price_series):
        """Test returns computation."""
        from data.features import compute_returns

        returns = compute_returns(simple_price_series, method="simple")

        # For linear increase (100, 101, 102, ...), check first return
        # First return: (101 - 100) / 100 = 0.01
        expected_first_return = 0.01
        actual_first_return = returns.iloc[1]
        np.testing.assert_allclose(actual_first_return, expected_first_return)

        # Verify returns are positive and decreasing (as prices compound)
        actual_returns = returns.dropna()
        assert (actual_returns > 0).all()
        assert (np.diff(actual_returns) < 0).all()  # Decreasing

    def test_no_leakage_moving_average(self, simple_price_series):
        """
        Verify moving average doesn't use future data.

        Critical test: at time t, MA should only use data from [t-window, t].
        """
        from data.features import compute_moving_average

        prices = simple_price_series
        ma = compute_moving_average(prices, window=5)

        # At t=10, MA should be mean(prices[6:11])
        # pandas rolling includes current point, so it's [10-5+1, 10] = [6, 10]
        t = 10
        expected_ma = prices.iloc[t - 4 : t + 1].mean()
        actual_ma = ma.iloc[t]

        np.testing.assert_allclose(actual_ma, expected_ma)

    def test_no_leakage_volatility(self):
        """
        Verify volatility uses only past returns (not future).
        """
        from data.features import compute_returns, compute_rolling_volatility

        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        # Create returns with known std
        returns_values = np.random.normal(0, 0.01, 50)
        returns = pd.Series(returns_values, index=dates)

        vol = compute_rolling_volatility(returns, window=20)

        # At t=25, vol should use returns[6:26] (window=20)
        t = 25
        window = 20
        expected_vol = returns.iloc[t - window + 1 : t + 1].std() * np.sqrt(252)
        actual_vol = vol.iloc[t]

        np.testing.assert_allclose(actual_vol, expected_vol)

    def test_no_leakage_momentum(self):
        """
        Verify momentum uses only past data.
        """
        from data.features import compute_momentum

        prices = pd.Series([100, 101, 102, 103, 104], index=range(5))

        momentum = compute_momentum(prices, window=2)

        # At t=4, momentum = (104 - 102) / 102
        expected = (104 - 102) / 102
        actual = momentum.iloc[4]

        np.testing.assert_allclose(actual, expected)

    def test_feature_engineering_shape(self):
        """Test that feature engineering produces correct output shape."""
        from data.features import engineer_features

        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        # Single asset, MultiIndex columns
        data = pd.DataFrame(
            {
                ("SPY", "Open"): np.random.uniform(100, 110, 50),
                ("SPY", "Close"): np.random.uniform(100, 110, 50),
                ("SPY", "Volume"): np.random.uniform(1e6, 2e6, 50),
                ("SPY", "High"): np.random.uniform(110, 120, 50),
                ("SPY", "Low"): np.random.uniform(90, 100, 50),
                ("SPY", "Adj Close"): np.random.uniform(100, 110, 50),
            },
            index=dates,
        )
        data.columns = pd.MultiIndex.from_tuples(data.columns)

        engineered = engineer_features(data)

        # Should have more features than input
        assert engineered.shape[1] > data.shape[1]
        # Same number of rows
        assert engineered.shape[0] == data.shape[0]
        # Should have returns, volatility, MA, momentum columns
        col_names = [col[1] for col in engineered.columns]
        assert "returns_log" in col_names
        assert "volatility_20" in col_names
        assert "ma_5" in col_names


def test_no_future_leakage_in_rolling_window():
    """Verify rolling windows don't include future information."""
    # Tested in TestFeatures above
    pass


class TestNormalization:
    """Test leakage-safe normalization."""

    @pytest.fixture
    def split_data(self):
        """Create train/val/test data with different statistics."""
        # Train: mean=100, std=5
        train = pd.DataFrame(
            {"price": np.random.normal(100, 5, 50), "volume": np.random.uniform(1e6, 2e6, 50)}
        )

        # Val: mean=110, std=8 (different distribution!)
        val = pd.DataFrame(
            {"price": np.random.normal(110, 8, 30), "volume": np.random.uniform(2e6, 3e6, 30)}
        )

        # Test: mean=95, std=6 (yet different)
        test = pd.DataFrame(
            {"price": np.random.normal(95, 6, 20), "volume": np.random.uniform(1e6, 2e6, 20)}
        )

        return train, val, test

    def test_normalizer_fit_only_on_train(self, split_data):
        """Verify normalizer fits only on training data."""
        from data.normalization import Normalizer

        train, val, test = split_data

        # Fit on train
        normalizer = Normalizer(method="zscore")
        normalizer.fit(train)

        # Extract fit statistics
        train_mean = normalizer.params["price"]["mean"]
        train_std = normalizer.params["price"]["std"]

        # Verify they match train, NOT val or test
        np.testing.assert_allclose(train_mean, train["price"].mean(), rtol=1e-5)
        np.testing.assert_allclose(train_std, train["price"].std(), rtol=1e-5)

        # Should NOT match val/test
        assert not np.isclose(train_mean, val["price"].mean())
        assert not np.isclose(train_mean, test["price"].mean())

    def test_normalizer_immutable_after_fit(self, split_data):
        """Verify normalizer cannot be refitted (is immutable)."""
        from data.normalization import Normalizer

        train, val, test = split_data

        normalizer = Normalizer(method="zscore")
        normalizer.fit(train)

        # Store original params
        orig_mean = normalizer.params["price"]["mean"]

        # Calling transform on val does NOT change the normalizer
        normalizer.transform(val)

        # Params should be unchanged
        assert normalizer.params["price"]["mean"] == orig_mean

    def test_normalize_dataset_no_leakage(self, split_data):
        """
        Critical test: Verify that train statistics don't leak into val/test.

        This is the core leakage prevention test.
        """
        from data.normalization import normalize_dataset

        train, val, test = split_data

        # Normalize using train stats only
        train_norm, val_norm, test_norm, normalizer = normalize_dataset(train, val, test)

        # After normalization with train stats, train should be centered near 0
        # (because we subtract train mean)
        train_norm_mean = train_norm["price"].mean()
        assert np.isclose(train_norm_mean, 0, atol=1e-10)

        # But val and test should NOT be centered at 0
        # (because we subtract train mean, not val/test mean)
        val_norm_mean = val_norm["price"].mean()
        test_norm_mean = test_norm["price"].mean()

        # These should be non-zero because val/test had different means
        assert not np.isclose(val_norm_mean, 0, atol=0.5)  # Allow some tolerance
        assert not np.isclose(test_norm_mean, 0, atol=0.5)

    def test_normalizer_zscore_correctness(self):
        """Verify z-score normalization formula."""
        from data.normalization import Normalizer

        data_train = pd.DataFrame({"x": [100, 102, 98, 101, 99]})
        data_test = pd.DataFrame({"x": [100, 105]})

        normalizer = Normalizer(method="zscore", epsilon=0)
        normalizer.fit(data_train)
        result = normalizer.transform(data_test)

        # Manual calculation
        mean = data_train["x"].mean()  # 100
        std = data_train["x"].std()    # ~1.41

        expected_0 = (100 - mean) / std  # 0
        expected_1 = (105 - mean) / std  # ~3.5

        np.testing.assert_allclose(result["x"].iloc[0], expected_0, rtol=1e-5)
        np.testing.assert_allclose(result["x"].iloc[1], expected_1, rtol=1e-5)

    def test_normalizer_inverse_transform(self):
        """Verify inverse_transform recovers original data."""
        from data.normalization import Normalizer

        data_train = pd.DataFrame({"x": np.random.uniform(0, 100, 50)})
        data_test = pd.DataFrame({"x": np.random.uniform(0, 100, 20)})

        normalizer = Normalizer(method="zscore", epsilon=1e-8)
        normalizer.fit(data_train)

        # Transform and inverse transform
        data_norm = normalizer.transform(data_test)
        data_recovered = normalizer.inverse_transform(data_norm)

        # Should recover original (approximately)
        np.testing.assert_allclose(data_recovered["x"].values, data_test["x"].values, rtol=1e-5)

    def test_normalizer_minmax_method(self):
        """Test min-max normalization."""
        from data.normalization import Normalizer

        data_train = pd.DataFrame({"x": [0, 50, 100]})
        data_test = pd.DataFrame({"x": [0, 50, 100]})

        normalizer = Normalizer(method="minmax", epsilon=0)
        normalizer.fit(data_train)
        result = normalizer.transform(data_test)

        # Min-max: (x - min) / (max - min)
        # 0 → (0 - 0) / 100 = 0
        # 50 → (50 - 0) / 100 = 0.5
        # 100 → (100 - 0) / 100 = 1
        expected = pd.DataFrame({"x": [0.0, 0.5, 1.0]})
        np.testing.assert_allclose(result["x"].values, expected["x"].values, rtol=1e-5)


def test_dataset_shapes():
    """Verify dataset produces correct tensor shapes."""
    # This will be tested in Phase 4 (dataset.py)
    pass
