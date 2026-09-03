"""
Feature engineering for financial time series.

Computes technical indicators and derived features from raw OHLCV data.

Key responsibilities:
  - Compute returns (simple and log)
  - Rolling volatility
  - Moving averages
  - Momentum indicators
  - Volume-based features
  - NO LEAKAGE: Features computed with proper lookback windows

Critical constraint:
  At time t, a feature can only use data from [0, t]. Never use [t+1, T].

  WRONG (uses future):
    vol[t] = std(returns[t-10:t+10])

  RIGHT (only past):
    vol[t] = std(returns[t-20:t])

  Implementation: Use .shift() to align past data correctly.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def compute_returns(prices: pd.Series, method: str = "log") -> pd.Series:
    """
    Compute returns from prices.

    Args:
        prices: pd.Series of closing prices (indexed by date)
        method: "simple" or "log"
                simple: (P_t - P_{t-1}) / P_{t-1}
                log: log(P_t / P_{t-1})

    Returns:
        pd.Series of returns (same index as prices)
        Note: First row will be NaN (no prior price)
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}")

    return returns


def compute_rolling_volatility(
    returns: pd.Series, window: int = 20
) -> pd.Series:
    """
    Compute rolling volatility (annualized standard deviation).

    Args:
        returns: pd.Series of returns
        window: lookback window in days

    Returns:
        pd.Series of rolling volatility

    Formula:
      vol[t] = std(returns[t-window:t])  ← only past data
      annualized = vol * sqrt(252)  ← 252 trading days/year

    Note:
      - First (window-1) rows will be NaN
      - Properly causal: only uses past returns
    """
    # rolling().std() uses data from [t-window+1, t] (includes current)
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    return vol


def compute_moving_average(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute moving average.

    Args:
        prices: pd.Series of prices
        window: lookback window in days

    Returns:
        pd.Series of moving average

    Formula:
      ma[t] = mean(prices[t-window:t])  ← only past data

    Note:
      - First (window-1) rows will be NaN
      - Properly causal
    """
    ma = prices.rolling(window=window).mean()
    return ma


def compute_momentum(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute momentum (price change over window).

    Args:
        prices: pd.Series of prices
        window: lookback window in days

    Returns:
        pd.Series of momentum (fractional change)

    Formula:
      momentum[t] = (prices[t] - prices[t-window]) / prices[t-window]

    Note:
      - First window rows will be NaN
      - Properly causal: t-window is in the past
    """
    momentum = (prices - prices.shift(window)) / prices.shift(window)
    return momentum


def compute_volume_change(volume: pd.Series, window: int = 5) -> pd.Series:
    """
    Compute volume change (% change from rolling mean).

    Args:
        volume: pd.Series of volume
        window: lookback window in days

    Returns:
        pd.Series of volume change ratio

    Formula:
      vol_change[t] = volume[t] / mean(volume[t-window:t])

    Interpretation:
      > 1.0: above average volume
      < 1.0: below average volume
    """
    vol_mean = volume.rolling(window=window).mean()
    vol_change = volume / vol_mean
    return vol_change


def engineer_features(
    data: pd.DataFrame,
    ma_windows: list = None,
    momentum_windows: list = None,
    vol_windows: list = None,
) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline.

    Adds derived columns: returns, volatility, moving averages, etc.
    All features are computed with NO LEAKAGE (only past data).

    Args:
        data: pd.DataFrame with MultiIndex columns (asset, OHLCV)
              Input from Phase 1: load_or_download()
        ma_windows: list of moving average windows (default: [5, 20, 60])
        momentum_windows: list of momentum windows (default: [5, 20, 60])
        vol_windows: list of volatility windows (default: [5, 20])

    Returns:
        pd.DataFrame with original OHLCV + engineered features
        Same index (dates), expanded columns

    Engineered features per asset:
      - returns_log: log returns
      - returns_simple: simple returns
      - volatility_{window}: rolling volatility
      - ma_{window}: moving average
      - momentum_{window}: momentum
      - vol_change_{window}: volume change ratio

    Shape after engineering:
      Input:  (n_dates, n_assets * 6 OHLCV)
      Output: (n_dates, n_assets * (6 OHLCV + n_engineered_features))
    """
    if ma_windows is None:
        ma_windows = [5, 20, 60]
    if momentum_windows is None:
        momentum_windows = [5, 20, 60]
    if vol_windows is None:
        vol_windows = [5, 20]

    features = data.copy()

    # Get unique assets from MultiIndex columns
    assets = features.columns.get_level_values(0).unique()

    logger.info(f"Engineering features for {len(assets)} assets...")

    for asset in assets:
        asset_data = features[asset]

        # Returns (from Close price)
        features[(asset, "returns_log")] = compute_returns(asset_data["Close"], method="log")
        features[(asset, "returns_simple")] = compute_returns(
            asset_data["Close"], method="simple"
        )

        # Volatility (from returns)
        returns = features[(asset, "returns_log")]
        for window in vol_windows:
            features[(asset, f"volatility_{window}")] = compute_rolling_volatility(
                returns, window=window
            )

        # Moving averages (from Close price)
        close = asset_data["Close"]
        for window in ma_windows:
            features[(asset, f"ma_{window}")] = compute_moving_average(close, window=window)

        # Momentum (from Close price)
        for window in momentum_windows:
            features[(asset, f"momentum_{window}")] = compute_momentum(close, window=window)

        # Volume change (from Volume)
        if "Volume" in asset_data.columns:
            volume = asset_data["Volume"]
            for window in ma_windows:
                features[(asset, f"vol_change_{window}")] = compute_volume_change(
                    volume, window=window
                )

    logger.info(
        f"Feature engineering complete. Output shape: {features.shape} "
        f"({len(assets)} assets × {features.shape[1] // len(assets)} features)"
    )

    return features
