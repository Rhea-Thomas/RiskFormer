"""
Data cleaning and quality assurance.

Handles missing values, outliers, and anomalies in market data.

Key responsibilities:
  - Detect and handle missing values (forward fill, interpolation, etc.)
  - Identify and handle price/volume anomalies
  - Handle corporate actions (splits, dividends) if needed
  - Ensure data consistency across assets

Philosophy:
  - Clean but preserve; flag outliers but don't remove
  - Conservative approach: forward fill for prices (minimal distortion)
  - Return both cleaned data and anomaly report
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def handle_missing_values(data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
    """
    Handle missing values in price data.

    Args:
        data: pd.DataFrame with OHLCV columns
        method: "forward_fill", "interpolate", or "drop"

    Returns:
        cleaned pd.DataFrame

    Note:
        For financial data:
        - forward_fill: safest (repeats last known price)
        - interpolate: assumes linear price path (risky for gaps)
        - drop: loses data (use only for small gaps)
    """
    data_clean = data.copy()

    if method == "forward_fill":
        data_clean = data_clean.ffill()
        # Forward fill doesn't propagate to all columns, so also bfill for first rows
        data_clean = data_clean.bfill()

    elif method == "interpolate":
        # Linear interpolation (risky; use only for small gaps)
        data_clean = data_clean.interpolate(method="linear")
        data_clean = data_clean.bfill().ffill()

    elif method == "drop":
        data_clean = data_clean.dropna()

    else:
        raise ValueError(f"Unknown method: {method}")

    n_missing = data.isnull().sum().sum()
    n_filled = n_missing - data_clean.isnull().sum().sum()
    logger.info(f"Handled missing values: {n_filled}/{n_missing} filled using {method}")

    return data_clean


def detect_outliers(
    data: pd.DataFrame,
    price_change_threshold: float = 0.20,
    volume_zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect price and volume outliers.

    Returns DataFrame of flagged rows (anomalies).

    Args:
        data: pd.DataFrame with OHLCV columns
              Can be single-asset (regular columns) or multi-asset (MultiIndex columns)
              Must have 'Close' and 'Volume' columns (or under each asset if MultiIndex)
        price_change_threshold: flag if |daily return| exceeds this (e.g., 0.20 = 20%)
        volume_zscore_threshold: flag if volume > mean + N_std deviations

    Returns:
        DataFrame of flagged rows with anomaly reasons

    Note:
        Outliers are flagged but NOT removed. This preserves real market events
        (stock splits, large moves) while alerting us to potential data issues.
    """
    anomalies = []

    # Handle MultiIndex columns (multiple assets)
    if isinstance(data.columns, pd.MultiIndex):
        assets = data.columns.get_level_values(0).unique()
        for asset in assets:
            asset_data = data[asset]
            close_prices = asset_data["Close"]
            daily_returns = close_prices.pct_change().abs()

            # Price outliers
            price_flags = daily_returns > price_change_threshold
            if price_flags.any():
                for idx in data.index[price_flags]:
                    anomalies.append(
                        {
                            "Date": idx,
                            "Asset": asset,
                            "Reason": f"Extreme price move: {daily_returns[idx]:.2%}",
                            "Type": "price",
                        }
                    )

            # Volume outliers
            if "Volume" in asset_data.columns:
                volume = asset_data["Volume"]
                vol_mean = volume.mean()
                vol_std = volume.std()
                vol_threshold = vol_mean + volume_zscore_threshold * vol_std

                volume_flags = volume > vol_threshold
                if volume_flags.any():
                    for idx in data.index[volume_flags]:
                        z_score = (volume[idx] - vol_mean) / vol_std
                        anomalies.append(
                            {
                                "Date": idx,
                                "Asset": asset,
                                "Reason": f"Volume spike: {z_score:.1f} std above mean",
                                "Type": "volume",
                            }
                        )
    else:
        # Single asset (regular columns)
        close_prices = data["Close"]
        daily_returns = close_prices.pct_change().abs()

        # Price outliers: extreme daily moves
        price_flags = daily_returns > price_change_threshold
        if price_flags.any():
            price_outlier_dates = data.index[price_flags]
            for date in price_outlier_dates:
                anomalies.append(
                    {
                        "Date": date,
                        "Reason": f"Extreme price move: {daily_returns[date]:.2%}",
                        "Type": "price",
                    }
                )

        # Volume outliers: unusual volume spikes
        if "Volume" in data.columns:
            volume = data["Volume"]
            vol_mean = volume.mean()
            vol_std = volume.std()
            vol_threshold = vol_mean + volume_zscore_threshold * vol_std

            volume_flags = volume > vol_threshold
            if volume_flags.any():
                volume_outlier_dates = data.index[volume_flags]
                for date in volume_outlier_dates:
                    z_score = (volume[date] - vol_mean) / vol_std
                    anomalies.append(
                        {
                            "Date": date,
                            "Reason": f"Volume spike: {z_score:.1f} std above mean",
                            "Type": "volume",
                        }
                    )

    anomaly_df = pd.DataFrame(anomalies) if anomalies else pd.DataFrame()
    logger.info(f"Detected {len(anomaly_df)} anomalies")

    return anomaly_df


def clean_data(
    data: pd.DataFrame,
    missing_method: str = "forward_fill",
    price_change_threshold: float = 0.20,
    volume_zscore_threshold: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply full cleaning pipeline.

    Args:
        data: raw market data (from Phase 1)
        missing_method: how to handle missing values
        price_change_threshold: outlier detection threshold for price moves
        volume_zscore_threshold: outlier detection threshold for volume

    Returns:
        (cleaned_data, anomaly_report)
        - cleaned_data: cleaned DataFrame, ready for feature engineering
        - anomaly_report: DataFrame with flagged anomalies (for inspection)
    """
    logger.info("Starting data cleaning pipeline...")

    # Step 1: Handle missing values
    data_clean = handle_missing_values(data, method=missing_method)

    # Step 2: Detect outliers (before removing them, for inspection)
    anomalies = detect_outliers(
        data_clean,
        price_change_threshold=price_change_threshold,
        volume_zscore_threshold=volume_zscore_threshold,
    )

    # Step 3: Log summary
    n_rows_lost = len(data) - len(data_clean)
    if n_rows_lost > 0:
        logger.warning(f"Cleaning removed {n_rows_lost} rows")
    else:
        logger.info(f"No rows removed during cleaning")

    logger.info(f"Cleaning complete: {len(data_clean)} rows, {len(anomalies)} anomalies flagged")

    return data_clean, anomalies
