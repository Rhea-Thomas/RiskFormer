"""
Phase 1 Demo: Market Data Ingestion

This script demonstrates how to use the data ingestion module.

Note: Requires yfinance to be installed:
    pip install yfinance
"""

import logging
from data.ingestion import load_or_download

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Define assets and date range
    assets = ["SPY", "QQQ"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    print(f"\nLoading data for {assets} from {start_date} to {end_date}")
    print("(Will download from yfinance if not cached)")

    # Load or download data
    # Returns aligned DataFrame with shape (n_dates, n_assets * n_features)
    # MultiIndex columns: (asset, OHLCV)
    data = load_or_download(assets, start_date, end_date, cache_dir="data/cache")

    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"\nFirst few rows:")
    print(data.head())

    print(f"\nColumn structure (MultiIndex):")
    print(data.columns)
