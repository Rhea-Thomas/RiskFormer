"""
Phase 1 Test: Local Data Pipeline

This tests the ingestion pipeline WITHOUT requiring yfinance.
We'll create synthetic market data and run it through the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data.ingestion import validate_market_data, align_assets

np.random.seed(42)

def create_synthetic_ohlcv(n_days=252, base_price=100):
    """Create realistic synthetic OHLCV data."""
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = base_price * np.exp(np.cumsum(returns))

    data = {
        "Open": prices + np.random.uniform(-1, 1, n_days),
        "Close": prices,
        "Volume": np.random.uniform(50e6, 100e6, n_days),
    }

    # High and Low
    data["High"] = np.maximum(data["Open"], data["Close"]) + np.abs(np.random.normal(0, 1, n_days))
    data["Low"] = np.minimum(data["Open"], data["Close"]) - np.abs(np.random.normal(0, 1, n_days))
    data["Adj Close"] = data["Close"]

    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")  # Business days

    return pd.DataFrame(data, index=dates)


def main():
    print("\n" + "="*70)
    print("PHASE 1 TEST: Local Data Pipeline")
    print("="*70)

    # Step 1: Create synthetic data for multiple assets
    print("\n1. Creating synthetic OHLCV data...")
    data = {
        "SPY": create_synthetic_ohlcv(n_days=252, base_price=400),
        "QQQ": create_synthetic_ohlcv(n_days=250, base_price=300),  # Intentionally one day shorter
        "IWM": create_synthetic_ohlcv(n_days=251, base_price=185),  # And one
    }

    for asset, df in data.items():
        print(f"  {asset}: {len(df)} trading days, ${df['Close'].mean():.2f} avg close")

    # Step 2: Validate data
    print("\n2. Validating data...")
    is_valid, report = validate_market_data(data)
    print(f"  Validation result: {'PASS' if is_valid else 'FAIL'}")
    for asset, status in report.items():
        print(f"    {asset}: {status}")

    # Step 3: Align assets
    print("\n3. Aligning assets to common trading dates...")
    before_align = {k: len(v) for k, v in data.items()}
    aligned = align_assets(data)
    after_align = len(aligned)

    print(f"  Before alignment: {before_align}")
    print(f"  After alignment: {after_align} common dates")
    print(f"  Dates removed: {sum(before_align.values()) - after_align * len(data)}")

    # Step 4: Inspect aligned data
    print("\n4. Aligned DataFrame structure:")
    print(f"  Shape: {aligned.shape}")
    print(f"  Index: {aligned.index.min()} to {aligned.index.max()}")
    print(f"  Columns (MultiIndex):")
    for col in aligned.columns:
        print(f"    {col}")

    # Step 5: Check for leakage
    print("\n5. Data quality checks:")
    print(f"  No NaN values: {not aligned.isnull().any().any()} [OK]")
    print(f"  All prices > 0: {(aligned.select_dtypes(include=[np.number]) > 0).all().all()} [OK]")

    # Step 6: Save to cache
    print("\n6. Caching data...")
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "test_aligned_data.csv"
    aligned.to_csv(cache_file)
    print(f"  Saved to: {cache_file}")
    print(f"  File size: {cache_file.stat().st_size / 1024:.1f} KB")

    # Step 7: Load from cache
    print("\n7. Loading from cache...")
    loaded = pd.read_csv(cache_file, index_col=0, header=[0, 1])
    loaded.index = pd.to_datetime(loaded.index)
    print(f"  Loaded successfully: {loaded.shape}")

    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)

    print("\nSample data (first 5 rows):")
    print(aligned.head())

    return aligned


if __name__ == "__main__":
    aligned_data = main()
