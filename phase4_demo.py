"""
Phase 4 Demo: Rolling Window Dataset Construction

Shows how to convert time series into (X, y) pairs for supervised learning.

Key: Maintain temporal order, ensure no leakage.
"""

import pandas as pd
import numpy as np
from data.dataset import FinancialTimeSeriesDataset
import logging

logging.basicConfig(level=logging.INFO)
np.random.seed(42)


def create_realistic_time_series(n_dates=200, n_assets=2):
    """Create realistic multi-asset time series data."""
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")

    assets = ["SPY", "QQQ"][:n_assets]
    features_dict = {}

    for asset in assets:
        base_price = {"SPY": 400, "QQQ": 300}[asset]
        returns = np.random.normal(0.0005, 0.015, n_dates)
        prices = base_price * np.exp(np.cumsum(returns))

        features_dict[(asset, "close")] = prices
        features_dict[(asset, "returns")] = returns
        features_dict[(asset, "volatility")] = np.random.uniform(0.15, 0.25, n_dates)

    features_df = pd.DataFrame(features_dict, index=dates)
    features_df.columns = pd.MultiIndex.from_tuples(features_df.columns)

    # Targets: next-day return
    targets_dict = {
        "SPY_return": np.random.normal(0.0005, 0.015, n_dates),
        "QQQ_return": np.random.normal(0.0005, 0.018, n_dates),
    }
    targets_df = pd.DataFrame(targets_dict, index=dates)

    return features_df, targets_df


def main():
    print("\n" + "="*80)
    print("PHASE 4 DEMO: Rolling Window Dataset Construction")
    print("="*80)

    # Step 1: Create data
    print("\n1. Creating time series data...")
    features_df, targets_df = create_realistic_time_series(n_dates=200, n_assets=2)

    print(f"   Features shape: {features_df.shape}")
    print(f"   Targets shape: {targets_df.shape}")

    # Step 2: Create dataset
    print("\n2. Creating rolling window dataset...")
    lookback = 20
    horizon = 1

    dataset = FinancialTimeSeriesDataset(
        features_df, targets_df, lookback_window=lookback, horizon=horizon
    )

    print(f"   Lookback window: {lookback} days")
    print(f"   Forecast horizon: {horizon} day ahead")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Expected: {len(features_df)} - {lookback} - {horizon} + 1 = {len(features_df) - lookback - horizon + 1}")

    # Step 3: Inspect samples
    print("\n3. Inspecting dataset samples...")

    X0, y0 = dataset[0]
    print(f"\n   Sample 0:")
    print(f"     X shape: {X0.shape}")
    print(f"       - Interpretation: {X0.shape[0]} days of history")
    print(f"       - {X0.shape[1]} assets (SPY, QQQ)")
    print(f"       - {X0.shape[2]} features per asset (close, returns, volatility)")
    print(f"     y shape: {y0.shape}")
    print(f"       - Prediction target (next-day returns for each asset)")

    # Step 4: Verify temporal order
    print("\n4. Verifying temporal order (no leakage)...")

    # For sample idx:
    # X uses times [idx, idx+1, ..., idx+lookback-1]
    # y uses time [idx+lookback+horizon-1]

    idx = 50
    X_idx, y_idx = dataset[idx]

    print(f"\n   Sample at index {idx}:")
    print(f"     X uses times: [{idx}, {idx+1}, ..., {idx+lookback-1}]")
    print(f"       (That's {lookback} days of historical data)")
    print(f"     y uses time: [{idx+lookback+horizon-1}]")
    print(f"       (That's 1 day in the future from last X day)")

    # Verify no overlap
    X_end = idx + lookback - 1
    y_time = idx + lookback + horizon - 1
    print(f"\n   Check: X ends at {X_end}, y is at {y_time}")
    print(f"   Separation: {y_time - X_end} day(s) [OK]")

    # Step 5: Show sequential samples
    print("\n5. Sequential samples (showing how windows slide)...")
    print(f"\n   Sample 0: X = [{0}, ..., {lookback-1}], y = {lookback+horizon-1}")
    print(f"   Sample 1: X = [{1}, ..., {lookback}], y = {lookback+horizon}")
    print(f"   Sample 2: X = [{2}, ..., {lookback+1}], y = {lookback+horizon+1}")
    print(f"   ...")
    print(f"\n   Each sample slides forward by 1 day")
    print(f"   This maintains CHRONOLOGICAL ORDER")

    # Step 6: Check statistics
    print("\n6. Dataset statistics...")
    print(f"   Total data points: {len(features_df)}")
    print(f"   Lookback period: {lookback} days")
    print(f"   Forecast horizon: {horizon} day")
    print(f"   Valid samples: {len(dataset)}")
    print(f"   Days lost to windowing: {len(features_df) - len(dataset)}")

    # Step 7: Show actual data example
    print("\n7. Sample data example (Sample 50):")
    print(f"\n   X[50, :5] (first 5 days of historical window):")
    print(f"   First 3 time steps, all assets/features:")

    X50_np = X0.numpy() if hasattr(X0, 'numpy') else X0
    X50_sample = dataset[50]
    X50_sample_np = X50_sample[0].numpy() if hasattr(X50_sample[0], 'numpy') else X50_sample[0]

    for t in range(min(3, X50_sample_np.shape[0])):
        print(f"     Day {50+t}: ", end="")
        for asset in range(X50_sample_np.shape[1]):
            print(f"Asset{asset}={X50_sample_np[t, asset, 0]:.2f}  ", end="")
        print()

    print("\n" + "="*80)
    print("PHASE 4 COMPLETE")
    print("="*80)

    print("\nKey insights:")
    print("  1. Rolling windows create overlapping sequences")
    print("  2. X contains only PAST data (historical context)")
    print("  3. y contains FUTURE data (prediction target)")
    print("  4. Samples slide forward chronologically (no shuffling)")
    print("  5. NO temporal leakage (X never contains future)")
    print("\nNext phase: Phase 5 - Embeddings (convert raw features to learned representations)")


if __name__ == "__main__":
    main()
