"""
Phase 3 Demo: Leakage-Safe Normalization

Shows the critical pipeline:
  Raw features → Chronological split → Fit on train → Apply to val/test

Demonstrates that normalization parameters come from training data ONLY.
"""

import pandas as pd
import numpy as np
from data.normalization import normalize_dataset
import logging

logging.basicConfig(level=logging.INFO)
np.random.seed(42)


def create_realistic_split_data(n_train=100, n_val=40, n_test=30):
    """
    Create data with train/val/test having DIFFERENT statistical properties.

    This is realistic: market regimes change over time, so val/test have
    different mean/std than the training period.
    """
    # Train period: calm market (low volatility)
    train = pd.DataFrame(
        {
            "returns": np.random.normal(0.0005, 0.01, n_train),
            "volume": np.random.normal(1e7, 2e6, n_train),
            "volatility": np.random.normal(0.15, 0.02, n_train),
        }
    )

    # Val period: market rally (higher mean return, similar vol)
    val = pd.DataFrame(
        {
            "returns": np.random.normal(0.0015, 0.012, n_val),  # Higher mean!
            "volume": np.random.normal(1.2e7, 2.5e6, n_val),     # Higher volume!
            "volatility": np.random.normal(0.18, 0.025, n_val),
        }
    )

    # Test period: market correction (negative mean, higher vol)
    test = pd.DataFrame(
        {
            "returns": np.random.normal(-0.0005, 0.015, n_test),  # Negative mean!
            "volume": np.random.normal(1.5e7, 3e6, n_test),       # Even higher!
            "volatility": np.random.normal(0.22, 0.03, n_test),   # Higher volatility!
        }
    )

    return train, val, test


def main():
    print("\n" + "="*80)
    print("PHASE 3 DEMO: Leakage-Safe Normalization")
    print("="*80)

    # Step 1: Create realistic split data
    print("\n1. Creating train/val/test with different statistical properties...")
    train, val, test = create_realistic_split_data()

    print(f"   Train shape: {train.shape}")
    print(f"   Val shape: {val.shape}")
    print(f"   Test shape: {test.shape}")

    # Step 2: Show that they have different statistics
    print("\n2. Statistical comparison (BEFORE normalization):")
    print("\n   Returns (raw):")
    print(f"     Train: mean={train['returns'].mean():.6f}, std={train['returns'].std():.6f}")
    print(f"     Val:   mean={val['returns'].mean():.6f}, std={val['returns'].std():.6f}")
    print(f"     Test:  mean={test['returns'].mean():.6f}, std={test['returns'].std():.6f}")

    print("\n   Volume (raw):")
    print(f"     Train: mean={train['volume'].mean():.0f}, std={train['volume'].std():.0f}")
    print(f"     Val:   mean={val['volume'].mean():.0f}, std={val['volume'].std():.0f}")
    print(f"     Test:  mean={test['volume'].mean():.0f}, std={test['volume'].std():.0f}")

    # Step 3: Normalize using train statistics ONLY
    print("\n3. Normalizing with Z-score (using train statistics only)...")
    train_norm, val_norm, test_norm, normalizer = normalize_dataset(train, val, test, method="zscore")

    # Step 4: Show that normalizer used train statistics
    print("\n4. Normalizer parameters (from train data only):")
    for col in ["returns", "volume"]:
        mean = normalizer.params[col]["mean"]
        std = normalizer.params[col]["std"]
        print(f"   {col:12s}: mean={mean:.6f}, std={std:.6f}")

    # Step 5: Compare statistics AFTER normalization
    print("\n5. Statistical comparison (AFTER normalization using train stats):")
    print("\n   Returns (normalized):")
    print(f"     Train: mean={train_norm['returns'].mean():.6f}, std={train_norm['returns'].std():.6f}")
    print(f"     Val:   mean={val_norm['returns'].mean():.6f}, std={val_norm['returns'].std():.6f}")
    print(f"     Test:  mean={test_norm['returns'].mean():.6f}, std={test_norm['returns'].std():.6f}")

    print("\n   Volume (normalized):")
    print(f"     Train: mean={train_norm['volume'].mean():.6f}, std={train_norm['volume'].std():.6f}")
    print(f"     Val:   mean={val_norm['volume'].mean():.6f}, std={val_norm['volume'].std():.6f}")
    print(f"     Test:  mean={test_norm['volume'].mean():.6f}, std={test_norm['volume'].std():.6f}")

    # Step 6: Key observation
    print("\n6. KEY INSIGHT: Leakage Prevention")
    print("\n   Notice:")
    print("     [OK] Train is centered at 0 (because we subtracted its mean)")
    print("     [OK] Val is NOT centered at 0 (because val had different mean than train)")
    print("     [OK] Test is NOT centered at 0 (because test had different mean than train)")
    print("\n   This is CORRECT. It means:")
    print("     - Normalizer only learned from train")
    print("     - Val/test did not influence normalization parameters")
    print("     - Val/test keep their own distributional properties (signal for learning)")

    # Step 7: Inverse transform (recovering original scale)
    print("\n7. Inverse transform (denormalizing predictions):")
    train_recovered = normalizer.inverse_transform(train_norm)
    error = (train_recovered["returns"] - train["returns"]).abs().max()
    print(f"   Max reconstruction error: {error:.2e}")
    print(f"   Status: {'PASS [OK]' if error < 1e-10 else 'FAIL'}")

    # Step 8: Production usage
    print("\n8. Production usage (simulated):")
    print("   Saved normalizer trained on 2020-2021 data")
    print("   New data arrives in 2024...")

    # Simulate new data (prod)
    prod_data = pd.DataFrame(
        {
            "returns": np.random.normal(0.0008, 0.011, 10),
            "volume": np.random.normal(1.3e7, 2.2e6, 10),
            "volatility": np.random.normal(0.19, 0.024, 10),
        }
    )

    # Apply saved normalizer (no refitting!)
    prod_norm = normalizer.transform(prod_data)
    print(f"   Applied 2020-2021 normalizer to 2024 data: {prod_norm.shape}")
    print(f"   First 3 rows of normalized 2024 data:")
    print(prod_norm.head(3))

    print("\n" + "="*80)
    print("PHASE 3 COMPLETE")
    print("="*80)

    print("\nKey takeaways:")
    print("  1. Fit ONLY on training data")
    print("  2. Apply to val/test WITHOUT refitting")
    print("  3. Val/test keep their distributional properties (no leakage)")
    print("  4. In production, use saved train-era normalizer")
    print("\nNext phase: Phase 4 - Rolling Window Dataset construction")


if __name__ == "__main__":
    main()
