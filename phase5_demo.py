"""
Phase 5 Demo: Embeddings

Shows how raw features are lifted into learned representations.

Pipeline: Raw features → Feature/Temporal/Asset embeddings → Unified representation
"""

import numpy as np
from models.embeddings import TemporalEmbedding, AssetEmbedding, FeatureEmbedding, EmbeddingCombiner
import logging

logging.basicConfig(level=logging.INFO)
np.random.seed(42)


def main():
    print("\n" + "="*80)
    print("PHASE 5 DEMO: Embeddings")
    print("="*80)

    # Configuration
    lookback = 20
    n_assets = 2
    n_features = 19
    d_model = 256

    print(f"\n Configuration:")
    print(f"   Lookback window: {lookback} days")
    print(f"   Assets: {n_assets} (SPY, QQQ)")
    print(f"   Features per asset: {n_features}")
    print(f"   Embedding dimension (d_model): {d_model}")

    # Step 1: Create sample input
    print(f"\n1. Creating sample input data...")
    X = np.random.randn(lookback, n_assets, n_features)
    print(f"   Shape: {X.shape} = (lookback x assets x features)")

    # Step 2: Feature Embedding
    print(f"\n2. Feature Embedding (raw features -> d_model)...")
    feature_emb = FeatureEmbedding(n_features, d_model)
    print(f"   Input: (lookback x assets x {n_features} features)")
    print(f"   Output: (lookback x assets x {d_model} dims)")

    # Step 3: Temporal Embedding
    print(f"\n3. Temporal Embedding (position encoding)...")
    temporal_emb = TemporalEmbedding(d_model, max_len=1000)
    te = temporal_emb(lookback)
    print(f"   Encodes: position 0, 1, ..., {lookback-1}")
    print(f"   Output shape: {te.shape}")
    print(f"   First 3 dimensions of temporal encoding at position 0:")
    te_np = te.numpy() if hasattr(te, 'numpy') else te
    print(f"     {te_np[0, :3]}")
    print(f"   Position 10:")
    print(f"     {te_np[10, :3]}")
    print(f"   Note: Different positions have DIFFERENT encodings (not all zeros)")

    # Step 4: Asset Embedding
    print(f"\n4. Asset Embedding (learned per-asset representation)...")
    asset_emb = AssetEmbedding(n_assets, d_model)
    ae0 = asset_emb(0)
    ae1 = asset_emb(1)
    print(f"   Asset 0 (SPY):  {ae0.shape}")
    print(f"   Asset 1 (QQQ):  {ae1.shape}")

    ae0_np = ae0.numpy() if hasattr(ae0, 'numpy') else ae0
    ae1_np = ae1.numpy() if hasattr(ae1, 'numpy') else ae1
    print(f"   Difference: {np.linalg.norm(ae0_np - ae1_np):.2f}")
    print(f"   (Different assets have DIFFERENT embeddings)")

    # Step 5: Combine all embeddings
    print(f"\n5. Combining embeddings (Feature + Temporal + Asset)...")
    combiner = EmbeddingCombiner(n_features, n_assets, d_model)

    X_embedded = combiner(X)
    print(f"   Input:  {X.shape} = (lookback, assets, features)")
    print(f"   Output: {X_embedded.shape} = (lookback, assets, d_model)")
    print(f"   Combined embedding is the UNIFIED representation")
    print(f"   that the Transformer backbone will process")

    # Step 6: Show what each embedding contributes
    print(f"\n6. What each embedding contributes:")
    print(f"   Feature emb:  Projects 19 features -> 256 dims")
    print(f"     Learned via backprop (random init -> trained)")
    print(f"   Temporal emb: Encodes position in sequence")
    print(f"     Fixed sinusoidal encoding (no learning)")
    print(f"   Asset emb:    Encodes asset identity (SPY vs QQQ)")
    print(f"     Learned via backprop")
    print(f"\n   Combined = Feature + Temporal + Asset")

    # Step 7: Batch processing
    print(f"\n7. Batch processing (multiple samples)...")
    batch_size = 32
    X_batch = np.random.randn(batch_size, lookback, n_assets, n_features)

    X_batch_embedded = combiner(X_batch)
    print(f"   Input:  {X_batch.shape} = (batch, lookback, assets, features)")
    print(f"   Output: {X_batch_embedded.shape} = (batch, lookback, assets, d_model)")

    # Step 8: Ready for Transformer
    print(f"\n8. Ready for Transformer backbone:")
    print(f"   Input shape: {X_batch_embedded.shape}")
    print(f"   This goes into: Attention layers -> Feed-forward -> Prediction heads")
    print(f"   The Transformer learns what each dimension of the embedding means")

    print("\n" + "="*80)
    print("PHASE 5 COMPLETE")
    print("="*80)

    print("\nKey insights:")
    print("  1. Feature embedding: learned projection (19 -> 256 dims)")
    print("  2. Temporal embedding: sinusoidal position encoding")
    print("  3. Asset embedding: learned per-asset representation")
    print("  4. Combined: single unified representation for Transformer")
    print("  5. No information loss: sum of embeddings preserves all info")
    print("\nNext phase: Phase 6 - Causal Self-Attention (the core of Transformer)")


if __name__ == "__main__":
    main()
