"""
Phase 8 Demo: Prediction Heads

Convert backbone's refined representations into task-specific forecasts.

Key insight: Multiple heads enable multi-task learning.
Each head learns different aspect (price, risk, direction).
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

print("\n" + "="*80)
print("PHASE 8 DEMO: Prediction Heads")
print("="*80)

# Configuration
d_model = 64
seq_len = 10
batch_size = 2

print(f"\nConfiguration:")
print(f"   Model dimension: {d_model}")
print(f"   Sequence length: {seq_len} time steps")
print(f"   Batch size: {batch_size} samples")

# Step 1: Backbone output
print(f"\n1. Backbone Output:")
print(f"   Shape: ({batch_size}, {seq_len}, {d_model})")
print(f"   - {batch_size} training samples")
print(f"   - {seq_len} time steps per sample")
print(f"   - {d_model} refined dimensions per step")
print(f"   This is the UNIFIED representation from all backbone layers")

# Step 2: Price head
print(f"\n2. Price Head (Return Prediction):")
print(f"   Input: ({batch_size}, {seq_len}, {d_model})")
print(f"   Architecture: Linear + ReLU + Dropout + Linear")
print(f"   - Expand: d_model ({d_model}) -> 2*d_model ({2*d_model})")
print(f"   - Non-linearity: ReLU")
print(f"   - Reduce: 2*d_model -> outputs (1 or more)")
print(f"\n   Output options:")
print(f"   - Point estimate (n_outputs=1): ({batch_size}, {seq_len}, 1)")
print(f"     Single prediction of expected return")
print(f"   - Quantiles (n_outputs=3): ({batch_size}, {seq_len}, 3)")
print(f"     q_0.05 (5% quantile), q_0.50 (median), q_0.95 (95% quantile)")
print(f"\n   Loss function:")
print(f"   - Point: MSE (mean squared error)")
print(f"   - Quantiles: Quantile loss (pinball loss)")
print(f"     L_q = (q - 1) * (y - y_pred) if y < y_pred")
print(f"           q * (y - y_pred) if y >= y_pred")
print(f"     Asymmetric penalty for over/under prediction")

# Step 3: Volatility head
print(f"\n3. Volatility Head (Risk Prediction):")
print(f"   Input: ({batch_size}, {seq_len}, {d_model})")
print(f"   Architecture: Linear + ReLU + Dropout + Linear + Softplus")
print(f"\n   Output: ({batch_size}, {seq_len}, 1) always positive")
print(f"   - Softplus(x) = log(1 + exp(x)) ensures output > 0")
print(f"   - Predicts next-period realized volatility")
print(f"\n   Interpretation:")
print(f"   - High vol prediction: market expects big moves (up or down)")
print(f"   - Low vol prediction: market expects small moves")
print(f"   - Independent of direction!")
print(f"\n   Loss function:")
print(f"   - MSE: (y_actual_vol - y_pred_vol)^2")
print(f"   - Gaussian NLL: -log(p(y | mu, sigma))")

# Step 4: Direction head
print(f"\n4. Direction Head (Classification):")
print(f"   Input: ({batch_size}, {seq_len}, {d_model})")
print(f"   Architecture: Linear + ReLU + Dropout + Linear + Sigmoid")
print(f"\n   Output: ({batch_size}, {seq_len}, 1) in [0, 1]")
print(f"   - Sigmoid squashes to probability")
print(f"   - Predicts P(next return > 0)")
print(f"\n   Interpretation:")
print(f"   - 0.0 = 'very likely down'")
print(f"   - 0.5 = 'equally likely up or down'")
print(f"   - 1.0 = 'very likely up'")
print(f"\n   Loss function:")
print(f"   - Binary cross-entropy:")
print(f"     L = -y*log(p) - (1-y)*log(1-p)")
print(f"     Asymmetric if y is 0 or 1")

# Step 5: Multi-task learning
print(f"\n5. Multi-Task Learning (All Heads Together):")
print(f"   Single backbone processes all tasks simultaneously")
print(f"\n   Forward pass:")
print(f"   backbone_output = Backbone(embeddings)")
print(f"   price_pred = PriceHead(backbone_output)")
print(f"   vol_pred = VolatilityHead(backbone_output)")
print(f"   dir_pred = DirectionHead(backbone_output)")
print(f"\n   Training loss (weighted combination):")
print(f"   L_total = w_p * L_price + w_v * L_volatility + w_d * L_direction")
print(f"\n   Common weights:")
print(f"   - w_p = 0.5 (price prediction is main task)")
print(f"   - w_v = 0.3 (volatility helps with risk)")
print(f"   - w_d = 0.2 (direction is auxiliary)")
print(f"   (weights sum to 1.0)")

# Step 6: Why multiple heads?
print(f"\n6. Why Multiple Heads? (Learning Benefits)")
print(f"   Different tasks provide different gradient signals:")
print(f"\n   Price head:")
print(f"   - Learns 'what causes magnitude of moves'")
print(f"   - Gradients emphasize features that predict returns")
print(f"\n   Volatility head:")
print(f"   - Learns 'what causes variability'")
print(f"   - Gradients emphasize features predicting risk")
print(f"   - May be different from price features!")
print(f"\n   Direction head:")
print(f"   - Learns 'what causes direction'")
print(f"   - Easier task (binary classification)")
print(f"   - Provides auxiliary supervision")
print(f"\n   Combined:")
print(f"   - Backbone learns features useful for ALL tasks")
print(f"   - Regularization: prevents overfitting to one task")
print(f"   - Implicit constraint: shared representation must be generalizable")

# Step 7: No temporal leakage
print(f"\n7. Temporal Leakage Check:")
print(f"   Each head is position-wise (fully connected only)")
print(f"   No cross-time communication in heads")
print(f"   Backbone already enforced causality via causal mask")
print(f"\n   Data alignment during training:")
print(f"   - X[t] (features at time t)")
print(f"   - y_price[t+1] (actual price at time t+1)")
print(f"   - y_vol[t+1] (actual vol at time t+1)")
print(f"   - y_dir[t+1] (actual direction at time t+1)")
print(f"\n   Head predictions at t use only X[0:t+1] (never future)")
print(f"   Targets are X[t+1] (shifted by 1)")
print(f"   No leakage: head cannot see its own target")

# Step 8: Inference flow
print(f"\n8. Inference (Forecasting New Data):")
print(f"   Given: new market data at times [0, 1, ..., T]")
print(f"   Want: predictions of prices, vols, directions at T+1")
print(f"\n   Process:")
print(f"   1. Embed: Create embeddings for times [0, 1, ..., T]")
print(f"      Shape: (1, T+1, d_model) batch_size=1")
print(f"   2. Backbone: Process through backbone")
print(f"      Output shape: (1, T+1, d_model)")
print(f"   3. Heads: Apply all prediction heads")
print(f"      Price: (1, T+1, 1)")
print(f"      Vol: (1, T+1, 1)")
print(f"      Dir: (1, T+1, 1)")
print(f"   4. Extract final time step (index T):")
print(f"      price_T+1 = price_pred[0, T, :]")
print(f"      vol_T+1 = vol_pred[0, T, :]")
print(f"      dir_T+1 = dir_pred[0, T, :]")
print(f"\n   Use these predictions for trading signals")

# Step 9: Practical interpretation
print(f"\n9. Practical Interpretation (Trading Example):")
print(f"   Suppose at time T:")
print(f"   - Price head predicts: return = +0.015 (+1.5%)")
print(f"   - Vol head predicts: sigma = 0.02 (2% daily vol)")
print(f"   - Dir head predicts: P(up) = 0.72 (72% likely to go up)")
print(f"\n   Interpretation:")
print(f"   - Expected move: ~1.5% up")
print(f"   - Risk band (1 sigma): [+1.5% +/- 2%] = [-0.5%, +3.5%]")
print(f"   - Confidence: 72% chance up (but 28% down is non-trivial)")
print(f"\n   Risk-aware decision:")
print(f"   - If portfolio volatility already high: reduce position")
print(f"   - If vol is low: can take larger position")
print(f"   - Use quantile predictions for stop-loss levels")

# Step 10: What comes next
print(f"\n10. Complete Architecture Summary:")
print(f"    Phase 1-5: Data -> Embeddings")
print(f"      Raw market data encoded as learnable representations")
print(f"    Phase 6:   Causal Self-Attention")
print(f"      Learn which past information matters")
print(f"    Phase 7:   Transformer Backbone")
print(f"      Stack attention + FFN for deep refinement")
print(f"    Phase 8:   Prediction Heads")
print(f"      Convert refined features to forecasts")
print(f"\n    Next phase: Phase 9 - Loss Functions")
print(f"      Define objectives for training (MSE, quantile loss, etc.)")

print("\n" + "="*80)
print("PHASE 8 COMPLETE")
print("="*80)

print("\nKey insights:")
print("  1. Heads convert backbone output to task-specific predictions")
print("  2. Price head: point or quantile predictions")
print("  3. Volatility head: positive predictions via softplus")
print("  4. Direction head: probability via sigmoid")
print("  5. Multi-task: shared backbone learns robust features")
print("  6. Each head provides gradient signal during training")
print("  7. Causality maintained: no temporal leakage")
print("  8. Inference: process new data through all layers for predictions")
print("\nNext phase: Phase 9 - Loss Functions and Training Objectives")
