"""
Phase 6 Demo: Causal Self-Attention

Shows the core attention mechanism with causal masking.

Key insight: at each position, attend only to PAST, never FUTURE.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

print("\n" + "="*80)
print("PHASE 6 DEMO: Causal Self-Attention")
print("="*80)

# Configuration
seq_len = 5
d_model = 64
n_heads = 4
d_k = d_model // n_heads

print(f"\nConfiguration:")
print(f"   Sequence length: {seq_len} time steps")
print(f"   Model dimension: {d_model}")
print(f"   Attention heads: {n_heads}")
print(f"   Dimension per head: {d_k}")

# Step 1: Explain the core equation
print(f"\n1. Core Attention Equation:")
print(f"   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V")
print(f"\n   Q (Query):  'What am I looking for?'")
print(f"   K (Key):    'Where is information?'")
print(f"   V (Value):  'What is the information?'")
print(f"   sqrt(d_k):  Scaling factor = sqrt({d_k}) = {np.sqrt(d_k):.2f}")

# Step 2: Create example queries and keys
print(f"\n2. Example Query-Key Similarity:")
print(f"   If Q and K are similar -> high attention")
print(f"   If Q and K are different -> low attention")
print(f"\n   Imagine:")
print(f"   - Position 0 (day 1): Q asks 'what are bullish signals?'")
print(f"   - Position 1 (day 2): K says 'here are bullish signals' -> HIGH similarity")
print(f"   - Position 2 (day 3): K says 'here are bearish signals' -> LOW similarity")

# Step 3: Explain attention scores
print(f"\n3. Attention Scores (QK^T / sqrt(d_k)):")
print(f"   For 5 time steps, this creates a 5x5 matrix of scores")
print(f"   scores[i,j] = how much position i attends to position j")

# Step 4: The CRITICAL causal mask
print(f"\n4. CRITICAL: Causal Mask (prevents future leakage):")
print(f"\n   Lower-triangular mask (True = attend, False = mask):")

causal_mask = np.tril(np.ones((5, 5), dtype=bool))
print(f"\n   Position | Can attend to positions:")
for i in range(5):
    can_attend = [str(j) for j in range(5) if causal_mask[i, j]]
    cannot_attend = [str(j) for j in range(5) if not causal_mask[i, j]]
    print(f"   {i}        | YES: {', '.join(can_attend):15s} NO: {', '.join(cannot_attend)}")

print(f"\n   Key insight:")
print(f"   - Position 0 can only attend to position 0 (itself)")
print(f"   - Position 1 can attend to positions 0, 1")
print(f"   - Position 4 can attend to positions 0, 1, 2, 3, 4 (all past + self)")
print(f"   - NO position can attend to FUTURE positions (row > col)")

# Step 5: Apply mask to attention scores
print(f"\n5. Applying Causal Mask to Attention Scores:")
print(f"\n   Before mask (random scores):")

scores = np.random.randn(5, 5)
print(f"   {scores.round(2)}")

print(f"\n   Masked scores (future = -inf):")
masked_scores = scores.copy()
for i in range(5):
    for j in range(5):
        if not causal_mask[i, j]:
            masked_scores[i, j] = -np.inf

print(f"   {masked_scores[:3].round(2)}")  # Show first 3 rows
print(f"   ... (future positions set to -inf)")

# Step 6: Softmax converts -inf to 0
print(f"\n6. Softmax Converts -inf to 0:")
print(f"   softmax(-inf) = 0")
print(f"   So future positions get ZERO attention weight")
print(f"\n   Example at position 3:")

scores_pos3 = scores[3, :]
masked_pos3 = masked_scores[3, :]
softmax_pos3 = np.zeros(5)
exp_scores = np.exp(masked_pos3 - np.max(masked_pos3[~np.isinf(masked_pos3)]))
exp_scores[np.isinf(masked_pos3)] = 0
softmax_pos3 = exp_scores / np.sum(exp_scores)

print(f"   Position 3 attention weights: {softmax_pos3.round(3)}")
print(f"   - Positions 0-3: normalized probabilities")
print(f"   - Positions 4: exactly 0.000 (masked)")

# Step 7: Multiply by values
print(f"\n7. Multiply by Values:")
print(f"   output[i] = sum(attention_weights[i,j] * V[j] for all j)")
print(f"   Since future positions have weight 0, they don't contribute")
print(f"   Therefore: output depends ONLY on past")

# Step 8: Multi-head attention
print(f"\n8. Multi-Head Attention (parallel heads):")
print(f"   Instead of 1 attention, use {n_heads} parallel heads:")
print(f"   Head 1: 'What price patterns matter?'")
print(f"   Head 2: 'What volatility patterns matter?'")
print(f"   Head 3: 'What correlation patterns matter?'")
print(f"   Head 4: 'What momentum patterns matter?'")
print(f"   Then concatenate: {d_k} * {n_heads} = {d_model} output dims")

# Step 9: Full pipeline
print(f"\n9. Complete Pipeline:")
print(f"   Input: ({seq_len}, {d_model}) = {seq_len} time steps x {d_model}-dim embeddings")
print(f"     v")
print(f"   Project to Q, K, V: ({seq_len}, {d_model}) each")
print(f"     v")
print(f"   Reshape for {n_heads} heads: ({n_heads}, {seq_len}, {d_k}) each")
print(f"     v")
print(f"   Compute attention scores: ({n_heads}, {seq_len}, {seq_len})")
print(f"     v")
print(f"   Apply causal mask + softmax")
print(f"     v")
print(f"   Weighted sum of values: ({n_heads}, {seq_len}, {d_k})")
print(f"     v")
print(f"   Concatenate heads: ({seq_len}, {d_model})")
print(f"     v")
print(f"   Output projection: ({seq_len}, {d_model})")

# Step 10: Why this prevents leakage
print(f"\n10. Why This Prevents Temporal Leakage:")
print(f"    Mathematical guarantee:")
print(f"    - Causal mask ensures future positions get exactly 0 weight")
print(f"    - Therefore output[t] = f(input[0:t+1]) only")
print(f"    - Future information cannot flow backward through attention")
print(f"    - Model learns from past only, never cheats by seeing future")

print("\n" + "="*80)
print("PHASE 6 COMPLETE")
print("="*80)

print("\nKey insights:")
print("  1. Attention learns WHAT to pay attention to")
print("  2. Causal mask enforces temporal order")
print("  3. -inf becomes 0 after softmax (masking)")
print("  4. Multi-head attention captures multiple patterns")
print("  5. Lower-triangular mask = no future leakage")
print("\nNext phase: Phase 7 - Transformer Backbone (stack attention + FFN)")
