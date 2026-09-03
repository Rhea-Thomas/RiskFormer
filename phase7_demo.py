"""
Phase 7 Demo: Transformer Backbone

Shows how blocks stack to refine representations.

Key insight: Each block learns to transform the sequence while preserving
temporal causality through residual connections.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

print("\n" + "="*80)
print("PHASE 7 DEMO: Transformer Backbone")
print("="*80)

# Configuration
d_model = 64
n_heads = 4
n_layers = 3
seq_len = 10
batch_size = 2

print(f"\nConfiguration:")
print(f"   Model dimension: {d_model}")
print(f"   Attention heads: {n_heads}")
print(f"   Backbone layers: {n_layers}")
print(f"   Sequence length: {seq_len} time steps")
print(f"   Batch size: {batch_size} samples")

# Step 1: Explain the block architecture
print(f"\n1. Single Transformer Block:")
print(f"   Input: (batch, seq_len, d_model)")
print(f"     v")
print(f"   Layer Norm")
print(f"     v")
print(f"   Causal Self-Attention (learns what to attend to)")
print(f"     v")
print(f"   Residual Connection: x + Attention(LN(x))")
print(f"   (skip connection preserves gradient flow)")
print(f"     v")
print(f"   Layer Norm")
print(f"     v")
print(f"   Feed-Forward Network (position-wise MLP)")
print(f"   - d_model -> 4*d_model (ReLU) -> d_model")
print(f"   - Applied independently at each time step")
print(f"     v")
print(f"   Residual Connection: x + FFN(LN(x))")
print(f"     v")
print(f"   Output: (batch, seq_len, d_model)")

# Step 2: Why layer normalization
print(f"\n2. Layer Normalization (pre-LN architecture):")
print(f"   Applied BEFORE each sub-layer (attention, FFN)")
print(f"   Purpose:")
print(f"   - Stabilize training: keep activations in consistent range")
print(f"   - Improve gradient flow: normalized inputs easier to learn")
print(f"   - No dependency on batch size (unlike batch norm)")
print(f"\n   Formula: LN(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta")
print(f"   Where gamma, beta are learnable scale/shift")

# Step 3: Why residual connections
print(f"\n3. Residual Connections (skip connections):")
print(f"   Without residuals:")
print(f"   - Layer 1 -> Layer 2 -> ... -> Layer N")
print(f"   - Early gradients may vanish in deep networks")
print(f"   - Hard to train deep models")
print(f"\n   With residuals: x -> x + f(x)")
print(f"   - Layer only needs to learn the DELTA (smaller change)")
print(f"   - Gradient flows directly back through skip")
print(f"   - Can train much deeper networks (100+ layers possible)")
print(f"\n   Example:")
print(f"   - Layer learns 5% change: output = x + 0.05*f(x)")
print(f"   - Easier than learning entire transformation from scratch")

# Step 4: Why feed-forward network
print(f"\n4. Feed-Forward Network (Position-wise MLP):")
print(f"   Complements attention's learning:")
print(f"   - Attention: learns WHICH information matters (dynamic)")
print(f"   - FFN: learns HOW to transform each piece (static)")
print(f"\n   Architecture: d_model -> 4*d_model -> d_model")
print(f"   - Expansion: {d_model} -> {4*d_model} dims (increase capacity)")
print(f"   - Non-linearity: ReLU activation")
print(f"   - Projection back: {4*d_model} -> {d_model} dims")
print(f"   - Result: learned non-linear transformation")

# Step 5: The stack
print(f"\n5. Stacking Blocks (Depth = {n_layers}):")
print(f"\n   Block 1: Learns initial patterns")
print(f"   - Input: embedded sequence")
print(f"   - Output: refined representations")
print(f"     v")
print(f"   Block 2: Refines further")
print(f"   - Input: Block 1 output (can attend to Block 1's features)")
print(f"   - Output: more abstract patterns")
print(f"     v")
print(f"   Block 3: Final refinement")
print(f"   - Input: Block 2 output")
print(f"   - Output: rich feature representations")
print(f"\n   Key: Each block's output becomes next block's input")
print(f"   - Deeper layers see refined representations")
print(f"   - Can attend to previous refinements via causal attention")
print(f"   - No new temporal leakage (causal mask already in place)")

# Step 6: Information flow
print(f"\n6. Information Flow in One Block:")
print(f"\n   Attention path:")
print(f"   - Learns dynamic weights based on content")
print(f"   - 'At this position, these past positions matter most'")
print(f"   - Output = weighted combination of past")
print(f"\n   Feed-forward path:")
print(f"   - Learns static non-linear transformation")
print(f"   - Applies same transformation at each position")
print(f"   - Complements attention's dynamic weighting")
print(f"\n   Combined (via residual):")
print(f"   - Original input preserved (via skip)")
print(f"   - Refined by both attention + FFN")
print(f"   - Gradients flow strongly back to input")

# Step 7: The complete pipeline
print(f"\n7. Complete Transformer Backbone Pipeline:")
print(f"\n   Input shape: ({batch_size}, {seq_len}, {d_model})")
print(f"   - {batch_size} samples (training batch)")
print(f"   - {seq_len} time steps each")
print(f"   - {d_model} dimensions per time step")
print(f"     v")
for i in range(1, n_layers+1):
    print(f"   Block {i}: LN -> Attn -> Residual + LN -> FFN -> Residual")
    print(f"   - {n_heads} attention heads")
    print(f"   - {4*d_model}D feed-forward hidden")
    if i < n_layers:
        print(f"     v")
print(f"     v")
print(f"   Output shape: ({batch_size}, {seq_len}, {d_model})")
print(f"   Ready for prediction heads (Phase 8)")

# Step 8: Why this architecture works
print(f"\n8. Why This Architecture Works:")
print(f"   - Causal masking: prevents future leakage (temporal integrity)")
print(f"   - Residual connections: enables deep learning (100+ layers)")
print(f"   - Multi-head attention: captures diverse patterns (4 heads = 4 experts)")
print(f"   - Layer normalization: stabilizes training (better convergence)")
print(f"   - Position-wise FFN: adds non-linearity (expressiveness)")
print(f"\n   Result: State-of-the-art architecture for time-series modeling")

# Step 9: No new leakage concerns
print(f"\n9. Temporal Leakage Check:")
print(f"   SAFE: Causal mask from Phase 6 is maintained")
print(f"   - Block 1 attention: only past (via causal mask)")
print(f"   - Block 1 FFN: position-wise, no cross-time communication")
print(f"   - Block 2 input: only depends on Block 1 output (which was causal)")
print(f"   - ... repeated for all blocks ...")
print(f"\n   Mathematical guarantee:")
print(f"   output[t] = f(input[0:t+1]) for all t")
print(f"   No future information can flow backward")

# Step 10: What comes next
print(f"\n10. What Comes Next (Phase 8):")
print(f"   Backbone output: (batch, seq_len, d_model) refined features")
print(f"   But we need predictions: prices, volatility, probabilities")
print(f"\n   Phase 8 adds prediction heads:")
print(f"   - Dense head: (seq_len, d_model) -> (seq_len, 1) price prediction")
print(f"   - Volatility head: (seq_len, d_model) -> (seq_len, 1) vol prediction")
print(f"   - Multi-task learning: joint optimization")

print("\n" + "="*80)
print("PHASE 7 COMPLETE")
print("="*80)

print("\nKey insights:")
print("  1. Transformer block = Attention + FFN + Residuals + LayerNorm")
print("  2. Residual connections enable deep networks (gradient flow)")
print("  3. Layer normalization stabilizes training")
print("  4. Stacking blocks refines representations progressively")
print("  5. Each block maintains causal ordering (no new leakage)")
print("  6. Backbone output is input to prediction heads")
print("\nNext phase: Phase 8 - Prediction Heads (output layer)")
