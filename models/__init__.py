"""
Models module for RiskFormer.

Contains:
  - Embeddings (temporal, asset, feature)
  - Attention mechanism (causal self-attention)
  - Transformer backbone
  - Prediction heads (return, volatility, quantile, regime)

Architecture:
  Input → Embeddings → Transformer Backbone → Heads → Outputs
"""
