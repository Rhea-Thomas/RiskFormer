"""
Evaluation module for RiskFormer.

Computes:
  - Forecasting metrics (MAE, RMSE, directional accuracy, correlation)
  - Probabilistic metrics (calibration, quantile coverage)
  - Risk metrics (volatility error, VaR, CVaR)
  - Walk-forward validation
  - Comparison against baselines
"""
