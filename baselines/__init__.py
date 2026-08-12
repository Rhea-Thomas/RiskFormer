"""
Baseline models for comparison.

Baselines:
  - Naive: yesterday's return
  - Historical mean: average historical return
  - Moving average: simple moving average
  - Linear regression: OLS on features
  - Ridge/Lasso: regularized linear regression
  - XGBoost: gradient boosting

Never present RiskFormer results without baseline comparison.
A model is only valuable if it beats simple baselines.
"""
