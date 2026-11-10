"""
Baseline models for comparison.

Baselines:
  - Naive: zero-mean random walk, yesterday's return
  - Historical mean: average historical return
  - Moving average: simple moving average
  - Linear regression: Ridge on features
  - Gradient boosting: sklearn HistGradientBoosting (XGBoost-style)
  - Volatility: rolling std, EWMA (RiskMetrics), GARCH(1,1), HAR

Never present RiskFormer results without baseline comparison.
A model is only valuable if it beats simple baselines.
"""

from baselines.models import (  # noqa: F401
    GradientBoostingReturn,
    HistoricalMean,
    MovingAverage,
    Persistence,
    RidgeReturn,
    ZeroMean,
    default_baselines,
)
