"""
Evaluation metrics for financial forecasting.

Metrics:
  - MAE (mean absolute error)
  - RMSE (root mean squared error)
  - Directional accuracy (% of correct signs)
  - Correlation with realized values
  - Quantile coverage (% within predicted intervals)
  - Volatility error
  - Calibration (predicted std vs actual error)
"""

import numpy as np
import torch


def mean_absolute_error(y_true, y_pred):
    """Compute MAE."""
    pass


def root_mean_squared_error(y_true, y_pred):
    """Compute RMSE."""
    pass


def directional_accuracy(y_true, y_pred):
    """
    Directional accuracy: % of times the sign is correct.

    Args:
        y_true: actual returns
        y_pred: predicted returns

    Returns:
        accuracy (0.0 to 1.0)
    """
    pass


def correlation(y_true, y_pred):
    """Compute Pearson correlation."""
    pass


def quantile_coverage(y_true, q_lower, q_upper):
    """
    Check if actual values fall within predicted quantile interval.

    Args:
        y_true: actual values
        q_lower: lower quantile (e.g., 0.05)
        q_upper: upper quantile (e.g., 0.95)

    Returns:
        coverage rate (should be ~0.90 for 0.05-0.95 interval)
    """
    pass


def volatility_error(vol_true, vol_pred):
    """
    Error in volatility forecast.

    Args:
        vol_true: realized volatility
        vol_pred: predicted volatility

    Returns:
        RMSE of volatility
    """
    pass


def forecast_metrics(y_true, y_pred, vol_pred=None, vol_true=None):
    """
    Compute comprehensive forecasting metrics.

    Args:
        y_true: actual returns
        y_pred: predicted returns
        vol_pred: predicted volatility (optional)
        vol_true: realized volatility (optional)

    Returns:
        dict with metrics
    """
    pass
