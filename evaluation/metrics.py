"""
Evaluation metrics for financial forecasting.

Metrics:
  - MAE (mean absolute error)
  - RMSE (root mean squared error)
  - Directional accuracy (% of correct signs)
  - Correlation with realized values (Pearson and rank / IC)
  - Quantile coverage (% within predicted intervals)
  - Pinball loss per quantile
  - Volatility error (RMSE) and QLIKE
  - Calibration (predicted std vs actual error)

All functions accept numpy arrays or torch tensors of any matching shape;
inputs are flattened, and NaNs are ignored pairwise.
"""

import numpy as np
import torch


def _np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _paired(*arrays):
    flat = [_np(a).ravel() for a in arrays]
    mask = np.ones_like(flat[0], dtype=bool)
    for a in flat:
        mask &= np.isfinite(a)
    return [a[mask] for a in flat]


def mean_absolute_error(y_true, y_pred):
    """Compute MAE."""
    t, p = _paired(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def root_mean_squared_error(y_true, y_pred):
    """Compute RMSE."""
    t, p = _paired(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def directional_accuracy(y_true, y_pred):
    """
    Directional accuracy: % of times the sign is correct.

    Zero true returns are excluded (no direction to get right).

    Args:
        y_true: actual returns
        y_pred: predicted returns

    Returns:
        accuracy (0.0 to 1.0)
    """
    t, p = _paired(y_true, y_pred)
    mask = t != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(t[mask]) == np.sign(p[mask])))


def correlation(y_true, y_pred):
    """Compute Pearson correlation."""
    t, p = _paired(y_true, y_pred)
    if len(t) < 2 or t.std() == 0 or p.std() == 0:
        return 0.0
    return float(np.corrcoef(t, p)[0, 1])


def rank_correlation(y_true, y_pred):
    """Spearman rank correlation (a.k.a. information coefficient)."""
    t, p = _paired(y_true, y_pred)
    rt = np.argsort(np.argsort(t)).astype(float)
    rp = np.argsort(np.argsort(p)).astype(float)
    return correlation(rt, rp)


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
    t, lo, hi = _paired(y_true, q_lower, q_upper)
    return float(np.mean((t >= lo) & (t <= hi)))


def pinball_loss(y_true, q_pred, tau):
    """Mean pinball loss for a single quantile level tau."""
    t, q = _paired(y_true, q_pred)
    e = t - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def volatility_error(vol_true, vol_pred):
    """
    Error in volatility forecast.

    Args:
        vol_true: realized volatility
        vol_pred: predicted volatility

    Returns:
        RMSE of volatility
    """
    return root_mean_squared_error(vol_true, vol_pred)


def qlike(vol_true, vol_pred, eps=1e-8):
    """
    QLIKE loss on variances (Patton, 2011) — robust to noise in the realized
    volatility proxy and the standard way to rank volatility forecasts.

      QLIKE = σ²_true/σ²_pred − log(σ²_true/σ²_pred) − 1   (0 = perfect)
    """
    t, p = _paired(vol_true, vol_pred)
    ratio = (t**2 + eps) / (p**2 + eps)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def gaussian_nll(y_true, mu, sigma):
    """Mean Gaussian negative log-likelihood (including the 0.5·log 2π constant)."""
    t, m, s = _paired(y_true, mu, sigma)
    s = np.maximum(s, 1e-8)
    return float(np.mean(0.5 * np.log(2 * np.pi * s**2) + 0.5 * ((t - m) / s) ** 2))


def calibration(y_true, mu, sigma):
    """
    Probabilistic calibration of a Gaussian forecast.

    Returns:
        dict with
          z_std:     std of standardized errors (1.0 = well calibrated;
                     > 1 overconfident, < 1 underconfident)
          cover_1s:  share of |z| <= 1   (Gaussian: 0.683)
          cover_2s:  share of |z| <= 1.96 (Gaussian: 0.950)
    """
    t, m, s = _paired(y_true, mu, sigma)
    z = (t - m) / np.maximum(s, 1e-8)
    return {
        "z_std": float(np.std(z)),
        "cover_1s": float(np.mean(np.abs(z) <= 1.0)),
        "cover_2s": float(np.mean(np.abs(z) <= 1.96)),
    }


def var_breach_rate(y_true, var_quantile):
    """Share of returns below the predicted lower quantile (VaR). Target = tau."""
    t, q = _paired(y_true, var_quantile)
    return float(np.mean(t < q))


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
    out = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "correlation": correlation(y_true, y_pred),
        "rank_ic": rank_correlation(y_true, y_pred),
    }
    if vol_pred is not None and vol_true is not None:
        out["vol_rmse"] = volatility_error(vol_true, vol_pred)
        out["vol_mae"] = mean_absolute_error(vol_true, vol_pred)
        out["vol_qlike"] = qlike(vol_true, vol_pred)
        out["vol_correlation"] = correlation(vol_true, vol_pred)
    return out
