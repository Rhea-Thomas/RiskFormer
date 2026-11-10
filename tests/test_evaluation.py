"""
Tests for evaluation metrics and walk-forward validation.

Focus:
  - Metric computation correctness
  - Walk-forward temporal order preservation
  - No leakage in walk-forward splits
  - Calibration metrics
"""

import numpy as np
import pytest

from evaluation.metrics import (
    calibration,
    correlation,
    directional_accuracy,
    forecast_metrics,
    mean_absolute_error,
    pinball_loss,
    qlike,
    quantile_coverage,
    rank_correlation,
    root_mean_squared_error,
    var_breach_rate,
)
from evaluation.walk_forward import WalkForwardValidator


def test_directional_accuracy():
    """Verify directional accuracy metric."""
    y_true = np.array([1.0, -1.0, 2.0, -0.5, 0.0])
    y_pred = np.array([0.5, 0.5, 1.0, -1.0, 3.0])  # last one excluded (true = 0)
    assert directional_accuracy(y_true, y_pred) == pytest.approx(3 / 4)


def test_basic_errors_and_nan_handling():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    p = np.array([1.0, 3.0, 9.0, 2.0])
    assert mean_absolute_error(y, p) == pytest.approx((0 + 1 + 2) / 3)
    assert root_mean_squared_error(y, p) == pytest.approx(np.sqrt((0 + 1 + 4) / 3))


def test_correlations():
    x = np.arange(50, dtype=float)
    assert correlation(x, 2 * x + 1) == pytest.approx(1.0)
    assert rank_correlation(x, x**3) == pytest.approx(1.0)
    assert correlation(x, np.zeros_like(x)) == 0.0


def test_walk_forward_no_leakage():
    """Verify walk-forward splits don't leak future data."""
    X = np.arange(100)
    wf = WalkForwardValidator(X, X, initial_train_size=50, test_size=10, gap=3)
    splits = list(wf.split())
    assert len(splits) >= 4
    prev_test_end = None
    for tr, te in splits:
        assert tr.max() + 3 < te.min()  # embargo respected
        assert np.all(np.diff(te) == 1)
        assert tr[0] == 0  # expanding window
        if prev_test_end is not None:
            assert te.min() == prev_test_end + 1  # test blocks tile without gaps/overlap
        prev_test_end = te.max()
    assert splits[0][1][0] == 53
    assert prev_test_end == 99


def test_walk_forward_rolling_and_max_splits():
    X = np.arange(100)
    wf = WalkForwardValidator(X, X, initial_train_size=40, test_size=20, expanding=False, max_splits=2)
    splits = list(wf.split())
    assert len(splits) == 2
    assert all(len(tr) == 40 for tr, _ in splits)


def test_walk_forward_validate_uses_only_past():
    X = np.arange(60, dtype=float)
    y = X * 2

    class LastValue:
        def __init__(self, Xtr, ytr):
            self.seen_max = Xtr.max()

        def predict(self, Xte):
            assert self.seen_max < Xte.min()
            return np.full(len(Xte), self.seen_max)

    res = WalkForwardValidator(X, y, initial_train_size=30, test_size=10).validate(
        LastValue, lambda t, p: {"mae": mean_absolute_error(t, p)}
    )
    assert [r["test_start"] for r in res] == [30, 40, 50]


def test_quantile_coverage():
    """Verify quantile coverage metric."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20000)
    cov = quantile_coverage(y, np.full_like(y, -1.6449), np.full_like(y, 1.6449))
    assert cov == pytest.approx(0.90, abs=0.01)
    assert var_breach_rate(y, np.full_like(y, -1.6449)) == pytest.approx(0.05, abs=0.006)


def test_calibration_of_true_model():
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.5, 3.0, 20000)
    y = rng.standard_normal(20000) * sigma
    c = calibration(y, np.zeros_like(y), sigma)
    assert c["z_std"] == pytest.approx(1.0, abs=0.02)
    assert c["cover_1s"] == pytest.approx(0.683, abs=0.01)
    # Overconfident model: σ halved -> z-std ≈ 2
    assert calibration(y, np.zeros_like(y), sigma / 2)["z_std"] == pytest.approx(2.0, abs=0.05)


def test_qlike_and_pinball():
    v = np.array([1.0, 2.0, 0.5])
    assert qlike(v, v) == pytest.approx(0.0, abs=1e-6)
    assert qlike(v, v * 2) > 0 and qlike(v, v / 2) > 0
    # Under-forecasting volatility is penalized more than over-forecasting
    assert qlike(v, v / 2) > qlike(v, v * 2)
    assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)


def test_forecast_metrics_keys():
    rng = np.random.default_rng(0)
    y, p = rng.standard_normal(100), rng.standard_normal(100)
    m = forecast_metrics(y, p, vol_pred=np.ones(100), vol_true=np.abs(y) + 0.1)
    for k in ["mae", "rmse", "directional_accuracy", "correlation", "vol_rmse", "vol_qlike"]:
        assert k in m and np.isfinite(m[k])
