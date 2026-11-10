"""
Tests for the end-to-end data pipeline, baselines, the full model and backtesting.

Uses synthetic OHLCV data (no network access).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from backtesting.engine import make_weights, performance_metrics, run_backtest
from baselines.models import GarchVol, ZeroMean, default_baselines, ewma_vol
from data.dataset import FinancialTimeSeriesDataset
from data.pipeline import build_panel, concat_splits, fit_normalizer, make_windows
from models.riskformer import RiskFormer


def synthetic_ohlcv(n_days=700, assets=("AAA", "BBB"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    frames = {}
    for a in assets:
        vol = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.05, n_days)) * 0.3)
        r = rng.standard_normal(n_days) * vol
        close = 100 * np.exp(np.cumsum(r))
        high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        frames[a] = pd.DataFrame(
            {"Open": close, "High": high, "Low": low, "Close": close,
             "Volume": rng.integers(1e6, 2e6, n_days).astype(float)},
            index=dates,
        )
    return pd.concat(frames, axis=1)


@pytest.fixture(scope="module")
def panel():
    return build_panel(synthetic_ohlcv())


@pytest.fixture(scope="module")
def splits(panel):
    d = panel.features.index
    train_end, val_end = d[400], d[550]
    norm = fit_normalizer(panel, train_end)
    tr = make_windows(panel, norm, d[0], train_end, lookback=30, embargo=4)
    va = make_windows(panel, norm, d[401], val_end, lookback=30, embargo=4)
    te = make_windows(panel, norm, d[551], d[-1], lookback=30)
    return tr, va, te


# ------------------------------------------------------------- pipeline
def test_panel_targets_are_forward_looking(panel):
    a = panel.assets[0]
    r = panel.returns_pct[a]
    fv = panel.targets[(a, "fwd_vol")]
    i = 100
    assert fv.iloc[i] == pytest.approx(r.iloc[i : i + 5].std(), rel=1e-9)
    assert panel.targets[(a, "ret")].iloc[i] == pytest.approx(r.iloc[i])
    assert not panel.features.isna().any().any()


def test_windows_never_include_target_day(panel, splits):
    tr, _, _ = splits
    a_idx, f_idx = 0, panel.feature_names.index("returns_log")
    dates = panel.features.index
    for k in [0, 50, len(tr) - 1]:
        r = dates.get_loc(tr.dates[k])
        # last row of the window is the day *before* the target
        assert np.allclose(tr.hist_returns[k, -1, a_idx], panel.returns_pct.iloc[r - 1, a_idx])
        assert tr.Y[k, a_idx, 0] == pytest.approx(panel.returns_pct.iloc[r, a_idx])
    assert tr.X.shape[1:] == (30, len(panel.assets), len(panel.feature_names))
    assert f_idx >= 0


def test_normalizer_uses_train_only(panel, splits):
    tr, _, te = splits
    # train features are ~standardized, test is not forced to be
    f = tr.X[:, -1].reshape(-1, tr.X.shape[-1])
    assert np.all(np.abs(f.mean(0)) < 0.5)


def test_embargo_and_chronology(splits):
    tr, va, te = splits
    assert tr.dates[-1] < va.dates[0] < va.dates[-1] < te.dates[0]
    full = concat_splits(tr, va, te)
    assert full.dates.is_monotonic_increasing
    assert len(full) == len(tr) + len(va) + len(te)


def test_pipeline_matches_original_dataset(panel):
    """Vectorized windows agree with Phase 4's FinancialTimeSeriesDataset."""
    norm = fit_normalizer(panel, panel.features.index[-1])
    split = make_windows(panel, norm, panel.features.index[0], panel.features.index[-1], lookback=20)
    feats = norm.transform(panel.features)
    ds = FinancialTimeSeriesDataset(feats, panel.targets.xs("ret", axis=1, level=1), lookback_window=20)
    X0, y0 = ds[0]
    assert np.allclose(split.X[0], np.clip(X0.numpy(), -10, 10), atol=1e-5)
    assert np.allclose(split.Y[0, :, 0], y0.numpy(), atol=1e-5)


# ------------------------------------------------------------- model
def test_riskformer_output_contract():
    torch.manual_seed(0)
    m = RiskFormer(n_features=6, n_assets=3, d_model=32, n_heads=4, n_layers=1)
    out = m(torch.randn(5, 20, 3, 6))
    assert out["mu"].shape == (5, 3)
    assert torch.all(out["sigma"] > 0)
    q = out["quantiles"]
    assert q.shape == (5, 3, 3)
    assert torch.all(q[..., 1:] >= q[..., :-1])  # no quantile crossing


def test_riskformer_is_causal_in_time():
    """Changing data after time t must not change the representation at t."""
    torch.manual_seed(0)
    m = RiskFormer(n_features=4, n_assets=2, d_model=32, n_heads=4, n_layers=2, dropout=0.0).eval()
    X = torch.randn(1, 15, 2, 4)
    X2 = X.clone()
    X2[:, 10:] += 5.0
    with torch.no_grad():
        h1 = m.backbone(m.embedder(X).permute(0, 2, 1, 3).reshape(2, 15, -1))
        h2 = m.backbone(m.embedder(X2).permute(0, 2, 1, 3).reshape(2, 15, -1))
    assert torch.allclose(h1[:, :10], h2[:, :10], atol=1e-5)
    assert not torch.allclose(h1[:, 10:], h2[:, 10:])


def test_embedding_parameters_are_trainable():
    m = RiskFormer(n_features=4, n_assets=2, d_model=16, n_heads=2, n_layers=1)
    names = {n for n, _ in m.named_parameters()}
    assert any(n.startswith("feature_proj") for n in names)
    assert any(n.startswith("asset_table") for n in names)


# ------------------------------------------------------------- baselines
def test_baselines_predict_valid_outputs(splits):
    tr, _, te = splits
    for bl in default_baselines():
        out = bl.fit(tr).predict(te)
        assert out["mu"].shape == te.Y.shape[:2], bl.name
        assert np.all(np.isfinite(out["mu"])), bl.name
        assert np.all(out["sigma"] > 0), bl.name
        assert np.all(np.diff(out["quantiles"], axis=-1) >= 0), bl.name


def test_ewma_weights_recent_more():
    hist = np.zeros((1, 50, 1))
    hist[0, -1, 0] = 10.0
    recent = ewma_vol(hist)[0, 0]
    hist2 = np.zeros((1, 50, 1))
    hist2[0, 0, 0] = 10.0
    assert recent > ewma_vol(hist2)[0, 0]


def test_garch_recovers_parameters():
    rng = np.random.default_rng(0)
    omega, alpha, beta = 0.05, 0.1, 0.85
    n = 4000
    r, h = np.zeros(n), np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t:
            h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()
    g = GarchVol().fit(r[:, None])
    o, a, b = g.params[0]
    assert a == pytest.approx(alpha, abs=0.04)
    assert b == pytest.approx(beta, abs=0.06)
    assert (a + b) == pytest.approx(alpha + beta, abs=0.03)


# ------------------------------------------------------------- backtest
def test_backtest_costs_and_returns():
    dates = pd.bdate_range("2020-01-01", periods=3)
    w = np.array([[1.0], [1.0], [0.0]])
    r_log = np.log1p(np.array([[0.01], [0.02], [0.03]])) * 100
    bt = run_backtest(w, r_log, dates, transaction_cost=0.001, initial_capital=100.0)
    # Day 1: buy 1.0 -> cost 0.001
    assert bt["net"].iloc[0] == pytest.approx(0.01 - 0.001)
    # Day 2: weight drifted to 1.0 already (single asset, fully invested) -> no cost
    assert bt["turnover"].iloc[1] == pytest.approx(0.0, abs=1e-12)
    # Day 3: sell everything
    assert bt["turnover"].iloc[2] == pytest.approx(1.0)
    assert bt["net"].iloc[2] == pytest.approx(-0.001)


def test_vol_target_scaling():
    sigma = np.full((10, 2), 1.0)  # 1% daily each
    w = make_weights("vol_target", np.zeros((10, 2)), sigma, target_vol=10.0, max_leverage=5)
    port_vol = np.sqrt(np.sum((w * 0.01) ** 2, axis=1)) * np.sqrt(252)
    assert np.allclose(port_vol, 0.10)
    w_capped = make_weights("vol_target", np.zeros((10, 2)), sigma / 100, target_vol=10.0, max_leverage=1.5)
    assert np.allclose(np.abs(w_capped).sum(1), 1.5)


def test_performance_metrics():
    r = np.array([0.01, -0.02, 0.015, 0.0, 0.005])
    m = performance_metrics(r)
    assert m["total_return"] == pytest.approx(np.prod(1 + r) - 1)
    assert m["max_drawdown"] == pytest.approx(-0.02, abs=1e-9)
    assert 0 < m["hit_rate"] < 1
