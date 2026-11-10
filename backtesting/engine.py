"""
Backtesting engine: forecasts -> positions -> P&L -> risk metrics.

Timing: the forecast for day r is made with data up to the close of r-1, the
position is set at that close, and it earns day r's return. Turnover is
charged at `transaction_cost` per unit of weight traded.

Strategies (all daily, long/short per asset, then scaled to a portfolio
volatility target using the *forecast* σ — no realized information):

  buy_and_hold      equal weight, never rebalanced to vol (benchmark)
  vol_target        long-only, w_a ∝ 1/σ̂_a          (uses only the vol forecast)
  signal            w_a ∝ μ̂_a / σ̂_a²               (mean-variance / Kelly-style)
  sign              w_a ∝ sign(μ̂_a) / σ̂_a          (direction only)

Simple returns are used for P&L (log returns are converted back).
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _to_simple(log_pct):
    return np.expm1(np.asarray(log_pct, dtype=np.float64) / 100.0)


def scale_to_target(raw_w, sigma_pct, target_vol_annual=10.0, max_leverage=1.5):
    """
    Scale raw weights so the forecast portfolio vol hits the target.

    Uses a diagonal covariance from the forecast σ (correlations ignored —
    conservative for long-short, slightly aggressive for long-only).
    """
    sig = np.asarray(sigma_pct, dtype=np.float64) / 100.0
    port_vol = np.sqrt(np.sum((raw_w * sig) ** 2, axis=1)) * np.sqrt(TRADING_DAYS)
    scale = (target_vol_annual / 100.0) / np.maximum(port_vol, 1e-8)
    w = raw_w * scale[:, None]
    gross = np.abs(w).sum(axis=1, keepdims=True)
    w = np.where(gross > max_leverage, w * max_leverage / np.maximum(gross, 1e-12), w)
    return w


def make_weights(strategy, mu, sigma, target_vol=10.0, max_leverage=1.5):
    N, A = sigma.shape
    if strategy == "buy_and_hold":
        return np.full((N, A), 1.0 / A)
    if strategy == "vol_target":
        raw = 1.0 / sigma
    elif strategy == "signal":
        raw = mu / sigma**2
    elif strategy == "sign":
        raw = np.sign(mu) / sigma
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return scale_to_target(raw, sigma, target_vol, max_leverage)


def run_backtest(weights, realized_log_pct, dates, transaction_cost=0.001, initial_capital=100_000.0):
    """
    Simulate a daily-rebalanced portfolio.

    Args:
        weights: (N, A) target weights held over day r
        realized_log_pct: (N, A) realized percent log returns on day r
        dates: (N,) dates of day r
        transaction_cost: cost per unit of turnover (0.001 = 10 bps)

    Returns:
        pd.DataFrame indexed by date with gross/net returns, turnover, equity,
        and per-asset P&L contributions.
    """
    r = _to_simple(realized_log_pct)
    w = np.asarray(weights, dtype=np.float64)
    prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    # Drift: yesterday's weights move with yesterday's returns before rebalancing
    prev_r = np.vstack([np.zeros((1, r.shape[1])), r[:-1]])
    drifted = prev * (1 + prev_r)
    drifted = drifted / np.maximum(1 + (prev * prev_r).sum(axis=1, keepdims=True), 1e-12)
    turnover = np.abs(w - drifted).sum(axis=1)
    contrib = w * r
    gross = contrib.sum(axis=1)
    net = gross - transaction_cost * turnover

    df = pd.DataFrame(
        {"gross": gross, "net": net, "turnover": turnover}, index=pd.DatetimeIndex(dates)
    )
    df["equity"] = initial_capital * np.cumprod(1 + net)
    for a in range(w.shape[1]):
        df[f"contrib_{a}"] = contrib[:, a]
    return df


def performance_metrics(returns, periods=TRADING_DAYS):
    """Annualized return/vol, Sharpe, Sortino, max drawdown, Calmar, hit rate."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return {}
    equity = np.cumprod(1 + r)
    years = len(r) / periods
    cagr = equity[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(periods)
    downside = r[r < 0]
    dvol = np.sqrt(np.mean(downside**2)) * np.sqrt(periods) if len(downside) else np.nan
    peak = np.maximum.accumulate(equity)
    mdd = float(np.min(equity / peak - 1))
    mean_ann = r.mean() * periods
    return {
        "total_return": float(equity[-1] - 1),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(mean_ann / vol) if vol > 0 else np.nan,
        "sortino": float(mean_ann / dvol) if dvol and dvol > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        "hit_rate": float(np.mean(r > 0)),
    }


def risk_attribution(bt: pd.DataFrame, asset_names):
    """Share of total P&L and of variance contributed by each asset."""
    cols = [c for c in bt.columns if c.startswith("contrib_")]
    C = bt[cols].to_numpy()
    total_pnl = C.sum()
    port = C.sum(axis=1)
    var = port.var(ddof=1)
    out = {}
    for i, name in enumerate(asset_names):
        cov = np.cov(C[:, i], port, ddof=1)[0, 1]
        out[name] = {
            "pnl_share": float(C[:, i].sum() / total_pnl) if total_pnl != 0 else np.nan,
            "risk_share": float(cov / var) if var > 0 else np.nan,
        }
    return out


def backtest_forecasts(mu, sigma, realized_log_pct, dates, asset_names, cfg, strategies=None):
    """Run every strategy for one forecaster. Returns ({strategy: metrics}, {strategy: bt_df})."""
    strategies = strategies or ["vol_target", "signal", "sign"]
    metrics, curves = {}, {}
    for s in strategies:
        w = make_weights(s, mu, sigma, cfg.get("target_vol", 10.0), cfg.get("max_leverage", 1.5))
        bt = run_backtest(w, realized_log_pct, dates, cfg.get("transaction_cost", 0.001),
                          cfg.get("initial_capital", 100_000.0))
        m = performance_metrics(bt["net"])
        m["avg_turnover"] = float(bt["turnover"].mean())
        m["gross_sharpe"] = performance_metrics(bt["gross"]).get("sharpe")
        m["attribution"] = risk_attribution(bt, asset_names)
        metrics[s], curves[s] = m, bt
    return metrics, curves
