"""
Main evaluation script for RiskFormer.

Usage:
    python evaluate.py --checkpoint checkpoints/riskformer.pt --config configs/base.yaml

Workflow:
    1. Load trained model and configuration
    2. Rebuild data (normalization statistics from the training period only)
    3. Walk-forward over the test period: refit RiskFormer and every baseline
       on all data before each block, forecast the block, move forward
    4. Compute forecasting, volatility and calibration metrics
    5. Compare against baselines
    6. Backtest forecast-driven strategies with transaction costs
    7. Write results/metrics.json, results/summary.md and figures
"""

import argparse
import copy
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from backtesting.engine import backtest_forecasts, performance_metrics, run_backtest
from baselines.models import default_baselines
from data.pipeline import build_splits, concat_splits, load_panel_from_config
from evaluation.metrics import (
    calibration,
    forecast_metrics,
    gaussian_nll,
    pinball_loss,
    quantile_coverage,
    var_breach_rate,
)
from evaluation.walk_forward import WalkForwardValidator
from models.riskformer import RiskFormer
from training.fit import fit_riskformer, predict, set_seed

logger = logging.getLogger("riskformer.evaluate")
SQRT252 = math.sqrt(252)


# ------------------------------------------------------------------ scoring
def score(pred, split, quantiles):
    ret, vol = split.Y[..., 0], split.Y[..., 1]
    m = forecast_metrics(ret, pred["mu"], vol_pred=pred["sigma"], vol_true=vol)
    m["nll"] = gaussian_nll(ret, pred["mu"], pred["sigma"])
    m.update(calibration(ret, pred["mu"], pred["sigma"]))
    q = pred["quantiles"]
    m["coverage_90"] = quantile_coverage(ret, q[..., 0], q[..., -1])
    m["var95_breach"] = var_breach_rate(ret, q[..., 0])
    m["pinball"] = float(np.mean([pinball_loss(ret, q[..., i], t) for i, t in enumerate(quantiles)]))
    return m


def concat_preds(preds):
    return {k: np.concatenate([p[k] for p in preds]) for k in preds[0]}


# ------------------------------------------------------------ walk-forward
class _Fitted:
    def __init__(self, fn):
        self.predict = fn


def walk_forward(full, n_history, block, gap, steps, model_fn):
    """Run WalkForwardValidator; return pooled predictions, the pooled test split, per-block results."""
    preds = []

    def metric_fn(y_true, pred):
        preds.append(pred)
        return {"n": int(len(y_true))}

    wf = WalkForwardValidator(
        full, full.Y, initial_train_size=n_history - gap, test_size=block, gap=gap, max_splits=steps
    )
    blocks = wf.validate(model_fn, metric_fn)
    test_idx = np.arange(blocks[0]["test_start"], blocks[-1]["test_end"] + 1)
    return concat_preds(preds), full.subset(test_idx), blocks, preds


# ----------------------------------------------------------------- figures
def make_figures(results_dir, test, forecasts, curves, blocks, assets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = test.dates
    a = 0  # first asset (SPY)
    rf, garch = forecasts["RiskFormer"], forecasts["zero+garch"]

    # 1. Volatility forecasts vs realized
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(d, test.Y[:, a, 1] * SQRT252, color="0.65", lw=1, label="realized (fwd 5-day)")
    ax.plot(d, garch["sigma"][:, a] * SQRT252, lw=1.3, label="GARCH(1,1)")
    ax.plot(d, rf["sigma"][:, a] * SQRT252, lw=1.6, label="RiskFormer")
    ax.set_ylabel("annualized vol (%)")
    ax.set_title(f"{assets[a]}: out-of-sample volatility forecasts (walk-forward)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "vol_forecast.png", dpi=130)
    plt.close(fig)

    # 2. Return forecast intervals
    fig, ax = plt.subplots(figsize=(11, 3.6))
    q = rf["quantiles"][:, a]
    ax.fill_between(d, q[:, 0], q[:, -1], alpha=0.25, label="RiskFormer 5–95% interval")
    ret = test.Y[:, a, 0]
    outside = (ret < q[:, 0]) | (ret > q[:, -1])
    ax.scatter(d[~outside], ret[~outside], s=5, color="0.3", label="realized return")
    ax.scatter(d[outside], ret[outside], s=9, color="crimson", label="outside interval")
    ax.set_ylabel("daily return (%)")
    ax.set_title(f"{assets[a]}: predictive intervals — coverage {np.mean(~outside):.1%} (target 90%)")
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "return_intervals.png", dpi=130)
    plt.close(fig)

    # 3. Calibration (reliability) curve for Gaussian central intervals
    levels = np.linspace(0.1, 0.98, 23)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name in ["RiskFormer", "zero+garch", "zero+ewma", "zero+rolling20"]:
        f = forecasts[name]
        z = np.abs((test.Y[..., 0] - f["mu"]) / f["sigma"]).ravel()
        emp = [np.mean(z <= norm.ppf(0.5 + l / 2)) for l in levels]
        ax.plot(levels, emp, marker=".", label=name)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration of predictive σ")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration.png", dpi=130)
    plt.close(fig)

    # 4. Equity curves
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for label, bt in curves.items():
        ax.plot(bt.index, 100 * np.cumprod(1 + bt["net"].to_numpy()), label=label)
    ax.set_ylabel("growth of 100 (net of costs)")
    ax.set_title("Out-of-sample backtests (10 bps per unit turnover)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "equity_curves.png", dpi=130)
    plt.close(fig)

    # 5. Per-block QLIKE
    names = ["RiskFormer", "zero+garch", "zero+har", "zero+ewma"]
    x = np.arange(len(blocks["RiskFormer"]))
    fig, ax = plt.subplots(figsize=(11, 3.4))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(x + i * w, [b["vol_qlike"] for b in blocks[n]], w, label=n)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([b["label"] for b in blocks["RiskFormer"]], rotation=0, fontsize=8)
    ax.set_ylabel("QLIKE (lower = better)")
    ax.set_title("Volatility forecast loss per walk-forward block")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "walk_forward_qlike.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- summary
def _fmt(v, pct=False, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v * 100:.1f}%" if pct else f"{v:.{digits}f}"


def write_summary(path, results, period):
    fm, bt = results["forecast_metrics"], results["backtests"]
    best = lambda key, lo=True: (min if lo else max)(fm, key=lambda n: fm[n][key])  # noqa: E731
    L = [f"# RiskFormer — out-of-sample results ({period})", ""]
    L += ["Walk-forward: every model is refit on all data before each quarter, then forecasts that quarter.", ""]

    L += ["## Volatility & probabilistic forecasts", "",
          "| Model | Vol QLIKE ↓ | Vol RMSE ↓ | Vol corr ↑ | NLL ↓ | z-std (→1) | 90% cover | VaR95 breach (→5%) |",
          "|---|---|---|---|---|---|---|---|"]
    for n, m in fm.items():
        tag = lambda k, lo=True: "**" if best(k, lo) == n else ""  # noqa: E731
        L.append(
            f"| {n} | {tag('vol_qlike')}{_fmt(m['vol_qlike'])}{tag('vol_qlike')} | "
            f"{tag('vol_rmse')}{_fmt(m['vol_rmse'])}{tag('vol_rmse')} | "
            f"{tag('vol_correlation', False)}{_fmt(m['vol_correlation'])}{tag('vol_correlation', False)} | "
            f"{tag('nll')}{_fmt(m['nll'])}{tag('nll')} | {_fmt(m['z_std'], digits=2)} | "
            f"{_fmt(m['coverage_90'], True)} | {_fmt(m['var95_breach'], True)} |"
        )

    L += ["", "## Return forecasts", "",
          "| Model | MAE ↓ | RMSE ↓ | Directional acc. | Corr | Rank IC |", "|---|---|---|---|---|---|"]
    for n, m in fm.items():
        L.append(f"| {n} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {_fmt(m['directional_accuracy'], True)} | "
                 f"{_fmt(m['correlation'])} | {_fmt(m['rank_ic'])} |")

    L += ["", "## Backtests (net of 10 bps costs, 10% vol target)", "",
          "| Forecaster | Strategy | Sharpe | CAGR | Ann. vol | Max DD | Turnover/day |", "|---|---|---|---|---|---|---|"]
    for n, strategies in bt.items():
        for s, m in strategies.items():
            if not m.get("ann_vol"):  # zero-mean forecasters never trade a μ signal
                continue
            L.append(f"| {n} | {s} | {_fmt(m['sharpe'], digits=2)} | {_fmt(m['cagr'], True)} | "
                     f"{_fmt(m['ann_vol'], True)} | {_fmt(m['max_drawdown'], True)} | {_fmt(m.get('avg_turnover', 0), digits=2)} |")
    L.append("")
    Path(path).write_text("\n".join(L))


# -------------------------------------------------------------------- main
def main(checkpoint_path, config_path, results_dir="results"):
    """
    Main evaluation entry point.

    Args:
        checkpoint_path: path to trained model checkpoint
        config_path: path to configuration file
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ev, bcfg = config["evaluation"], config["backtesting"]
    quantiles = config["heads"]["quantiles"]
    device = config["system"]["device"]

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_state = ckpt["model_state"]

    panel = load_panel_from_config(config)
    train, val, test, _ = build_splits(config, panel)
    assets = panel.assets
    full = concat_splits(train, val, test)
    n_history = len(train) + len(val)
    gap = config["data"].get("vol_horizon", 5) - 1
    steps = ev["walk_forward_steps"]
    block = math.ceil(len(test) / steps)
    logger.info(f"Walk-forward: {steps} blocks of {block} days over {test.dates[0].date()}→{test.dates[-1].date()}")

    forecasts, blocks_out = {}, {}

    # --- RiskFormer, static (trained once on 2015–2021) ---
    model = RiskFormer.from_config(config, n_features=train.X.shape[3], n_assets=train.X.shape[2])
    model.load_state_dict(base_state)
    static_pred = predict(model, test, device)

    # --- RiskFormer, walk-forward fine-tuning from the trained checkpoint ---
    def rf_fn(tr_split, _y):
        set_seed(config["system"]["seed"])
        m, _ = fit_riskformer(config, tr_split, None, max_epochs=ev["walk_forward_epochs"],
                              init_state=copy.deepcopy(base_state), lr_scale=0.3, log_every=0)
        logger.info(f"  refit on {len(tr_split)} samples through {tr_split.dates[-1].date()}")
        return _Fitted(lambda s: predict(m, s, device))

    rf_pred, test_wf, rf_blocks, rf_block_preds = walk_forward(full, n_history, block, gap, steps, rf_fn)
    forecasts["RiskFormer"] = rf_pred
    blocks_raw = {"RiskFormer": (rf_blocks, rf_block_preds)}
    # Static model scored on the same days as the walk-forward
    offset = np.searchsorted(test.dates, test_wf.dates[0])
    forecasts["RiskFormer (static)"] = {k: v[offset: offset + len(test_wf)] for k, v in static_pred.items()}

    # --- Baselines, same walk-forward protocol ---
    for bl in default_baselines():
        def bl_fn(tr_split, _y, bl=bl):
            return bl.fit(tr_split)
        pred, _, b, bp = walk_forward(full, n_history, block, gap, steps, bl_fn)
        forecasts[bl.name] = pred
        blocks_raw[bl.name] = (b, bp)
        logger.info(f"  baseline {bl.name} done")

    # --- Metrics ---
    fm = {n: score(p, test_wf, quantiles) for n, p in forecasts.items()}
    for n, (b, bp) in blocks_raw.items():
        blocks_out[n] = []
        for info, p in zip(b, bp):
            seg = full.subset(np.arange(info["test_start"], info["test_end"] + 1))
            s = score(p, seg, quantiles)
            s["label"] = f"{seg.dates[0]:%Y-%m}"
            s.update({k: info[k] for k in ("test_start", "test_end")})
            blocks_out[n].append(s)

    # --- Backtests ---
    realized = test_wf.Y[..., 0]
    bh = run_backtest(np.full_like(realized, 1.0 / len(assets)), realized, test_wf.dates,
                      bcfg["transaction_cost"], bcfg["initial_capital"])
    backtests = {"Benchmark": {"buy_and_hold": {**performance_metrics(bh["net"]),
                                                  "avg_turnover": float(bh["turnover"].mean())}}}
    curves = {"Buy & hold (equal weight)": bh}
    for n, p in forecasts.items():
        m, c = backtest_forecasts(p["mu"], p["sigma"], realized, test_wf.dates, assets, bcfg)
        backtests[n] = m
        if n == "RiskFormer":
            curves["RiskFormer vol-target"] = c["vol_target"]
            curves["RiskFormer μ/σ² signal"] = c["signal"]
        if n == "zero+garch":
            curves["GARCH vol-target"] = c["vol_target"]

    period = f"{test_wf.dates[0]:%b %Y} – {test_wf.dates[-1]:%b %Y}, {', '.join(assets)}"
    results = {
        "period": period,
        "n_test_days": int(len(test_wf)),
        "walk_forward": {"blocks": steps, "block_days": block, "embargo_days": gap},
        "forecast_metrics": fm,
        "walk_forward_blocks": blocks_out,
        "backtests": backtests,
    }
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    write_summary(out / "summary.md", results, period)
    make_figures(out, test_wf, forecasts, curves, blocks_out, assets)
    logger.info(f"Wrote {out / 'metrics.json'}, {out / 'summary.md'} and figures")
    print((out / "summary.md").read_text())
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for noisy in ("models", "data", "training"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/riskformer.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to configuration file")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.results)
