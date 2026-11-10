"""
End-to-end data pipeline: raw prices -> model-ready rolling windows.

Phases 1–4 provide the building blocks; this module wires them together
for training, evaluation and walk-forward validation.

Timeline convention (row r = trading day r):
  features[r]  use information available at the close of day r
  targets[r]   describe day r itself (return close r-1 -> r) and the
               next few days (forward realized volatility)

A sample whose target row is r uses the feature window [r-L, r-1], so the
model never sees the day it is predicting.

Targets per asset (percent units):
  ret       next-day log return × 100
  fwd_vol   std of daily log returns over days r .. r+vol_horizon-1, × 100
  up        1.0 if ret > 0 else 0.0

Because fwd_vol looks `vol_horizon - 1` days past r, the last rows of the
train and validation periods are purged (embargo) so no target overlaps the
next split.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.cleaning import clean_data
from data.features import engineer_features
from data.ingestion import load_or_download
from data.normalization import Normalizer

logger = logging.getLogger(__name__)

# Stationary features only: price levels (Close, MA) drift out of the training
# range after z-scoring, so they are expressed as ratios instead.
DEFAULT_FEATURES = [
    "returns_log",
    "volatility_5",
    "volatility_20",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "price_to_ma_20",
    "price_to_ma_60",
    "hl_range",
    "vol_change_5",
    "vol_change_20",
]
TARGETS = ["ret", "fwd_vol", "up"]


@dataclass
class Panel:
    """Aligned feature and target panels for all assets."""

    features: pd.DataFrame  # MultiIndex columns (asset, feature)
    targets: pd.DataFrame  # MultiIndex columns (asset, target)
    returns_pct: pd.DataFrame  # (date, asset) daily log returns × 100
    assets: List[str]
    feature_names: List[str]


@dataclass
class SplitData:
    """Rolling-window arrays for one chronological split."""

    X: np.ndarray  # (N, L, A, F) normalized features
    Y: np.ndarray  # (N, A, 3) targets [ret, fwd_vol, up]
    dates: pd.DatetimeIndex  # (N,) date of the target row
    hist_returns: np.ndarray  # (N, L, A) raw percent returns in the window
    meta: Dict = field(default_factory=dict)

    def __len__(self):
        return len(self.dates)

    def subset(self, idx) -> "SplitData":
        return SplitData(self.X[idx], self.Y[idx], self.dates[idx], self.hist_returns[idx], self.meta)

    __getitem__ = subset  # lets WalkForwardValidator index a split directly


def concat_splits(*splits: SplitData) -> SplitData:
    """Concatenate chronologically ordered splits into one sample sequence."""
    return SplitData(
        X=np.concatenate([s.X for s in splits]),
        Y=np.concatenate([s.Y for s in splits]),
        dates=pd.DatetimeIndex(np.concatenate([s.dates.values for s in splits])),
        hist_returns=np.concatenate([s.hist_returns for s in splits]),
        meta=splits[0].meta,
    )


def add_ratio_features(features: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Add scale-free price features that engineer_features doesn't provide."""
    out = features.copy()
    for asset in features.columns.get_level_values(0).unique():
        close = raw[(asset, "Close")]
        for w in (20, 60):
            out[(asset, f"price_to_ma_{w}")] = close / features[(asset, f"ma_{w}")] - 1.0
        out[(asset, "hl_range")] = np.log(raw[(asset, "High")] / raw[(asset, "Low")])
    return out


def build_panel(
    raw: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    vol_horizon: int = 5,
) -> Panel:
    """
    Raw OHLCV (MultiIndex columns asset, field) -> feature + target panels.

    Rows with any NaN (feature warm-up at the start, forward-vol at the end)
    are dropped so every remaining row is a valid sample target.
    """
    feature_names = feature_names or DEFAULT_FEATURES
    cleaned, _ = clean_data(raw)
    feats = add_ratio_features(engineer_features(cleaned), cleaned)
    assets = list(feats.columns.get_level_values(0).unique())

    feat_cols = pd.MultiIndex.from_tuples([(a, f) for a in assets for f in feature_names])
    missing = [c for c in feat_cols if c not in feats.columns]
    if missing:
        raise KeyError(f"Features not produced by the pipeline: {missing[:5]}")
    features = feats[feat_cols].replace([np.inf, -np.inf], np.nan)

    returns_pct = pd.DataFrame({a: 100.0 * feats[(a, "returns_log")] for a in assets})

    targets = {}
    for a in assets:
        r = returns_pct[a]
        targets[(a, "ret")] = r
        # std over rows r .. r+h-1  (rolling window ending at r+h-1, shifted back)
        targets[(a, "fwd_vol")] = r.rolling(vol_horizon).std().shift(-(vol_horizon - 1))
        targets[(a, "up")] = (r > 0).astype(float)
    targets = pd.DataFrame(targets)
    targets.columns = pd.MultiIndex.from_tuples(targets.columns)

    # Drop the feature warm-up period (rolling windows not yet full). Trailing
    # rows with NaN targets stay; make_windows skips them as sample targets.
    first_valid = features.notna().all(axis=1).idxmax()
    features = features.loc[first_valid:].ffill()  # interior gaps: causal fill
    targets, returns_pct = targets.loc[first_valid:], returns_pct.loc[first_valid:]

    logger.info(
        f"Panel: {len(features)} days, {len(assets)} assets, {len(feature_names)} features, "
        f"{features.index[0].date()} -> {features.index[-1].date()}"
    )
    return Panel(features, targets, returns_pct, assets, list(feature_names))


def fit_normalizer(panel: Panel, train_end: str, method: str = "zscore") -> Normalizer:
    """Fit normalization statistics on rows up to train_end only (no leakage)."""
    train_rows = panel.features.loc[: pd.Timestamp(train_end)]
    return Normalizer(method=method).fit(train_rows)


def make_windows(
    panel: Panel,
    normalizer: Normalizer,
    start: str,
    end: str,
    lookback: int,
    embargo: int = 0,
    clip: float = 10.0,
) -> SplitData:
    """
    Build rolling windows whose *target* date lies in [start, end].

    The look-back window may reach into earlier periods (that's past data and
    allowed). The last `embargo` target rows are dropped to purge overlap of
    forward-looking targets with the next split.
    """
    feats_norm = normalizer.transform(panel.features)
    A, F = len(panel.assets), len(panel.feature_names)
    feat_arr = feats_norm.to_numpy(dtype=np.float32).reshape(len(feats_norm), A, F)
    feat_arr = np.clip(np.nan_to_num(feat_arr), -clip, clip)

    targ = panel.targets
    targ_arr = np.stack(
        [targ.xs(t, axis=1, level=1)[panel.assets].to_numpy(dtype=np.float32) for t in TARGETS], -1
    )  # (T, A, 3)
    ret_arr = panel.returns_pct[panel.assets].to_numpy(dtype=np.float32)  # (T, A)

    dates = panel.features.index
    rows = np.where((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))[0]
    rows = rows[rows >= lookback]
    rows = rows[np.isfinite(targ_arr[rows]).all(axis=(1, 2))]
    if embargo > 0:
        rows = rows[:-embargo] if len(rows) > embargo else rows[:0]
    if len(rows) == 0:
        raise ValueError(f"No samples with target dates in [{start}, {end}]")

    win = np.lib.stride_tricks.sliding_window_view(np.arange(len(dates)), lookback)
    idx = win[rows - lookback]  # (N, L) rows r-L .. r-1
    return SplitData(
        X=feat_arr[idx],
        Y=targ_arr[rows],
        dates=dates[rows],
        hist_returns=ret_arr[idx],
        meta={"assets": panel.assets, "features": panel.feature_names, "lookback": lookback},
    )


def load_panel_from_config(config: dict, cache_dir: str = "data/cache") -> Panel:
    d = config["data"]
    # Extra history before train_start so rolling features (up to 60d) warm up
    start = (pd.Timestamp(d["train_start_date"]) - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    # yfinance end date is exclusive, plus room for the forward-vol target
    end = (pd.Timestamp(d["test_end_date"]) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    raw = load_or_download(d["assets"], start, end, cache_dir=cache_dir)
    return build_panel(raw, d.get("features") or None, d.get("vol_horizon", 5))


def build_splits(config: dict, panel: Panel):
    """Return (train, val, test, normalizer) using the dates in config['data']."""
    d = config["data"]
    L = d["lookback_window"]
    emb = d.get("vol_horizon", 5) - 1
    normalizer = fit_normalizer(panel, d["train_end_date"], config["normalization"]["method"])
    train = make_windows(panel, normalizer, d["train_start_date"], d["train_end_date"], L, embargo=emb)
    val = make_windows(panel, normalizer, d["val_start_date"], d["val_end_date"], L, embargo=emb)
    test = make_windows(panel, normalizer, d["test_start_date"], d["test_end_date"], L)
    logger.info(f"Splits: train={len(train)}, val={len(val)}, test={len(test)} samples")
    return train, val, test, normalizer


def to_loader(split: SplitData, batch_size: int, shuffle: bool = False):
    """
    Wrap a split in a DataLoader.

    Shuffling *across* windows is fine for training: every window only
    contains its own past, so order between samples carries no leakage.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.from_numpy(split.X), torch.from_numpy(split.Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
