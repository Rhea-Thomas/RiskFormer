"""
Walk-forward validation for time series.

Standard approach: train on [1, ..., t], predict [t+1, ..., t+k], then move window forward.

This prevents look-ahead bias and mimics real deployment:
  1. At each step, (re)train on all available history
  2. Forecast the next block
  3. Observe realization
  4. Move window forward and repeat

An optional `gap` (embargo) is left between train and test indices so that
targets that look a few days ahead (e.g. forward volatility) never overlap
the test block.

Important: Never train on validation or test data.
           Never use future information in feature engineering.
"""

import numpy as np


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time series.
    """

    def __init__(
        self,
        data,
        targets,
        initial_train_size=None,
        test_size=None,
        forecast_horizon=1,
        gap=0,
        expanding=True,
        max_splits=None,
    ):
        """
        Initialize walk-forward validator.

        Args:
            data: features (T x ...) — any array-like indexable on axis 0
            targets: targets (T x ...)
            initial_train_size: how much data to train on initially
            test_size: size of each test period
            forecast_horizon: how many steps ahead to predict
            gap: extra samples skipped between train end and test start
            expanding: True = train on all history; False = rolling window
                       of fixed size initial_train_size
            max_splits: stop after this many splits (None = until data ends)
        """
        if len(data) != len(targets):
            raise ValueError(f"data ({len(data)}) and targets ({len(targets)}) differ in length")
        self.data = data
        self.targets = targets
        self.initial_train_size = initial_train_size or int(0.7 * len(data))
        self.test_size = test_size or 30  # 30 days
        self.forecast_horizon = forecast_horizon
        self.gap = gap
        self.expanding = expanding
        self.max_splits = max_splits

    def split(self):
        """
        Generate train/test splits in forward order.

        Yields:
            (train_indices, test_indices) for each split
        """
        n = len(self.data)
        # Samples between train end and test start: the embargo plus the
        # extra steps a multi-step target looks ahead.
        offset = self.gap + self.forecast_horizon - 1
        k = 0
        while self.max_splits is None or k < self.max_splits:
            # Test blocks tile the data after the initial training period;
            # each block's training set ends `offset` samples before it.
            test_start = self.initial_train_size + offset + k * self.test_size
            if test_start >= n:
                break
            test_end = min(test_start + self.test_size, n)
            train_end = test_start - offset  # exclusive
            train_start = 0 if self.expanding else max(0, train_end - self.initial_train_size)
            yield np.arange(train_start, train_end), np.arange(test_start, test_end)
            k += 1

    def validate(self, model_fn, metric_fn):
        """
        Run walk-forward validation.

        Args:
            model_fn: function that trains and returns a fitted model
                     signature: model_fn(X_train, y_train) -> object with .predict(X)
            metric_fn: function to compute metrics on predictions
                      signature: metric_fn(y_true, y_pred) -> dict

        Returns:
            list of metric dicts, one per walk-forward step (with split info)
        """
        results = []
        for step, (tr, te) in enumerate(self.split()):
            model = model_fn(self.data[tr], self.targets[tr])
            y_pred = model.predict(self.data[te])
            metrics = dict(metric_fn(self.targets[te], y_pred))
            metrics.update(
                step=step,
                train_start=int(tr[0]),
                train_end=int(tr[-1]),
                test_start=int(te[0]),
                test_end=int(te[-1]),
            )
            results.append(metrics)
        return results
