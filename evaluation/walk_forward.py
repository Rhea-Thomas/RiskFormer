"""
Walk-forward validation for time series.

Standard approach: train on [1, ..., t], predict [t+1], then move window forward.

This prevents look-ahead bias and mimics real deployment:
  1. At each time step, retrain on all available history
  2. Generate h-step-ahead forecast
  3. Observe realization
  4. Move window forward and repeat

Important: Never train on validation or test data.
           Never use future information in feature engineering.
"""


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
    ):
        """
        Initialize walk-forward validator.

        Args:
            data: features (T x features)
            targets: targets (T x targets)
            initial_train_size: how much data to train on initially
            test_size: size of each test period
            forecast_horizon: how many steps ahead to predict
        """
        self.data = data
        self.targets = targets
        self.initial_train_size = initial_train_size or int(0.7 * len(data))
        self.test_size = test_size or 30  # 30 days
        self.forecast_horizon = forecast_horizon

    def split(self):
        """
        Generate train/test splits in forward order.

        Yields:
            (train_indices, test_indices) for each split
        """
        pass

    def validate(self, model_fn, metric_fn):
        """
        Run walk-forward validation.

        Args:
            model_fn: function that trains and returns a fitted model
                     signature: model_fn(X_train, y_train)
            metric_fn: function to compute metrics on predictions
                      signature: metric_fn(y_true, y_pred)

        Returns:
            list of metric dicts, one per walk-forward step
        """
        pass
