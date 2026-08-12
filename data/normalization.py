"""
Leakage-safe normalization (standardization/scaling).

CRITICAL: Normalization parameters (mean, std) are fit on training data ONLY,
then applied to validation and test data.

This prevents information leakage where future statistics influence
historical predictions.

Pipeline:
  Train data → fit scaler (compute mean/std) → transform train
  Val data → apply scaler (using train mean/std) → transform val
  Test data → apply scaler (using train mean/std) → transform test
"""

class Normalizer:
    """
    Fits normalization parameters on training data.

    Stores mean/std for reproducible forward pass on new data.
    """

    def __init__(self, method="zscore", epsilon=1e-8):
        """
        Initialize normalizer.

        Args:
            method: "zscore" or "minmax"
            epsilon: small constant for numerical stability
        """
        self.method = method
        self.epsilon = epsilon
        self.params = {}  # Will store mean, std, min, max

    def fit(self, data):
        """
        Fit normalization parameters on training data.

        Args:
            data: np.ndarray or pd.DataFrame (shape: n_samples x n_features)

        Returns:
            self
        """
        pass

    def transform(self, data):
        """
        Apply learned normalization to new data.

        Args:
            data: np.ndarray or pd.DataFrame

        Returns:
            normalized np.ndarray or pd.DataFrame
        """
        pass

    def fit_transform(self, data):
        """Fit and transform in one call."""
        self.fit(data)
        return self.transform(data)


def normalize_dataset(train_data, val_data, test_data, method="zscore"):
    """
    Normalize dataset using training statistics.

    Args:
        train_data: pd.DataFrame (training set)
        val_data: pd.DataFrame (validation set)
        test_data: pd.DataFrame (test set)
        method: normalization method

    Returns:
        tuple of (normalized_train, normalized_val, normalized_test, normalizer)
    """
    pass
