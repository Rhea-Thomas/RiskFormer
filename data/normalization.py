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

Key principle:
  Once normalizer is fit on train, it is IMMUTABLE.
  Val/test never inform the normalizer.
  In production, use the saved train-era normalizer on new data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Union

logger = logging.getLogger(__name__)


class Normalizer:
    """
    Fits normalization parameters on training data only.

    Stores mean, std (or min, max) for reproducible forward pass on new data.
    After fitting, the normalizer is immutable—val/test cannot change it.

    Attributes:
        method: "zscore" or "minmax"
        epsilon: small constant for numerical stability
        params: dict storing fit statistics {feature: {mean, std} or {min, max}}
        is_fit: whether normalizer has been fit
    """

    def __init__(self, method: str = "zscore", epsilon: float = 1e-8):
        """
        Initialize normalizer.

        Args:
            method: "zscore" or "minmax"
            epsilon: small constant for numerical stability (avoids division by 0)
        """
        if method not in ["zscore", "minmax"]:
            raise ValueError(f"Unknown method: {method}")

        self.method = method
        self.epsilon = epsilon
        self.params = {}  # Will store {feature: {mean, std}} or {feature: {min, max}}
        self.is_fit = False
        self.feature_names = None

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> "Normalizer":
        """
        Fit normalization parameters on training data.

        CRITICAL: Call this ONLY on training data.
        After fitting, this normalizer is locked. Val/test cannot change it.

        Args:
            data: training data (n_samples, n_features)
                  Can be pd.DataFrame or np.ndarray

        Returns:
            self (for chaining)
        """
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Store feature names for later validation
        self.feature_names = list(data.columns)

        # Compute statistics
        if self.method == "zscore":
            self.params = {
                col: {"mean": data[col].mean(), "std": data[col].std()}
                for col in data.columns
            }
        elif self.method == "minmax":
            self.params = {
                col: {"min": data[col].min(), "max": data[col].max()}
                for col in data.columns
            }

        self.is_fit = True
        logger.info(f"Normalizer fit on {len(data)} training samples using {self.method}")

        return self

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply learned normalization to new data.

        Uses statistics from training data. Does NOT refit.

        Args:
            data: data to normalize (n_samples, n_features)

        Returns:
            normalized data (same type as input: DataFrame or ndarray)

        Raises:
            RuntimeError if normalizer has not been fit
            ValueError if data columns don't match fit data
        """
        if not self.is_fit:
            raise RuntimeError("Normalizer must be fit before calling transform()")

        # Convert to DataFrame if needed
        is_ndarray = isinstance(data, np.ndarray)
        if is_ndarray:
            data = pd.DataFrame(data, columns=self.feature_names)

        # Validate columns match
        if list(data.columns) != self.feature_names:
            raise ValueError(
                f"Data columns {list(data.columns)} don't match fit columns {self.feature_names}"
            )

        # Apply normalization
        data_norm = data.copy()

        if self.method == "zscore":
            for col in data.columns:
                mean = self.params[col]["mean"]
                std = self.params[col]["std"]
                data_norm[col] = (data[col] - mean) / (std + self.epsilon)

        elif self.method == "minmax":
            for col in data.columns:
                min_val = self.params[col]["min"]
                max_val = self.params[col]["max"]
                range_val = max_val - min_val + self.epsilon
                data_norm[col] = (data[col] - min_val) / range_val

        # Convert back to ndarray if input was ndarray
        if is_ndarray:
            return data_norm.to_numpy()

        return data_norm

    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit and transform in one call (convenience for training data).

        Args:
            data: training data

        Returns:
            normalized training data
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data_norm: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Reverse normalization (denormalize).

        Useful for converting predictions back to original scale.

        Args:
            data_norm: normalized data

        Returns:
            data in original scale
        """
        if not self.is_fit:
            raise RuntimeError("Normalizer must be fit before calling inverse_transform()")

        is_ndarray = isinstance(data_norm, np.ndarray)
        if is_ndarray:
            data_norm = pd.DataFrame(data_norm, columns=self.feature_names)

        data_orig = data_norm.copy()

        if self.method == "zscore":
            for col in data_norm.columns:
                mean = self.params[col]["mean"]
                std = self.params[col]["std"]
                data_orig[col] = data_norm[col] * (std + self.epsilon) + mean

        elif self.method == "minmax":
            for col in data_norm.columns:
                min_val = self.params[col]["min"]
                max_val = self.params[col]["max"]
                range_val = max_val - min_val + self.epsilon
                data_orig[col] = data_norm[col] * range_val + min_val

        if is_ndarray:
            return data_orig.to_numpy()

        return data_orig


def normalize_dataset(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    method: str = "zscore",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Normalizer]:
    """
    Normalize train/val/test using training statistics only.

    This is the standard way to prevent leakage.

    Args:
        train_data: pd.DataFrame (training set)
        val_data: pd.DataFrame (validation set)
        test_data: pd.DataFrame (test set)
        method: "zscore" or "minmax"

    Returns:
        tuple of (train_norm, val_norm, test_norm, normalizer)

    Example:
        >>> train_norm, val_norm, test_norm, scaler = normalize_dataset(train, val, test)
        >>> # In production, reuse scaler:
        >>> prod_data_norm = scaler.transform(prod_data)
    """
    # Create and fit normalizer on training data only
    normalizer = Normalizer(method=method)
    normalizer.fit(train_data)

    # Transform all splits using train statistics (no refitting)
    train_norm = normalizer.transform(train_data)
    val_norm = normalizer.transform(val_data)
    test_norm = normalizer.transform(test_data)

    logger.info(
        f"Normalized dataset using {method}: "
        f"train={train_norm.shape}, val={val_norm.shape}, test={test_norm.shape}"
    )

    return train_norm, val_norm, test_norm, normalizer
