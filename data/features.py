"""
Feature engineering for financial time series.

Computes technical indicators and derived features from raw OHLCV data.

Key responsibilities:
  - Compute returns (simple and log)
  - Rolling volatility
  - Moving averages
  - Momentum indicators (RSI, MACD, etc.)
  - Volume-based features
  - NO LEAKAGE: Features computed with proper lookback windows

Important: All rolling statistics are computed with full historical data
available at each time step. Once chronological split occurs, only
training data informs feature computation for val/test.
"""

def compute_returns(prices, method="log"):
    """
    Compute returns from prices.

    Args:
        prices: pd.Series of closing prices
        method: "simple" or "log"

    Returns:
        pd.Series of returns
    """
    pass


def compute_rolling_volatility(returns, window=20):
    """
    Compute rolling volatility.

    Args:
        returns: pd.Series of returns
        window: lookback window in days

    Returns:
        pd.Series of rolling volatility
    """
    pass


def compute_moving_average(prices, window=20):
    """
    Compute moving average.

    Args:
        prices: pd.Series of prices
        window: lookback window in days

    Returns:
        pd.Series of moving average
    """
    pass


def compute_momentum(prices, window=20):
    """
    Compute momentum (price change over window).

    Args:
        prices: pd.Series of prices
        window: lookback window in days

    Returns:
        pd.Series of momentum
    """
    pass


def engineer_features(data):
    """
    Apply full feature engineering pipeline.

    Adds derived columns: returns, volatility, moving averages, etc.

    Args:
        data: pd.DataFrame with OHLCV columns

    Returns:
        pd.DataFrame with original + engineered columns
    """
    pass
