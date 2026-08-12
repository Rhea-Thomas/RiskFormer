"""
Market data ingestion.

Downloads historical OHLCV data from public sources (e.g., yfinance).
Handles caching, validation, and versioning.

Key responsibilities:
  - Download market data for specified assets and date ranges
  - Cache to avoid redundant downloads
  - Validate data completeness and consistency
  - Align data across multiple assets
"""

def download_market_data(assets, start_date, end_date):
    """
    Download OHLCV data for specified assets.

    Args:
        assets: list of ticker symbols (e.g., ["SPY", "QQQ"])
        start_date: start date (YYYY-MM-DD)
        end_date: end date (YYYY-MM-DD)

    Returns:
        dict mapping asset -> pd.DataFrame with OHLCV columns
    """
    pass


def validate_market_data(data):
    """
    Check for missing values, duplicates, unreasonable prices.

    Args:
        data: dict mapping asset -> pd.DataFrame

    Returns:
        validation report (pass/fail and statistics)
    """
    pass
