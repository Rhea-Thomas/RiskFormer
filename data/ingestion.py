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

import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def download_market_data(assets: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data from yfinance for specified assets.

    Args:
        assets: list of ticker symbols (e.g., ["SPY", "QQQ"])
        start_date: start date (YYYY-MM-DD)
        end_date: end date (YYYY-MM-DD)

    Returns:
        dict mapping asset -> pd.DataFrame with columns [Open, High, Low, Close, Volume, Adj Close]
        Index: DatetimeIndex (trading dates)

    Raises:
        RuntimeError if yfinance is not installed
    """
    if not HAS_YFINANCE:
        raise RuntimeError(
            "yfinance is required for download_market_data. "
            "Install it with: pip install yfinance"
        )

    data = {}
    for asset in assets:
        logger.info(f"Downloading {asset} from {start_date} to {end_date}")
        df = yf.download(asset, start=start_date, end=end_date, progress=False)

        if df.empty:
            logger.warning(f"No data found for {asset}")
            continue

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        data[asset] = df

    return data


def validate_market_data(data: Dict[str, pd.DataFrame]) -> Tuple[bool, Dict]:
    """
    Validate market data for completeness and quality.

    Checks:
      - No NaN values in OHLCV
      - Close price between High and Low
      - High >= Low
      - All prices > 0
      - Volume >= 0
      - No duplicate dates

    Args:
        data: dict mapping asset -> pd.DataFrame

    Returns:
        (is_valid, report) where report is dict with issues per asset
    """
    report = {}
    is_valid = True

    for asset, df in data.items():
        issues = []

        # Check for NaN
        nan_cols = df.isnull().sum()
        if nan_cols.any():
            issues.append(f"NaN values: {nan_cols[nan_cols > 0].to_dict()}")

        # Check duplicate dates
        if df.index.duplicated().any():
            issues.append(f"Duplicate dates: {df.index[df.index.duplicated()].tolist()}")

        # Check price sanity
        # Use .to_numpy().any() to handle both single-asset and multi-asset DataFrames
        if (df["Close"] < df["Low"]).to_numpy().any() or (df["Close"] > df["High"]).to_numpy().any():
            issues.append("Close price outside High-Low range")

        if (df["High"] < df["Low"]).to_numpy().any():
            issues.append("High < Low")

        if (df[["Open", "High", "Low", "Close"]] <= 0).to_numpy().any():
            issues.append("Price <= 0")

        if (df["Volume"] < 0).to_numpy().any():
            issues.append("Volume < 0")

        if issues:
            is_valid = False
            report[asset] = issues
        else:
            report[asset] = "OK"

    return is_valid, report


def align_assets(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align multiple assets to common trading dates.

    Returns the intersection of trading dates (safest approach: no NaN from alignment).

    Args:
        data: dict mapping asset -> pd.DataFrame

    Returns:
        aligned_df with MultiIndex (date, asset) or wide format depending on use
        For now: concatenate into single DataFrame with MultiIndex columns (asset, OHLCV)
    """
    if not data:
        return pd.DataFrame()

    # Get common dates (intersection)
    all_dates = [df.index for df in data.values()]
    common_dates = all_dates[0]
    for dates in all_dates[1:]:
        common_dates = common_dates.intersection(dates)

    logger.info(f"Common trading dates: {len(common_dates)} (from {min(len(df) for df in data.values())} to {max(len(df) for df in data.values())} individual assets)")

    # Align all assets to common dates
    aligned = {}
    for asset, df in data.items():
        aligned[asset] = df.loc[common_dates]

    # Concatenate into single DataFrame with MultiIndex columns
    # Column structure: (asset, OHLCV)
    combined = pd.concat(aligned, axis=1)
    combined.index.name = "Date"

    return combined


def load_or_download(
    assets: List[str],
    start_date: str,
    end_date: str,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Load market data from cache if available, otherwise download and cache.

    Args:
        assets: list of ticker symbols
        start_date: start date (YYYY-MM-DD)
        end_date: end date (YYYY-MM-DD)
        cache_dir: directory to store cached data

    Returns:
        aligned DataFrame with shape (n_dates, n_assets * n_features)
        MultiIndex columns: (asset, OHLCV)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    all_cached = True
    data = {}

    for asset in assets:
        cache_file = cache_dir / f"{asset}_{start_date}_{end_date}.parquet"

        if cache_file.exists():
            logger.info(f"Loading {asset} from cache: {cache_file}")
            data[asset] = pd.read_parquet(cache_file)
        else:
            all_cached = False
            break

    # If all cached, return aligned data
    if all_cached:
        logger.info("All assets loaded from cache")
        return align_assets(data)

    # Otherwise, download all
    logger.info(f"Cache miss or incomplete. Downloading {len(assets)} assets...")
    data = download_market_data(assets, start_date, end_date)

    # Validate
    is_valid, report = validate_market_data(data)
    logger.info(f"Validation report: {report}")

    if not is_valid:
        logger.warning("Validation issues detected. Proceeding anyway.")

    # Cache individual assets
    for asset, df in data.items():
        cache_file = cache_dir / f"{asset}_{start_date}_{end_date}.parquet"
        df.to_parquet(cache_file)
        logger.info(f"Cached {asset} to {cache_file}")

    # Align and return
    return align_assets(data)
