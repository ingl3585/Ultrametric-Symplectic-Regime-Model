"""
Data loading and transformation utilities for 15-minute bar data.

This module handles:
- Loading OHLCV data from CSV
- Resampling to 15-minute bars
- Computing log prices
- Computing smoothed, normalized volume
- Building phase-space gamma vectors
"""

import numpy as np
import pandas as pd
from typing import Tuple


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.

    Expected columns: timestamp, open, high, low, close, volume

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with DateTime index and OHLCV columns
    """
    df = pd.read_csv(path)

    # Parse timestamp and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def resample_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample OHLCV data to 15-minute bars.

    If data is already 15m, this is essentially a no-op (but validates structure).

    Args:
        df: DataFrame with DateTime index and OHLCV columns

    Returns:
        DataFrame resampled to 15-minute bars
    """
    resampled = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Drop any rows with NaN (incomplete bars)
    resampled = resampled.dropna()

    return resampled


def compute_log_price(df: pd.DataFrame) -> np.ndarray:
    """
    Compute log prices from close prices.

    Args:
        df: DataFrame with 'close' column

    Returns:
        1D array of log prices: p_t = log(close_t)
    """
    return np.log(df['close'].values)


def compute_smoothed_volume(
    df: pd.DataFrame,
    normalization_window: int,
    ema_period: int
) -> np.ndarray:
    """
    Compute smoothed, normalized volume.

    Steps:
    1. Normalize volume by rolling mean over normalization_window
    2. Apply EMA smoothing with ema_period

    Args:
        df: DataFrame with 'volume' column
        normalization_window: Window for rolling mean normalization (e.g., 200)
        ema_period: Period for EMA smoothing (e.g., 5)

    Returns:
        1D array of smoothed, normalized volume
    """
    volume = df['volume'].values

    # Step 1: Normalize by rolling mean
    rolling_mean = df['volume'].rolling(
        window=normalization_window,
        min_periods=1
    ).mean().values

    vol_norm = volume / (rolling_mean + 1e-8)

    # Step 2: Apply EMA smoothing
    alpha = 2.0 / (ema_period + 1.0)
    v = np.zeros_like(vol_norm)
    v[0] = vol_norm[0]

    for t in range(1, len(vol_norm)):
        v[t] = alpha * vol_norm[t] + (1 - alpha) * v[t-1]

    return v


def build_gamma(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Build phase-space vector gamma from price and volume.

    Args:
        p: Log prices, shape (N,)
        v: Smoothed normalized volume, shape (N,)

    Returns:
        Phase-space array gamma, shape (N, 2) where gamma[t] = [p[t], v[t]]
    """
    if len(p) != len(v):
        raise ValueError(f"p and v must have same length, got {len(p)} and {len(v)}")

    return np.stack([p, v], axis=1)
