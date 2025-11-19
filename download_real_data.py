#!/usr/bin/env python3
"""
Download real 15-minute QQQ data using yfinance.

Note: Requires 'yfinance' package:
    pip install yfinance

Usage:
    python download_real_data.py
"""

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance not installed")
    print("Install with: pip install yfinance")
    exit(1)

import pandas as pd
from datetime import datetime, timedelta


def download_qqq_15m(days_back: int = 60):
    """
    Download QQQ 15-minute data using yfinance.

    Args:
        days_back: Number of days of history to fetch

    Returns:
        DataFrame with timestamp, OHLCV
    """
    print(f"Downloading {days_back} days of QQQ 15-minute data...")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Download data
    ticker = yf.Ticker("QQQ")
    df = ticker.history(
        start=start_date,
        end=end_date,
        interval="15m"
    )

    if df.empty:
        print("Error: No data downloaded")
        return None

    # Rename columns to match our format
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Reset index to get timestamp as column
    df.reset_index(inplace=True)
    df.rename(columns={'Datetime': 'timestamp'}, inplace=True)

    # Keep only required columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Remove any rows with missing data
    df = df.dropna()

    return df


def main():
    """Download and save real QQQ data."""
    df = download_qqq_15m(days_back=60)

    if df is None or len(df) == 0:
        print("Failed to download data")
        return

    print(f"\n✓ Downloaded {len(df)} bars")
    print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

    # Save to CSV
    output_path = "data/sample_qqq_15m.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")

    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()
