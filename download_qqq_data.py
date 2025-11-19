#!/usr/bin/env python3
"""
Download real QQQ 15-minute bar data from yfinance.

QQQ tracks the Nasdaq-100 index (same underlying as NQ futures)
with much better data reliability on yfinance.

Usage:
    python download_qqq_data.py [--months 3] [--output data/qqq_15m.csv]
"""

import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def download_qqq_15m(
    months_back: int = 2,
    output_path: str = "data/sample_data_template.csv",
    fallback_ticker: str = "SPY"
) -> pd.DataFrame:
    """
    Download QQQ 15-minute bar data from yfinance.

    Note: yfinance limits 15-minute data to the last 60 days.
    For longer history, use daily data or a different data source.

    Args:
        months_back: How many months of history to download (max ~2 months for 15m data)
        output_path: Where to save the CSV
        fallback_ticker: Ticker to try if QQQ fails (e.g., SPY)

    Returns:
        DataFrame with OHLCV data
    """
    print("="*70)
    print("Downloading Real Market Data from yfinance")
    print("="*70)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)

    print(f"\nTicker: QQQ (Nasdaq-100 ETF)")
    print(f"Interval: 15 minutes")
    print(f"Period: {start_date.date()} to {end_date.date()} (~{months_back} months)")
    print(f"\nDownloading...")

    # Try QQQ first
    try:
        df = yf.download(
            "QQQ",
            start=start_date,
            end=end_date,
            interval="15m",
            progress=True
        )

        if df.empty:
            raise ValueError("No data returned for QQQ")

        ticker_used = "QQQ"
        print(f"\n✓ Successfully downloaded {len(df)} bars for QQQ")

    except Exception as e:
        print(f"\n⚠ Failed to download QQQ: {e}")
        print(f"Trying fallback ticker: {fallback_ticker}...")

        try:
            df = yf.download(
                fallback_ticker,
                start=start_date,
                end=end_date,
                interval="15m",
                progress=True
            )

            if df.empty:
                raise ValueError(f"No data returned for {fallback_ticker}")

            ticker_used = fallback_ticker
            print(f"\n✓ Successfully downloaded {len(df)} bars for {fallback_ticker}")

        except Exception as e2:
            print(f"\n✗ Failed to download {fallback_ticker}: {e2}")
            print("\nTroubleshooting:")
            print("  1. Check your internet connection")
            print("  2. Try: pip install --upgrade yfinance")
            print("  3. yfinance servers may be down - try again later")
            return None

    # Process data
    print(f"\nProcessing data...")

    # Rename columns to lowercase for consistency
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Reset index to get timestamp as column
    df = df.reset_index()
    df = df.rename(columns={'Datetime': 'timestamp', 'Date': 'timestamp'})

    # Ensure timestamp is datetime
    if 'timestamp' not in df.columns:
        # Sometimes the index name is different
        df['timestamp'] = df.index

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Keep only required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[required_cols]

    # Remove any NaN rows
    df = df.dropna()

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"✓ Processed {len(df)} bars")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    print(f"  Avg volume: {df['volume'].mean():,.0f}")

    # Check data quality
    print(f"\nData quality checks:")

    # Check for gaps
    time_diffs = df['timestamp'].diff()
    expected_diff = pd.Timedelta(minutes=15)

    # Allow some flexibility for market hours (weekends, holidays)
    large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=24)]
    if len(large_gaps) > 0:
        print(f"  ⚠ Found {len(large_gaps)} large gaps (>24h) - normal for weekends/holidays")
    else:
        print(f"  ✓ No large gaps detected")

    # Check for duplicate timestamps
    duplicates = df['timestamp'].duplicated().sum()
    if duplicates > 0:
        print(f"  ⚠ Found {duplicates} duplicate timestamps - removing...")
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
    else:
        print(f"  ✓ No duplicate timestamps")

    # Check for zero/negative prices
    invalid_prices = ((df['close'] <= 0) | (df['open'] <= 0)).sum()
    if invalid_prices > 0:
        print(f"  ⚠ Found {invalid_prices} invalid prices - removing...")
        df = df[(df['close'] > 0) & (df['open'] > 0)]
    else:
        print(f"  ✓ All prices valid")

    # Save to CSV
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df)} bars")

    # Show sample
    print(f"\nFirst 5 bars:")
    print(df.head().to_string(index=False))

    print(f"\nLast 5 bars:")
    print(df.tail().to_string(index=False))

    print("\n" + "="*70)
    print(f"SUCCESS: Downloaded {ticker_used} data")
    print(f"You can now run: python example_step0.py")
    print("="*70 + "\n")

    return df


def main():
    """Command-line interface for downloading QQQ data."""
    parser = argparse.ArgumentParser(
        description="Download QQQ 15-minute bar data from yfinance"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=2,
        help="Number of months of history to download (default: 2, max ~2 for 15m data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_data_template.csv",
        help="Output CSV path (default: data/sample_data_template.csv)"
    )
    parser.add_argument(
        "--fallback",
        type=str,
        default="SPY",
        help="Fallback ticker if QQQ fails (default: SPY)"
    )

    args = parser.parse_args()

    df = download_qqq_15m(
        months_back=args.months,
        output_path=args.output,
        fallback_ticker=args.fallback
    )

    if df is None:
        print("\n✗ Download failed")
        exit(1)
    else:
        print("\n✓ Download complete")
        exit(0)


if __name__ == "__main__":
    main()
