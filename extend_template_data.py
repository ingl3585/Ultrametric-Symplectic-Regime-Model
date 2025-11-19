#!/usr/bin/env python3
"""
Extend the existing template data with more synthetic bars.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def extend_data(input_path: str, output_path: str, total_bars: int = 1560):
    """
    Extend existing CSV data with synthetic bars.

    Args:
        input_path: Path to existing CSV
        output_path: Path to save extended CSV
        total_bars: Total bars desired (~60 days * 26 bars/day)
    """
    # Read existing data
    df_existing = pd.read_csv(input_path)
    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])

    print(f"Existing data: {len(df_existing)} bars")
    print(f"Last timestamp: {df_existing['timestamp'].iloc[-1]}")
    print(f"Last close: ${df_existing['close'].iloc[-1]:.2f}")

    # Get last values
    last_timestamp = df_existing['timestamp'].iloc[-1]
    last_close = df_existing['close'].iloc[-1]
    last_volume = df_existing['volume'].iloc[-1]

    # Calculate how many bars to add
    bars_to_add = total_bars - len(df_existing)

    if bars_to_add <= 0:
        print(f"Data already has {len(df_existing)} bars, no need to extend")
        return

    print(f"Adding {bars_to_add} bars...")

    # Generate continuation data
    np.random.seed(42)

    # 15-minute volatility (roughly 20% annualized)
    bar_vol = 0.20 / np.sqrt(252 * 26)

    # Generate returns with slight mean reversion
    log_prices = np.zeros(bars_to_add + 1)
    log_prices[0] = np.log(last_close)

    for i in range(1, bars_to_add + 1):
        ret = np.random.normal(0, bar_vol)
        mean_rev = -0.01 * (log_prices[i-1] - np.log(last_close))
        log_prices[i] = log_prices[i-1] + ret + mean_rev

    close_prices = np.exp(log_prices[1:])

    # Generate OHLC
    new_data = []
    current_time = last_timestamp + timedelta(minutes=15)
    bars_today = 5  # We already have 5 bars on first day

    for i in range(bars_to_add):
        close = close_prices[i]

        # Open (previous close with small gap)
        if i == 0:
            open_price = last_close * (1 + np.random.normal(0, bar_vol * 0.3))
        else:
            open_price = close_prices[i-1] * (1 + np.random.normal(0, bar_vol * 0.3))

        # High and low
        bar_range = abs(np.random.normal(0, bar_vol)) * close
        high = max(open_price, close) + abs(np.random.uniform(0, bar_range))
        low = min(open_price, close) - abs(np.random.uniform(0, bar_range))

        # Volume
        volume = int(last_volume * np.random.gamma(2, 0.5))

        new_data.append({
            'timestamp': current_time,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        # Advance time
        current_time += timedelta(minutes=15)
        bars_today += 1

        # Skip to next day after 26 bars (6.5 hours)
        if bars_today >= 26:
            current_time += timedelta(days=1)
            # Skip weekends
            while current_time.weekday() >= 5:
                current_time += timedelta(days=1)
            current_time = current_time.replace(hour=9, minute=30)
            bars_today = 0

    # Combine existing and new data
    df_new = pd.DataFrame(new_data)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Save
    df_combined.to_csv(output_path, index=False)

    print(f"\n✓ Extended data saved to {output_path}")
    print(f"  Total bars: {len(df_combined)}")
    print(f"  Date range: {df_combined['timestamp'].iloc[0]} to {df_combined['timestamp'].iloc[-1]}")
    print(f"  Price range: ${df_combined['close'].min():.2f} to ${df_combined['close'].max():.2f}")


if __name__ == "__main__":
    extend_data(
        input_path="data/sample_data_template.csv",
        output_path="data/sample_data_template.csv",  # Overwrite
        total_bars=1560  # ~60 days
    )
