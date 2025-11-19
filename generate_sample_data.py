#!/usr/bin/env python3
"""
Generate synthetic 15-minute OHLCV data for testing STEP 0.

This creates a realistic-looking price series with:
- Trend + mean reversion
- Realistic OHLC relationships
- Volume with patterns

Output: data/sample_qqq_15m.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_15m_data(
    num_days: int = 60,
    bars_per_day: int = 26,  # 6.5 hours * 4 bars/hour
    initial_price: float = 400.0,
    annual_vol: float = 0.20
) -> pd.DataFrame:
    """
    Generate synthetic 15-minute bar data.

    Args:
        num_days: Number of trading days
        bars_per_day: Bars per trading day (26 for 6.5hr session)
        initial_price: Starting price
        annual_vol: Annualized volatility

    Returns:
        DataFrame with timestamp, OHLCV
    """
    total_bars = num_days * bars_per_day

    # Calculate 15-minute volatility from annual
    # 252 trading days * 26 bars/day = 6552 bars/year
    bars_per_year = 252 * bars_per_day
    bar_vol = annual_vol / np.sqrt(bars_per_year)

    # Generate price process (GBM with slight mean reversion)
    np.random.seed(42)
    returns = np.random.normal(0, bar_vol, total_bars)

    # Add slight mean reversion
    log_prices = np.zeros(total_bars)
    log_prices[0] = np.log(initial_price)

    for i in range(1, total_bars):
        # Mean reversion toward initial price
        mean_reversion = -0.01 * (log_prices[i-1] - np.log(initial_price))
        log_prices[i] = log_prices[i-1] + returns[i] + mean_reversion

    close_prices = np.exp(log_prices)

    # Generate OHLC from close prices
    high_prices = np.zeros(total_bars)
    low_prices = np.zeros(total_bars)
    open_prices = np.zeros(total_bars)

    open_prices[0] = close_prices[0]

    for i in range(total_bars):
        # Open is previous close with small gap
        if i > 0:
            gap = np.random.normal(0, bar_vol * 0.3)
            open_prices[i] = close_prices[i-1] * (1 + gap)

        # High and low around open/close
        bar_range = abs(np.random.normal(0, bar_vol)) * close_prices[i]
        high_prices[i] = max(open_prices[i], close_prices[i]) + abs(np.random.uniform(0, bar_range))
        low_prices[i] = min(open_prices[i], close_prices[i]) - abs(np.random.uniform(0, bar_range))

    # Generate volume with patterns
    base_volume = 1_500_000
    volume = np.random.gamma(2, base_volume/2, total_bars)

    # Add some volume clustering (high volume tends to cluster)
    for i in range(1, total_bars):
        if abs(returns[i]) > bar_vol:  # Big move = more volume
            volume[i] *= 1.5

    # Generate timestamps (trading hours: 9:30 AM - 4:00 PM ET)
    start_date = datetime(2024, 1, 2, 9, 30)
    timestamps = []

    current_time = start_date
    bars_today = 0

    for _ in range(total_bars):
        timestamps.append(current_time)
        current_time += timedelta(minutes=15)
        bars_today += 1

        # Skip to next day after market close
        if bars_today >= bars_per_day:
            # Move to next trading day (skip weekends roughly)
            current_time += timedelta(days=1)
            while current_time.weekday() >= 5:  # Skip Sat/Sun
                current_time += timedelta(days=1)
            current_time = current_time.replace(hour=9, minute=30)
            bars_today = 0

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume.astype(int)
    })

    return df


def main():
    """Generate and save synthetic data."""
    print("Generating synthetic 15-minute OHLCV data...")

    df = generate_synthetic_15m_data(
        num_days=60,        # ~3 months of data
        initial_price=400.0,  # QQQ-like price
        annual_vol=0.20     # 20% annualized vol
    )

    print(f"Generated {len(df)} bars")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

    # Save to CSV
    output_path = "data/sample_qqq_15m.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")

    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    main()
