#!/usr/bin/env python3
"""
STEP 0 Example – Demonstration of Base Pipeline + AR(1) Baseline

This script demonstrates:
1. Loading and processing 15-minute OHLCV data
2. Computing log prices and smoothed volume
3. Fitting an AR(1) model
4. Running a backtest with realistic costs
5. Displaying performance metrics

Usage:
    python example_step0.py

Note: You'll need to place a sample CSV file at data/sample_qqq_15m.csv
with columns: timestamp, open, high, low, close, volume
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from model.data_utils import (
    load_ohlcv_csv,
    resample_to_15m,
    compute_log_price,
    compute_smoothed_volume,
    build_gamma
)
from model.signal_api import AR1Model
from model.backtest import run_ar1_backtest


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_metrics(metrics: dict, title: str = "Backtest Metrics"):
    """Pretty print backtest metrics."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"Total Trades:          {metrics['num_trades']:>10}")
    print(f"Winning Trades:        {metrics['num_wins']:>10}")
    print(f"Losing Trades:         {metrics['num_losses']:>10}")
    print(f"Win Rate:              {metrics['win_rate']:>10.2%}")
    print(f"-" * 60)
    print(f"Avg Gross PnL/Trade:   {metrics['avg_gross_pnl']:>10.6f}")
    print(f"Avg Net PnL/Trade:     {metrics['avg_net_pnl']:>10.6f}")
    print(f"Total Net PnL:         {metrics['total_net_pnl']:>10.6f}")
    print(f"-" * 60)
    print(f"Avg Win:               {metrics['avg_win']:>10.6f}")
    print(f"Avg Loss:              {metrics['avg_loss']:>10.6f}")
    print(f"Profit Factor:         {metrics['profit_factor']:>10.2f}")
    print(f"-" * 60)
    print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:          {metrics['max_drawdown']:>10.6f}")
    print(f"{'='*60}\n")


def main():
    """Run STEP 0 example pipeline."""
    print("\n" + "="*60)
    print("STEP 0: Base Pipeline + AR(1) Baseline Demonstration")
    print("="*60 + "\n")

    # Load configuration
    print("Loading configuration...")
    config = load_config()
    print(f"✓ Configuration loaded from configs/config.yaml")

    # Check if sample data exists
    data_path = Path("data/sample_data_template.csv")
    if not data_path.exists():
        print(f"\n⚠ Sample data not found at {data_path}")
        print("\nTo use this example, create a CSV file with columns:")
        print("  timestamp, open, high, low, close, volume")
        print("\nExample format:")
        print("  timestamp,open,high,low,close,volume")
        print("  2024-01-02 09:30:00,400.50,401.20,400.30,400.80,1500000")
        print("  2024-01-02 09:45:00,400.80,401.50,400.70,401.20,1600000")
        print("  ...")
        return

    # Load data
    print(f"\nLoading OHLCV data from {data_path}...")
    df = load_ohlcv_csv(str(data_path))
    print(f"✓ Loaded {len(df)} bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Resample to 15-minute bars (if needed)
    print("\nResampling to 15-minute bars...")
    df_15m = resample_to_15m(df)
    print(f"✓ Resampled to {len(df_15m)} 15-minute bars")

    # Compute log price
    print("\nComputing log prices...")
    p = compute_log_price(df_15m)
    print(f"✓ Log prices computed, shape: {p.shape}")
    print(f"  Range: [{p.min():.4f}, {p.max():.4f}]")

    # Compute smoothed volume
    print("\nComputing smoothed, normalized volume...")
    v = compute_smoothed_volume(
        df_15m,
        normalization_window=config['volume']['normalization_window'],
        ema_period=config['volume']['ema_period']
    )
    print(f"✓ Smoothed volume computed, shape: {v.shape}")
    print(f"  Range: [{v.min():.4f}, {v.max():.4f}]")

    # Build gamma (phase-space vector)
    print("\nBuilding phase-space vector gamma...")
    gamma = build_gamma(p, v)
    print(f"✓ Gamma built, shape: {gamma.shape}")
    print(f"  First 3 rows:")
    for i in range(min(3, len(gamma))):
        print(f"    [{gamma[i, 0]:.4f}, {gamma[i, 1]:.4f}]")

    # Split data into train/test
    split_idx = int(len(p) * 0.7)
    p_train = p[:split_idx]
    p_test = p[split_idx:]
    print(f"\n✓ Data split: {len(p_train)} train bars, {len(p_test)} test bars")

    # Fit AR(1) model
    print("\nFitting AR(1) model on training data...")
    ar1_model = AR1Model(config)
    ar1_model.fit(p_train)
    print(f"✓ AR(1) fitted:")
    print(f"  Mean return: {ar1_model.mean_ret:.6f}")
    print(f"  AR(1) coeff (phi): {ar1_model.phi:.4f}")
    print(f"  Threshold (theta): {config['signal']['theta_ar1']:.6f}")

    # Run backtest on test data
    print("\nRunning AR(1) backtest on test data...")
    cost_log = config['costs']['cost_log_15m']
    print(f"  Cost per round trip: {cost_log:.6f} ({cost_log*100:.4f}%)")

    results = run_ar1_backtest(
        model=ar1_model,
        p=p_test,
        cost_log=cost_log
    )

    # Display results
    print_metrics(results['metrics'], title="AR(1) Baseline – Test Set Results")

    # Summary statistics
    if results['metrics']['num_trades'] > 0:
        print("Sample trades (first 5):")
        print(f"{'Entry':>8} {'Exit':>8} {'Dir':>4} {'Entry P':>10} {'Exit P':>10} {'Net PnL':>12}")
        print("-" * 60)
        for trade in results['trades'][:5]:
            print(f"{trade.entry_idx:>8} {trade.exit_idx:>8} "
                  f"{trade.direction:>4} {trade.entry_price:>10.4f} "
                  f"{trade.exit_price:>10.4f} {trade.net_pnl:>12.6f}")

        # Equity curve summary
        equity = results['equity_curve']
        print(f"\nEquity curve:")
        print(f"  Initial: {equity[0]:>10.6f}")
        print(f"  Final:   {equity[-1]:>10.6f}")
        print(f"  Peak:    {equity.max():>10.6f}")
        print(f"  Trough:  {equity.min():>10.6f}")

    print("\n" + "="*60)
    print("STEP 0 demonstration complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
