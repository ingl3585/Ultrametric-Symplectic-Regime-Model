#!/usr/bin/env python3
"""
Ultrametric-Symplectic Regime Model - Main Entry Point

This script trains and backtests the pure math trading model on 3-minute QQQ/NQ data.

Usage:
    python run.py

The model combines:
- Ultrametric clustering for regime detection
- Symplectic dynamics for price forecasting
- Gating based on cluster quality

Results show trades, PnL, and performance metrics.
"""

import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

from model.data_utils import (
    load_ohlcv_csv,
    compute_log_price,
    compute_smoothed_volume,
    build_gamma
)
from model.trainer import build_segments
from model.clustering import (
    cluster_segments_ultrametric,
    assign_to_nearest_centroid,
    compute_centroids,
    compute_persistence
)
from model.symplectic_model import (
    estimate_global_kappa,
    estimate_kappa_per_cluster
)
from model.signal_api import SymplecticUltrametricModel
from model.backtest import run_hybrid_backtest


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_section(title: str, width: int = 80):
    """Print formatted section header."""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def main():
    """Run model training and backtest."""

    print_section("ULTRAMETRIC-SYMPLECTIC REGIME MODEL")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load configuration
    config = load_config()
    K = config['segments']['K']
    encoding = config['symplectic']['encoding']
    cost_log = config['costs']['cost_per_trade']
    num_clusters = config['clustering']['num_clusters']
    min_cluster_size = config['clustering']['min_cluster_size']
    base_b = config['ultrametric']['base_b']
    bar_size_minutes = config.get('timeframe', {}).get('bar_size_minutes', 3)

    print(f"Configuration:")
    print(f"  Timeframe: {bar_size_minutes}-minute bars")
    print(f"  Segment length (K): {K} bars ({K * bar_size_minutes} minutes)")
    print(f"  Encoding: {encoding}")
    print(f"  Target clusters: {num_clusters}")
    print(f"  Cost per trade: {cost_log:.6f} ({cost_log*100:.4f}%)")

    # Load data
    data_path = Path("data/sample_data_template.csv")
    if not data_path.exists():
        print(f"\n✗ Data not found at {data_path}")
        print("Please run: python download_qqq_1m_max.py")
        return

    print(f"\nLoading data from {data_path}...")
    df = load_ohlcv_csv(str(data_path))

    # Resample to target timeframe
    if bar_size_minutes != 1:
        df = df.resample(f'{bar_size_minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    print(f"✓ Loaded {len(df)} {bar_size_minutes}-minute bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Compute phase-space data
    p = compute_log_price(df)
    v = compute_smoothed_volume(
        df,
        normalization_window=config['volume']['normalization_window'],
        ema_period=config['volume']['ema_period']
    )
    gamma = build_gamma(p, v)
    segments = build_segments(gamma, K)
    print(f"✓ Built {len(segments)} segments")

    # Train/Test split (80/20)
    train_end = int(len(p) * 0.8)

    p_train = p[:train_end]
    v_train = v[:train_end]
    p_test = p[train_end:]
    v_test = v[train_end:]

    seg_train_end = train_end - K + 1
    segments_train = segments[:seg_train_end]

    print(f"\n✓ Data split (80/20):")
    print(f"  Train: {len(p_train)} bars, {len(segments_train)} segments")
    print(f"  Test:  {len(p_test)} bars")

    # Train model
    print_section("MODEL TRAINING")

    # Ultrametric clustering
    print("Clustering training segments...")
    labels_subsample, Z = cluster_segments_ultrametric(
        segments_train,
        num_clusters=num_clusters,
        base_b=base_b
    )

    centroids = compute_centroids(
        segments_train[:len(labels_subsample)],
        labels_subsample,
        min_cluster_size
    )
    print(f"✓ Found {len(centroids)} valid clusters")

    labels_train = assign_to_nearest_centroid(segments_train, centroids, base_b)
    persistence = compute_persistence(labels_train)
    mean_persist = np.mean(list(persistence.values())) if len(persistence) > 0 else 0.0
    print(f"  Mean persistence: {mean_persist:.2%}")

    # Estimate κ
    print("\nEstimating κ parameters...")
    kappa_global = estimate_global_kappa(segments_train, encoding=encoding)
    print(f"  Global κ: {kappa_global:.4f}")

    kappa_per_cluster = estimate_kappa_per_cluster(
        segments_train,
        labels_train,
        encoding=encoding
    )
    print(f"  Per-cluster κ: {len(kappa_per_cluster)} clusters")
    for cluster_id, kappa in kappa_per_cluster.items():
        count = np.sum(labels_train == cluster_id)
        persist = persistence.get(cluster_id, 0.0)
        print(f"    Cluster {cluster_id} (n={count}, persist={persist:.1%}): κ = {kappa:.4f}")

    # Compute hit rates
    from model.trainer import compute_hit_rates_from_data
    hit_rates = compute_hit_rates_from_data(segments_train, labels_train, p_train, config)

    print("\n  Cluster hit rates:")
    for cluster_id, hit_rate in hit_rates.items():
        print(f"    Cluster {cluster_id}: {hit_rate:.2%}")

    # Create model
    model = SymplecticUltrametricModel(
        config, centroids, kappa_per_cluster, hit_rates, encoding=encoding
    )
    print(f"\n✓ Model trained: {len(centroids)} clusters, gating enabled")

    # Backtest on test set
    print_section("BACKTEST RESULTS (TEST SET)")

    print("Running backtest...")
    results = run_hybrid_backtest(model, p_test, v_test, K, cost_log)

    metrics = results['metrics']
    trades = results['trades']

    print(f"\nPerformance Metrics:")
    print(f"  Total trades: {metrics['num_trades']}")
    print(f"  Win rate: {metrics['win_rate']:.1%}")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Total net PnL: {metrics['total_net_pnl']:.6f} ({metrics['total_net_pnl']*100:.2f}%)")
    print(f"  Avg PnL per trade: {metrics['avg_net_pnl']:.6f}")
    print(f"  Max drawdown: {metrics['max_drawdown']:.6f} ({metrics['max_drawdown']*100:.2f}%)")

    if metrics['num_trades'] > 0:
        print(f"\nTrade Quality:")
        print(f"  Wins: {metrics['num_wins']}/{metrics['num_trades']}")
        print(f"  Avg win: {metrics['avg_win']:.6f}")
        print(f"  Avg loss: {metrics['avg_loss']:.6f}")
        if metrics['avg_loss'] != 0:
            print(f"  Win/Loss ratio: {abs(metrics['avg_win'] / metrics['avg_loss']):.2f}")
        print(f"  Profit factor: {metrics['profit_factor']:.2f}")

    # Show individual trades
    if len(trades) > 0:
        print(f"\nTrades ({len(trades)} total):")
        print(f"  {'Entry':>6} {'Exit':>6} {'Dir':>4} {'PnL':>12} {'Net PnL':>12}")
        print("  " + "-"*48)
        for trade in trades[:10]:  # Show first 10
            dir_str = "LONG" if trade.direction == 1 else "SHORT"
            print(f"  {trade.entry_idx:>6} {trade.exit_idx:>6} {dir_str:>4} "
                  f"{trade.pnl:>12.6f} {trade.net_pnl:>12.6f}")
        if len(trades) > 10:
            print(f"  ... ({len(trades - 10)} more trades not shown)")

    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80 + "\n")

    # Assessment
    if metrics['sharpe_ratio'] > 1.0 and metrics['total_net_pnl'] > 0:
        print("✓ Model shows positive edge on test set")
        print("  Consider further validation or paper trading")
    elif metrics['num_trades'] < 10:
        print("⚠ Too few trades for statistical significance")
        print("  Consider adjusting gating parameters or collecting more data")
    else:
        print("✗ Model does not show positive edge on test set")
        print("  Review gating parameters and cluster quality")


if __name__ == "__main__":
    main()
