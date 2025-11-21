#!/usr/bin/env python3
"""
ML Combiner Layer Validation

Tests the HybridMLModel (ML combiner on top of pure math model) and compares
performance against the pure math SymplecticUltrametricModel baseline.

Usage:
    python validation_ml.py
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from model.data_utils import (
    load_ohlcv_csv,
    resample_to_15m,
    compute_log_price,
    compute_smoothed_volume,
    build_gamma
)
from model.trainer import build_segments, compute_cluster_stats, compute_hit_rates_from_data
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
from model.signal_api import SymplecticUltrametricModel, HybridMLModel
from model.backtest import run_hybrid_backtest, run_hybrid_ml_backtest


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_section(title: str, width: int = 80):
    """Print formatted section header."""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def print_subsection(title: str, width: int = 80):
    """Print formatted subsection header."""
    print("\n" + "-"*width)
    print(title)
    print("-"*width + "\n")


def main():
    """Run ML combiner validation."""

    print_section("ML COMBINER LAYER VALIDATION")
    print(f"Validation run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # 1. Load Configuration and Data
    # ========================================================================
    print_subsection("1. CONFIGURATION & DATA LOADING")

    config = load_config()
    K = config['segments']['K']
    encoding = config['symplectic']['encoding']

    # Cost model (backward compatible)
    cost_log = config['costs'].get('cost_per_trade', config['costs'].get('cost_log_15m', -0.00048))

    num_clusters = config['clustering']['num_clusters']
    min_cluster_size = config['clustering']['min_cluster_size']
    base_b = config['ultrametric']['base_b']

    # Timeframe configuration
    bar_size_minutes = config.get('timeframe', {}).get('bar_size_minutes', 15)

    # ML config
    ml_enabled = config.get('ml', {}).get('enabled', True)
    ml_type = config.get('ml', {}).get('type', 'classifier')
    ml_hidden_sizes = config.get('ml', {}).get('hidden_sizes', [48, 24])
    ml_retrain_interval = config.get('ml', {}).get('retrain_interval', 1000)
    ml_min_samples = config.get('ml', {}).get('min_training_samples', 500)

    print(f"Configuration:")
    print(f"  Timeframe: {bar_size_minutes}-minute bars")
    print(f"  Segment length (K): {K} bars ({K * bar_size_minutes} minutes)")
    print(f"  Encoding: {encoding}")
    print(f"  Target clusters: {num_clusters}")
    print(f"  Min cluster size: {min_cluster_size}")
    print(f"  Ultrametric base: {base_b}")
    print(f"  Cost per trade: {cost_log:.6f} ({cost_log*100:.4f}%)")
    print(f"\n  ML enabled: {ml_enabled}")
    print(f"  ML type: {ml_type}")
    print(f"  ML hidden layers: {ml_hidden_sizes}")
    print(f"  ML retrain interval: {ml_retrain_interval} bars")
    print(f"  ML min samples: {ml_min_samples}")

    # Load data
    data_path = Path("data/sample_data_template.csv")
    if not data_path.exists():
        print(f"\n⚠ Data not found at {data_path}")
        print("Please run: python download_qqq_simple.py")
        return

    print(f"\nLoading data from {data_path}...")
    df = load_ohlcv_csv(str(data_path))

    # Resample to target timeframe
    if bar_size_minutes == 1:
        df_resampled = df  # Assume data is already 1-minute
        print(f"✓ Loaded {len(df_resampled)} 1-minute bars (no resampling)")
    elif bar_size_minutes == 15:
        df_resampled = resample_to_15m(df)
        print(f"✓ Loaded {len(df_resampled)} 15-minute bars (resampled)")
    else:
        # Generic resampling for other timeframes
        df_resampled = df.resample(f'{bar_size_minutes}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        print(f"✓ Loaded {len(df_resampled)} {bar_size_minutes}-minute bars (resampled)")

    print(f"  Date range: {df_resampled.index[0]} to {df_resampled.index[-1]}")
    print(f"  Days: {(df_resampled.index[-1] - df_resampled.index[0]).days}")

    # Compute phase-space data
    p = compute_log_price(df_resampled)
    v = compute_smoothed_volume(
        df_resampled,
        normalization_window=config['volume']['normalization_window'],
        ema_period=config['volume']['ema_period']
    )
    gamma = build_gamma(p, v)
    segments = build_segments(gamma, K)
    print(f"✓ Built {len(segments)} segments")

    # Train/Val/Test split (60/20/20)
    train_frac = config['backtest']['train_split']
    val_frac = config['backtest']['val_split']

    train_end = int(len(p) * train_frac)
    val_end = int(len(p) * (train_frac + val_frac))

    p_train = p[:train_end]
    p_val = p[train_end:val_end]
    p_test = p[val_end:]

    v_train = v[:train_end]
    v_val = v[train_end:val_end]
    v_test = v[val_end:]

    seg_train_end = train_end - K + 1
    seg_val_end = val_end - K + 1

    segments_train = segments[:seg_train_end]
    segments_val = segments[seg_train_end:seg_val_end]
    segments_test = segments[seg_val_end:]

    print(f"\n✓ Data split (60/20/20):")
    print(f"  Train: {len(p_train)} bars, {len(segments_train)} segments")
    print(f"  Val:   {len(p_val)} bars, {len(segments_val)} segments")
    print(f"  Test:  {len(p_test)} bars, {len(segments_test)} segments")

    # ========================================================================
    # 2. Clustering & Model Training
    # ========================================================================
    print_subsection("2. CLUSTERING & BASE MODEL TRAINING")

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
        print(f"    Cluster {cluster_id} (n={count}): κ = {kappa:.4f}")

    # Compute cluster statistics (for ML features)
    print("\nComputing cluster statistics...")
    cluster_stats = compute_hit_rates_from_data(segments_train, labels_train, p_train, config)

    # Build full cluster_stats dict
    cluster_stats_dict = {}
    for cluster_id in np.unique(labels_train):
        if cluster_id == -1:
            continue
        mask = labels_train == cluster_id
        cluster_stats_dict[cluster_id] = {
            'persistence': persistence.get(cluster_id, 0.5),
            'hit_rate': cluster_stats.get(cluster_id, 0.5),
            'size': int(np.sum(mask)),
            'raw_kappa': kappa_per_cluster.get(cluster_id, kappa_global)
        }

    print(f"✓ Cluster stats computed for {len(cluster_stats_dict)} clusters")
    for cluster_id, stats in cluster_stats_dict.items():
        print(f"    Cluster {cluster_id}: hit_rate={stats['hit_rate']:.2%}, "
              f"persist={stats['persistence']:.2%}, size={stats['size']}")

    # Train base model (pure math)
    print("\nTraining base model (Pure Math SymplecticUltrametric)...")
    base_model = SymplecticUltrametricModel(
        config, centroids, kappa_per_cluster, cluster_stats, encoding=encoding
    )
    print(f"✓ Base model ready: {len(centroids)} clusters, gating enabled")

    # Create ML combiner model
    print(f"\nCreating ML combiner model (type={ml_type})...")
    ml_model = HybridMLModel(
        config=config,
        base_model=base_model,
        ml_type=ml_type
    )
    print(f"✓ ML model created (not yet fitted)")

    # ========================================================================
    # 3. Validation Set Testing
    # ========================================================================
    print_subsection("3. VALIDATION SET PERFORMANCE")

    print("Testing Pure Math model on validation set...")
    pure_val_results = run_hybrid_backtest(
        base_model, p_val, v_val, K, cost_log
    )

    print("\nTesting ML-Enhanced model on validation set (with online learning)...")
    ml_val_results = run_hybrid_ml_backtest(
        ml_model, p_val, v_val, K, cluster_stats_dict, cost_log, warmup_bars=250
    )

    # Print validation results
    print("\nValidation Set Results:")
    print(f"{'Model':<30} {'Trades':>8} {'Win%':>8} {'Sharpe':>10} {'Net PnL':>12} {'Notes':>20}")
    print("-"*100)

    pure_m = pure_val_results['metrics']
    print(f"{'Pure Math':<30} {pure_m['num_trades']:>8} {pure_m['win_rate']:>7.1%} "
          f"{pure_m['sharpe_ratio']:>10.2f} {pure_m['total_net_pnl']:>12.6f} {'(baseline)':>20}")

    ml_m = ml_val_results['metrics']
    ml_fitted = ml_m.get('ml_model_fitted', False)
    ml_samples = ml_m.get('training_samples_collected', 0)
    print(f"{'ML-Enhanced':<30} {ml_m['num_trades']:>8} {ml_m['win_rate']:>7.1%} "
          f"{ml_m['sharpe_ratio']:>10.2f} {ml_m['total_net_pnl']:>12.6f} "
          f"{'fitted=' + str(ml_fitted):>20}")

    print(f"\n  ML training samples collected: {ml_samples}")
    print(f"  ML model fitted: {ml_fitted}")

    # ========================================================================
    # 4. Test Set Performance (Final)
    # ========================================================================
    print_subsection("4. TEST SET PERFORMANCE (FINAL)")

    print("Testing Pure Math model on test set...")
    pure_test_results = run_hybrid_backtest(
        base_model, p_test, v_test, K, cost_log
    )

    # Reset ML model for clean test (or create new instance)
    print("\nCreating fresh ML model for test set...")
    ml_model_test = HybridMLModel(
        config=config,
        base_model=base_model,
        ml_type=ml_type
    )

    print("Testing ML-Enhanced model on test set (with online learning)...")
    ml_test_results = run_hybrid_ml_backtest(
        ml_model_test, p_test, v_test, K, cluster_stats_dict, cost_log, warmup_bars=250
    )

    # Print test results
    print("\nTest Set Results:")
    print(f"{'Model':<30} {'Trades':>8} {'Win%':>8} {'Sharpe':>10} {'Net PnL':>12} {'Avg/Trade':>12}")
    print("-"*100)

    pure_t = pure_test_results['metrics']
    print(f"{'Pure Math':<30} {pure_t['num_trades']:>8} {pure_t['win_rate']:>7.1%} "
          f"{pure_t['sharpe_ratio']:>10.2f} {pure_t['total_net_pnl']:>12.6f} "
          f"{pure_t['avg_net_pnl']:>12.6f}")

    ml_t = ml_test_results['metrics']
    ml_fitted_test = ml_t.get('ml_model_fitted', False)
    ml_samples_test = ml_t.get('training_samples_collected', 0)
    print(f"{'ML-Enhanced':<30} {ml_t['num_trades']:>8} {ml_t['win_rate']:>7.1%} "
          f"{ml_t['sharpe_ratio']:>10.2f} {ml_t['total_net_pnl']:>12.6f} "
          f"{ml_t['avg_net_pnl']:>12.6f}")

    print(f"\n  ML training samples collected: {ml_samples_test}")
    print(f"  ML model fitted: {ml_fitted_test}")

    # ========================================================================
    # 5. Performance Comparison
    # ========================================================================
    print_subsection("5. PERFORMANCE COMPARISON")

    print("Comparing Pure Math vs ML-Enhanced (Test Set):\n")

    # Sharpe improvement
    sharpe_pure = pure_t['sharpe_ratio']
    sharpe_ml = ml_t['sharpe_ratio']
    sharpe_improvement = ((sharpe_ml - sharpe_pure) / abs(sharpe_pure) * 100) if sharpe_pure != 0 else 0
    print(f"Sharpe Ratio:")
    print(f"  Pure Math:    {sharpe_pure:>8.2f}")
    print(f"  ML-Enhanced:  {sharpe_ml:>8.2f}")
    print(f"  Improvement:  {sharpe_improvement:>7.1f}%")

    # Win rate improvement
    win_pure = pure_t['win_rate']
    win_ml = ml_t['win_rate']
    win_improvement = (win_ml - win_pure) * 100  # percentage point difference
    print(f"\nWin Rate:")
    print(f"  Pure Math:    {win_pure:>7.1%}")
    print(f"  ML-Enhanced:  {win_ml:>7.1%}")
    print(f"  Improvement:  {win_improvement:>6.1f} ppts")

    # PnL improvement
    pnl_pure = pure_t['total_net_pnl']
    pnl_ml = ml_t['total_net_pnl']
    pnl_improvement = ((pnl_ml - pnl_pure) / abs(pnl_pure) * 100) if pnl_pure != 0 else 0
    print(f"\nTotal Net PnL:")
    print(f"  Pure Math:    {pnl_pure:>12.6f}")
    print(f"  ML-Enhanced:  {pnl_ml:>12.6f}")
    print(f"  Improvement:  {pnl_improvement:>7.1f}%")

    # Trade count
    trades_pure = pure_t['num_trades']
    trades_ml = ml_t['num_trades']
    print(f"\nNumber of Trades:")
    print(f"  Pure Math:    {trades_pure:>8}")
    print(f"  ML-Enhanced:  {trades_ml:>8}")

    # ========================================================================
    # 6. Assessment
    # ========================================================================
    print_subsection("6. ML LAYER ASSESSMENT")

    print("ML Layer Performance:\n")

    # Check if ML improved things
    improvements = 0
    if sharpe_ml > sharpe_pure:
        improvements += 1
        print("✓ Sharpe ratio improved")
    else:
        print("✗ Sharpe ratio did not improve")

    if win_ml > win_pure:
        improvements += 1
        print("✓ Win rate improved")
    else:
        print("✗ Win rate did not improve")

    if pnl_ml > pnl_pure:
        improvements += 1
        print("✓ Net PnL improved")
    else:
        print("✗ Net PnL did not improve")

    if ml_fitted_test:
        improvements += 1
        print("✓ ML model successfully fitted")
    else:
        print("✗ ML model failed to fit")

    print(f"\n{'='*80}")
    print(f"Improvements: {improvements}/4")
    print(f"{'='*80}\n")

    # Final verdict
    if improvements >= 3:
        print("✓✓✓ ML LAYER SUCCESSFUL")
        print("The ML combiner layer provides meaningful improvements over the pure math model.")
        print("\nRecommendations:")
        print("  - Consider deploying ML-enhanced model")
        print("  - Monitor online learning performance in production")
        print("  - Test with different ML architectures (hidden layer sizes)")
    elif improvements >= 2:
        print("⚠ ML LAYER SHOWS PROMISE")
        print("The ML layer provides some benefit but results are mixed.")
        print("\nRecommendations:")
        print("  - Test with more diverse data periods")
        print("  - Experiment with regressor mode instead of classifier")
        print("  - Try different feature combinations")
    else:
        print("✗ ML LAYER UNDERPERFORMING")
        print("The ML layer does not improve over the pure math model.")
        print("\nPossible reasons:")
        print("  - Insufficient training data")
        print("  - Pure math model already optimal for this regime")
        print("  - ML features may need refinement")
        print("\nRecommendations:")
        print("  - Stick with pure math model for now")
        print("  - Revisit ML layer with longer data history")
        print("  - Consider simpler ML approaches (e.g., logistic regression)")

    print("\n" + "="*80)
    print("ML VALIDATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
