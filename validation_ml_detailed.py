#!/usr/bin/env python3
"""
ML Combiner Layer Validation - DETAILED TRADE ANALYSIS

Provides trade-by-trade comparison between Pure Math and ML-Enhanced models
to diagnose performance differences.

Usage:
    python validation_ml_detailed.py
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
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


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detailed_backtest_comparison(
    pure_model: SymplecticUltrametricModel,
    ml_model: HybridMLModel,
    p: np.ndarray,
    v: np.ndarray,
    K: int,
    cluster_stats_dict: Dict,
    cost_log: float,
    warmup_bars: int = 250
) -> pd.DataFrame:
    """
    Run both models bar-by-bar and capture detailed trade information.

    Returns DataFrame with columns:
    - bar_idx: Bar index
    - pure_dir: Pure math direction
    - pure_size: Pure math size_factor
    - ml_dir: ML direction
    - ml_size: ML size_factor
    - ml_confidence: ML model confidence
    - pure_pnl: Pure math PnL this bar
    - ml_pnl: ML PnL this bar
    - cluster_id: Assigned cluster
    - distance: Distance to centroid
    - hit_rate: Cluster hit rate
    """

    N = len(p)
    trades = []

    # State tracking
    pure_pos = 0
    ml_pos = 0
    pure_cum_pnl = 0.0
    ml_cum_pnl = 0.0

    print(f"\nRunning detailed bar-by-bar comparison...")
    print(f"Total bars: {N}, warmup: {warmup_bars}, tradeable: {N - K + 1 - warmup_bars}")

    for t in range(K-1, N):
        bar_idx = t - K + 1  # segment index

        # Build last K bars as segment (K, 2)
        last_k_p = p[t-K+1:t+1]
        last_k_v = v[t-K+1:t+1]
        segment = np.stack([last_k_p, last_k_v], axis=1)

        # Add training example for ML model (if not first bar)
        if t > K-1:
            # Realized return from last bar
            realized_return = p[t] - p[t-1]

            # Extract features from previous segment for training
            if t >= K:
                from model.features import extract_feature_vector
                from model.symplectic_model import extract_state_from_segment, leapfrog_step

                prev_segment = np.stack([p[t-K:t], v[t-K:t]], axis=1)

                # Get base model internals
                nearest = ml_model.base_model._nearest_cluster(prev_segment)
                c_id, dist = (nearest[0], nearest[1]) if nearest[0] is not None else (None, None)
                gating_passed = ml_model.base_model._passes_gating(c_id, dist, cluster_stats_dict)
                kappa = ml_model.base_model.kappa_per_cluster.get(c_id, ml_model.base_model.kappa_global)

                # Extract state and forecasts
                q, pi = extract_state_from_segment(prev_segment, ml_model.base_model.encoding)
                q_next, pi_next_1step = leapfrog_step(q, pi, kappa, dt=1.0)
                q_next2, pi_next_2step = leapfrog_step(q_next, pi_next_1step, kappa, dt=1.0)

                # Extract features
                features = extract_feature_vector(
                    segment=prev_segment,
                    nearest_cluster_id=c_id,
                    distance_to_centroid=dist,
                    cluster_stats=cluster_stats_dict,
                    kappa_shrunk=kappa,
                    pi_next_1step=pi_next_1step,
                    pi_next_2step=pi_next_1step + pi_next_2step,
                    gating_passed=gating_passed,
                    p_full=p,
                    v_full=v,
                    t=t-1,
                    config=ml_model.config
                )

                # Add training example
                ml_model._add_training_example(features, realized_return)

        # Retrain ML model periodically
        ml_model.retrain_if_needed(current_bar_idx=t, force=False)

        # Skip warmup period for trading
        if bar_idx < warmup_bars:
            continue

        # Get signals from both models
        pure_signal = pure_model.get_signal(segment)
        pure_dir = pure_signal['direction']
        pure_size = pure_signal['size_factor']

        # Get ML signal with cluster stats (needs full p, v arrays and time index)
        ml_signal = ml_model.get_signal(segment, p, v, t, cluster_stats_dict)
        ml_dir = ml_signal['direction']
        ml_size = ml_signal['size_factor']
        ml_confidence = ml_signal.get('ml_confidence', 0.0)
        cluster_id, distance = pure_model._nearest_cluster(segment)
        hit_rate = pure_model.hit_rates.get(cluster_id, 0.0) if cluster_id is not None else 0.0

        # Calculate PnL for this bar
        if t < N - 1:
            next_return = p[t+1] - p[t]

            # Pure math PnL
            pure_bar_pnl = 0.0
            pure_cost = 0.0
            if pure_dir != pure_pos:
                # Position change = pay cost
                pure_cost = cost_log
                if pure_dir != 0:
                    # Entering position
                    pure_bar_pnl = pure_dir * next_return + pure_cost
                else:
                    # Exiting to flat
                    pure_bar_pnl = pure_cost
            elif pure_pos != 0:
                # Holding position
                pure_bar_pnl = pure_pos * next_return

            # ML PnL
            ml_bar_pnl = 0.0
            ml_cost = 0.0
            if ml_dir != ml_pos:
                # Position change = pay cost
                ml_cost = cost_log
                if ml_dir != 0:
                    # Entering position
                    ml_bar_pnl = ml_dir * next_return + ml_cost
                else:
                    # Exiting to flat
                    ml_bar_pnl = ml_cost
            elif ml_pos != 0:
                # Holding position
                ml_bar_pnl = ml_pos * next_return

            pure_cum_pnl += pure_bar_pnl
            ml_cum_pnl += ml_bar_pnl

            # Record trade data
            trades.append({
                'bar_idx': bar_idx,
                'bar_global': t,
                'pure_dir': pure_dir,
                'pure_size': pure_size,
                'pure_pos_before': pure_pos,
                'ml_dir': ml_dir,
                'ml_size': ml_size,
                'ml_confidence': ml_confidence,
                'ml_pos_before': ml_pos,
                'next_return': next_return,
                'pure_bar_pnl': pure_bar_pnl,
                'ml_bar_pnl': ml_bar_pnl,
                'pure_cum_pnl': pure_cum_pnl,
                'ml_cum_pnl': ml_cum_pnl,
                'cluster_id': cluster_id if cluster_id is not None else -1,
                'distance': distance if distance is not None else -1,
                'hit_rate': hit_rate,
                'pure_cost_paid': pure_cost,
                'ml_cost_paid': ml_cost,
                'position_agree': int(pure_dir == ml_dir)
            })

            # Update positions
            pure_pos = pure_dir
            ml_pos = ml_dir

    df = pd.DataFrame(trades)
    print(f"✓ Captured {len(df)} bars of trading data")

    return df


def analyze_trade_differences(df: pd.DataFrame):
    """Analyze where and why Pure Math and ML models differ."""

    print("\n" + "="*80)
    print("DETAILED TRADE ANALYSIS")
    print("="*80 + "\n")

    # 1. Position agreement
    agree_pct = df['position_agree'].mean() * 100
    print(f"1. POSITION AGREEMENT")
    print(f"   Models agree on direction: {agree_pct:.1f}% of the time")
    print(f"   Models disagree: {100-agree_pct:.1f}% of the time\n")

    # 2. Find trades where they took opposite positions
    disagreements = df[df['position_agree'] == 0].copy()
    if len(disagreements) > 0:
        print(f"2. DISAGREEMENT ANALYSIS ({len(disagreements)} bars)")
        print(f"   {'Bar':>6} {'Pure':>6} {'ML':>6} {'Return':>10} {'Pure PnL':>12} {'ML PnL':>12} {'Winner':>10}")
        print("   " + "-"*72)
        for _, row in disagreements.head(20).iterrows():
            winner = "Pure" if row['pure_bar_pnl'] > row['ml_bar_pnl'] else "ML" if row['ml_bar_pnl'] > row['pure_bar_pnl'] else "Tie"
            print(f"   {int(row['bar_idx']):>6} {int(row['pure_dir']):>6} {int(row['ml_dir']):>6} "
                  f"{row['next_return']:>10.6f} {row['pure_bar_pnl']:>12.6f} {row['ml_bar_pnl']:>12.6f} {winner:>10}")
        if len(disagreements) > 20:
            print(f"   ... ({len(disagreements) - 20} more disagreements not shown)\n")

    # 3. Trades taken by each model
    pure_trades = df[df['pure_dir'] != df['pure_pos_before']].copy()
    ml_trades = df[df['ml_dir'] != df['ml_pos_before']].copy()

    print(f"\n3. TRADE COUNTS")
    print(f"   Pure Math trades: {len(pure_trades)}")
    print(f"   ML trades: {len(ml_trades)}\n")

    # 4. Trade outcomes
    if len(pure_trades) > 0:
        pure_wins = (pure_trades['pure_bar_pnl'] > 0).sum()
        pure_losses = (pure_trades['pure_bar_pnl'] < 0).sum()
        pure_win_rate = pure_wins / len(pure_trades) * 100
        pure_avg_win = pure_trades[pure_trades['pure_bar_pnl'] > 0]['pure_bar_pnl'].mean() if pure_wins > 0 else 0
        pure_avg_loss = pure_trades[pure_trades['pure_bar_pnl'] < 0]['pure_bar_pnl'].mean() if pure_losses > 0 else 0

        print(f"4. PURE MATH TRADE QUALITY")
        print(f"   Wins: {pure_wins}/{len(pure_trades)} ({pure_win_rate:.1f}%)")
        print(f"   Avg win: {pure_avg_win:.6f}")
        print(f"   Avg loss: {pure_avg_loss:.6f}")
        if pure_avg_loss != 0:
            print(f"   Win/Loss ratio: {pure_avg_win / abs(pure_avg_loss):.2f}")

    if len(ml_trades) > 0:
        ml_wins = (ml_trades['ml_bar_pnl'] > 0).sum()
        ml_losses = (ml_trades['ml_bar_pnl'] < 0).sum()
        ml_win_rate = ml_wins / len(ml_trades) * 100
        ml_avg_win = ml_trades[ml_trades['ml_bar_pnl'] > 0]['ml_bar_pnl'].mean() if ml_wins > 0 else 0
        ml_avg_loss = ml_trades[ml_trades['ml_bar_pnl'] < 0]['ml_bar_pnl'].mean() if ml_losses > 0 else 0

        print(f"\n5. ML TRADE QUALITY")
        print(f"   Wins: {ml_wins}/{len(ml_trades)} ({ml_win_rate:.1f}%)")
        print(f"   Avg win: {ml_avg_win:.6f}")
        print(f"   Avg loss: {ml_avg_loss:.6f}")
        if ml_avg_loss != 0:
            print(f"   Win/Loss ratio: {ml_avg_win / abs(ml_avg_loss):.2f}")

    # 5. Find which trades Pure took that ML skipped
    print(f"\n6. TRADES PURE TOOK THAT ML SKIPPED")
    pure_only = pure_trades[~pure_trades.index.isin(ml_trades.index)].copy()
    if len(pure_only) > 0:
        print(f"   Count: {len(pure_only)}")
        print(f"   Total PnL from these: {pure_only['pure_bar_pnl'].sum():.6f}")
        print(f"   Avg PnL: {pure_only['pure_bar_pnl'].mean():.6f}")
        print(f"   Top 5 by PnL:")
        print(f"   {'Bar':>6} {'Dir':>5} {'Return':>10} {'PnL':>12} {'Confidence':>12} {'Hit Rate':>10}")
        print("   " + "-"*65)
        for _, row in pure_only.nlargest(5, 'pure_bar_pnl').iterrows():
            print(f"   {int(row['bar_idx']):>6} {int(row['pure_dir']):>5} {row['next_return']:>10.6f} "
                  f"{row['pure_bar_pnl']:>12.6f} {row['ml_confidence']:>12.3f} {row['hit_rate']:>10.2%}")
    else:
        print(f"   None - ML took all of Pure's trades")

    # 6. Find which trades ML took that Pure skipped
    print(f"\n7. TRADES ML TOOK THAT PURE SKIPPED")
    ml_only = ml_trades[~ml_trades.index.isin(pure_trades.index)].copy()
    if len(ml_only) > 0:
        print(f"   Count: {len(ml_only)}")
        print(f"   Total PnL from these: {ml_only['ml_bar_pnl'].sum():.6f}")
        print(f"   Avg PnL: {ml_only['ml_bar_pnl'].mean():.6f}")
        print(f"   Bottom 5 by PnL:")
        print(f"   {'Bar':>6} {'Dir':>5} {'Return':>10} {'PnL':>12} {'Confidence':>12} {'Hit Rate':>10}")
        print("   " + "-"*65)
        for _, row in ml_only.nsmallest(5, 'ml_bar_pnl').iterrows():
            print(f"   {int(row['bar_idx']):>6} {int(row['ml_dir']):>5} {row['next_return']:>10.6f} "
                  f"{row['ml_bar_pnl']:>12.6f} {row['ml_confidence']:>12.3f} {row['hit_rate']:>10.2%}")
    else:
        print(f"   None - Pure took all of ML's trades")

    # 7. Confidence analysis for ML
    print(f"\n8. ML CONFIDENCE ANALYSIS")
    ml_active = df[df['ml_dir'] != 0].copy()
    if len(ml_active) > 0:
        print(f"   Bars where ML wanted to trade: {len(ml_active)}")
        print(f"   Avg confidence when trading: {ml_active['ml_confidence'].mean():.3f}")
        print(f"   Min confidence: {ml_active['ml_confidence'].min():.3f}")
        print(f"   Max confidence: {ml_active['ml_confidence'].max():.3f}")

        # Correlation between confidence and PnL
        if len(ml_trades) > 0:
            corr = ml_trades[['ml_confidence', 'ml_bar_pnl']].corr().iloc[0, 1]
            print(f"   Correlation(confidence, PnL): {corr:.3f}")

    # 8. Summary
    print(f"\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    pure_final = df['pure_cum_pnl'].iloc[-1] if len(df) > 0 else 0
    ml_final = df['ml_cum_pnl'].iloc[-1] if len(df) > 0 else 0

    print(f"\nFinal Cumulative PnL:")
    print(f"  Pure Math: {pure_final:>12.6f} ({pure_final*100:>6.2f}%)")
    print(f"  ML:        {ml_final:>12.6f} ({ml_final*100:>6.2f}%)")
    print(f"  Difference: {ml_final - pure_final:>11.6f} ({(ml_final - pure_final)*100:>6.2f}%)")

    if ml_final < pure_final:
        print(f"\n⚠ ML UNDERPERFORMED by {abs(ml_final - pure_final)*100:.2f}%")
        print("\nLikely causes:")
        if len(ml_only) > 0 and ml_only['ml_bar_pnl'].mean() < 0:
            print(f"  - ML took {len(ml_only)} extra trades that Pure skipped")
            print(f"  - These extra trades averaged {ml_only['ml_bar_pnl'].mean():.6f} PnL")
            print(f"  - ML may be overconfident and trading too aggressively")
        if len(pure_only) > 0 and pure_only['pure_bar_pnl'].sum() > 0:
            print(f"  - Pure took {len(pure_only)} trades that ML skipped")
            print(f"  - ML missed {pure_only['pure_bar_pnl'].sum():.6f} in PnL")
            print(f"  - ML may be too conservative on high-conviction setups")
    else:
        print(f"\n✓ ML OUTPERFORMED by {abs(ml_final - pure_final)*100:.2f}%")

    print("\n" + "="*80 + "\n")

    return df


def main():
    """Run detailed ML validation."""

    print("="*80)
    print("ML COMBINER LAYER - DETAILED TRADE ANALYSIS")
    print("="*80)
    print(f"\nValidation run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load configuration
    config = load_config()
    K = config['segments']['K']
    encoding = config['symplectic']['encoding']
    cost_log = config['costs'].get('cost_per_trade', -0.00048)
    num_clusters = config['clustering']['num_clusters']
    min_cluster_size = config['clustering']['min_cluster_size']
    base_b = config['ultrametric']['base_b']
    bar_size_minutes = config.get('timeframe', {}).get('bar_size_minutes', 15)
    ml_type = config.get('ml', {}).get('type', 'classifier')

    print(f"Configuration: {bar_size_minutes}min bars, K={K}, encoding={encoding}, cost={cost_log:.6f}")

    # Load data
    data_path = Path("data/sample_data_template.csv")
    if not data_path.exists():
        print(f"\n⚠ Data not found at {data_path}")
        print("Please run: python download_qqq_1m_max.py")
        return

    print(f"\nLoading data from {data_path}...")
    df = load_ohlcv_csv(str(data_path))

    # Resample
    if bar_size_minutes == 1:
        df_resampled = df
    elif bar_size_minutes == 15:
        df_resampled = resample_to_15m(df)
    else:
        df_resampled = df.resample(f'{bar_size_minutes}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    print(f"✓ Resampled to {len(df_resampled)} {bar_size_minutes}min bars")

    # Compute phase-space
    p = compute_log_price(df_resampled)
    v = compute_smoothed_volume(
        df_resampled,
        normalization_window=config['volume']['normalization_window'],
        ema_period=config['volume']['ema_period']
    )
    gamma = build_gamma(p, v)
    segments = build_segments(gamma, K)

    # Train/test split (use only test set for this analysis)
    train_frac = config['backtest']['train_split']
    val_frac = config['backtest']['val_split']

    train_end = int(len(p) * train_frac)
    val_end = int(len(p) * (train_frac + val_frac))

    p_train = p[:train_end]
    v_train = v[:train_end]
    p_test = p[val_end:]
    v_test = v[val_end:]

    seg_train_end = train_end - K + 1
    segments_train = segments[:seg_train_end]

    print(f"✓ Using test set: {len(p_test)} bars")

    # Clustering & training
    print("\nClustering training data...")
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
    print(f"✓ Found {len(centroids)} clusters")

    labels_train = assign_to_nearest_centroid(segments_train, centroids, base_b)
    persistence = compute_persistence(labels_train)

    # Estimate κ
    kappa_global = estimate_global_kappa(segments_train, encoding=encoding)
    kappa_per_cluster = estimate_kappa_per_cluster(
        segments_train,
        labels_train,
        encoding=encoding
    )

    # Compute cluster stats
    cluster_stats = compute_hit_rates_from_data(segments_train, labels_train, p_train, config)
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

    print(f"✓ Cluster stats computed")

    # Create models
    print("\nCreating models...")
    pure_model = SymplecticUltrametricModel(
        config, centroids, kappa_per_cluster, cluster_stats, encoding=encoding
    )

    ml_model = HybridMLModel(
        config=config,
        base_model=pure_model,
        ml_type=ml_type
    )

    print("✓ Models created")

    # Run detailed comparison
    df_trades = detailed_backtest_comparison(
        pure_model=pure_model,
        ml_model=ml_model,
        p=p_test,
        v=v_test,
        K=K,
        cluster_stats_dict=cluster_stats_dict,
        cost_log=cost_log,
        warmup_bars=250
    )

    # Analyze differences
    analyze_trade_differences(df_trades)

    # Save detailed data
    output_path = "validation_ml_detailed_trades.csv"
    df_trades.to_csv(output_path, index=False)
    print(f"✓ Detailed trade data saved to {output_path}")

    print("\n" + "="*80)
    print("DETAILED VALIDATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
