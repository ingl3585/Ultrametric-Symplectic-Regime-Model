#!/usr/bin/env python3
"""
STEP 2 Example – Clustering + Regime Persistence

This script demonstrates:
1. Clustering segments using ultrametric distance
2. Computing regime persistence for each cluster
3. Comparing against baselines (random, k-means, volatility)
4. Evaluating regime quality

Note: We still use AR(1) for trading in this step.
The clustering is just being tested, not used for signals yet.

Usage:
    python example_step2.py
"""

import yaml
import numpy as np
from pathlib import Path

from model.data_utils import (
    load_ohlcv_csv,
    resample_to_15m,
    compute_log_price,
    compute_smoothed_volume,
    build_gamma
)
from model.trainer import build_segments
from model.clustering import (
    cluster_segments_ultrametric,
    compute_centroids,
    compute_persistence,
    compute_cluster_stats,
    assign_to_nearest_centroid,
    cluster_random,
    cluster_kmeans,
    cluster_volatility
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_persistence_comparison(results: dict):
    """Pretty print persistence comparison across methods."""
    print("\n" + "="*70)
    print("Regime Persistence Comparison")
    print("="*70)
    print(f"{'Method':<25} {'Avg Persistence':>15} {'# Clusters':>12} {'Best':>12}")
    print("-"*70)

    for method_name, data in results.items():
        avg_pers = data['avg_persistence']
        num_clusters = data['num_clusters']
        max_pers = data['max_persistence']
        print(f"{method_name:<25} {avg_pers:>14.2%} {num_clusters:>12} {max_pers:>11.2%}")

    print("="*70)


def main():
    """Run STEP 2 demonstration."""
    print("\n" + "="*70)
    print("STEP 2: Clustering + Regime Persistence Demonstration")
    print("="*70 + "\n")

    # Load configuration
    print("Loading configuration...")
    config = load_config()
    K = config['segments']['K']
    base_b = config['ultrametric']['base_b']
    eps = config['ultrametric']['eps']
    num_clusters = config['clustering']['num_clusters']
    min_cluster_size = config['clustering']['min_cluster_size']
    subsample_size = config['clustering']['subsample_size']

    print(f"✓ Configuration loaded")
    print(f"  Segment length (K): {K}")
    print(f"  Target clusters: {num_clusters}")
    print(f"  Min cluster size: {min_cluster_size}")
    print(f"  Subsample size: {subsample_size}")

    # Load data
    data_path = Path("data/sample_data_template.csv")
    if not data_path.exists():
        print(f"\n⚠ Data not found at {data_path}")
        return

    print(f"\nLoading data from {data_path}...")
    df = load_ohlcv_csv(str(data_path))
    df_15m = resample_to_15m(df)
    print(f"✓ Loaded {len(df_15m)} 15-minute bars")

    # Compute phase-space data
    print("\nComputing phase-space data...")
    p = compute_log_price(df_15m)
    v = compute_smoothed_volume(
        df_15m,
        normalization_window=config['volume']['normalization_window'],
        ema_period=config['volume']['ema_period']
    )
    gamma = build_gamma(p, v)

    # Build segments
    print(f"\nBuilding {K}-bar segments...")
    segments = build_segments(gamma, K)
    print(f"✓ Built {len(segments)} segments")

    # ========================================================================
    # Ultrametric Clustering
    # ========================================================================
    print("\n" + "-"*70)
    print("1. ULTRAMETRIC CLUSTERING")
    print("-"*70)

    labels_ultra_sub, Z = cluster_segments_ultrametric(
        segments,
        num_clusters=num_clusters,
        base_b=base_b,
        eps=eps,
        subsample_size=subsample_size
    )

    # Compute centroids from subsample
    print(f"\nComputing centroids...")
    if len(segments) > subsample_size:
        indices_sub = np.linspace(0, len(segments)-1, min(subsample_size, len(segments)), dtype=int)
        segments_sub = segments[indices_sub]
    else:
        segments_sub = segments

    centroids_ultra = compute_centroids(segments_sub, labels_ultra_sub, min_cluster_size=20)
    print(f"✓ Computed {len(centroids_ultra)} centroids")

    # Assign all segments to nearest centroid
    print(f"Assigning all {len(segments)} segments to nearest centroid...")
    labels_ultra = assign_to_nearest_centroid(segments, centroids_ultra, base_b, eps)

    # Recompute centroids with full data
    centroids_ultra = compute_centroids(segments, labels_ultra, min_cluster_size)
    print(f"✓ Final centroids: {len(centroids_ultra)} clusters")

    # Compute persistence
    persistence_ultra = compute_persistence(labels_ultra)
    stats_ultra = compute_cluster_stats(labels_ultra)

    print(f"\nUltrametric Cluster Summary:")
    for cluster_id in sorted(centroids_ultra.keys()):
        pers = persistence_ultra.get(cluster_id, 0.0)
        size = stats_ultra[cluster_id]['size']
        print(f"  Cluster {cluster_id}: size={size:4d}, persistence={pers:.2%}")

    avg_pers_ultra = np.mean(list(persistence_ultra.values()))
    print(f"\nAverage persistence: {avg_pers_ultra:.2%}")

    # ========================================================================
    # Baseline 1: Random Clustering
    # ========================================================================
    print("\n" + "-"*70)
    print("2. RANDOM CLUSTERING (Baseline)")
    print("-"*70)

    np.random.seed(42)
    labels_random = cluster_random(segments, num_clusters)
    persistence_random = compute_persistence(labels_random)
    avg_pers_random = np.mean(list(persistence_random.values()))

    print(f"Random clustering persistence: {avg_pers_random:.2%}")

    # ========================================================================
    # Baseline 2: K-means (Euclidean)
    # ========================================================================
    print("\n" + "-"*70)
    print("3. K-MEANS CLUSTERING (Euclidean Baseline)")
    print("-"*70)

    labels_kmeans = cluster_kmeans(segments, num_clusters, random_state=42)
    persistence_kmeans = compute_persistence(labels_kmeans)
    avg_pers_kmeans = np.mean(list(persistence_kmeans.values()))

    print(f"K-means clustering persistence: {avg_pers_kmeans:.2%}")

    # ========================================================================
    # Baseline 3: Volatility Regimes
    # ========================================================================
    print("\n" + "-"*70)
    print("4. VOLATILITY REGIMES (Baseline)")
    print("-"*70)

    labels_vol = cluster_volatility(p, K, num_buckets=num_clusters, window=20)
    persistence_vol = compute_persistence(labels_vol)
    avg_pers_vol = np.mean(list(persistence_vol.values()))

    print(f"Volatility regime persistence: {avg_pers_vol:.2%}")

    # ========================================================================
    # Comparison
    # ========================================================================
    results = {
        'Ultrametric': {
            'avg_persistence': avg_pers_ultra,
            'num_clusters': len(centroids_ultra),
            'max_persistence': max(persistence_ultra.values()) if persistence_ultra else 0.0
        },
        'Random': {
            'avg_persistence': avg_pers_random,
            'num_clusters': len(persistence_random),
            'max_persistence': max(persistence_random.values()) if persistence_random else 0.0
        },
        'K-means (Euclidean)': {
            'avg_persistence': avg_pers_kmeans,
            'num_clusters': len(persistence_kmeans),
            'max_persistence': max(persistence_kmeans.values()) if persistence_kmeans else 0.0
        },
        'Volatility Buckets': {
            'avg_persistence': avg_pers_vol,
            'num_clusters': len(persistence_vol),
            'max_persistence': max(persistence_vol.values()) if persistence_vol else 0.0
        }
    }

    print_persistence_comparison(results)

    # ========================================================================
    # Success Criteria Check
    # ========================================================================
    print("\n" + "="*70)
    print("Success Criteria Evaluation (from CLAUDE.md)")
    print("="*70)

    # Target: ultrametric persistence > 0.65
    target_persistence = 0.65
    print(f"\n1. Average Persistence:")
    print(f"   Target: > {target_persistence:.0%}")
    print(f"   Ultrametric: {avg_pers_ultra:.2%} {'✓' if avg_pers_ultra > target_persistence else '✗'}")

    # Better than random (~0.50)
    print(f"\n2. Better than Random:")
    print(f"   Random: {avg_pers_random:.2%}")
    print(f"   Improvement: {(avg_pers_ultra - avg_pers_random)*100:.1f} percentage points {'✓' if avg_pers_ultra > avg_pers_random + 0.1 else '✗'}")

    # Better than k-means (~0.55)
    print(f"\n3. Better than K-means:")
    print(f"   K-means: {avg_pers_kmeans:.2%}")
    print(f"   Improvement: {(avg_pers_ultra - avg_pers_kmeans)*100:.1f} percentage points {'✓' if avg_pers_ultra > avg_pers_kmeans + 0.05 else '✗'}")

    # At least 3 clusters with >= 100 segments and persistence >= 0.60
    print(f"\n4. High-Quality Clusters:")
    print(f"   Target: ≥3 clusters with size≥100 and persistence≥0.60")

    good_clusters = []
    for cluster_id in centroids_ultra.keys():
        size = stats_ultra[cluster_id]['size']
        pers = persistence_ultra[cluster_id]
        if size >= 100 and pers >= 0.60:
            good_clusters.append((cluster_id, size, pers))
            print(f"   Cluster {cluster_id}: size={size}, persistence={pers:.2%} ✓")

    print(f"   Found {len(good_clusters)} good clusters {'✓' if len(good_clusters) >= 3 else '✗'}")

    print("\n" + "="*70)
    print("STEP 2 demonstration complete!")
    print("\nKey Findings:")
    print(f"  - Ultrametric persistence: {avg_pers_ultra:.2%}")
    print(f"  - Beats random by: {(avg_pers_ultra - avg_pers_random)*100:.1f} pp")
    print(f"  - Beats k-means by: {(avg_pers_ultra - avg_pers_kmeans)*100:.1f} pp")
    print(f"  - High-quality clusters: {len(good_clusters)}")

    if avg_pers_ultra > 0.65 and len(good_clusters) >= 3:
        print("\n✓ SUCCESS: Regime detection looks promising!")
        print("  Ready to proceed to STEP 3 (Symplectic Global Model)")
    else:
        print("\n⚠ MARGINAL: Regimes are detectable but not strongly persistent")
        print("  Can proceed to STEP 3, but may want to tune clustering parameters")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
