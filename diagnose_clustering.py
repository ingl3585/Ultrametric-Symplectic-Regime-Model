#!/usr/bin/env python3
"""
Diagnose why ultrametric clustering is failing.
"""

import yaml
import numpy as np
from pathlib import Path

from model.data_utils import (
    load_ohlcv_csv, resample_to_15m,
    compute_log_price, compute_smoothed_volume, build_gamma
)
from model.trainer import build_segments
from model.ultrametric import condensed_distance_matrix
from scipy.cluster.hierarchy import linkage, fcluster

# Load config
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load data
df = load_ohlcv_csv("data/sample_data_template.csv")
df_15m = resample_to_15m(df)
p = compute_log_price(df_15m)
v = compute_smoothed_volume(
    df_15m,
    config['volume']['normalization_window'],
    config['volume']['ema_period']
)
gamma = build_gamma(p, v)
segments = build_segments(gamma, config['segments']['K'])

print(f"Total segments: {len(segments)}")

# Sample for distance matrix (use subset for speed)
n_sample = min(500, len(segments))
indices = np.linspace(0, len(segments)-1, n_sample, dtype=int)
segments_sample = segments[indices]

print(f"\n1. Computing distance matrix for {n_sample} segments...")
base_b = config['ultrametric']['base_b']
condensed_dists = condensed_distance_matrix(
    segments_sample,
    base_b=base_b,
    eps=config['ultrametric']['eps']
)

print(f"\nDistance statistics:")
print(f"  Min:     {condensed_dists.min():.6f}")
print(f"  Max:     {condensed_dists.max():.6f}")
print(f"  Mean:    {condensed_dists.mean():.6f}")
print(f"  Median:  {np.median(condensed_dists):.6f}")
print(f"  Std:     {condensed_dists.std():.6f}")
print(f"  Zeros:   {np.sum(condensed_dists == 0)} / {len(condensed_dists)} ({np.sum(condensed_dists == 0)/len(condensed_dists)*100:.1f}%)")

# Check distribution
print(f"\nDistance distribution:")
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(condensed_dists, p)
    print(f"  {p:2d}th percentile: {val:.6f}")

# Try different linkage methods
print(f"\n2. Testing different linkage methods (target: {config['clustering']['num_clusters']} clusters):\n")

for method in ['ward', 'average', 'complete', 'single']:
    try:
        Z = linkage(condensed_dists, method=method)
        labels = fcluster(Z, config['clustering']['num_clusters'], criterion='maxclust')
        labels = labels - 1

        unique, counts = np.unique(labels, return_counts=True)

        print(f"  {method:10s}: {len(unique)} clusters, sizes = {sorted(counts, reverse=True)[:5]}")

        # Check how balanced
        largest = counts.max()
        smallest = counts.min()
        ratio = largest / smallest if smallest > 0 else np.inf
        print(f"               Balance ratio (largest/smallest): {ratio:.1f}")

    except Exception as e:
        print(f"  {method:10s}: FAILED - {e}")

# Try different num_clusters
print(f"\n3. Testing different number of clusters (with ward linkage):\n")

for n_clust in [3, 4, 5, 6, 8, 10]:
    Z = linkage(condensed_dists, method='ward')
    labels = fcluster(Z, n_clust, criterion='maxclust')
    labels = labels - 1

    unique, counts = np.unique(labels, return_counts=True)
    print(f"  {n_clust} clusters: actual={len(unique)}, sizes={sorted(counts, reverse=True)[:8]}")

print(f"\n4. Recommendation:")

# Check if distances are too uniform
unique_dists = len(np.unique(condensed_dists))
if unique_dists < 10:
    print("  ⚠ Very few unique distance values!")
    print(f"    Only {unique_dists} unique distances")
    print("  → Try smaller base_b (e.g., 1.2) for finer granularity")
elif condensed_dists.std() < 0.1:
    print("  ⚠ Distances have very low variance!")
    print(f"    Std dev: {condensed_dists.std():.6f}")
    print("  → Try different base_b or check data variation")
else:
    print("  ✓ Distance distribution looks reasonable")
    print("  → Try 'average' or 'complete' linkage instead of 'ward'")
    print("  → Or reduce num_clusters to 4-6")
