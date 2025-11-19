#!/usr/bin/env python3
"""
Check ultrametric distances between non-adjacent segments.
"""

import yaml
import numpy as np
from pathlib import Path

from model.data_utils import (
    load_ohlcv_csv, resample_to_15m,
    compute_log_price, compute_smoothed_volume, build_gamma
)
from model.trainer import build_segments
from model.ultrametric import ultrametric_dist

# Load config
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load data
df = load_ohlcv_csv("data/sample_data_template.csv")
df_15m = resample_to_15m(df)
p = compute_log_price(df_15m)
v = compute_smoothed_volume(
    df_15m,
    normalization_window=config['volume']['normalization_window'],
    ema_period=config['volume']['ema_period']
)
gamma = build_gamma(p, v)

# Build segments
K = config['segments']['K']
segments = build_segments(gamma, K)

print(f"Total segments: {len(segments)}\n")

# Check distances between non-adjacent segments
print("Distances between non-adjacent segments:")
print("(Every 50th segment to avoid overlap)\n")

indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
indices = [i for i in indices if i < len(segments)]

base_b = config['ultrametric']['base_b']
eps = config['ultrametric']['eps']

# Print header
print("     ", end="")
for idx in indices:
    print(f"{idx:7d}", end="")
print("\n     " + "-" * (7 * len(indices)))

# Compute and print distances
for i, idx_i in enumerate(indices):
    print(f"{idx_i:3d} |", end="")
    for j, idx_j in enumerate(indices):
        if i == j:
            print("   -   ", end="")
        else:
            d = ultrametric_dist(segments[idx_i], segments[idx_j], base_b, eps)
            print(f"{d:7.4f}", end="")
    print()

# Statistics on non-adjacent pairs
print("\nStatistics for these non-adjacent segments:")
distances = []
for i, idx_i in enumerate(indices):
    for j, idx_j in enumerate(indices):
        if i < j:
            d = ultrametric_dist(segments[idx_i], segments[idx_j], base_b, eps)
            distances.append(d)

distances = np.array(distances)
print(f"  Min:    {distances.min():.6f}")
print(f"  Max:    {distances.max():.6f}")
print(f"  Mean:   {distances.mean():.6f}")
print(f"  Median: {np.median(distances):.6f}")
print(f"  Std:    {distances.std():.6f}")
print(f"  Non-zero: {np.sum(distances > 0)} / {len(distances)}")

# Check a random sample
print("\n\nRandom sample of 20 segment pairs:")
np.random.seed(42)
n_samples = min(20, len(segments) - 1)
for _ in range(n_samples):
    i = np.random.randint(0, len(segments))
    j = np.random.randint(0, len(segments))
    if i != j:
        d = ultrametric_dist(segments[i], segments[j], base_b, eps)
        print(f"  d(seg {i:4d}, seg {j:4d}) = {d:.6f}")
