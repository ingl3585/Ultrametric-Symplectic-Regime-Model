#!/usr/bin/env python3
"""
Debug: Check what valuations are being computed.
"""

import yaml
import numpy as np
from math import floor, log

from model.data_utils import (
    load_ohlcv_csv, resample_to_15m,
    compute_log_price, compute_smoothed_volume, build_gamma
)
from model.trainer import build_segments

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

print("Phase-space data ranges:")
print(f"  Log price (p): [{p.min():.4f}, {p.max():.4f}], range = {p.max()-p.min():.4f}")
print(f"  Norm volume (v): [{v.min():.4f}, {v.max():.4f}], range = {v.max()-v.min():.4f}")

# Build segments
K = config['segments']['K']
segments = build_segments(gamma, K)

# Check first few segments
print(f"\nAnalyzing first segment (bars 0-{K-1}):")
seg = segments[0]

base_b = config['ultrametric']['base_b']
eps = config['ultrametric']['eps']

print("\nBar | p      | v      | norm      | valuation")
print("-" * 55)

for i in range(K):
    p_i = seg[i, 0]
    v_i = seg[i, 1]
    norm = np.sqrt(p_i**2 + v_i**2)
    val_float = log(max(norm, eps)) / log(base_b)
    val = floor(val_float)
    print(f"{i:3d} | {p_i:.4f} | {v_i:.4f} | {norm:.6f} | {val:3d} (raw: {val_float:.4f})")

# Check another segment far away
print(f"\nAnalyzing segment 500 (bars 500-{500+K-1}):")
seg = segments[500]

print("\nBar | p      | v      | norm      | valuation")
print("-" * 55)

for i in range(K):
    p_i = seg[i, 0]
    v_i = seg[i, 1]
    norm = np.sqrt(p_i**2 + v_i**2)
    val_float = log(max(norm, eps)) / log(base_b)
    val = floor(val_float)
    print(f"{i:3d} | {p_i:.4f} | {v_i:.4f} | {norm:.6f} | {val:3d} (raw: {val_float:.4f})")

# Check all norms across all segments
all_norms = []
all_valuations = []

for seg in segments[:100]:  # Sample first 100
    for i in range(K):
        norm = np.sqrt(seg[i, 0]**2 + seg[i, 1]**2)
        val_float = log(max(norm, eps)) / log(base_b)
        val = floor(val_float)
        all_norms.append(norm)
        all_valuations.append(val)

all_norms = np.array(all_norms)
all_valuations = np.array(all_valuations)

print(f"\n\nNorm statistics (first 100 segments, {len(all_norms)} points):")
print(f"  Min:    {all_norms.min():.6f}")
print(f"  Max:    {all_norms.max():.6f}")
print(f"  Mean:   {all_norms.mean():.6f}")
print(f"  Std:    {all_norms.std():.6f}")
print(f"  Range:  {all_norms.max() - all_norms.min():.6f}")

print(f"\nValuation statistics:")
print(f"  Unique valuations: {np.unique(all_valuations)}")
print(f"  Most common: {np.bincount(all_valuations - all_valuations.min()).argmax() + all_valuations.min()}")
print(f"  Counts: {np.bincount(all_valuations - all_valuations.min())}")

print("\n" + "="*55)
print("DIAGNOSIS:")
if len(np.unique(all_valuations)) == 1:
    print("  ⚠ ALL valuations are identical!")
    print("  This means base_b=2.0 is too coarse for this data.")
    print("  Solution: Use smaller base_b (e.g., 1.2 or 1.5)")
else:
    print("  ✓ Multiple valuations found")
    print(f"  Number of distinct levels: {len(np.unique(all_valuations))}")
print("="*55)
