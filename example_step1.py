#!/usr/bin/env python3
"""
STEP 1 Example – Segments + Ultrametric Distance

This script demonstrates:
1. Building K-bar segments from phase-space data
2. Computing ultrametric distances between segments
3. Validating distance properties
4. Visualizing a small distance matrix

Note: We still use AR(1) for trading in this step.
The ultrametric distance is just being tested, not used for signals yet.

Usage:
    python example_step1.py
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
from model.trainer import build_segments, validate_segments
from model.ultrametric import (
    ultrametric_dist,
    ultrametric_dist_matrix,
    condensed_distance_matrix
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Run STEP 1 demonstration."""
    print("\n" + "="*70)
    print("STEP 1: Segments + Ultrametric Distance Demonstration")
    print("="*70 + "\n")

    # Load configuration
    print("Loading configuration...")
    config = load_config()
    K = config['segments']['K']
    base_b = config['ultrametric']['base_b']
    eps = config['ultrametric']['eps']
    print(f"✓ Configuration loaded")
    print(f"  Segment length (K): {K}")
    print(f"  Ultrametric base (b): {base_b}")
    print(f"  Epsilon: {eps}")

    # Check if data exists
    data_path = Path("data/sample_data_template.csv")
    if not data_path.exists():
        print(f"\n⚠ Data not found at {data_path}")
        print("Run example_step0.py first to verify the pipeline works.")
        return

    # Load and process data
    print(f"\nLoading data from {data_path}...")
    df = load_ohlcv_csv(str(data_path))
    df_15m = resample_to_15m(df)
    print(f"✓ Loaded {len(df_15m)} 15-minute bars")

    # Compute phase-space variables
    print("\nComputing phase-space data...")
    p = compute_log_price(df_15m)
    v = compute_smoothed_volume(
        df_15m,
        normalization_window=config['volume']['normalization_window'],
        ema_period=config['volume']['ema_period']
    )
    gamma = build_gamma(p, v)
    print(f"✓ Gamma shape: {gamma.shape}")

    # Build segments
    print(f"\nBuilding {K}-bar segments...")
    segments = build_segments(gamma, K)
    print(f"✓ Built {len(segments)} segments")
    print(f"  Segment shape: {segments.shape}")
    print(f"  (num_segments, bars_per_segment, features)")

    # Validate segments
    try:
        validate_segments(segments, K)
        print("✓ Segments validated successfully")
    except ValueError as e:
        print(f"✗ Segment validation failed: {e}")
        return

    # Show sample segment
    print(f"\nSample segment (first segment):")
    print("  Bar | Log Price | Norm Volume")
    print("  " + "-"*35)
    for i in range(min(K, len(segments[0]))):
        print(f"  {i:3d} | {segments[0][i, 0]:9.4f} | {segments[0][i, 1]:11.4f}")

    # Compute distances between first few segments
    print(f"\nComputing ultrametric distances...")
    print("Distance matrix for first 10 segments:")
    print("(Row/Col indices, lower = more similar)")
    print()

    # Take first 10 segments for demonstration
    n_demo = min(10, len(segments))
    demo_segments = segments[:n_demo]

    dist_matrix = ultrametric_dist_matrix(demo_segments, base_b, eps)

    # Print matrix header
    print("     ", end="")
    for j in range(n_demo):
        print(f"{j:7d}", end="")
    print()
    print("     " + "-" * (7 * n_demo))

    # Print matrix rows
    for i in range(n_demo):
        print(f"{i:3d} |", end="")
        for j in range(n_demo):
            if i == j:
                print("   -   ", end="")
            else:
                print(f"{dist_matrix[i, j]:7.4f}", end="")
        print()

    # Analyze distance distribution
    print(f"\nDistance statistics (excluding diagonal):")
    non_diag = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    if len(non_diag) > 0:
        print(f"  Min:    {non_diag.min():.6f}")
        print(f"  Max:    {non_diag.max():.6f}")
        print(f"  Mean:   {non_diag.mean():.6f}")
        print(f"  Median: {np.median(non_diag):.6f}")
        print(f"  Std:    {non_diag.std():.6f}")

    # Find most similar and most different pairs
    if len(non_diag) > 0:
        # Most similar
        min_idx = np.argmin(dist_matrix + np.eye(n_demo) * 1e10)
        i_min, j_min = min_idx // n_demo, min_idx % n_demo
        print(f"\nMost similar pair: segments {i_min} and {j_min}")
        print(f"  Distance: {dist_matrix[i_min, j_min]:.6f}")

        # Most different
        max_idx = np.argmax(dist_matrix)
        i_max, j_max = max_idx // n_demo, max_idx % n_demo
        print(f"\nMost different pair: segments {i_max} and {j_max}")
        print(f"  Distance: {dist_matrix[i_max, j_max]:.6f}")

    # Demonstrate condensed distance matrix for scipy
    print(f"\nCondensed distance vector for scipy clustering:")
    condensed = condensed_distance_matrix(demo_segments, base_b, eps)
    print(f"  Length: {len(condensed)} (expected: {n_demo*(n_demo-1)//2})")
    print(f"  First 10 values: {condensed[:10]}")

    # Property checks
    print(f"\nValidating ultrametric properties on sample segments...")

    # 1. Self-distance
    seg_sample = segments[0]
    self_dist = ultrametric_dist(seg_sample, seg_sample, base_b, eps)
    print(f"  Self-distance: {self_dist:.10f} (should be 0)")

    # 2. Symmetry
    if len(segments) > 1:
        d_01 = ultrametric_dist(segments[0], segments[1], base_b, eps)
        d_10 = ultrametric_dist(segments[1], segments[0], base_b, eps)
        print(f"  Symmetry: d(0,1)={d_01:.6f}, d(1,0)={d_10:.6f} (should match)")

    # 3. Ultrametric inequality (sample check)
    if len(segments) > 2:
        d_02 = ultrametric_dist(segments[0], segments[2], base_b, eps)
        d_01 = ultrametric_dist(segments[0], segments[1], base_b, eps)
        d_12 = ultrametric_dist(segments[1], segments[2], base_b, eps)
        max_dist = max(d_01, d_12)
        inequality_holds = d_02 <= max_dist + 1e-9
        print(f"  Ultrametric inequality: d(0,2)={d_02:.6f} <= max({d_01:.6f}, {d_12:.6f}) = {max_dist:.6f}")
        print(f"    {'✓ Holds' if inequality_holds else '✗ Violated'}")

    print("\n" + "="*70)
    print("STEP 1 demonstration complete!")
    print("\nKey Observations:")
    print("  - Segments capture K-bar 'shapes' in [price, volume] space")
    print("  - Ultrametric distance measures scale of first difference")
    print("  - Distance = 0 when valuations match at all bars")
    print("  - Satisfies strong triangle inequality")
    print("\nNext: Run tests/test_ultrametric.py to validate all properties")
    print("      Then proceed to STEP 2 for clustering")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
