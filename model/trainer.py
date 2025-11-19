"""
Training orchestration and segment building utilities.

This module handles:
- Building K-bar segments from phase-space data
- Orchestrating training flows (segments → clustering → kappa estimation)
- Preparing data for regime detection and symplectic models
"""

import numpy as np
from typing import Tuple


def build_segments(gamma: np.ndarray, K: int) -> np.ndarray:
    """
    Build sliding K-bar segments from phase-space data.

    Creates overlapping windows of K consecutive bars, where each segment
    captures a "shape" in [price, volume] space.

    Args:
        gamma: Phase-space array, shape (N, 2) with columns [p, v]
               where p = log price, v = smoothed normalized volume
        K: Number of bars per segment (e.g., 10)

    Returns:
        segments: Array of shape (N-K+1, K, 2)
                  where segments[i] = gamma[i:i+K]

    Example:
        >>> gamma = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> segments = build_segments(gamma, K=2)
        >>> segments.shape
        (3, 2, 2)
        >>> segments[0]
        array([[1, 2],
               [3, 4]])
    """
    N = len(gamma)

    if K > N:
        raise ValueError(f"K ({K}) cannot be larger than data length ({N})")

    if K < 1:
        raise ValueError(f"K must be at least 1, got {K}")

    # Number of segments
    num_segments = N - K + 1

    # Preallocate segments array
    segments = np.zeros((num_segments, K, 2))

    # Build sliding windows
    for i in range(num_segments):
        segments[i] = gamma[i:i+K]

    return segments


def get_segment_at_index(gamma: np.ndarray, K: int, idx: int) -> np.ndarray:
    """
    Extract a single K-bar segment ending at index idx.

    Useful for online inference where you want the segment ending at the
    current bar.

    Args:
        gamma: Phase-space array, shape (N, 2)
        K: Segment length
        idx: End index (inclusive) for the segment

    Returns:
        segment: Array of shape (K, 2) containing gamma[idx-K+1:idx+1]

    Raises:
        ValueError: If idx < K-1 (not enough history)
    """
    if idx < K - 1:
        raise ValueError(f"Not enough history: need K={K} bars, but idx={idx}")

    return gamma[idx-K+1:idx+1]


def validate_segments(segments: np.ndarray, K: int) -> bool:
    """
    Validate segment array shape and properties.

    Args:
        segments: Segment array to validate
        K: Expected segment length

    Returns:
        True if valid

    Raises:
        ValueError: If segments are invalid
    """
    if segments.ndim != 3:
        raise ValueError(f"Segments must be 3D, got shape {segments.shape}")

    M, K_actual, D = segments.shape

    if K_actual != K:
        raise ValueError(f"Expected segment length {K}, got {K_actual}")

    if D != 2:
        raise ValueError(f"Expected 2 features [p, v], got {D}")

    if not np.all(np.isfinite(segments)):
        raise ValueError("Segments contain NaN or Inf values")

    return True
