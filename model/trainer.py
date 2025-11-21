"""
Training orchestration and segment building utilities.

This module handles:
- Building K-bar segments from phase-space data
- Orchestrating training flows (segments → clustering → kappa estimation)
- Preparing data for regime detection and symplectic models
- Computing cluster statistics for ML feature extraction
"""

import numpy as np
from typing import Tuple, Dict, Any


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


def compute_cluster_stats(
    segments: np.ndarray,
    labels: np.ndarray,
    config: dict,
    forecasts: np.ndarray = None,
    actual_returns: np.ndarray = None
) -> Dict[int, Dict[str, Any]]:
    """
    Compute cluster statistics for ML feature extraction.

    Computes per-cluster:
    - persistence: P(cluster_t+1 = c | cluster_t = c)
    - hit_rate: Directional accuracy (if forecasts provided)
    - size: Number of segments in cluster
    - raw_kappa: Cluster-specific kappa value

    Args:
        segments: Segment array, shape (M, K, 2)
        labels: Cluster labels, shape (M,)
        config: Configuration dict
        forecasts: Optional forecast values, shape (M,) for hit rate calculation
        actual_returns: Optional actual returns, shape (M,) for hit rate calculation

    Returns:
        cluster_stats: Dict mapping cluster_id to statistics dict
            {
                cluster_id: {
                    'persistence': float,
                    'hit_rate': float,
                    'size': int,
                    'raw_kappa': float
                },
                ...
            }
    """
    from .clustering import compute_persistence
    from .symplectic_model import estimate_kappa_per_cluster

    # Get unique clusters
    unique_clusters = np.unique(labels)
    cluster_stats = {}

    # Compute persistence for all clusters
    persistence_dict = compute_persistence(labels)

    # Compute raw kappa per cluster
    encoding = config.get('symplectic', {}).get('encoding', 'A')
    kappa_dict = estimate_kappa_per_cluster(segments, labels, encoding=encoding)

    # Compute hit rate if forecasts and actual returns provided
    hit_rate_dict = {}
    if forecasts is not None and actual_returns is not None:
        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            if mask.sum() > 0:
                cluster_forecasts = forecasts[mask]
                cluster_actuals = actual_returns[mask]

                # Directional hit rate
                forecast_direction = np.sign(cluster_forecasts)
                actual_direction = np.sign(cluster_actuals)
                correct = (forecast_direction == actual_direction)
                hit_rate = np.mean(correct)
                hit_rate_dict[cluster_id] = hit_rate
            else:
                hit_rate_dict[cluster_id] = 0.5  # Default

    # Build cluster stats dict
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        size = int(np.sum(mask))

        cluster_stats[cluster_id] = {
            'persistence': persistence_dict.get(cluster_id, 0.5),
            'hit_rate': hit_rate_dict.get(cluster_id, 0.5) if hit_rate_dict else 0.5,
            'size': size,
            'raw_kappa': kappa_dict.get(cluster_id, 0.01)
        }

    return cluster_stats


def compute_hit_rates_from_data(
    segments: np.ndarray,
    labels: np.ndarray,
    p_full: np.ndarray,
    config: dict
) -> Dict[int, float]:
    """
    Compute directional hit rates per cluster using symplectic forecasts.

    For each segment, computes 1-step forecast and compares to actual return.

    Args:
        segments: Segment array, shape (M, K, 2)
        labels: Cluster labels, shape (M,)
        p_full: Full log price series, shape (N,) where N >= M+K
        config: Configuration dict

    Returns:
        hit_rates: Dict mapping cluster_id to directional hit rate
    """
    from .symplectic_model import extract_state_from_segment, leapfrog_step, estimate_kappa_per_cluster

    M = len(segments)
    K = segments.shape[1]

    # Compute kappa per cluster
    encoding = config.get('symplectic', {}).get('encoding', 'A')
    kappa_dict = estimate_kappa_per_cluster(segments, labels, encoding=encoding)

    # Get global kappa fallback
    kappa_global = np.mean(list(kappa_dict.values()))

    # Compute forecasts and actual returns
    forecasts = []
    actuals = []
    forecast_labels = []

    for i in range(M):
        segment = segments[i]
        cluster_id = labels[i]

        # Extract state
        q, pi = extract_state_from_segment(segment, encoding=encoding)

        # Get kappa
        kappa = kappa_dict.get(cluster_id, kappa_global)

        # Forecast next momentum
        _, pi_next = leapfrog_step(q, pi, kappa, dt=1.0)
        forecasts.append(pi_next)
        forecast_labels.append(cluster_id)

        # Actual return (if available)
        t = i + K  # Bar index after segment
        if t < len(p_full):
            actual_return = p_full[t] - p_full[t - 1]
            actuals.append(actual_return)
        else:
            actuals.append(0.0)

    forecasts = np.array(forecasts)
    actuals = np.array(actuals)
    forecast_labels = np.array(forecast_labels)

    # Compute hit rate per cluster
    unique_clusters = np.unique(labels)
    hit_rates = {}

    for cluster_id in unique_clusters:
        mask = forecast_labels == cluster_id
        if mask.sum() > 0:
            cluster_forecasts = forecasts[mask]
            cluster_actuals = actuals[mask]

            # Directional accuracy
            forecast_direction = np.sign(cluster_forecasts)
            actual_direction = np.sign(cluster_actuals)
            correct = (forecast_direction == actual_direction)
            hit_rate = np.mean(correct)
            hit_rates[cluster_id] = float(hit_rate)
        else:
            hit_rates[cluster_id] = 0.5

    return hit_rates
