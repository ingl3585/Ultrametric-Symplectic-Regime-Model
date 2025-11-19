"""
Ultrametric distance computation for regime detection.

The ultrametric distance measures "at what scale do two patterns first differ?"
- Segments that match on large structure but differ in fine details → small distance
- Segments that diverge early on large moves → large distance

This distance has the strong triangle inequality property:
    d(x, z) <= max(d(x, y), d(y, z))

which makes it suitable for hierarchical clustering and creates a natural
tree structure over market regimes.
"""

import numpy as np
from math import floor, log


def ultrametric_dist(
    seg1: np.ndarray,
    seg2: np.ndarray,
    base_b: float = 2.0,
    eps: float = 1e-10
) -> float:
    """
    Compute ultrametric distance between two K-bar segments.

    Algorithm:
    1. For each bar in the segments, compute the norm: sqrt(p^2 + v^2)
    2. Compute valuation: floor(log_b(max(norm, eps)))
    3. Find the first index j where valuations differ
    4. Distance = base_b^(-j) if they differ, else 0

    The valuation represents the "scale" of each bar - larger moves have
    higher valuations. The distance is determined by the earliest bar where
    the scales differ.

    Args:
        seg1: First segment, shape (K, 2) with [p, v]
        seg2: Second segment, shape (K, 2) with [p, v]
        base_b: Base for valuation (default 2.0)
        eps: Small value to avoid log(0) (default 1e-10)

    Returns:
        distance: Ultrametric distance in range [0, infinity)
                  0 means identical valuations at all bars
                  Higher values mean divergence at earlier bars

    Example:
        >>> seg1 = np.array([[1.0, 0.5], [1.1, 0.6]])
        >>> seg2 = np.array([[1.0, 0.5], [1.2, 0.7]])
        >>> d = ultrametric_dist(seg1, seg2)
        >>> d > 0  # Different at second bar
        True
    """
    # Validate inputs
    if seg1.shape != seg2.shape:
        raise ValueError(f"Segments must have same shape, got {seg1.shape} and {seg2.shape}")

    if seg1.ndim != 2 or seg1.shape[1] != 2:
        raise ValueError(f"Segments must be (K, 2), got shape {seg1.shape}")

    K = seg1.shape[0]

    # Step 1: Compute norms per bar
    # norm_i = sqrt(p_i^2 + v_i^2)
    norms1 = np.sqrt(seg1[:, 0]**2 + seg1[:, 1]**2)
    norms2 = np.sqrt(seg2[:, 0]**2 + seg2[:, 1]**2)

    # Step 2: Compute valuations
    # val_i = floor(log_b(max(norm_i, eps)))
    def compute_valuation(norm: float, base: float, epsilon: float) -> int:
        """Compute valuation for a single norm."""
        val = max(norm, epsilon)
        return floor(log(val) / log(base))

    valuations1 = np.array([compute_valuation(n, base_b, eps) for n in norms1])
    valuations2 = np.array([compute_valuation(n, base_b, eps) for n in norms2])

    # Step 3: Find first index where valuations differ
    differences = valuations1 != valuations2

    if not np.any(differences):
        # All valuations match → distance is 0
        return 0.0

    # Find first differing index
    first_diff_idx = np.argmax(differences)  # argmax returns first True

    # Step 4: Compute distance = base_b^(-j)
    distance = base_b ** (-first_diff_idx)

    return float(distance)


def compute_norm(point: np.ndarray) -> float:
    """
    Compute Euclidean norm of a point in phase space.

    Args:
        point: Array of shape (2,) with [p, v]

    Returns:
        norm: sqrt(p^2 + v^2)
    """
    return float(np.sqrt(np.sum(point**2)))


def ultrametric_dist_matrix(
    segments: np.ndarray,
    base_b: float = 2.0,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute pairwise ultrametric distance matrix for a set of segments.

    Warning: This is O(M^2 * K) and can be slow for large M.
    Consider using a subsample for clustering.

    Args:
        segments: Array of shape (M, K, 2)
        base_b: Base for valuation
        eps: Small value for numerical stability

    Returns:
        dist_matrix: Symmetric matrix of shape (M, M) with distances
    """
    M = len(segments)
    dist_matrix = np.zeros((M, M))

    for i in range(M):
        for j in range(i+1, M):
            d = ultrametric_dist(segments[i], segments[j], base_b, eps)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d  # Symmetric

    return dist_matrix


def condensed_distance_matrix(
    segments: np.ndarray,
    base_b: float = 2.0,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute condensed (upper triangular) distance matrix for scipy clustering.

    This is the format expected by scipy.cluster.hierarchy.linkage().

    Args:
        segments: Array of shape (M, K, 2)
        base_b: Base for valuation
        eps: Small value for numerical stability

    Returns:
        condensed: 1D array of length M*(M-1)/2 with upper triangular distances
    """
    M = len(segments)
    n_pairs = M * (M - 1) // 2

    condensed = np.zeros(n_pairs)
    idx = 0

    for i in range(M):
        for j in range(i+1, M):
            condensed[idx] = ultrametric_dist(segments[i], segments[j], base_b, eps)
            idx += 1

    return condensed
