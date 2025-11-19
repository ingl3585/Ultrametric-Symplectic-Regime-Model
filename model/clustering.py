"""
Clustering and regime detection using ultrametric distances.

This module handles:
- Hierarchical clustering on ultrametric distances
- Computing cluster centroids
- Measuring regime persistence
- Baseline comparisons (random, k-means, volatility)
"""

import numpy as np
from typing import Dict, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

from .ultrametric import condensed_distance_matrix


def cluster_segments_ultrametric(
    segments: np.ndarray,
    num_clusters: int,
    base_b: float = 1.3,
    eps: float = 1e-10,
    subsample_size: int = 5000,
    method: str = 'ward'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster segments using hierarchical clustering on ultrametric distances.

    Uses a subsample to keep computation tractable (O(M^2 * K) for full matrix).

    Args:
        segments: Array of shape (M, K, 2)
        num_clusters: Target number of clusters
        base_b: Base for ultrametric valuation
        eps: Epsilon for numerical stability
        subsample_size: Max segments to use for distance matrix
        method: Linkage method ('ward', 'average', 'complete', 'single')

    Returns:
        labels: Cluster labels for the subsample, shape (subsample_size,)
        Z: Linkage matrix from scipy
    """
    M = len(segments)

    # Subsample if needed
    if M > subsample_size:
        print(f"  Subsampling {subsample_size} of {M} segments for clustering...")
        indices = np.linspace(0, M-1, subsample_size, dtype=int)
        segments_sub = segments[indices]
    else:
        print(f"  Using all {M} segments for clustering...")
        segments_sub = segments
        indices = np.arange(M)

    # Compute condensed distance matrix
    print(f"  Computing {len(segments_sub)}x{len(segments_sub)} distance matrix...")
    condensed_dists = condensed_distance_matrix(segments_sub, base_b, eps)

    # Hierarchical clustering
    print(f"  Running hierarchical clustering (method={method})...")
    Z = linkage(condensed_dists, method=method)

    # Cut tree to get cluster labels
    labels = fcluster(Z, num_clusters, criterion='maxclust')

    # Convert to 0-indexed
    labels = labels - 1

    return labels, Z


def assign_to_nearest_centroid(
    segments: np.ndarray,
    centroids: Dict[int, np.ndarray],
    base_b: float = 1.3,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Assign all segments to nearest centroid using ultrametric distance.

    Args:
        segments: Array of shape (M, K, 2)
        centroids: Dict mapping cluster_id -> centroid_segment
        base_b: Base for ultrametric valuation
        eps: Epsilon for numerical stability

    Returns:
        labels: Cluster labels for all segments, shape (M,)
                -1 if no valid centroid found
    """
    from .ultrametric import ultrametric_dist

    M = len(segments)
    labels = np.full(M, -1, dtype=int)

    for i in range(M):
        min_dist = np.inf
        best_cluster = -1

        for cluster_id, centroid in centroids.items():
            d = ultrametric_dist(segments[i], centroid, base_b, eps)
            if d < min_dist:
                min_dist = d
                best_cluster = cluster_id

        labels[i] = best_cluster

    return labels


def compute_centroids(
    segments: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int = 50
) -> Dict[int, np.ndarray]:
    """
    Compute centroid (mean segment) for each cluster.

    Only includes clusters with size >= min_cluster_size.

    Args:
        segments: Array of shape (M, K, 2)
        labels: Cluster labels, shape (M,)
        min_cluster_size: Minimum segments per valid cluster

    Returns:
        centroids: Dict mapping cluster_id -> centroid_segment of shape (K, 2)
    """
    centroids = {}
    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip invalid labels
            continue

        # Get all segments in this cluster
        mask = labels == cluster_id
        cluster_segments = segments[mask]

        # Check size threshold
        if len(cluster_segments) < min_cluster_size:
            continue

        # Compute mean segment (centroid)
        centroid = np.mean(cluster_segments, axis=0)
        centroids[cluster_id] = centroid

    return centroids


def compute_persistence(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute regime persistence for each cluster.

    Persistence = P(cluster_t+1 = c | cluster_t = c)

    Args:
        labels: Time-ordered cluster labels, shape (M,)

    Returns:
        persistence: Dict mapping cluster_id -> persistence probability
    """
    persistence = {}
    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip invalid labels
            continue

        # Find all timesteps where we're in this cluster
        in_cluster = (labels == cluster_id)

        # Count transitions
        stays = 0  # c -> c
        leaves = 0  # c -> other

        for t in range(len(labels) - 1):
            if in_cluster[t]:
                if in_cluster[t + 1]:
                    stays += 1
                else:
                    leaves += 1

        # Compute persistence probability
        total = stays + leaves
        if total > 0:
            persistence[cluster_id] = stays / total
        else:
            persistence[cluster_id] = 0.0

    return persistence


def compute_cluster_stats(
    labels: np.ndarray
) -> Dict[int, Dict[str, int]]:
    """
    Compute statistics for each cluster.

    Args:
        labels: Cluster labels, shape (M,)

    Returns:
        stats: Dict mapping cluster_id -> {size, first_idx, last_idx}
    """
    stats = {}
    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue

        mask = labels == cluster_id
        indices = np.where(mask)[0]

        stats[cluster_id] = {
            'size': len(indices),
            'first_idx': int(indices[0]) if len(indices) > 0 else -1,
            'last_idx': int(indices[-1]) if len(indices) > 0 else -1,
        }

    return stats


# ============================================================================
# Baseline Comparisons
# ============================================================================


def cluster_random(
    segments: np.ndarray,
    num_clusters: int,
    preserve_distribution: bool = True
) -> np.ndarray:
    """
    Random clustering baseline.

    Args:
        segments: Array of shape (M, K, 2)
        num_clusters: Number of clusters
        preserve_distribution: If True, match cluster size distribution

    Returns:
        labels: Random cluster labels, shape (M,)
    """
    M = len(segments)

    if preserve_distribution:
        # Assign roughly equal numbers to each cluster
        labels = np.random.choice(num_clusters, size=M)
    else:
        # Completely uniform random
        labels = np.random.randint(0, num_clusters, size=M)

    return labels


def cluster_kmeans(
    segments: np.ndarray,
    num_clusters: int,
    random_state: int = 42
) -> np.ndarray:
    """
    K-means clustering baseline using Euclidean distance.

    Flattens segments to (M, K*2) and runs standard k-means.

    Args:
        segments: Array of shape (M, K, 2)
        num_clusters: Number of clusters
        random_state: Random seed for reproducibility

    Returns:
        labels: K-means cluster labels, shape (M,)
    """
    M, K, D = segments.shape

    # Flatten segments to (M, K*D)
    segments_flat = segments.reshape(M, K * D)

    # Run k-means
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = kmeans.fit_predict(segments_flat)

    return labels


def cluster_volatility(
    p: np.ndarray,
    K: int,
    num_buckets: int = 3,
    window: int = 20
) -> np.ndarray:
    """
    Volatility regime clustering baseline.

    Computes rolling volatility and assigns segments to low/med/high vol buckets.

    Args:
        p: Log prices, shape (N,)
        K: Segment length
        num_buckets: Number of volatility buckets (e.g., 3 for low/med/high)
        window: Window for rolling volatility

    Returns:
        labels: Volatility regime labels, shape (N-K+1,)
    """
    # Compute returns
    returns = np.diff(p)

    # Compute rolling volatility (std dev over window)
    N = len(p)
    vol = np.zeros(N)

    for i in range(window, N):
        vol[i] = np.std(returns[i-window:i])

    # For first 'window' bars, use expanding window
    for i in range(1, window):
        if i > 1:
            vol[i] = np.std(returns[:i])

    # Get volatility at end of each segment
    num_segments = N - K + 1
    segment_vols = vol[K-1:K-1+num_segments]

    # Assign to buckets based on percentiles
    if num_buckets == 3:
        percentiles = [33.33, 66.67]
    else:
        percentiles = np.linspace(100/num_buckets, 100*(1-1/num_buckets), num_buckets-1)

    thresholds = np.percentile(segment_vols, percentiles)

    # Assign labels
    labels = np.digitize(segment_vols, thresholds)

    return labels
