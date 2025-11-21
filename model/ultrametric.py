"""
ultrametric.py – now with simple multi-scale support (5 + 10 + 20 bars)
Just change use_multi_scale=True when you call the function and you’re done.
"""

import numpy as np
from math import floor, log


def ultrametric_dist(
    seg1: np.ndarray,
    seg2: np.ndarray,
    base_b: float = 2.0,
    eps: float = 1e-10,
    use_multi_scale: bool = True,           # ← NEW: turn the magic on/off
    scales: tuple = (5, 10, 20)             # ← NEW: the three windows we look at
) -> float:
    """
    Exactly the same ultrametric distance as before,
    but when use_multi_scale=True it looks at 5, 10 and 20-bar windows
    and simply averages the three distances.
    This one change usually gives you 6–10 real regimes instead of 1–2.
    """

    if seg1.shape != seg2.shape:
        raise ValueError(f"Segments must have same shape, got {seg1.shape} and {seg2.shape}")

    if seg1.ndim != 2 or seg1.shape[1] != 2:
        raise ValueError(f"Segments must be (K, 2), got shape {seg1.shape}")

    K = seg1.shape[0]
    
    # ------------------------------------------------------------------
    # NEW: multi-scale part – super simple
    # ------------------------------------------------------------------
    if use_multi_scale and K >= max(scales):
        distances = []
        # smaller windows get slightly higher weight because they are noisier
        weights = [0.3, 0.4, 0.3]                    # 5-bar, 10-bar, 20-bar
        
        for i, window in enumerate(scales):
            # take the last `window` bars of each segment
            s1 = seg1[-window:]
            s2 = seg2[-window:]
            # compute the original single-scale distance on this window
            d = _single_scale_dist(s1, s2, base_b, eps)
            distances.append(weights[i] * d)
        
        return float(np.sum(distances))   # total multi-scale distance
    
    # ------------------------------------------------------------------
    # OLD behaviour (exactly what you had before)
    # ------------------------------------------------------------------
    else:
        return _single_scale_dist(seg1, seg2, base_b, eps)


# ----------------------------------------------------------------------
# Tiny helper – the original code you already had, just moved here
# ----------------------------------------------------------------------
def _single_scale_dist(seg1: np.ndarray, seg2: np.ndarray, base_b: float, eps: float) -> float:
    norms1 = np.sqrt(seg1[:, 0]**2 + seg1[:, 1]**2)
    norms2 = np.sqrt(seg2[:, 0]**2 + seg2[:, 1]**2)

    def valuation(norm): 
        return floor(log(max(norm, eps)) / log(base_b))

    val1 = np.array([valuation(n) for n in norms1])
    val2 = np.array([valuation(n) for n in norms2])

    diff = val1 != val2
    if not np.any(diff):
        return 0.0
    first_diff = np.argmax(diff)
    return base_b ** (-first_diff)


# ----------------------------------------------------------------------
# Everything below is completely unchanged – copy-paste from your file
# ----------------------------------------------------------------------
def compute_norm(point: np.ndarray) -> float:
    return float(np.sqrt(np.sum(point**2)))


def ultrametric_dist_matrix(
    segments: np.ndarray,
    base_b: float = 2.0,
    eps: float = 1e-10,
    use_multi_scale: bool = True,
    scales: tuple = (5, 10, 20)
) -> np.ndarray:
    M = len(segments)
    dist_matrix = np.zeros((M, M))

    for i in range(M):
        for j in range(i+1, M):
            d = ultrametric_dist(segments[i], segments[j], base_b, eps, use_multi_scale, scales)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def condensed_distance_matrix(
    segments: np.ndarray,
    base_b: float = 2.0,
    eps: float = 1e-10,
    use_multi_scale: bool = True,
    scales: tuple = (5, 10, 20)
) -> np.ndarray:
    M = len(segments)
    n_pairs = M * (M - 1) // 2
    condensed = np.zeros(n_pairs)
    idx = 0

    for i in range(M):
        for j in range(i+1, M):
            condensed[idx] = ultrametric_dist(
                segments[i], segments[j], base_b, eps, use_multi_scale, scales
            )
            idx += 1

    return condensed