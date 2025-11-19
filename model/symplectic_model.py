"""
Symplectic (Hamiltonian) dynamics for market forecasting.

This module treats market state like a simple physical system:
- Position q: Deviation from equilibrium (price or volume based)
- Momentum π: Recent price change (return)
- Hamiltonian H = 0.5*π² + 0.5*κ*q²

The parameter κ (kappa) controls the "stiffness":
- Low κ: Momentum-driven, trends continue
- High κ: Mean-reverting, price snaps back

Uses symplectic (leapfrog) integration to preserve energy and forecast
the next bar's return.
"""

import numpy as np
from typing import Tuple, Dict


def hamiltonian(q: float, pi: float, kappa: float) -> float:
    """
    Compute Hamiltonian (total energy) of the system.

    H = 0.5 * π² + 0.5 * κ * q²

    Args:
        q: Position (deviation from equilibrium)
        pi: Momentum (recent price change)
        kappa: Stiffness parameter

    Returns:
        H: Total energy
    """
    return 0.5 * pi**2 + 0.5 * kappa * q**2


def leapfrog_step(
    q: float,
    pi: float,
    kappa: float,
    dt: float = 1.0
) -> Tuple[float, float]:
    """
    Single leapfrog integration step for Hamiltonian dynamics.

    For H = 0.5*π² + 0.5*κ*q²:
    - dq/dt = ∂H/∂π = π
    - dπ/dt = -∂H/∂q = -κ*q

    Leapfrog algorithm (symplectic, energy-preserving):
    1. Half-step momentum: π_{1/2} = π - 0.5*dt*κ*q
    2. Full-step position: q_next = q + dt*π_{1/2}
    3. Half-step momentum: π_next = π_{1/2} - 0.5*dt*κ*q_next

    Args:
        q: Current position
        pi: Current momentum
        kappa: Stiffness parameter
        dt: Time step (default 1.0 for 1-bar forecast)

    Returns:
        (q_next, pi_next): State after one time step
    """
    # Half-step momentum update
    pi_half = pi - 0.5 * dt * kappa * q

    # Full-step position update
    q_next = q + dt * pi_half

    # Half-step momentum update
    pi_next = pi_half - 0.5 * dt * kappa * q_next

    return q_next, pi_next


def extract_state_from_segment(
    segment: np.ndarray,
    encoding: str = 'A',
    alpha_q: float = 0.5
) -> Tuple[float, float]:
    """
    Extract (q, π) state from a K-bar segment.

    Three encoding schemes:

    **Encoding A (volume-based):**
    - q = v[-1] (last bar's normalized volume)
    - π = p[-1] - p[-2] (last bar's log-return)

    **Encoding B (price-only):**
    - q = p[-1] - mean(p) (deviation from segment mean)
    - π = p[-1] - p[-2] (last bar's log-return)

    **Encoding C (hybrid):**
    - q = α*v[-1] + (1-α)*(p[-1] - mean(p))
    - π = p[-1] - p[-2]

    Args:
        segment: Shape (K, 2) with columns [log_price, norm_volume]
        encoding: 'A', 'B', or 'C'
        alpha_q: Weight for hybrid encoding (only used if encoding='C')

    Returns:
        (q, π): Position and momentum
    """
    if segment.shape[0] < 2:
        raise ValueError(f"Segment must have at least 2 bars, got {segment.shape[0]}")

    if segment.shape[1] != 2:
        raise ValueError(f"Segment must have 2 features [p, v], got {segment.shape[1]}")

    p = segment[:, 0]  # Log prices
    v = segment[:, 1]  # Normalized volumes

    # Momentum is always last bar's return
    pi = p[-1] - p[-2]

    # Position depends on encoding
    if encoding == 'A':
        # Volume-based
        q = v[-1]

    elif encoding == 'B':
        # Price deviation from mean
        q = p[-1] - np.mean(p)

    elif encoding == 'C':
        # Hybrid
        q = alpha_q * v[-1] + (1 - alpha_q) * (p[-1] - np.mean(p))

    else:
        raise ValueError(f"Unknown encoding: {encoding}. Use 'A', 'B', or 'C'")

    return float(q), float(pi)


def estimate_global_kappa(
    segments: np.ndarray,
    encoding: str = 'A',
    epsilon: float = 1e-8
) -> float:
    """
    Estimate global κ from all training segments.

    Uses the relationship: κ ≈ E[π²] / E[q²]

    This approximates the "stiffness" of the system - how strongly
    momentum relates to position.

    Args:
        segments: Shape (M, K, 2) with all training segments
        encoding: Encoding scheme ('A', 'B', or 'C')
        epsilon: Regularization to avoid division by zero

    Returns:
        kappa: Global stiffness parameter, clipped to [0.01, 10.0]
    """
    M = len(segments)

    q_squared_sum = 0.0
    pi_squared_sum = 0.0

    for i in range(M):
        q, pi = extract_state_from_segment(segments[i], encoding)
        q_squared_sum += q**2
        pi_squared_sum += pi**2

    mean_q2 = q_squared_sum / M
    mean_pi2 = pi_squared_sum / M

    # Estimate kappa
    kappa = mean_pi2 / (mean_q2 + epsilon)

    # Clip to reasonable range
    kappa = np.clip(kappa, 0.01, 10.0)

    return float(kappa)


def estimate_kappa_per_cluster(
    segments: np.ndarray,
    labels: np.ndarray,
    encoding: str = 'A',
    epsilon: float = 1e-8
) -> Dict[int, float]:
    """
    Estimate κ per cluster with shrinkage toward global κ.

    For STEP 4 (not used in STEP 3).

    For each cluster c:
    1. Compute raw κ_c from cluster segments
    2. Compute global κ_global from all segments
    3. Shrink κ_c toward κ_global based on cluster size
    4. Clip to [0.01, 10.0]

    Shrinkage formula:
        λ_c = 0.5 + 0.3 * min(n_c / 200, 1.0)
        κ_c_final = λ_c * κ_c_raw + (1 - λ_c) * κ_global

    Args:
        segments: Shape (M, K, 2)
        labels: Cluster labels, shape (M,)
        encoding: Encoding scheme
        epsilon: Regularization

    Returns:
        Dict mapping cluster_id -> κ_c_final
    """
    # Compute global kappa
    kappa_global = estimate_global_kappa(segments, encoding, epsilon)

    kappa_per_cluster = {}
    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue

        # Get segments in this cluster
        mask = labels == cluster_id
        cluster_segments = segments[mask]
        n_c = len(cluster_segments)

        if n_c < 10:
            # Too few segments, just use global
            kappa_per_cluster[cluster_id] = kappa_global
            continue

        # Estimate raw kappa for this cluster
        kappa_c_raw = estimate_global_kappa(cluster_segments, encoding, epsilon)

        # Compute shrinkage factor
        # Small clusters shrink more toward global
        lambda_c = 0.5 + 0.3 * min(n_c / 200.0, 1.0)

        # Apply shrinkage
        kappa_c_final = lambda_c * kappa_c_raw + (1 - lambda_c) * kappa_global

        # Clip to safe range
        kappa_c_final = np.clip(kappa_c_final, 0.01, 10.0)

        kappa_per_cluster[cluster_id] = float(kappa_c_final)

    return kappa_per_cluster


def forecast_next_return(
    segment: np.ndarray,
    kappa: float,
    encoding: str = 'A',
    dt: float = 1.0
) -> float:
    """
    Forecast next bar's log-return using symplectic dynamics.

    Steps:
    1. Extract (q, π) from segment
    2. Apply one leapfrog step
    3. Return π_next as forecast Δp

    Args:
        segment: Shape (K, 2)
        kappa: Stiffness parameter
        encoding: Encoding scheme
        dt: Time step

    Returns:
        forecast_return: Predicted Δp for next bar
    """
    q, pi = extract_state_from_segment(segment, encoding)
    q_next, pi_next = leapfrog_step(q, pi, kappa, dt)

    return float(pi_next)
