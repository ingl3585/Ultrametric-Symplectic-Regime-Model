"""
Signal generation APIs for trading models.

This module provides a uniform interface for different forecasting models:
- AR1Model: Simple AR(1) baseline on returns
- SymplecticGlobalModel: Symplectic dynamics with global kappa
- SymplecticUltrametricModel: Full ultrametric-symplectic hybrid model

All models expose a get_signal(...) method that returns:
{
    "direction": -1 | 0 | 1,  # short, flat, long
    "size_factor": float      # 0.0 to 1.0
}
"""

import numpy as np
from typing import Dict, Tuple

from .symplectic_model import (
    extract_state_from_segment,
    leapfrog_step,
    forecast_next_return
)


class AR1Model:
    """
    Simple AR(1) baseline model for returns.

    Fits an autoregressive model on log-price returns:
        r_t = mu + phi * (r_{t-1} - mu) + epsilon_t

    Generates signals based on predicted next return with a threshold.
    """

    def __init__(self, config: dict):
        """
        Initialize AR1 model.

        Args:
            config: Configuration dict containing signal parameters
        """
        self.config = config
        self.phi = 0.0
        self.mean_ret = 0.0
        self._fitted = False

    def fit(self, p: np.ndarray) -> None:
        """
        Fit AR(1) model on log-price returns.

        Uses simple OLS estimation:
        - Compute returns r_t = p_t - p_{t-1}
        - Estimate mean and AR(1) coefficient

        Args:
            p: Log prices, shape (N,)
        """
        # Compute returns
        returns = p[1:] - p[:-1]

        # Estimate mean return
        self.mean_ret = np.mean(returns)

        # Demean returns
        r_demean = returns - self.mean_ret

        # Estimate phi via OLS: r_t = phi * r_{t-1}
        # phi = sum(r_t * r_{t-1}) / sum(r_{t-1}^2)
        r_t = r_demean[1:]
        r_t_minus_1 = r_demean[:-1]

        numerator = np.sum(r_t * r_t_minus_1)
        denominator = np.sum(r_t_minus_1 ** 2)

        if denominator > 1e-10:
            self.phi = numerator / denominator
        else:
            self.phi = 0.0

        # Clamp phi to reasonable range for stability
        self.phi = np.clip(self.phi, -0.99, 0.99)

        self._fitted = True

    def get_signal(self, last_price: float, prev_price: float) -> Dict[str, float]:
        """
        Generate trading signal based on AR(1) forecast.

        Args:
            last_price: Most recent log price (p_t)
            prev_price: Previous log price (p_{t-1})

        Returns:
            {
                "direction": -1 (short), 0 (flat), or 1 (long),
                "size_factor": Position sizing factor (0.0 to 1.0)
            }
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before generating signals")

        # Current return
        r_t = last_price - prev_price

        # Predict next return using AR(1)
        r_hat = self.mean_ret + self.phi * (r_t - self.mean_ret)

        # Get threshold from config
        theta = self.config.get("signal", {}).get("theta_ar1", 0.0001)

        # Generate signal
        if abs(r_hat) <= theta:
            # Forecast too small, stay flat
            direction = 0
            size_factor = 0.0
        else:
            # Take directional position
            direction = int(np.sign(r_hat))
            size_factor = 1.0

        return {
            "direction": direction,
            "size_factor": size_factor
        }


class SymplecticGlobalModel:
    """
    Symplectic dynamics model with single global κ.

    Treats market state as a Hamiltonian system:
    - Position q: Deviation from equilibrium (volume or price based)
    - Momentum π: Recent price change
    - H = 0.5*π² + 0.5*κ*q²

    Uses leapfrog integration to forecast next bar's return.
    """

    def __init__(self, config: dict, kappa: float, encoding: str = 'A'):
        """
        Initialize symplectic global model.

        Args:
            config: Configuration dict
            kappa: Global stiffness parameter
            encoding: State encoding ('A', 'B', or 'C')
        """
        self.config = config
        self.kappa = kappa
        self.encoding = encoding
        self.dt = config.get("symplectic", {}).get("dt", 1.0)

    def get_signal(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Generate trading signal from K-bar segment.

        Steps:
        1. Extract (q, π) from segment using encoding
        2. Apply one leapfrog step: (q, π) → (q_next, π_next)
        3. Use π_next as forecast Δp
        4. Apply threshold and generate signal

        Args:
            segment: Shape (K, 2) with columns [log_price, norm_volume]

        Returns:
            {
                "direction": -1 (short), 0 (flat), or 1 (long),
                "size_factor": Position sizing factor (0.0 to 1.0)
            }
        """
        # Forecast next return
        forecast = forecast_next_return(segment, self.kappa, self.encoding, self.dt)

        # Get threshold from config
        theta = self.config.get("signal", {}).get("theta_symplectic", 0.0001)

        # Generate signal
        if abs(forecast) <= theta:
            # Forecast too small, stay flat
            direction = 0
            size_factor = 0.0
        else:
            # Take directional position
            direction = int(np.sign(forecast))
            size_factor = 1.0

        return {
            "direction": direction,
            "size_factor": size_factor
        }


class SymplecticUltrametricModel:
    """
    Full hybrid model: Ultrametric regimes + per-cluster κ + gating.

    Combines regime detection with symplectic dynamics:
    - Uses ultrametric distance to find nearest cluster
    - Applies per-cluster κ for forecasting
    - Gates signals based on cluster quality (hit rate, distance)
    - Adjusts position sizing by cluster confidence

    This is the production model for trading.
    """

    def __init__(
        self,
        config: dict,
        centroids: Dict[int, np.ndarray],
        kappa_per_cluster: Dict[int, float],
        hit_rates: Dict[int, float],
        encoding: str = 'A'
    ):
        """
        Initialize hybrid ultrametric-symplectic model.

        Args:
            config: Configuration dict
            centroids: Dict mapping cluster_id -> centroid_segment (K, 2)
            kappa_per_cluster: Dict mapping cluster_id -> κ value
            hit_rates: Dict mapping cluster_id -> historical hit rate (0-1)
            encoding: State encoding ('A', 'B', or 'C')
        """
        self.config = config
        self.centroids = centroids
        self.kappa_per_cluster = kappa_per_cluster
        self.hit_rates = hit_rates
        self.encoding = encoding
        self.dt = config.get("symplectic", {}).get("dt", 1.0)

        # Gating parameters
        self.epsilon_gate = config.get("signal", {}).get("epsilon_gate", 1.0)
        self.hit_threshold = config.get("signal", {}).get("hit_threshold", 0.50)

        # Fallback kappa if cluster not found
        if len(kappa_per_cluster) > 0:
            self.kappa_global = np.mean(list(kappa_per_cluster.values()))
        else:
            self.kappa_global = 1.0

    def _nearest_cluster(
        self,
        seg: np.ndarray
    ) -> Tuple[int | None, float | None]:
        """
        Find nearest cluster centroid using ultrametric distance.

        Args:
            seg: Segment array, shape (K, 2)

        Returns:
            (cluster_id, distance) or (None, None) if no valid centroids
        """
        from .ultrametric import ultrametric_dist

        if len(self.centroids) == 0:
            return None, None

        base_b = self.config.get("ultrametric", {}).get("base_b", 1.2)
        eps = self.config.get("ultrametric", {}).get("eps", 1e-10)

        min_dist = np.inf
        best_cluster = None

        for cluster_id, centroid in self.centroids.items():
            d = ultrametric_dist(seg, centroid, base_b, eps, use_multi_scale=True)
            if d < min_dist:
                min_dist = d
                best_cluster = cluster_id

        return best_cluster, float(min_dist)

    def _passes_gating(
        self,
        cluster_id: int | None,
        distance: float | None,
        cluster_stats: Dict[int, Dict] = None
    ) -> bool:
        """
        Check if segment passes gating criteria.

        Args:
            cluster_id: Cluster ID (or None if no cluster found)
            distance: Distance to centroid (or None)
            cluster_stats: Optional override cluster stats dict

        Returns:
            True if passes gating, False otherwise
        """
        # No valid cluster
        if cluster_id is None:
            return False

        # Too far from centroid
        if distance is not None and distance > self.epsilon_gate:
            return False

        # Check hit rate
        if cluster_stats is not None:
            # Use provided cluster_stats (for compatibility)
            hit_rate = cluster_stats.get(cluster_id, {}).get('hit_rate', 0.0)
        else:
            # Use model's internal hit_rates
            hit_rate = self.hit_rates.get(cluster_id, 0.0)

        if hit_rate < self.hit_threshold:
            return False

        return True

    def get_signal(self, segment: np.ndarray) -> Dict[str, float]:
        """
        Generate trading signal from K-bar segment with regime gating.

        Steps:
        1. Find nearest cluster via ultrametric distance
        2. Apply gating checks (cluster quality, distance, hit rate)
        3. Extract (q, π) from segment
        4. Look up cluster-specific κ
        5. Apply leapfrog step to forecast
        6. Generate signal with confidence-based sizing

        Args:
            segment: Shape (K, 2) with columns [log_price, norm_volume]

        Returns:
            {
                "direction": -1 (short), 0 (flat), or 1 (long),
                "size_factor": Position sizing factor (0.0 to 1.0)
            }
        """
        # 1. Find nearest cluster
        c_id, dist = self._nearest_cluster(segment)

        # 2. Gating checks
        if not self._passes_gating(c_id, dist):
            return {"direction": 0, "size_factor": 0.0}

        hit_rate = self.hit_rates.get(c_id, 0.0)

        # 3. Extract state
        q, pi = extract_state_from_segment(segment, self.encoding)

        # 4. Get cluster-specific kappa
        kappa = self.kappa_per_cluster.get(c_id, self.kappa_global)

        # 5. Forecast via leapfrog
        q_next, pi_next = leapfrog_step(q, pi, kappa, self.dt)
        forecast = pi_next

        # 6. Apply threshold
        theta = self.config.get("signal", {}).get("theta_symplectic", 0.0001)

        if abs(forecast) <= theta:
            return {"direction": 0, "size_factor": 0.0}

        direction = int(np.sign(forecast))

        # 7. Confidence-based sizing
        # Scale linearly from hit_threshold to hit_threshold + 0.10
        # e.g., if hit_threshold=0.52, then:
        #   0.52 -> 0.0
        #   0.57 -> 0.5
        #   0.62+ -> 1.0
        size_factor = (hit_rate - self.hit_threshold) / 0.10
        size_factor = max(0.0, min(1.0, size_factor))

        return {
            "direction": direction,
            "size_factor": size_factor
        }
