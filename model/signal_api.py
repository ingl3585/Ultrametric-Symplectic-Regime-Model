"""
Signal generation APIs for trading models.

This module provides a uniform interface for different forecasting models:
- AR1Model: Simple AR(1) baseline on returns
- SymplecticGlobalModel: Symplectic dynamics with global kappa (STEP 3)
- SymplecticUltrametricModel: Full hybrid model (STEP 4)

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

    This is the full STEP 4 model.
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
            d = ultrametric_dist(seg, centroid, base_b, eps)
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
            # Use provided cluster_stats (for ML model)
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
        # e.g., if hit_threshold=0.50, then:
        #   0.50 -> 0.0
        #   0.55 -> 0.5
        #   0.60+ -> 1.0
        size_factor = (hit_rate - self.hit_threshold) / 0.10
        size_factor = max(0.0, min(1.0, size_factor))

        return {
            "direction": direction,
            "size_factor": size_factor
        }


class HybridMLModel:
    """
    ML Combiner Layer on top of SymplecticUltrametricModel.

    Uses scikit-learn MLPClassifier or MLPRegressor to learn from the pure math
    model's intermediate features + market microstructure.

    Training is done online (incremental) with periodic retraining.
    """

    def __init__(
        self,
        config: dict,
        base_model: "SymplecticUltrametricModel",   # The pure math model
        ml_type: str = "classifier"                  # "classifier" or "regressor"
    ):
        """
        Initialize HybridMLModel.

        Args:
            config: Full config dict
            base_model: Trained SymplecticUltrametricModel instance
            ml_type: "classifier" (predict direction) or "regressor" (predict return)
        """
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler

        self.config = config
        self.base_model = base_model
        self.ml_type = ml_type
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.X_hist = []
        self.y_hist = []

        # Get ML config
        ml_config = config.get("ml", {})
        hidden_sizes = tuple(ml_config.get("hidden_sizes", [48, 24]))

        if ml_type == "classifier":
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_sizes,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42,
                verbose=False
            )
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_sizes,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42,
                verbose=False
            )

    def _add_training_example(self, features: np.ndarray, target: float):
        """
        Add a training example (feature vector, realized return).

        Args:
            features: Feature vector (25-dim)
            target: Realized next-bar return (for training)
        """
        self.X_hist.append(features)
        self.y_hist.append(target)

    def retrain_if_needed(self, current_bar_idx: int, force: bool = False):
        """
        Retrain ML model if enough examples accumulated.

        Args:
            current_bar_idx: Current bar index in backtest
            force: Force retraining regardless of interval
        """
        ml_config = self.config.get("ml", {})
        retrain_interval = ml_config.get("retrain_interval", 1000)
        min_samples = ml_config.get("min_training_samples", 500)

        # Check if we should retrain
        should_retrain = (
            force or
            (not self.is_fitted and len(self.X_hist) >= min_samples) or  # Initial training
            (current_bar_idx % retrain_interval == 0 and len(self.X_hist) >= min_samples)  # Periodic retraining
        )

        if not should_retrain:
            return

        if len(self.X_hist) < min_samples:
            return

        # Prepare training data
        X = np.array(self.X_hist)
        y = np.array(self.y_hist)

        if self.ml_type == "classifier":
            # Convert returns to directional labels: -1, 0, +1
            y_class = np.sign(y)
        else:
            y_class = y

        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)

        try:
            self.model.fit(X_scaled, y_class)
            self.is_fitted = True
        except Exception as e:
            print(f"Warning: ML model training failed: {e}")
            self.is_fitted = False

        # Optional: Keep only recent N examples to save memory
        # max_history = 100_000
        # if len(self.X_hist) > max_history:
        #     self.X_hist = self.X_hist[-max_history:]
        #     self.y_hist = self.y_hist[-max_history:]

    def get_signal(
        self,
        segment: np.ndarray,
        p_full: np.ndarray,
        v_full: np.ndarray,
        t: int,
        cluster_stats: Dict[int, any],
        force_fallback: bool = False
    ) -> Dict[str, float]:
        """
        Generate trading signal using ML model.

        Args:
            segment: (K, 2) segment [p, v]
            p_full: Full log-price series
            v_full: Full normalized volume series
            t: Current bar index
            cluster_stats: Dict of cluster statistics
            force_fallback: If True, use pure math model

        Returns:
            {"direction": int, "size_factor": float}
        """
        from .features import extract_feature_vector

        # Get base model's intermediate values
        nearest = self.base_model._nearest_cluster(segment)
        c_id, dist = (nearest[0], nearest[1]) if nearest[0] is not None else (None, None)

        gating_passed = self.base_model._passes_gating(c_id, dist, cluster_stats)

        kappa = self.base_model.kappa_per_cluster.get(
            c_id,
            self.base_model.kappa_global
        )

        # Extract state and compute forecasts
        q, pi = extract_state_from_segment(segment, self.base_model.encoding)

        # 1-step and 2-step forecasts
        q_next, pi_next_1step = leapfrog_step(q, pi, kappa, dt=1.0)
        q_next2, pi_next_2step = leapfrog_step(q_next, pi_next_1step, kappa, dt=1.0)

        # Extract features
        features = extract_feature_vector(
            segment=segment,
            nearest_cluster_id=c_id,
            distance_to_centroid=dist,
            cluster_stats=cluster_stats,
            kappa_shrunk=kappa,
            pi_next_1step=pi_next_1step,
            pi_next_2step=pi_next_1step + pi_next_2step,  # Cumulative 2-step
            gating_passed=gating_passed,
            p_full=p_full,
            v_full=v_full,
            t=t,
            config=self.config
        )

        # Fallback conditions
        if force_fallback or not self.is_fitted or not gating_passed:
            # Use pure math model
            result = self.base_model.get_signal(segment)
            result['ml_confidence'] = 0.0  # No ML confidence in fallback
            return result

        # Use ML model
        X = self.scaler.transform(features.reshape(1, -1))

        if self.ml_type == "classifier":
            # Predict direction
            try:
                proba = self.model.predict_proba(X)[0]
                # Assumes class order: [-1, 0, +1] after sklearn's label sorting
                # Get class labels
                classes = self.model.classes_
                direction_idx = np.argmax(proba)
                direction = int(classes[direction_idx])
                confidence = np.max(proba)
            except Exception as e:
                # Fallback to base model if prediction fails
                result = self.base_model.get_signal(segment)
                result['ml_confidence'] = 0.0
                return result

        else:  # regressor
            # Predict return
            try:
                pred_ret = self.model.predict(X)[0]
                theta_ml = self.config.get("signal", {}).get("theta_ml", 0.0005)

                if abs(pred_ret) <= theta_ml:
                    direction = 0
                    confidence = 0.0
                else:
                    direction = 1 if pred_ret > 0 else -1
                    confidence = min(1.0, abs(pred_ret) / (3 * theta_ml))
            except Exception as e:
                result = self.base_model.get_signal(segment)
                result['ml_confidence'] = 0.0
                return result

        # Size factor based on confidence and cluster hit rate
        hit_rate = cluster_stats.get(c_id, {}).get('hit_rate', 0.5) if c_id is not None else 0.5
        size_factor = confidence * (hit_rate - 0.5) / 0.05
        size_factor = np.clip(size_factor, 0.0, 1.0)

        # ML-specific gating: Check minimum confidence and size_factor thresholds
        ml_config = self.config.get('ml', {})
        min_confidence = ml_config.get('min_confidence', 0.5)
        min_size_factor = ml_config.get('min_size_factor', 0.0)

        # If confidence or size_factor too low, don't trade (or fallback to base model)
        if confidence < min_confidence or size_factor < min_size_factor:
            return {"direction": 0, "size_factor": 0.0, "ml_confidence": float(confidence)}

        return {
            "direction": int(direction) if abs(direction) == 1 else 0,
            "size_factor": float(size_factor),
            "ml_confidence": float(confidence)
        }
