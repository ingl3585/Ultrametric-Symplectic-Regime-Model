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
from typing import Dict


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
