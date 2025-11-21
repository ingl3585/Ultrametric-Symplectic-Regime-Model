"""
Feature extraction for ML combiner layer.

Extracts ~25 normalized features combining:
- Ultrametric cluster information
- Symplectic dynamics (forecasts, energy)
- Market microstructure (volume, volatility)
- Temporal features (time of day, day of week)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from .symplectic_model import extract_state_from_segment, hamiltonian


def extract_feature_vector(
    segment: np.ndarray,                    # (K, 2) [p_log, v_norm]
    nearest_cluster_id: Optional[int],
    distance_to_centroid: Optional[float],
    cluster_stats: Dict[int, Any],          # From trainer: persistence, hit_rate, size, raw_kappa
    kappa_shrunk: float,
    pi_next_1step: float,
    pi_next_2step: float,
    gating_passed: bool,
    p_full: np.ndarray,                     # Full log-price series for context
    v_full: np.ndarray,                     # Full volume series
    t: int,                                 # Current bar index
    config: dict
) -> np.ndarray:
    """
    Extract feature vector for ML model.

    Returns a flat np.ndarray of ~25 normalized features.

    Feature categories:
    1. Cluster info (5): cluster_id, distance, persistence, hit_rate, cluster_size
    2. Kappa values (2): raw_kappa, shrunk_kappa
    3. Forecasts (2): pi_next_1step, pi_next_2step
    4. Gating (1): gating_passed flag
    5. Volume (2): current_volume_regime, volume_trend
    6. Time (2): hour_of_day, day_of_week
    7. Volatility (1): realized_vol_20
    8. Price position (2): dist_to_daily_high, dist_to_daily_low
    9. Recent returns (3): last 3 bar returns
    10. Recent volumes (3): last 3 volume ratios
    11. Energy (1): Hamiltonian
    12. Divergence (1): forecast vs recent momentum

    Total: 25 features
    """
    features = []
    K = segment.shape[0]

    # ========================================================================
    # 1. Cluster Information (5 features)
    # ========================================================================

    # Cluster ID (normalized to [0, 1])
    max_clusters = config.get('clustering', {}).get('n_clusters', 20)
    cluster_id_norm = (nearest_cluster_id if nearest_cluster_id is not None else -1) / max_clusters
    features.append(cluster_id_norm)

    # Distance to centroid (clipped to [0, 5])
    dist_norm = np.clip(distance_to_centroid if distance_to_centroid is not None else 5.0, 0, 5) / 5.0
    features.append(dist_norm)

    # Cluster persistence
    persistence = 0.5  # Default
    if nearest_cluster_id is not None and nearest_cluster_id in cluster_stats:
        persistence = cluster_stats[nearest_cluster_id].get('persistence', 0.5)
    features.append(persistence)

    # Cluster hit rate
    hit_rate = 0.5  # Default
    if nearest_cluster_id is not None and nearest_cluster_id in cluster_stats:
        hit_rate = cluster_stats[nearest_cluster_id].get('hit_rate', 0.5)
    features.append(hit_rate)

    # Cluster size (log-normalized)
    cluster_size = 1
    if nearest_cluster_id is not None and nearest_cluster_id in cluster_stats:
        cluster_size = cluster_stats[nearest_cluster_id].get('size', 1)
    cluster_size_norm = np.log1p(cluster_size) / 10.0  # Normalize by log(~20k)
    features.append(cluster_size_norm)

    # ========================================================================
    # 2. Kappa Values (2 features)
    # ========================================================================

    # Raw kappa (from cluster)
    raw_kappa = 0.01  # Default global
    if nearest_cluster_id is not None and nearest_cluster_id in cluster_stats:
        raw_kappa = cluster_stats[nearest_cluster_id].get('raw_kappa', 0.01)
    raw_kappa_norm = np.clip(raw_kappa, 0.001, 1.0)  # Already in reasonable range
    features.append(raw_kappa_norm)

    # Shrunk kappa (after shrinkage)
    shrunk_kappa_norm = np.clip(kappa_shrunk, 0.001, 1.0)
    features.append(shrunk_kappa_norm)

    # ========================================================================
    # 3. Symplectic Forecasts (2 features)
    # ========================================================================

    # 1-step forecast (clipped to [-0.1, 0.1])
    pi_1step_norm = np.clip(pi_next_1step, -0.1, 0.1) / 0.1
    features.append(pi_1step_norm)

    # 2-step forecast (clipped to [-0.2, 0.2])
    pi_2step_norm = np.clip(pi_next_2step, -0.2, 0.2) / 0.2
    features.append(pi_2step_norm)

    # ========================================================================
    # 4. Gating (1 feature)
    # ========================================================================

    features.append(1.0 if gating_passed else 0.0)

    # ========================================================================
    # 5. Volume Features (2 features)
    # ========================================================================

    # Current volume regime (relative to recent average)
    v_recent = v_full[max(0, t-20):t+1] if t >= 20 else v_full[:t+1]
    v_mean_recent = np.mean(v_recent) if len(v_recent) > 0 else 1.0
    current_vol = v_full[t] if t < len(v_full) else 1.0
    volume_regime = np.clip(current_vol / (v_mean_recent + 1e-8), 0.1, 3.0)
    volume_regime_norm = (volume_regime - 1.0) / 2.0  # Center at 0, range ~[-0.45, +1.0]
    features.append(volume_regime_norm)

    # Volume trend (EMA slope over last 5 bars)
    if t >= 5:
        v_last5 = v_full[t-4:t+1]
        volume_trend = (v_last5[-1] - v_last5[0]) / (v_last5[0] + 1e-8)
        volume_trend_norm = np.clip(volume_trend, -1.0, 1.0)
    else:
        volume_trend_norm = 0.0
    features.append(volume_trend_norm)

    # ========================================================================
    # 6. Time Features (2 features)
    # ========================================================================

    # Hour of day (normalized to [0, 1])
    # Note: If timestamps not available, default to 0.5
    hour_norm = 0.5  # Default (midday)
    features.append(hour_norm)

    # Day of week (normalized to [0, 1])
    dow_norm = 0.5  # Default (midweek)
    features.append(dow_norm)

    # ========================================================================
    # 7. Realized Volatility (1 feature)
    # ========================================================================

    # 20-bar realized volatility
    if t >= 20:
        p_last20 = p_full[t-19:t+1]
        returns = np.diff(p_last20)
        realized_vol = np.std(returns)
    elif t >= 2:
        p_recent = p_full[:t+1]
        returns = np.diff(p_recent)
        realized_vol = np.std(returns) if len(returns) > 0 else 0.001
    else:
        realized_vol = 0.001

    realized_vol_norm = np.clip(realized_vol, 0, 0.05) / 0.05  # Normalize by typical NQ vol
    features.append(realized_vol_norm)

    # ========================================================================
    # 8. Price Position (2 features)
    # ========================================================================

    # Distance to daily high (last 100 bars as proxy for "daily")
    lookback = min(100, t+1)
    p_recent = p_full[max(0, t-lookback+1):t+1]
    p_current = p_full[t]
    p_high = np.max(p_recent) if len(p_recent) > 0 else p_current
    p_low = np.min(p_recent) if len(p_recent) > 0 else p_current

    dist_to_high = (p_high - p_current) / (p_high - p_low + 1e-8) if p_high > p_low else 0.5
    dist_to_high_norm = np.clip(dist_to_high, 0, 1)
    features.append(dist_to_high_norm)

    dist_to_low = (p_current - p_low) / (p_high - p_low + 1e-8) if p_high > p_low else 0.5
    dist_to_low_norm = np.clip(dist_to_low, 0, 1)
    features.append(dist_to_low_norm)

    # ========================================================================
    # 9. Recent Returns (3 features)
    # ========================================================================

    # Last 3 bar returns
    for lag in [1, 2, 3]:
        if t >= lag:
            ret = p_full[t] - p_full[t-lag]
            ret_norm = np.clip(ret, -0.05, 0.05) / 0.05
        else:
            ret_norm = 0.0
        features.append(ret_norm)

    # ========================================================================
    # 10. Recent Volume Ratios (3 features)
    # ========================================================================

    # Last 3 volume ratios (current / t-lag)
    for lag in [1, 2, 3]:
        if t >= lag:
            v_ratio = v_full[t] / (v_full[t-lag] + 1e-8)
            v_ratio_norm = np.clip(v_ratio, 0.1, 3.0)
            v_ratio_norm = (v_ratio_norm - 1.0) / 2.0  # Center at 0
        else:
            v_ratio_norm = 0.0
        features.append(v_ratio_norm)

    # ========================================================================
    # 11. Hamiltonian Energy (1 feature)
    # ========================================================================

    # Extract current state and compute energy
    encoding = config.get('symplectic', {}).get('encoding', 'A')
    q, pi = extract_state_from_segment(segment, encoding=encoding)
    H = hamiltonian(q, pi, kappa_shrunk)
    H_norm = np.clip(H, 0, 0.1) / 0.1  # Normalize by typical energy scale
    features.append(H_norm)

    # ========================================================================
    # 12. Forecast Divergence (1 feature)
    # ========================================================================

    # Divergence between forecast and recent momentum
    # Recent momentum = average of last 3 returns
    if t >= 3:
        recent_momentum = np.mean([p_full[t] - p_full[t-i] for i in [1, 2, 3]])
    else:
        recent_momentum = 0.0

    divergence = pi_next_1step - recent_momentum
    divergence_norm = np.clip(divergence, -0.05, 0.05) / 0.05
    features.append(divergence_norm)

    # ========================================================================
    # Convert to numpy array
    # ========================================================================

    feature_vector = np.array(features, dtype=np.float32)

    # Sanity check
    assert feature_vector.shape[0] == 25, f"Expected 25 features, got {feature_vector.shape[0]}"
    assert not np.any(np.isnan(feature_vector)), "Feature vector contains NaN values"
    assert not np.any(np.isinf(feature_vector)), "Feature vector contains Inf values"

    return feature_vector


def get_feature_names() -> list:
    """Return human-readable feature names for analysis."""
    return [
        'cluster_id_norm',
        'distance_to_centroid',
        'cluster_persistence',
        'cluster_hit_rate',
        'cluster_size_log',
        'raw_kappa',
        'shrunk_kappa',
        'pi_forecast_1step',
        'pi_forecast_2step',
        'gating_passed',
        'volume_regime',
        'volume_trend',
        'hour_of_day',
        'day_of_week',
        'realized_vol_20',
        'dist_to_daily_high',
        'dist_to_daily_low',
        'return_lag1',
        'return_lag2',
        'return_lag3',
        'volume_ratio_lag1',
        'volume_ratio_lag2',
        'volume_ratio_lag3',
        'hamiltonian',
        'forecast_divergence'
    ]
