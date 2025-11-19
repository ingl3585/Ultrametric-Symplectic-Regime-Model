# CLAUDE.md – AI Implementation Spec

**Project:** Ultrametric–Symplectic Regime Model for NQ (15-Minute Bars)  
**Version:** 1.0  
**Last Updated:** 2025-11-19  
**Status:** Ready for implementation

---

## Change Policy

- Update this file whenever:
  - Validation results contradict assumptions, or
  - The design meaningfully changes.
- Add "Lessons Learned" sections after major milestones (e.g., Phase 1 complete).
- Keep Git history so we can see how understanding evolved over time.

---

## 0. Context & High-Level Intent

The goal is to build a **research engine** (not a production bot) that:

1. Clusters recent price/volume **shapes** on 15-minute bars into regimes using an **ultrametric distance**.
2. Within those regimes, uses a **symplectic (Hamiltonian)** model to forecast the **next bar's** return.
3. Evaluates whether this hybrid approach offers any edge over simple baselines after **realistic 15m NQ costs**.
4. Optionally exposes a thin HTTP API so NinjaTrader (Simulation / Playback) can call into it as a signal engine.

Timeframe: **15-minute bars only** in v1.  
Instruments: QQQ for initial research; NQ (via NinjaTrader) later.

---

## 1. Division of Labor (Python vs NinjaTrader)

Two distinct contexts:

### 1.1 Offline Research (Python-Only)

- Python:
  - Loads historical **15m OHLCV** data from CSV (QQQ / NQ).
  - Builds log-price and smoothed, normalized volume.
  - Constructs K-bar segments and ultrametric distances.
  - Clusters segments into regimes.
  - Fits global and per-cluster κ for the symplectic model.
  - Runs **backtests with costs** on 15m bars for:
    - AR(1) baseline,
    - SymplecticGlobal (no regimes),
    - SymplecticUltrametric (full hybrid).
- No NinjaTrader involvement in this phase.

### 1.2 NinjaTrader Simulation / Playback (Phase 2, Optional)

- NinjaTrader / NinjaScript:
  - Connects to data feeds.
  - Pulls **historical NQ data**.
  - Builds **15-minute bars**.
  - Calls `OnBarUpdate()` on **bar close**.
  - Manages `Account`, `Position`, and PnL state.
  - Fills trades on **Sim101 / Playback101**.

- Python (when integrated):
  - Does **not** fetch data or build bars.
  - Receives last K 15m bars (close + volume, plus optional timestamps and account snapshot).
  - Returns a signal:

    ```json
    {
      "direction": -1 | 0 | 1,
      "size_factor": 0.0–1.0
    }
    ```

  - Optionally receives trade logs from Ninja (`/trade_log`).

---

## 2. Quick Start for AI Implementation

This is the "do this first" section for you as an AI (Claude Code).

### 2.1 Absolute Priority Order

1. **Base pipeline + AR(1) baseline** (CSV → 15m bars → log price → AR(1) → backtest).
2. **Segments + ultrametric distance** (no clustering used for trading yet).
3. **Clustering + regime persistence** (compare against random, k-means, vol regimes).
4. **Symplectic forecasting (global κ)** — compare vs AR(1).
5. **Full hybrid** (ultrametric regimes + per-cluster κ + gating).
6. **Cost-aware backtests** and validation.
7. **Optional**: HTTP service for NinjaTrader integration.

Do not try to implement everything at once. Follow `IMPLEMENTATION_STEPS.md` in strict order.

### 2.2 Files to Create / Use

Expected layout (can be adapted, but keep the separation):

```text
project_root/
  README.md
  CLAUDE.md
  IMPLEMENTATION_STEPS.md

  configs/
    config.yaml

  data/
    sample_qqq_15m.csv  # optional example

  model/
    __init__.py
    data_utils.py       # loading/resampling/p,v,gamma
    trainer.py          # segments, train orchestration
    ultrametric.py      # ultrametric_dist
    clustering.py       # regimes, centroids, persistence
    symplectic_model.py # leapfrog, encodings, kappa estimators
    signal_api.py       # AR1Model, SymplecticGlobalModel, SymplecticUltrametricModel
    backtest.py         # all backtest harnesses

  server/
    app.py              # optional HTTP API (FastAPI/Flask) for Ninja integration

  tests/
    test_ultrametric.py # optional but recommended
    test_symplectic.py  # optional sanity checks

  notebooks/
    exploration.ipynb   # optional EDA / plots
```

### 2.3 Critical Implementation Notes

- Use `np.log(close)` for price everywhere in the models.
- Handle all edge cases:
  - Division by zero in κ estimation → add epsilon.
  - Tiny clusters → heavy shrinkage or ignore (do not trust tiny regimes).
  - Insufficient history → no signal until K bars available.
- Write at least basic tests for:
  - Ultrametric distance properties.
  - Leapfrog integrator stability on simple synthetic cases.
- Apply the **same cost model** to all strategies (hybrid and baselines) when backtesting.

---

## 3. Data Model (15-Minute Bars)

### 3.1 Input Bars

Each bar should contain:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

If the raw data is not 15-minute, resample to 15m OHLCV.

### 3.2 Transformations

Let `df` be a 15m OHLCV DataFrame.

- Log price:

  ```python
  p = np.log(df["close"].values)  # shape (N,)
  ```

- Smoothed, normalized volume:

  ```python
  # Normalization
  norm_window = config["volume"]["normalization_window"]  # e.g. 200 bars
  rolling_mean = df["volume"].rolling(norm_window, min_periods=1).mean().values
  vol_norm = df["volume"].values / (rolling_mean + 1e-8)

  # EMA smoothing
  ema_period = config["volume"]["ema_period"]  # e.g. 5
  alpha = 2.0 / (ema_period + 1.0)
  v = np.zeros_like(vol_norm)
  v[0] = vol_norm[0]
  for t in range(1, len(vol_norm)):
      v[t] = alpha * vol_norm[t] + (1 - alpha) * v[t-1]
  ```

- Phase-space vector:

  ```python
  gamma = np.stack([p, v], axis=1)  # shape (N, 2)
  ```

### 3.3 Segments

For window length `K` (e.g., 10 bars):

```python
def build_segments(gamma: np.ndarray, K: int) -> np.ndarray:
    """
    gamma: (N, 2)
    returns: segments: (N-K+1, K, 2)
    """
```

Used for both ultrametric regime detection and symplectic state extraction.

---

## 4. Ultrametric Distance & Clustering

### 4.1 Ultrametric Distance

In `model/ultrametric.py`:

```python
def ultrametric_dist(
    seg1: np.ndarray,
    seg2: np.ndarray,
    base_b: float = 2.0,
    eps: float = 1e-10
) -> float:
    """
    seg1, seg2: (K, 2) arrays [p, v].

    Steps:
    1) Compute norms per bar: n_i = sqrt(p_i^2 + v_i^2)
    2) Valuations: val_i = floor(log_b(max(n_i, eps)))
    3) Find first index j where val1[j] != val2[j].
       - If none differ: distance = 0.0
       - Else: distance = base_b ** (-j)
    """
```

Tests (recommended):

- `d(seg, seg) == 0`.
- Symmetry: `d(a, b) ≈ d(b, a)`.
- Approximate ultrametric inequality: for random x,y,z, check:

  ```python
  d(x,z) <= max(d(x,y), d(y,z)) + tol
  ```

### 4.2 Clustering via Hierarchical Methods

In `model/clustering.py`:

- Use a **subsample** of segments (e.g., first 3000–5000) to build a distance matrix with `pdist` and `ultrametric_dist`.
- Run hierarchical clustering (e.g., `linkage(..., method="ward")`).
- Assign cluster labels to the subsample with `fcluster`.

Then compute **centroids**:

```python
def compute_centroids(
    segments: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int
) -> Dict[int, np.ndarray]:
    """
    segments: (M, K, 2)
    labels: (M,) cluster ids
    return: {cluster_id: centroid_segment}
    """
```

Centroid = mean of all segments in that cluster (only if size ≥ `min_cluster_size`).

### 4.3 Regime Persistence

Define persistence for cluster `c`:

$$
P_{\text{persist}}(c) = P(c_{t+1} = c \mid c_t = c)
$$

Implement:

```python
def compute_persistence(labels: np.ndarray) -> Dict[int, float]:
    """
    labels: time-ordered cluster labels for segments.
    For each cluster c, compute persistence probability.
    """
```

### 4.4 Baseline Comparisons

To confirm ultrametric regimes are meaningful, compare persistence against:

1. **Random clustering** with same cluster size distribution.
2. **K-means** (Euclidean) on flattened segments.
3. **Volatility regimes**:
   - Compute rolling vol over last M bars (e.g., 20).
   - Sort into buckets (e.g., low/med/high).
   - Compare persistence.

---

## 5. Symplectic Dynamics (Global κ)

### 5.1 Encodings

From a K-bar segment, extract a single state `(q, π)`:

**Encoding A (volume-based):**

```python
q = v[-1]  # last bar's normalized, smoothed volume
π = p[-1] - p[-2]  # last bar's log-return
```

**Encoding B (price-only):**

```python
q = p[-1] - np.mean(p)  # deviation from segment mean
π = p[-1] - p[-2]
```

**Encoding C (hybrid):**

```python
q = alpha_q * v[-1] + (1 - alpha_q) * (p[-1] - np.mean(p))
π = p[-1] - p[-2]
```

Start with **Encoding A** for simplicity. Later test B and C.

### 5.2 Hamiltonian

```python
def hamiltonian(q: float, pi: float, kappa: float) -> float:
    return 0.5 * pi**2 + 0.5 * kappa * q**2
```

### 5.3 Leapfrog Integrator

```python
def leapfrog_step(
    q: float,
    pi: float,
    kappa: float,
    dt: float = 1.0
) -> Tuple[float, float]:
    """
    One leapfrog step for Hamiltonian H = 0.5*pi^2 + 0.5*kappa*q^2.
    Returns (q_next, pi_next).
    """
    pi_half = pi - 0.5 * dt * kappa * q
    q_next = q + dt * pi_half
    pi_next = pi_half - 0.5 * dt * kappa * q_next
    return q_next, pi_next
```

### 5.4 Global κ Estimation

From all training segments:

```python
def estimate_global_kappa(segments: np.ndarray, encoding: str) -> float:
    """
    segments: (M, K, 2)
    For each segment:
      - extract (q, pi) using chosen encoding
      - accumulate pi^2 and q^2
    kappa_global = mean(pi^2) / (mean(q^2) + epsilon)
    Clamp to [0.01, 10.0].
    """
```

---

## 6. Per-Cluster κ with Shrinkage

For the full hybrid model we need κ per cluster, with shrinkage and sanity bounds.

### 6.1 Raw κ per Cluster

For each cluster `c` with `n_c` segments:

1. Collect all segments in cluster `c`.
2. For each segment, extract `(q_t, pi_t)` using the chosen encoding.
3. Approximate:

   ```python
   mean_dp2_c = mean(pi_t^2 over all segments in c)
   mean_q2_c  = mean(q_t^2 over all segments in c)
   kappa_c_raw = mean_dp2_c / (mean_q2_c + epsilon)
   ```

### 6.2 Global κ

Compute a global baseline:

```python
kappa_global = weighted_mean(kappa_c_raw, weights=n_c)
```

### 6.3 Shrinkage

Define cluster-specific shrinkage:

```python
λ_c = 0.5 + 0.3 * min(n_c / 200.0, 1.0)
# small clusters shrink more toward kappa_global
```

Then:

```python
kappa_c_final = λ_c * kappa_c_raw + (1 - λ_c) * kappa_global
```

### 6.4 Bounds

Clamp κ to avoid numerical blowups:

```python
kappa_c_final = np.clip(kappa_c_final, 0.01, 10.0)
```

Implement in `model/symplectic_model.py`:

```python
def estimate_kappa_per_cluster(
    segments: np.ndarray,
    labels: np.ndarray,
    encoding: str,
    epsilon: float = 1e-8
) -> Dict[int, float]:
    """
    Return {cluster_id: kappa_c_final} using the algorithm above.
    """
```

---

## 7. Models & Signal API

All models live in `model/signal_api.py` and expose a uniform `get_signal(...)` interface.

### 7.1 AR1Model (Baseline)

- Fit AR(1) on returns `r_t = p_t - p_{t-1}`.
- Use predicted next return to generate direction/thresh-based signal.
- Used as the primary "simple baseline" for comparison.

### 7.2 SymplecticGlobalModel

- Uses **global κ** (no regimes).
- For each K-bar segment:
  - Extract `(q, pi)` from last bar.
  - One leapfrog step → `pi_next`.
  - `pi_next` ≈ predicted Δp.
- Threshold like AR(1) to decide long/short/flat.

### 7.3 SymplecticUltrametricModel (Full Hybrid)

Constructor:

```python
class SymplecticUltrametricModel:
    def __init__(
        self,
        config: dict,
        centroids: Dict[int, np.ndarray],
        kappa_per_cluster: Dict[int, float],
        hit_rates: Dict[int, float],
        encoding: str = 'A'
    ):
        ...
```

Internal:

- `_nearest_cluster(seg)`:
  - Compute ultrametric distance from `seg` to each `centroid`.
  - Return `(cluster_id, distance)` for nearest valid centroid or `(None, None)`.

`get_signal(last_k_bars)`:

1. Convert `last_k_bars` into `(K, 2)` segment `[p, v]`.
2. Find nearest cluster `(c_id, dist)`.
3. Gating:
   - If `c_id` is `None` → **flat**.
   - If `dist > epsilon_gate` → flat.
   - If `hit_rates[c_id] < hit_threshold` → flat.
4. Extract `(q, pi)` from segment using `encoding`.
5. Look up `kappa = kappa_per_cluster.get(c_id, kappa_global_fallback)`.
6. Run `leapfrog_step(q, pi, kappa, dt=1.0)` → `(q_next, pi_next)`.
7. Treat `pi_next` as `Δp_hat`.
   - If `abs(Δp_hat) <= theta` → flat.
   - Else `direction = sign(Δp_hat)`.
8. Confidence sizing:

   ```python
   size_factor = max(0.0, min(1.0, (hit_rates[c_id] - 0.5) / 0.05))
   ```

Return:

```python
{"direction": direction, "size_factor": size_factor}
```

---

## 8. Backtesting & Cost Model

All backtests live in `model/backtest.py`.

### 8.1 15-Minute NQ-like Costs

Use a conservative log-return cost per **round trip**:

```python
cost_log_15m = -0.00048  # ≈ -0.048% per round trip
```

Apply this whenever position changes (entering/exiting/flip).

### 8.2 Backtests to Implement

- `run_ar1_backtest(...)`
- `run_symplectic_global_backtest(...)`
- `run_hybrid_backtest(...)` for SymplecticUltrametric model.

Each should:

- Replay 15m bars.
- Track:
  - Position (−1, 0, +1).
  - Per-trade PnL and equity curve.
  - Hit rate (directional).
  - Avg net return per trade.
  - Sharpe ratio (basic: mean / std * √trades).
  - Max drawdown.
  - Trade count.

Use **the same cost model** across all backtests and baselines.

---

## 9. Success Criteria (Phase 1)

Before moving to Ninja integration or deeper engineering, the hybrid model should roughly satisfy:

### 9.1 Regime Quality

- Ultrametric cluster persistence:

  $$
  \mathbb{E}_c[P_{\text{persist}}(c)] > 0.65
  $$

- Random clustering ≈ 0.50, k-means baseline ≈ 0.55.

- At least **3 clusters** with:
  - ≥ 100 segments,
  - Persistence ≥ 0.60.

### 9.2 Forecast Quality

- Ungated 1-bar directional hit rate > 52%.
- Gated hit rate > 55%.
- RMSE on 1-bar forecast improved by at least 15% over AR(1).

### 9.3 Economic Viability

With 15m NQ-like cost model:

- Post-cost Sharpe > 1.0 on the test period.
- Average net return per trade > 2× per-trade cost.
- Max drawdown < 15% of cumulative profits.
- At least ~200 gated trades.

These are guidelines; if results are significantly worse, treat as research outcome, not deployment candidate.

---

## 10. Red Flags (Stop and Diagnose)

If you see **multiple** of these, pause and investigate before proceeding.

### 10.1 Data / Regime Issues

- Most clusters have `< 20` segments (too small to trust).
- Regime persistence similar for:
  - Ultrametric clustering,
  - Random labels,
  - K-means,
  - Volatility buckets.
- κ values vary by **> 10×** across clusters without any clear regime interpretation.

### 10.2 Overfitting Signals

- Train hit rate > 60%, but test hit rate < 52%.
- Performance collapses when you extend the test window.
- Edge appears only at very specific hyperparameter settings and vanishes with small changes.

### 10.3 Cost Model Issues

- Good pre-cost performance, but after costs, Sharpe ~ 0 or negative.
- Very few trades (< 100/year), making conclusions fragile.
- Average trade duration < 2 bars (likely trading noise and paying costs too often).

### 10.4 Implementation Bugs

- Ultrametric distance fails triangle inequality tests significantly.
- Leapfrog integrator misbehaves on simple synthetic tests (e.g., κ=1 constant-energy scenario).
- Look-ahead bias:
  - Using bar t data to decide actions credited to bar t,
  - Using future cluster labels when evaluating test performance.

If 2+ red flags show up, consider:

- Simplifying:
  - Fewer clusters,
  - One encoding instead of many.
- Changing timeframe to 1-hour bars for testing.
- Using simpler regime definitions (volatility buckets, trend filters, time-of-day) as a sanity comparison.

---

## 11. Implementation Steps (Pointer)

All the **small, ordered steps** for implementation are in:

```text
IMPLEMENTATION_STEPS.md
```

That file breaks the project into:

- STEP 0: Base pipeline + AR(1) baseline.
- STEP 1: Segments + ultrametric distance.
- STEP 2: Clustering + regime persistence vs baselines.
- STEP 3: SymplecticGlobalModel (single κ).
- STEP 4: SymplecticUltrametricModel (full hybrid + gating).
- STEP 5: Validation & cleanup.
- STEP 6: Optional NinjaTrader HTTP integration.

Follow that staircase directly.
Use this `CLAUDE.md` and `README.md` as the conceptual spec and reference.

---

## 12. Philosophy

- The point is not to prove markets are "ultrametric" or "Hamiltonian."
- The point is to see if these structures yield **useful, testable edge** on **15-minute NQ/QQQ** data, **after** realistic costs.
- It's totally acceptable for the final conclusion to be:
  - "This doesn't make money, but the implementation is correct and the experiment was worthwhile."

The math can be unusual.
The code and validation should be simple, clear, and brutally honest.