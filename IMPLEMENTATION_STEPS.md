# IMPLEMENTATION_STEPS.md

Ultrametric–Symplectic NQ Project – Step-by-Step Implementation Plan

This file is a **strict staircase** for implementation.

- Each step is **small and self-contained**.  
- Only touch the files listed in that step.  
- Do **not** jump ahead or mix steps.  
- After each step, run a small sanity test before moving on.

The idea:  
**Base pipeline → AR(1) baseline → segments+ultrametric → clustering → symplectic → hybrid → backtests → (optional) Ninja integration.**

---

## STEP 0 – Bare Bones Pipeline + Simple AR(1) Baseline

**Goal:** Go from CSV → log prices → AR(1) forecast → simple backtest.  
No ultrametric. No symplectic. No regimes.

### 0.0 Files you may touch

- `model/data_utils.py`
- `model/signal_api.py`
- `model/backtest.py`
- `configs/config.yaml`
- (Optional) `notebooks/exploration.ipynb`

Do **not** edit other modules in this step.

---

### 0.1 Data Loading & Transform (15-Minute Bars)

In `model/data_utils.py` implement:

1. `load_ohlcv_csv(path: str) -> pd.DataFrame`  
   - Load CSV with columns: `timestamp, open, high, low, close, volume`.  
   - Parse `timestamp` as DateTime.  
   - Set DateTime as index.

2. `resample_to_15m(df: pd.DataFrame) -> pd.DataFrame`  
   - If the data is not already 15m, resample to 15-minute bars:
     - `open` = first, `high` = max, `low` = min, `close` = last, `volume` = sum.
   - Return 15m OHLCV DataFrame.

3. `compute_log_price(df: pd.DataFrame) -> np.ndarray`  
   - Returns `p` = `np.log(df["close"].values)`.

4. `compute_smoothed_volume(df, normalization_window, ema_period) -> np.ndarray`  
   - Rolling mean volume over `normalization_window` bars.
   - `vol_norm = volume / (rolling_mean + eps)`.
   - Apply simple EMA with `ema_period`.
   - Return `v` as 1D array, same length as `p`.

5. `build_gamma(p: np.ndarray, v: np.ndarray) -> np.ndarray`  
   - Stack into shape `(N, 2)` where `gamma[t] = [p[t], v[t]]`.

**Checkpoint 0.1:**  
Create a tiny script or notebook to:

- Load `data/sample_qqq_15m.csv`.
- Call all above functions.
- Print shapes of `p`, `v`, `gamma` and first few values.

---

### 0.2 Simple AR(1) Model for Returns

In `model/signal_api.py` implement:

```python
class AR1Model:
    def __init__(self, config: dict):
        self.config = config
        self.phi = 0.0
        self.mean_ret = 0.0

    def fit(self, p: np.ndarray) -> None:
        """
        Fit AR(1) on returns r_t = p_t - p_{t-1}.
        Estimate phi and mean_ret.
        """
        ...

    def get_signal(self, last_price: float, prev_price: float) -> dict:
        """
        Predict next return using AR(1).
        Returns:
          {
            "direction": -1 | 0 | 1,
            "size_factor": float in [0, 1]
          }
        """
        ...
```

Logic:

- `fit`:
  - Compute `r_t = p[1:] - p[:-1]`.
  - Estimate AR(1) parameters (e.g. via OLS).
  - Set `self.mean_ret` and `self.phi`.

- `get_signal`:
  - Compute `r_t = last_price - prev_price`.
  - Predict `r_hat = mean_ret + phi * (r_t - mean_ret)`.
  - Read `theta` from `config["signal"]["theta_ar1"]`.
  - If `abs(r_hat) <= theta` → `direction = 0`, `size_factor = 0.0`.
  - Else `direction = sign(r_hat)`, `size_factor = 1.0`.

---

### 0.3 Minimal AR(1) Backtest

In `model/backtest.py` implement:

```python
def run_ar1_backtest(
    model: AR1Model,
    p: np.ndarray,
    cost_log: float
) -> dict:
    """
    Replays log prices p.
    On each bar, uses AR1Model.get_signal() to decide position.
    Applies cost_log when changing position.
    Returns stats and trade log.
    """
    ...
```

Rules:

- Position ∈ {-1, 0, 1}.
- Start flat.
- Loop t from 1 to N-1:
  - Call `signal = model.get_signal(p[t], p[t-1])`.
  - If `signal["direction"] != current_position`:
    - If current_position != 0 → realize PnL on that bar.
    - Enter new position if direction != 0.
    - Apply `cost_log` on position change.
- Track:
  - Per-bar PnL and equity,
  - Trade entries/exits,
  - Win rate,
  - Average net return per trade,
  - Sharpe ratio (simple: mean / std * √trades),
  - Max drawdown.

**Checkpoint 0.3:**

- Fit AR(1) model on `p`.
- Run `run_ar1_backtest(...)` on 15m QQQ/NQ.
- Confirm:
  - Code runs without errors.
  - Metrics look reasonable (not NaN; trades being taken).

Stop here before touching ultrametric/symplectic.

---

## STEP 1 – Segments & Ultrametric Distance (No Clustering, No Symplectic)

**Goal:** Build K-bar segments and implement ultrametric distances.
Still use **only AR(1)** for trading.

### 1.0 Files you may touch

- `model/trainer.py`
- `model/ultrametric.py`
- `tests/test_ultrametric.py` (optional but ideal)
- (You may add a tiny helper function in a notebook for demos)

Do **not** change `signal_api` or `backtest` logic here.

---

### 1.1 Segment Builder

In `model/trainer.py`:

```python
def build_segments(gamma: np.ndarray, K: int) -> np.ndarray:
    """
    gamma: (N, 2) with columns [p, v]
    returns: segments of shape (N-K+1, K, 2)
    where segments[i] = gamma[i : i+K]
    """
    ...
```

**Checkpoint 1.1:**

- On `gamma` from STEP 0, K=10:
  - Verify `segments.shape == (N-9, 10, 2)`.
  - Print the first and last segment to check.

---

### 1.2 Ultrametric Distance

In `model/ultrametric.py`:

```python
import numpy as np
from math import floor, log
from typing import Optional

def ultrametric_dist(
    seg1: np.ndarray,
    seg2: np.ndarray,
    base_b: float = 2.0,
    eps: float = 1e-10
) -> float:
    """
    Compute ultrametric distance between two segments (K, 2) [p, v].

    Sketch:
    - Compute norms per bar: sqrt(p^2 + v^2)
    - Take valuation = floor(log_b(norm + eps))
    - Find first index j where valuations differ
      * if none differ: distance = 0
      * else: distance = base_b ** (-j)
    """
    ...
```

Optional `tests/test_ultrametric.py`:

- `test_self_distance_zero`: `d(seg, seg) == 0`.
- `test_symmetry`: `d(seg1, seg2) ≈ d(seg2, seg1)`.
- `test_ultrametric_inequality`:
  - For random segs x,y,z:
    - Check `d(x,z) <= max(d(x,y), d(y,z)) + tol`.

**Checkpoint 1.2:**

- Compute a small distance matrix for first ~20 segments.
- Confirm diagonal is 0, values are non-negative and not all identical.
- Ultrametric inequality test passes (within tolerance).

AR(1) remains the only model used for trading at this point.

---

## STEP 2 – Clustering & Regime Persistence (Still No Symplectic)

**Goal:** Use ultrametric distance to form clusters, then test regime persistence vs baselines.
Trading is **still** AR(1) only.

### 2.0 Files you may touch

- `model/clustering.py`
- `model/trainer.py` (small helper to run clustering)
- Optional: `notebooks/regime_persistence.ipynb`

Do **not** touch `signal_api` or `backtest` models yet.

---

### 2.1 Clustering via Ultrametric Distances

In `model/clustering.py`:

```python
import numpy as np
from typing import Tuple, Dict
from scipy.cluster.hierarchy import linkage, fcluster

from .ultrametric import ultrametric_dist

def cluster_segments_ultrametric(
    segments: np.ndarray,
    num_clusters: int,
    base_b: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use a subsample of segments.
    - Compute pairwise distances via ultrametric_dist.
    - Apply hierarchical clustering (e.g., 'ward').
    Returns:
      labels: (M,) cluster ids for subsample indices
      Z: linkage matrix
    """
    ...
```

Also:

```python
def compute_centroids(
    segments: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int
) -> Dict[int, np.ndarray]:
    """
    Compute mean segment per cluster with size >= min_cluster_size.
    Return {cluster_id: centroid_segment}.
    """
    ...
```

Use a subsample (e.g., first 5000 segments) for the distance matrix to keep it fast.

---

### 2.2 Regime Persistence

```python
def compute_persistence(labels: np.ndarray) -> Dict[int, float]:
    """
    labels: time-ordered cluster labels for segments.
    For each cluster c:
      count transitions c -> c vs c -> other.
      return {cluster_id: persistence_prob}
    """
    ...
```

---

### 2.3 Baseline Comparisons

Implement alternative clustering baselines:

1. **Random clustering:**

```python
def cluster_random(segments: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Randomly assign cluster labels while preserving cluster size distribution.
    """
    ...
```

2. **K-means (Euclidean):**

```python
def cluster_kmeans(segments: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Flatten segments to (M, K*2) and run sklearn KMeans.
    """
    ...
```

3. **Volatility regimes:**

```python
def cluster_volatility(p: np.ndarray, num_buckets: int = 3, window: int = 20) -> np.ndarray:
    """
    Compute rolling volatility over window.
    Assign segments to low/med/high vol buckets.
    """
    ...
```

**Checkpoint 2.3:**

- Compare persistence across:
  - Ultrametric clustering,
  - Random clustering,
  - K-means,
  - Volatility regimes.
- Expected: ultrametric should have noticeably higher persistence than random and k-means.
- If not, this is a red flag.

AR(1) still used for trading.

---

## STEP 3 – Symplectic Model (Global κ Only)

**Goal:** Add symplectic forecasting with a **single global κ** (no per-cluster κ yet).
Compare SymplecticGlobalModel vs AR(1) baseline.

### 3.0 Files you may touch

- `model/symplectic_model.py`
- `model/signal_api.py`
- `model/backtest.py`

---

### 3.1 Leapfrog & Encodings

In `model/symplectic_model.py`:

```python
from typing import Tuple

def hamiltonian(q: float, pi: float, kappa: float) -> float:
    """H = 0.5 * pi^2 + 0.5 * kappa * q^2"""
    return 0.5 * pi**2 + 0.5 * kappa * q**2

def leapfrog_step(q: float, pi: float, kappa: float, dt: float = 1.0) -> Tuple[float, float]:
    """
    Single leapfrog step for:
      H = 0.5 * pi^2 + 0.5 * kappa * q^2
    Returns (q_next, pi_next).
    """
    ...
```

Encodings:

```python
def extract_state_from_segment(
    segment: np.ndarray,
    encoding: str
) -> Tuple[float, float]:
    """
    segment: (K, 2) [p, v]
    encoding:
      'A' - volume-based,
      'B' - price-only,
      'C' - hybrid.
    Returns (q_t, pi_t) based on last bar.
    """
    ...
```

Use the same logic we outlined earlier.

---

### 3.2 Estimate Global κ

In `model/symplectic_model.py`:

```python
def estimate_global_kappa(
    segments: np.ndarray,
    encoding: str,
    epsilon: float = 1e-8
) -> float:
    """
    Compute a single κ for all segments.
    1) For each segment, extract (q_t, pi_t).
    2) Approximate Δp_t ~ pi_t.
    3) Use mean(dp^2) / mean(q^2) with regularization.
    """
    ...
```

---

### 3.3 SymplecticGlobalModel

In `model/signal_api.py`:

```python
class SymplecticGlobalModel:
    def __init__(self, config: dict, kappa: float, encoding: str = 'A'):
        self.config = config
        self.kappa = kappa
        self.encoding = encoding

    def get_signal(self, segment: np.ndarray) -> dict:
        """
        segment: (K, 2) [p, v]
        1) Extract (q, pi),
        2) Apply leapfrog_step once,
        3) Use predicted pi_next as Δp_hat,
        4) Threshold & sign like AR(1).
        """
        ...
```

---

### 3.4 Symplectic Global Backtest

In `model/backtest.py`:

```python
def run_symplectic_global_backtest(
    model: SymplecticGlobalModel,
    segments: np.ndarray,
    p: np.ndarray,
    K: int,
    cost_log: float
) -> dict:
    """
    Use sliding K-bar segments as inputs.
    On each eligible bar:
      - Build segment ending at t,
      - Call model.get_signal(segment),
      - Update position & PnL like AR(1) backtest.
    """
    ...
```

**Checkpoint 3.4:**

- Run AR(1) vs SymplecticGlobalModel on same 15m data.
- Compare hit rates and post-cost metrics.
- If Symplectic is much worse everywhere, another red flag. If it's comparable or slightly better, proceed.

---

## STEP 4 – Full Hybrid: Clusters + Per-Cluster κ + Gating

**Goal:** Combine ultrametric regimes + per-cluster κ + symplectic forecast + gating into `SymplecticUltrametricModel`.

### 4.0 Files you may touch

- `model/symplectic_model.py` (add per-cluster κ estimation)
- `model/signal_api.py` (main hybrid model)
- `model/trainer.py` (train pipeline wrapper)
- `model/backtest.py` (hybrid backtest)

---

### 4.1 κ per Cluster (with Regularization)

In `model/symplectic_model.py`:

```python
from typing import Dict

def estimate_kappa_per_cluster(
    segments: np.ndarray,
    labels: np.ndarray,
    encoding: str,
    epsilon: float = 1e-8
) -> Dict[int, float]:
    """
    For each cluster c:
      1) Collect segments belonging to c,
      2) Extract (q, pi) for each,
      3) Compute raw κ_c = mean(dp^2) / mean(q^2 + eps),
      4) Compute global κ_global across all clusters,
      5) Shrink κ_c toward κ_global based on cluster size,
      6) Clip κ_c_final to [0.01, 10.0].
    """
    ...
```

---

### 4.2 SymplecticUltrametricModel

In `model/signal_api.py`:

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
        self.config = config
        self.centroids = centroids
        self.kappa_per_cluster = kappa_per_cluster
        self.hit_rates = hit_rates
        self.encoding = encoding

    def _nearest_cluster(self, seg: np.ndarray) -> Tuple[int | None, float | None]:
        """
        Return (cluster_id, ultrametric_distance_to_centroid).
        If no valid centroid, return (None, None).
        """
        ...
```

`get_signal(last_k_bars)`:

- `last_k_bars`: `(K, 2)` [p, v].
- Steps:
  1. `_nearest_cluster(seg)` → `(c_id, dist)`.
  2. Gating:
     - If `c_id is None` → flat.
     - If `dist > epsilon` → flat.
     - If `hit_rates[c_id] < hit_threshold` → flat.
  3. Extract `(q, pi)` via `extract_state_from_segment(seg, encoding)`.
  4. Look up `kappa = kappa_per_cluster.get(c_id, kappa_fallback)`.
  5. `(_, pi_next) = leapfrog_step(q, pi, kappa, dt=1.0)`.
  6. Use `pi_next` as `Δp_hat`.
     - If `|Δp_hat| <= theta` → flat.
     - Else `direction = sign(Δp_hat)`.
  7. `size_factor = (hit_rates[c_id] - 0.5) / 0.05` → clip to `[0, 1]`.

Return:

```python
{"direction": direction, "size_factor": size_factor}
```

---

### 4.3 Hybrid Backtest

In `model/backtest.py`:

```python
def run_hybrid_backtest(
    model: SymplecticUltrametricModel,
    p: np.ndarray,
    v: np.ndarray,
    K: int,
    cost_log: float
) -> dict:
    """
    Similar to symplectic global backtest, but:
      - Build (K, 2) segment per bar,
      - Call model.get_signal(segment),
      - Apply cost + position logic.
    """
    ...
```

**Checkpoint 4.3:**

- Compare:
  - AR(1),
  - SymplecticGlobal,
  - SymplecticUltrametric (hybrid),
    on same 15m dataset with same `cost_log`.
- Evaluate vs success criteria (hit rate, Sharpe, drawdown, trade count).

---

## STEP 5 – Final Phase 1 Validation & Cleanup

**Goal:** Lock down Phase 1: solid experiments, clear results, clean code.

### 5.0 Files you may touch

- `notebooks/` (for plots and summary)
- `README.md` (document results)
- `CLAUDE.md` (lessons learned)
- Small refactors in `model/*` as needed (no big changes).

Tasks:

- [ ] Run all three models on train/val/test splits.
- [ ] Summarize:
  - Regime persistence results vs baselines.
  - AR(1) vs SymplecticGlobal vs Hybrid performance.
  - Impact of cost model.
- [ ] Decide:
  - Is this behaving like **interesting research** or **something with real edge**?
- [ ] If results are weak:
  - Document that clearly as a finding.
- [ ] If results are promising:
  - Note which config/encoding looks best.

---

## STEP 6 – Optional: NinjaTrader Integration (15-Minute Only)

**Goal:** Let Tony watch this trade in Ninja Simulation / Playback.

This is Phase 2 and can wait until Phase 1 is stable.

### 6.0 Files you may touch

- `server/app.py` (or similar) – small FastAPI/Flask server
- `model/signal_api.py` (to load model + expose `get_signal` to API)

Outline:

- [ ] Implement `POST /signal` that:
  - Accepts JSON with last K 15m bars + optional account snapshot.
  - Rebuilds `(K, 2)` segment.
  - Calls `SymplecticUltrametricModel.get_signal`.
  - Returns JSON `{direction, size_factor}`.

- [ ] Implement `POST /trade_log` for Ninja to send fills.

The NinjaScript strategy will:

- Build last K bars from its 15m series,
- Call Python `/signal` in `OnBarUpdate`,
- Submit orders accordingly in Sim101/Playback101,
- Call `/trade_log` from `OnExecutionUpdate`.

---

## Summary

Follow the steps in order:

- **STEP 0:** CSV → 15m bars → AR(1) baseline + backtest.
- **STEP 1:** Segments + ultrametric distance (no trading with it yet).
- **STEP 2:** Clustering + regime persistence vs baselines.
- **STEP 3:** Symplectic "physics" model with one global κ.
- **STEP 4:** Full hybrid (clusters + per-cluster κ + gating).
- **STEP 5:** Validate everything on 15m bars with costs, clean up docs.
- **STEP 6:** (Optional) Wire the signal API into NinjaTrader Simulation / Playback.

Each step is bite-sized and self-contained, so an LLM can implement them one by one without getting lost.