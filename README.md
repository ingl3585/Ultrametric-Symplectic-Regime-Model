# Ultrametric–Symplectic Regime Model for NQ (15-Minute Bars)

Research project for a bar-based trading engine that:

- Clusters recent price/volume **shapes** into regimes using an **ultrametric distance**.
- Within each regime, uses a simple **Hamiltonian (symplectic)** model to forecast the **next bar's** direction.
- Exposes a clean **signal API** that can later be wired into **NinjaTrader Simulation / Playback** for NQ.

> Status: **Research / Experimental**  
> Timeframe: **15-minute bars only** (no tick or 2-minute logic in v1)  
> Instruments: **QQQ** for initial research, **NQ** via NinjaTrader for playback

---

## 0. Philosophy

This project is **research-first**, not "plug in and get rich."

We're testing:

- Whether **ultrametric clustering** finds regimes that **persist** and differ meaningfully.
- Whether a simple **symplectic / Hamiltonian** forecast inside a regime **beats simple baselines**.
- Whether any apparent edge survives **realistic trading costs** on **15-minute bar data**.

The math is allowed to be fancy.  
The **results** must be tested, boring, and honest.

---

## 1. What This Repo Is / Is Not

**This repo IS:**

- A Python research engine for:
  - Loading 15-minute OHLCV data (CSV),
  - Transforming into price/volume phase space,
  - Building K-bar segments,
  - Clustering via ultrametric distance,
  - Fitting symplectic models,
  - Backtesting everything on 15-minute bars with costs.
- A **ready-to-wire** signal API that can be called by NinjaTrader (Simulation / Playback) as:
  - `POST /signal` → `{"direction": -1|0|1, "size_factor": 0–1}`.

**This repo is NOT (yet):**

- A full production trading system.
- A complete NinjaTrader strategy implementation (C# lives in Ninja).
- A tick-level HFT model (we stay on 15-minute bars in v1).

---

## 2. Data & Timeframe

### 2.1 Timeframe & Bars

Version 1 is **strictly 15-minute bar based**:

- Offline research:
  - 15-minute QQQ or NQ OHLCV from CSV (e.g., exports from Ninja or another source).
- NinjaTrader integration (later):
  - NinjaTrader pulls historical NQ data from its data provider,
  - Builds 15-minute bars,
  - Feeds them to your strategy via `OnBarUpdate` on **bar close**.

Tick data and sub-15m horizons are **out of scope** for v1.

### 2.2 Required Fields

Each bar must have at least:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

---

## 3. Core Idea (Plain English)

Every 15 minutes we want to answer:

1. **What "market mood" are we in?**  
   (Detect regime based on the last K bars of price + volume.)

2. **Given that mood, is the next bar more likely up or down?**  
   (Use a tiny physics-like model to forecast direction.)

We do this in two main layers:

### 3.1 Ultrametric Regimes

- Take the last **K** 15m bars (e.g., last 10 bars = 2.5 hours).
- For each bar, track:
  - `p_t` = log price,
  - `v_t` = smoothed, normalized volume.
- That window of K bars is a **shape** in `[p, v]` space.
- The **ultrametric distance** says:
  - "At what scale do these two shapes first stop looking alike?"
  - If they match on big structure and differ only in small wiggles → very close.
  - If they diverge early or on big moves → far apart.
- Using this distance, we cluster many historical windows into regimes, and then measure:
  - **Persistence:**  
    `P(cluster_t+1 = same | cluster_t)`  
  - **Outcome bias:**  
    e.g. average next-bar return by cluster.

### 3.2 Symplectic Forecast (Tiny Physics Engine)

Inside a regime:

- Treat the current state like a **mass on a spring**:
  - `q` (position) = something like deviation from normal, or volume signal.
  - `π` (momentum) = recent price change.
- Use a simple Hamiltonian:

  $$
  H(q,\pi) = \tfrac{1}{2}\pi^2 + \tfrac{1}{2}\kappa q^2
  $$

- `κ` (kappa) is the "stiffness" of the regime: how strongly price tends to snap back vs continue.
- Run one step of a **leapfrog integrator** to get a forecast for the next bar's Δprice:
  - If forecast small → do nothing,
  - If forecast large enough:
    - sign(Δprice^) → long or short,
    - "confidence" → how big to size (as a fraction of max size).

So in practice:

> "Look up past situations where the last K bars looked like this.  
> If those situations tended to have a clear directional bias for the next bar, and our little physics model agrees, and the forecast is big enough to beat costs, then take a trade. Otherwise, sit flat."

---

## 4. Architecture Overview

### 4.1 Python Modules (Conceptual)

- `model/data_utils.py`
  - Load OHLCV CSV.
  - Resample to 15-minute bars (if needed).
  - Compute:
    - `p_t = log(close_t)`
    - `v_t` = smoothed, normalized volume.
  - Build `gamma_t = [p_t, v_t]`.

- `model/trainer.py`
  - Build K-bar segments from `gamma`.
  - Orchestrate training flows (segments → clustering → kappa, etc).

- `model/ultrametric.py`
  - Implements `ultrametric_dist(seg1, seg2)` for segments `(K, 2)`.

- `model/clustering.py`
  - Hierarchical clustering on ultrametric distances (subsampled).
  - Compute centroids (mean segment per cluster).
  - Compute regime persistence.
  - Compare to baselines: random, k-means, volatility buckets.

- `model/symplectic_model.py`
  - Hamiltonian / leapfrog integrator.
  - Encodings A/B/C (volume-based, price-only, hybrid).
  - κ estimation:
    - Global κ,
    - Per-cluster κ with shrinkage toward global.

- `model/signal_api.py`
  - `AR1Model` (simple baseline).
  - `SymplecticGlobalModel` (symplectic without regimes).
  - `SymplecticUltrametricModel` (full hybrid).
  - Each exposes `get_signal(...) -> {"direction": -1|0|1, "size_factor": 0–1}`.

- `model/backtest.py`
  - Backtesting harness for:
    - AR(1),
    - Symplectic global,
    - Symplectic + ultrametric hybrid.
  - Applies **15m NQ-like costs** to all strategies and baselines.

- `server/app.py` (optional, Phase 2)
  - Simple FastAPI/Flask service exposing `/signal` and `/trade_log` for NinjaTrader.

---

## 5. Phase 1 – 15-Minute Research Engine

Phase 1 is purely Python-based, using CSV data and backtests.

### 5.1 Minimum Viable Staircase

Implementation is broken into **small steps** (see `IMPLEMENTATION_STEPS.md` for details):

1. **STEP 0:**  
   - Data pipeline (15m bars, `p`, `v`),  
   - Simple AR(1) model,  
   - Basic backtest with costs.

2. **STEP 1:**  
   - Build K-bar segments and ultrametric distance function.  
   - No clustering yet – just test distances.

3. **STEP 2:**  
   - Clustering on segments using ultrametric.  
   - Measure regime persistence vs random / k-means / vol buckets.

4. **STEP 3:**  
   - Symplectic "physics" model with **global κ** (no regimes).  
   - Backtest vs AR(1).

5. **STEP 4:**  
   - Full hybrid:
     - Clusters,
     - Per-cluster κ,
     - Gating by distance and cluster quality,
     - Symplectic forecast inside regimes.

6. **STEP 5:**  
   - Full validation on 15m data with costs.  
   - Summaries, plots, and decision: "Interesting research" vs "Potential edge".

---

### 5.2 Phase 1 Success Criteria (Guidelines)

These are **go/no-go** style guidelines before investing time in execution/integration.

**Regime Quality (15m segments)**

- Ultrametric cluster persistence:

  $$
  P(c_{t+1}=c \mid c_t=c) > 0.65
  $$

  on average, where:
  - Random cluster baseline ≈ 0.50,
  - k-means baseline ≈ 0.55.

- At least **3 clusters** with:
  - ≥ 100 segments each,
  - persistence ≥ 0.60.

**Forecast Quality (per 15m bar, within regimes)**

- Ungated 1-bar hit rate **> 52%** (directional).
- Gated hit rate (distance + cluster quality filters) **> 55%**.
- RMSE improvement **> 15%** vs an AR(1) baseline on returns.

**Economic Viability (15m NQ-like, with costs)**

Using a conservative 15m cost model (see next section):

- Post-cost Sharpe **> 1.0** on test set.
- Average net return per trade > **2× cost** (e.g., > 0.10% net for ≈0.048% cost).
- Max drawdown < **15%** of total profits.
- At least **~200 gated trades** on test data (enough samples to care).

If the model misses these by a wide margin, treat this as interesting research, not something to deploy.

---

## 6. Cost Model (15-Minute NQ Approximation)

We need a realistic approximation of NQ costs, even if we test on QQQ.

**Components per contract, per round trip (ballpark):**

- Exchange + NFA + broker fees: ≈ \$3.50 per round trip.
- Slippage:
  - On 15m NQ bars, effective slippage might be ≈ 0.25–0.375 points.
  - With NQ around ~21,000, that's ≈ 0.024–0.036%.

We simplify this into a **single log-return cost**:

```python
cost_log_15m ≈ -0.00048  # ≈ -0.048% per round trip (conservative)
```

This cost should be applied to **every trade** in the backtests (for both the hybrid model and all baselines).

**Break-even intuition (example):**

- If you only trade when forecast magnitude ≈ 0.0012 (0.12%):
  - With 0.048% costs, you need:
    - ~62% hit rate to break even,
    - ~65%+ hit rate for a decent Sharpe.

---

## 7. Implementation Steps

The detailed, low-hallucination implementation staircase is in:

```text
IMPLEMENTATION_STEPS.md
```

That file breaks the build into:

- Very small, ordered steps,
- Each touching only a few files,
- With clear checkpoints at each stage.

Rough outline:

- **Step 0:** CSV → 15m bar pipeline + AR(1) baseline backtest.
- **Step 1:** Segments + ultrametric distance (no clustering).
- **Step 2:** Clustering + regime persistence vs baselines.
- **Step 3:** SymplecticGlobalModel vs AR(1).
- **Step 4:** SymplecticUltrametricModel (full hybrid).
- **Step 5:** Final validation / cleanup.
- **Step 6:** Optional NinjaTrader integration.

LLMs (Claude, ChatGPT, etc.) should follow that file step-by-step rather than trying to "build everything" from this README at once.

---

## 8. NinjaTrader Integration (Phase 2, 15m Only)

Once Phase 1 looks promising, we can wire the model into **NinjaTrader Simulation / Playback**.

### 8.1 Division of Labor

**NinjaTrader (NinjaScript, C#) handles:**

- Connecting to your data provider.
- Pulling **historical NQ data**.
- Building **15-minute bars**.
- Calling `OnBarUpdate()` on each bar close.
- Tracking:
  - Account balance,
  - Positions,
  - Realized/Unrealized PnL.
- Simulated fills in **Sim101** or **Playback**.

**Python handles:**

- No data fetching.
- No bar construction.
- No order routing.

Python is a pure **math service**:

- Input: last **K** 15m bars (close + volume, maybe time) and optionally a small account snapshot.
- Output: `{"direction": -1|0|1, "size_factor": 0–1}`.

### 8.2 Suggested HTTP API

Python (FastAPI/Flask):

- `POST /signal`

  ```json
  {
    "bars": [
      { "time": "2025-01-01T14:00:00Z", "close": 21000.0, "volume": 1500.0 },
      { "time": "2025-01-01T14:15:00Z", "close": 21020.0, "volume": 1600.0 }
      // ... last K 15m bars
    ],
    "account": {
      "accountId": "Sim101",
      "cashValue": 100000.0,
      "realizedPnL": 250.50,
      "unrealizedPnL": -75.25,
      "totalBuyingPower": 200000.0,
      "positionQuantity": 2,
      "positionAvgPrice": 21000.0
    }
  }
  ```

  Return:

  ```json
  {
    "direction": 1,
    "size_factor": 0.6
  }
  ```

- `POST /trade_log`

  ```json
  {
    "time": "2025-01-01T15:00:00Z",
    "instrument": "NQ 03-25",
    "side": "Long",
    "quantity": 1,
    "price": 21050.0,
    "realizedPnL": 25.0,
    "strategy": "UltrametricSymplectic15m"
  }
  ```

NinjaScript strategy:

- `OnStateChange()`:
  - Configure 15m NQ data series,
  - Set `Calculate.OnBarClose`.

- `OnBarUpdate()`:
  - Only for primary BarsInProgress (15m).
  - Build last K 15m bars from Ninja's bar series.
  - Build small account snapshot from `Account`.
  - Async POST to `/signal`.
  - Use returned signal to:
    - Enter/exit long or short via `EnterLong()`, `EnterShort()`, `ExitLong()`, etc.

- `OnExecutionUpdate()`:
  - For each fill:
    - Build trade log JSON.
    - Async POST to `/trade_log`.

---

## 9. Known Limitations

- **Regime model:**
  - Assumes regimes are discrete and somewhat hierarchical.
  - Real markets may have overlapping or continuous regimes.
  - Ultrametric geometry is a modeling choice, not a claim of truth.

- **Dynamics model:**
  - Hamiltonian assumes approximate energy conservation and smooth dynamics.
  - Real markets have friction, jumps, and information shocks.
  - We use this as a **compact forecast structure**, not literal physics.

- **Scope:**
  - Single-instrument, single-timeframe (NQ, 15m).
  - Only directional strategies (no market-making).
  - No fundamental or sentiment inputs.

- **Edge realism:**
  - 15m bars are less noisy than 2m, but still noisy.
  - After realistic costs, true edge may be small or nonexistent.
  - Backtests must be walk-forward, out-of-sample, and cost-aware.

---

## 10. Summary

This repo defines a clear research path to:

- Test ultrametric clustering as a way to detect persistent regimes on 15-minute bar data.
- Test symplectic (Hamiltonian) forecasting inside those regimes.
- Compare results against sane baselines, with realistic NQ-like costs.
- If it looks promising, plug the model into NinjaTrader Simulation / Playback to watch it trade NQ in a safe environment.

For step-by-step implementation details, see:

```text
IMPLEMENTATION_STEPS.md
```

The math can be exotic.
The implementation and validation should be simple, transparent, and honest.