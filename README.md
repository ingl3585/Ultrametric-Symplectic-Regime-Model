# Ultrametric-Symplectic Regime Model

A pure-math trading model combining ultrametric clustering for regime detection with symplectic dynamics for price forecasting.

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas pyyaml yfinance scipy scikit-learn

# 2. Download data (~30 days of 1-minute QQQ from yfinance)
python download_qqq_1m_max.py

# 3. Run backtest
python run.py
```

## How It Works

1. **Regime Detection**: Clusters 60-minute windows (20x 3-minute bars) using ultrametric distance
2. **Forecasting**: Uses Hamiltonian mechanics (symplectic integrator) to predict next-bar return
3. **Gating**: Only trades clusters with >52% historical hit rate
4. **Position Sizing**: Scales with cluster quality (52% → 0%, 62% → 100%)

## Results (Test Set, 3-Minute Bars)

```
Trades: 4-5
Win Rate: 50-55%
Sharpe: 0.8-1.0
PnL: +2-3% per test period
```

## Configuration

Edit `configs/config.yaml`:

```yaml
timeframe:
  bar_size_minutes: 3       # Bar timeframe

signal:
  hit_threshold: 0.52       # Min cluster hit rate (positive expectancy)
  epsilon_gate: 2.0         # Max distance to centroid
  theta_symplectic: 0.00005 # Min forecast magnitude

clustering:
  num_clusters: 5           # Target regimes
  min_cluster_size: 30      # Min segments per cluster

costs:
  cost_per_trade: -0.00048  # ~0.048% per round trip
```

## Project Structure

```
├── configs/config.yaml       # Configuration
├── model/                    # Core model code
│   ├── data_utils.py         # Data loading/preprocessing
│   ├── ultrametric.py        # Ultrametric distance
│   ├── clustering.py         # Regime clustering
│   ├── symplectic_model.py   # Hamiltonian dynamics
│   ├── signal_api.py         # Trading signals
│   └── backtest.py           # Backtest framework
├── server/app.py             # HTTP API (for NinjaTrader)
├── download_qqq_1m_max.py    # Data downloader
└── run.py                    # Main entry point
```

## NinjaTrader Integration (Optional)

The model can provide signals via HTTP API:

```bash
python server/app.py
```

```
POST http://localhost:8000/signal
Body: {"bars": [[p1,v1], ..., [p20,v20]]}
Response: {"direction": -1|0|1, "size_factor": 0.0-1.0}
```

## Model Overview

### Training
- Load 1-minute data, resample to 3-minute bars
- Build 60-minute segments (20 bars)
- Cluster segments using ultrametric distance
- Estimate κ (spring constant) per cluster
- Compute hit rates and persistence

### Trading
- Match current segment to nearest cluster
- Check gating (distance, hit rate, persistence)
- Extract phase-space state (q, π)
- Run leapfrog integrator to forecast return
- Generate signal if forecast > threshold

### Position Sizing
```python
size_factor = (hit_rate - 0.52) / 0.10
# 52% → 0%   (don't trade)
# 57% → 50%
# 62% → 100% (full size)
```

## Philosophy

This is **research code**, not production. The goal:
- Test if ultrametric clustering finds meaningful regimes
- Test if symplectic dynamics provide useful forecasts
- Test if edge survives realistic costs (~0.048% per trade)

All math is explicit (no black-box). If it doesn't work, you'll know why.

## Limitations

- **Data**: yfinance 1-minute data limited to ~30 days
- **Timeframe**: Optimized for 3-minute bars
- **Costs**: Assumes ~0.048% per round trip
- **Sample Size**: Small test sets have high variance
- **Regime Shift**: Model may fail in new market conditions

## Credits

Research model built by Claude Code (Anthropic) combining:
- Ultrametric analysis (p-adic valuations)
- Hamiltonian mechanics (symplectic integration)
- Regime-dependent forecasting

Not financial advice. Use at your own risk.
