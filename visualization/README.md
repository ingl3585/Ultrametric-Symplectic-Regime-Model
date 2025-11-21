# Phase Space Visualization System

Real-time and offline visualization tools for understanding the Ultrametric-Symplectic trading model's internal state.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r visualization/requirements.txt
```

### 2. Run the API Server

The server must be running to generate phase space logs:

```bash
python server/app.py
```

### 3. Launch Real-Time Dashboard

```bash
streamlit run visualization/dashboard.py
```

Opens in browser at `http://localhost:8501`

**Dashboard Features:**
- 🌐 **Phase Space Tab**: Live (q, π) scatter with energy contours
- ⚡ **Energy Tab**: Hamiltonian and energy drift monitoring
- 📈 **Forecast Tab**: Forecast magnitude over time
- 📊 **Statistics Tab**: Signal distribution and summary stats

**Settings:**
- Auto-refresh (default: 60 seconds)
- Hours of data to show (default: 24)
- Plot window size (default: 100 points)

### 4. Offline Analysis

Analyze historical data:

```bash
# Today's data
python visualization/visualize_phase_space.py

# Specific date
python visualization/visualize_phase_space.py --date 20251119

# Save plots as HTML
python visualization/visualize_phase_space.py --save
```

**Output:**
- Phase space scatter plot
- Energy conservation analysis
- Forecast quality metrics
- Volume-momentum distributions
- Summary statistics

---

## What the Visualizations Show

### Phase Space (q, π)

**Interpretation:**
- **q (position)**: Normalized volume (Encoding A) or price deviation
- **π (momentum)**: Last bar's log-return
- **Points colored by signal**: 🟢 Long, 🔴 Short, ⚪ Flat
- **Energy contours**: H = 0.5π² + 0.5κq²

**What to look for:**
- ✅ Points clustered in specific regions = model finding structure
- ✅ Trajectory follows contours = energy conservation working
- ❌ Random scatter = model not capturing meaningful dynamics
- ❌ Points far from contours = integration issues

### Energy Monitoring

**Interpretation:**
- **Hamiltonian (H)**: Total "energy" of the system
- **Energy Drift (|ΔH|)**: How much energy changes per step

**What to look for:**
- ✅ H relatively stable over time = valid symplectic model
- ✅ Mean drift < 0.001 = excellent energy conservation
- ⚠️ Mean drift < 0.01 = acceptable
- ❌ Large drift or trending H = integration breaking down

### Forecast Analysis

**Interpretation:**
- **π_next**: Predicted next-bar return (from leapfrog step)
- **Larger |π_next|** = stronger conviction

**What to look for:**
- ✅ Large forecasts correlate with correct direction = good calibration
- ✅ Forecast magnitude varies = model adapting to conditions
- ❌ All forecasts same magnitude = model stuck
- ❌ No correlation with outcomes = model not predictive

---

## Data Files

### CSV Logs

Located in `logs/phase_space_YYYYMMDD.csv`:

```csv
timestamp,instrument,q,pi,q_next,pi_next,H_before,H_after,energy_drift,kappa,encoding,forecast,direction,size_factor,last_close,last_volume
2025-11-19T16:00:00,NQ 12-25,0.9234,-0.0012,0.9233,-0.0013,0.000421,0.000422,0.000001,0.0100,A,-0.0013,-1,1.0,25080.5,1500.0
```

**Columns:**
- `timestamp`: Signal generation time
- `instrument`: Trading instrument
- `q, pi`: Current phase space state
- `q_next, pi_next`: Predicted next state
- `H_before, H_after`: Hamiltonian before/after leapfrog
- `energy_drift`: |H_after - H_before|
- `kappa`: Model parameter κ
- `encoding`: Which encoding used (A/B/C)
- `forecast`: π_next (predicted return)
- `direction`: Signal (-1/0/1)
- `size_factor`: Position size (0-1)

---

## Advanced Usage

### Custom Time Windows

```python
# Load specific date range
from visualization.visualize_phase_space import load_phase_space_data
df = load_phase_space_data(date='20251119')
df = df[(df['timestamp'] >= '2025-11-19 09:00') &
        (df['timestamp'] <= '2025-11-19 16:00')]
```

### Export Plots

```python
# Save all plots
python visualization/visualize_phase_space.py --save

# Outputs to: visualization/output/
# - phase_space_20251119.html
# - energy_20251119.html
# - forecast_20251119.html
# - distributions_20251119.html
```

### Dashboard Customization

Edit `visualization/dashboard.py` to:
- Add custom metrics
- Change color schemes
- Add alerts/thresholds
- Integrate with external data sources

---

## Interpreting Results

### Good Signs ✅

1. **Phase space shows structure**
   - Points cluster in specific (q, π) regions
   - Profitable trades concentrate in certain areas
   - Clear separation between long/short signals

2. **Energy conservation**
   - Mean drift < 0.001
   - H stable over time
   - Trajectory follows contours

3. **Forecasts are meaningful**
   - Larger |forecast| → better outcomes
   - Direction correlates with realized returns
   - Magnitude varies with market conditions

### Warning Signs ⚠️

1. **Random phase space**
   - No clustering or structure
   - Uniform scatter across (q, π)
   - No relationship between region and performance

2. **Poor energy conservation**
   - Drift > 0.01
   - H trending up/down
   - Points far from energy contours

3. **Uninformative forecasts**
   - All forecasts similar magnitude
   - No correlation with outcomes
   - Model always predicting same direction

### Action Items

**If energy drift is high:**
- Reduce dt (currently 1.0)
- Adjust κ estimation
- Check for numerical instability

**If phase space shows no structure:**
- Try different encoding (A → B → C)
- Adjust K (segment length)
- Check if volume is informative

**If forecasts don't predict:**
- Model may not be capturing dynamics
- Consider regime-based approach (hybrid model)
- Validate data preprocessing

---

## Troubleshooting

### No data showing

```bash
# Check if logs directory exists
ls logs/

# Check if server is running and generating logs
tail -f logs/phase_space_*.csv

# Verify API is being called
# (should see "Signal generated" in server output)
```

### Dashboard not updating

- Check "Auto-refresh" is enabled
- Verify refresh interval setting
- Click "🔄 Refresh Now" manually
- Check server is running

### Plots not opening

```bash
# Make sure plotly is installed
pip install plotly

# Try saving instead of showing
python visualization/visualize_phase_space.py --save
```

---

## Next Steps

Once you have visualization data:

1. **Identify sweet spots**: Which (q, π) regions have best hit rates?
2. **Tune parameters**: Adjust κ based on energy drift
3. **Test encodings**: Compare A vs B vs C performance
4. **Add filters**: Gate signals based on phase space location
5. **Build alerts**: Notify when energy drift exceeds threshold

Happy visualizing! 📊
