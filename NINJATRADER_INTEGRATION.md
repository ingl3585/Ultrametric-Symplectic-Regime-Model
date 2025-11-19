# NinjaTrader Integration Guide (STEP 6)

**Status:** Phase 2 - NinjaTrader HTTP API Integration
**Purpose:** Connect Python trading model to NinjaTrader for automated execution
**Requirements:** NinjaTrader 8, Python 3.8+, FastAPI server

---

## Overview

This integration allows NinjaTrader to call your Python trading model via HTTP REST API. The architecture separates concerns:

- **Python (FastAPI server):** Pure math/signal generation
- **NinjaTrader (C# strategy):** Data feeds, bar building, order execution, account management

```
┌──────────────┐         HTTP POST          ┌─────────────────┐
│              │  /signal (last K bars)     │                 │
│ NinjaTrader  │ ──────────────────────────>│  Python API     │
│ (C#)         │                             │  (FastAPI)      │
│              │ <──────────────────────────│                 │
│              │  {direction, size_factor}   │                 │
└──────────────┘                             └─────────────────┘
  │                                                   │
  │ • Market data                                   │ • Model inference
  │ • 15m bars                                      │ • Symplectic dynamics
  │ • Order execution                               │ • Regime detection
  │ • Position tracking                             │ • Signal generation
  └─────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install Python Dependencies

```bash
# Install FastAPI and dependencies
pip install fastapi uvicorn pydantic pyyaml

# Or use the requirements file
pip install -r server/requirements.txt
```

### 2. Start Python API Server

```bash
# From project root
python server/app.py
```

You should see:
```
================================================================================
Ultrametric-Symplectic Trading API Server
================================================================================

Starting server on http://localhost:8000

Endpoints:
  GET  /           - Health check
  GET  /health     - Detailed health status
  POST /signal     - Get trading signal
  POST /trade_log  - Log trade execution
  POST /reload_model - Reload model

Docs: http://localhost:8000/docs
================================================================================

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Model loaded successfully
INFO:     Application startup complete.
```

**Test the API:**
```bash
# Health check
curl http://localhost:8000/health

# View interactive docs
# Open browser to: http://localhost:8000/docs
```

### 3. Import Strategy to NinjaTrader

**Option A: Via NinjaScript Editor**

1. Open NinjaTrader 8
2. Tools → NinjaScript Editor
3. File → New → Strategy
4. Replace entire content with `ninjatrader/UltrametricSymplecticStrategy.cs`
5. **Add References:**
   - Right-click strategy → References
   - Add: `System.Net.Http`
   - Add: `Newtonsoft.Json` (download from NuGet if needed)
6. Compile (F5)

**Option B: Copy to Documents**

1. Copy `UltrametricSymplecticStrategy.cs` to:
   ```
   Documents/NinjaTrader 8/bin/Custom/Strategies/
   ```
2. Restart NinjaTrader or Tools → Compile All

### 4. Apply Strategy to Chart

1. Open 15-minute NQ chart (or QQQ for testing)
2. Right-click chart → Strategies
3. Select "UltrametricSymplecticStrategy"
4. Configure parameters:
   - **API URL:** `http://localhost:8000` (default)
   - **Bars To Send:** `10` (must match K in config.yaml)
   - **Default Quantity:** `1` (adjust based on account size)
   - **Enable Trade Logging:** `True`
5. Click "OK"

### 5. Run in Simulation

1. Switch to **Sim101** account
2. Strategy should start making API calls on bar close
3. Monitor:
   - **Output window** for logs
   - **Chart** for signal arrows
   - **Strategies tab** for PnL

---

## Configuration

### Python API Configuration

Edit `configs/config.yaml` if needed:

```yaml
segments:
  K: 10  # Must match "Bars To Send" in NinjaTrader

symplectic:
  encoding: 'A'  # Volume-based encoding
  dt: 1.0

signal:
  theta_symplectic: 0.0001  # Signal threshold
  epsilon_gate: 1.0         # Hybrid model gating
  hit_threshold: 0.50       # Minimum hit rate

costs:
  cost_log_15m: -0.00048  # Per round-trip cost
```

### NinjaScript Strategy Parameters

Accessible via strategy properties panel:

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| **API URL** | Python server endpoint | `http://localhost:8000` | Change if server on different machine |
| **Bars To Send** | Number of bars for model (K) | `10` | Must match config.yaml |
| **Default Quantity** | Base position size | `1` | Scale based on account |
| **Enable Trade Logging** | Send fills to API | `True` | Useful for monitoring |

---

## Testing Workflow

### 1. Paper Trading (Sim101)

**Recommended first step:**

1. Start Python server
2. Apply strategy to 15m chart with Sim101 account
3. Watch for 30-60 minutes (2-4 bars)
4. Verify:
   - API calls succeed (check Output window)
   - Signals display on chart (arrows)
   - Orders execute correctly
   - Position tracking accurate

**Expected behavior:**
- Strategy calls API on every 15-minute bar close
- Python logs "Signal generated: direction=X, size=Y"
- NinjaTrader prints "Signal received: Direction=X, Size=Y"
- Trades execute based on signal

### 2. Market Replay (Playback101)

**For faster testing:**

1. Tools → Market Replay Connection
2. Load historical data (NinjaTrader includes NQ data)
3. Set date range (e.g., last 30 days)
4. Apply strategy to Playback101
5. Speed up replay (2x, 5x, 10x)
6. Observe behavior over many bars quickly

**Advantages:**
- Test weeks of data in minutes
- Repeatable (same data each time)
- No risk, no real-time pressure
- Good for debugging

### 3. Live Simulation

**After successful Playback testing:**

1. Keep Sim101 account
2. Apply to live 15m chart (real-time data)
3. Monitor for full trading day
4. Check:
   - API response times (<1 second)
   - Signal quality
   - Position management
   - No errors or exceptions

**Do NOT go live until:**
- ✅ Sim/Playback testing complete (30+ days)
- ✅ Strategy behavior understood
- ✅ Extended data validation done (see TODO.md)
- ✅ Multiple market conditions tested
- ✅ Risk management validated

---

## API Endpoints Reference

### POST /signal

**Request trading signal from Python model.**

**Request:**
```json
{
  "bars": [
    {
      "timestamp": "2024-11-19T14:00:00Z",
      "open": 21000.0,
      "high": 21010.0,
      "low": 20995.0,
      "close": 21005.0,
      "volume": 1500.0
    },
    ...  // Last K bars (minimum 10)
  ],
  "instrument": "NQ 03-25",
  "account": {
    "account_id": "Sim101",
    "cash_value": 100000.0,
    "realized_pnl": 250.0,
    "unrealized_pnl": -50.0,
    "total_buying_power": 200000.0,
    "position_quantity": 1,
    "position_avg_price": 21000.0
  }
}
```

**Response:**
```json
{
  "direction": 1,          // -1 (short), 0 (flat), 1 (long)
  "size_factor": 1.0,      // 0.0 to 1.0
  "model_used": "global",  // "global" or "hybrid"
  "timestamp": "2024-11-19T14:00:05Z",
  "metadata": {
    "bars_received": 10,
    "bars_used": 10,
    "instrument": "NQ 03-25",
    "last_close": 21005.0,
    "last_volume": 1500.0
  }
}
```

### POST /trade_log

**Log trade execution for monitoring.**

**Request:**
```json
{
  "timestamp": "2024-11-19T14:00:10Z",
  "instrument": "NQ 03-25",
  "side": "Long",
  "quantity": 1,
  "price": 21005.0,
  "realized_pnl": 25.0,
  "strategy": "UltrametricSymplecticStrategy"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Trade logged successfully",
  "timestamp": "2024-11-19T14:00:10Z"
}
```

### GET /health

**Check API server status.**

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "global",
  "timestamp": "2024-11-19T14:00:00Z"
}
```

---

## Troubleshooting

### Python Server Issues

**Problem:** Server won't start

```
ERROR:     Model not found
```

**Solution:**
- Ensure `configs/config.yaml` exists
- Run from project root directory: `python server/app.py`

---

**Problem:** Import errors

```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
pip install -r server/requirements.txt
```

---

### NinjaScript Compilation Errors

**Problem:** `HttpClient` not found

```
Error CS0246: The type or namespace name 'HttpClient' could not be found
```

**Solution:**
1. Open NinjaScript Editor
2. Right-click strategy → References
3. Add: `System.Net.Http`
4. Compile again (F5)

---

**Problem:** `Newtonsoft.Json` not found

```
Error CS0246: The type or namespace name 'Newtonsoft' could not be found
```

**Solution:**
1. Download Newtonsoft.Json from NuGet
2. Copy `Newtonsoft.Json.dll` to:
   ```
   Documents/NinjaTrader 8/bin/Custom/
   ```
3. Add reference in strategy
4. Compile again

**Alternative:**
- Use built-in JSON if Newtonsoft not available
- Modify strategy to use `System.Text.Json` instead

---

### Runtime Errors

**Problem:** API calls timing out

```
ERROR calling API: The operation has timed out
```

**Solutions:**
- Verify Python server is running: `curl http://localhost:8000/health`
- Check firewall settings (allow port 8000)
- Increase timeout in strategy (currently 5 seconds)
- Check Python server logs for errors

---

**Problem:** "Environment is invalid" after async call

**Root cause:** NinjaScript's threading model + async/await

**Solution:** Strategy uses `.Result` for synchronous blocking calls instead of async/await

---

**Problem:** Signals not executing

```
Signal received: Direction=1, Size=1.00, Model=global
(but no trades)
```

**Check:**
1. `BarsRequiredToTrade` met (default 20 bars)
2. Account connected (Sim101 or Playback101)
3. Market hours (if using live connection)
4. Strategy state is "Running" not "Paused"
5. Output window for error messages

---

### Connectivity Issues

**Problem:** Connection refused

```
ERROR calling API: No connection could be made because the target machine actively refused it
```

**Solutions:**
1. Start Python server first
2. Check URL: `http://localhost:8000` (not `https`)
3. Test with curl: `curl http://localhost:8000/health`
4. Check Windows Firewall rules
5. If server on different machine, update API URL

---

## Best Practices

### Development

1. **Test with Sim101 first** - Never start with live account
2. **Use Market Replay** - Test weeks of data quickly
3. **Monitor Python logs** - Watch API calls and responses
4. **Start with 1 contract** - Scale up after validation
5. **Test during market hours** - Different behavior than replay

### Production (When Ready)

1. **Run Python server as service** - Use systemd/Windows Service
2. **Add error recovery** - Restart server automatically
3. **Log everything** - Save API calls and responses
4. **Monitor performance** - Track latency, errors, fills
5. **Have kill switch** - Manual override to flatten position

### Risk Management

1. **Max position size** - Don't rely solely on size_factor
2. **Daily loss limit** - Stop trading after X% loss
3. **API timeout handling** - Flatten on extended API failure
4. **Heartbeat checks** - Verify API health periodically
5. **Manual oversight** - Don't leave fully unattended initially

---

## Advanced Configuration

### Running Server as Windows Service

**Using NSSM (Non-Sucking Service Manager):**

```bash
# Download NSSM from nssm.cc
# Install as service
nssm install UltrametricAPI "C:\Python39\python.exe" "C:\path\to\algo\server\app.py"
nssm start UltrametricAPI
```

### Running on Different Machine

If Python server runs on different computer:

**1. Update Python server** (`server/app.py`):
```python
uvicorn.run(
    app,
    host="0.0.0.0",  # Allow external connections
    port=8000,
    log_level="info"
)
```

**2. Update NinjaScript strategy:**
```csharp
ApiUrl = "http://192.168.1.100:8000"  // IP of Python machine
```

**3. Configure firewall:**
- Allow incoming TCP port 8000 on Python machine
- Test with: `curl http://192.168.1.100:8000/health`

### HTTPS / Authentication (Optional)

For production deployments with security:

**1. Add API key authentication to FastAPI:**
```python
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/signal")
async def get_signal(request: SignalRequest, api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of code
```

**2. Update NinjaScript to send API key:**
```csharp
httpClient.DefaultRequestHeaders.Add("X-API-Key", "your-secret-key");
```

---

## Performance Optimization

### Python Server

- **Pre-load model** on startup (already done)
- **Use production ASGI server:** gunicorn with uvicorn workers
- **Add caching** for repeated requests (if needed)
- **Monitor memory** usage for leaks

### NinjaScript Strategy

- **Minimize API calls** - only on bar close (already configured)
- **Cache results** within bar if needed
- **Batch operations** if possible
- **Handle timeouts gracefully**

### Network

- **Run on same machine** - Lowest latency
- **Use wired connection** - More reliable than WiFi
- **Monitor latency** - Should be <100ms typically
- **Have fallback** - What to do if API unavailable

---

## Next Steps

After successful STEP 6 testing:

1. **Extended validation** (see `TODO.md`)
   - Test on 1-2 years of Market Replay data
   - Validate across different market conditions
   - Achieve >200 trades for statistical confidence

2. **Risk management enhancements**
   - Add stop-loss / take-profit levels
   - Implement position sizing rules
   - Add daily/weekly PnL limits
   - Create emergency flatten function

3. **Monitoring & alerts**
   - Discord/Telegram notifications
   - Performance dashboard
   - Real-time PnL tracking
   - Error alerting

4. **Multi-instrument support**
   - Test on ES, SPY, other markets
   - Compare performance across instruments
   - Portfolio-level risk management

5. **Production deployment**
   - Dedicated trading machine
   - Automated server restart
   - Backup power supply
   - Comprehensive logging
   - Regular strategy reviews

---

## Support & Resources

### Documentation
- `README.md` - Project overview
- `CLAUDE.md` - Implementation specification
- `RESULTS.md` - Phase 1 validation results
- `TODO.md` - Future enhancements

### NinjaTrader Resources
- [NinjaScript Documentation](https://ninjatrader.com/support/helpguides/nt8/)
- [Developer Community](https://developer.ninjatrader.com/)
- [Support Forum](https://forum.ninjatrader.com/)

### FastAPI Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Models](https://docs.pydantic.dev/)
- [Uvicorn ASGI Server](https://www.uvicorn.org/)

---

**STEP 6 Complete!** You now have a full integration between NinjaTrader and your Python trading model. Test thoroughly in simulation before considering live deployment.
