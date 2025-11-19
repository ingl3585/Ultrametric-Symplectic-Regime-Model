# Project TODO List

## Phase 1 Complete ✅
- ✅ STEP 0-5 implemented and validated
- ✅ Core symplectic model working (Sharpe 1.96)
- ✅ Documentation complete

## Current: Phase 2 - NinjaTrader Integration
- 🔄 STEP 6: HTTP API server + NinjaScript strategy (IN PROGRESS)

## Future Extended Validation (Data Acquisition Needed)

### Priority 1: Extended Historical Data Testing
**Status:** ⏸ On Hold (requires data acquisition)

**Goal:** Validate regime-switching with diverse market conditions

**Data Requirements:**
1. **1-2 years of 15-minute OHLCV data** for:
   - QQQ (Nasdaq-100)
   - SPY (S&P 500)
   - ES futures (if accessible)
   - NQ futures (if accessible)

2. **Target Time Periods** (to capture regime diversity):
   - 2020 Q1: COVID crash (Feb-Mar 2020, VIX >80)
   - 2020 Q2-Q4: Recovery rally
   - 2021: Low volatility grind higher
   - 2022: Bear market (Fed tightening, VIX 25-35)
   - 2023: Recovery and AI rally
   - 2024: Current conditions

**Why This Matters:**
- Current test (Aug-Nov 2024) only has 1 regime
- Need multiple distinct volatility/trend regimes to validate:
  - Per-cluster κ differentiation
  - Regime-switching value proposition
  - Gating logic under diverse conditions
  - Statistical significance (target: >200 trades)

**Data Source Options to Investigate:**

### Free/Low-Cost Sources:
1. **yfinance limitations:** 60 days max for 15-minute data
   - Could download rolling 60-day windows and stitch together
   - Time-consuming but free

2. **Yahoo Finance CSV downloads:**
   - May have longer history available
   - Need to verify 15-minute availability

3. **Alpha Vantage API:**
   - Free tier: 500 calls/day
   - Intraday data available
   - Need to check 15-minute support

4. **Polygon.io:**
   - Free tier: 5 API calls/minute
   - Has historical intraday data
   - Check pricing for extended access

5. **IEX Cloud:**
   - Free tier available
   - Intraday data
   - Verify 15-minute granularity

6. **Tiingo:**
   - Free tier with registration
   - Intraday data
   - Check historical depth

### Paid Sources (If Budget Available):
1. **NinjaTrader Market Replay:**
   - Built-in to NinjaTrader
   - Can replay historical NQ data
   - Good for testing STEP 6 integration
   - Cost: Included with NinjaTrader license

2. **Kinetick / Continuum (via NinjaTrader):**
   - Real-time and historical data
   - Directly integrated
   - Cost: ~$60-100/month

3. **Interactive Brokers:**
   - Historical data access with account
   - Quality data for multiple instruments
   - Cost: Account required + market data fees

4. **QuantConnect / Quantopian alternatives:**
   - Research platforms with data included
   - Could export for local testing

5. **Commercial data vendors:**
   - Quandl, Norgate, etc.
   - Expensive but comprehensive

### Action Items (When Ready to Proceed):

- [ ] Research free API tiers for each source above
- [ ] Test yfinance rolling window approach (60 days at a time)
- [ ] Check if NinjaTrader Market Replay includes historical data
- [ ] Evaluate cost/benefit of paid vs stitched-together free data
- [ ] Write automated data collection script for chosen source
- [ ] Validate data quality (check for gaps, anomalies)
- [ ] Update validation_step5.py to handle longer datasets
- [ ] Run extended validation on 2020-2024 data
- [ ] Compare performance across different market regimes

### Success Criteria for Extended Validation:
- ✅ Test on 1+ year of data
- ✅ Capture 3+ distinct market regimes
- ✅ Achieve >200 trades for statistical confidence
- ✅ Validate that per-cluster κ values differ meaningfully
- ✅ Confirm regime-switching adds value over global model
- ✅ Performance holds across different market conditions

### Estimated Timeline:
- Data acquisition: 1-3 days (depending on approach)
- Data validation: 1 day
- Extended testing: 1-2 days
- Analysis: 1 day
- **Total: ~1 week** when ready to proceed

---

## Other Future Enhancements

### Optional Features:
- [ ] Real-time regime detection (rolling window updates)
- [ ] Confidence intervals on performance metrics
- [ ] Walk-forward optimization framework
- [ ] Multi-instrument testing
- [ ] Alternative encodings (B, C) comparison on extended data
- [ ] Parameter sensitivity analysis
- [ ] Regime transition detection
- [ ] Position sizing based on regime confidence

### Documentation:
- [ ] Video walkthrough of system
- [ ] Trading journal integration
- [ ] Performance tracking dashboard
- [ ] Risk management guidelines

---

**Note:** This TODO list tracks future work items that require resources (data acquisition costs or time investment). The core Phase 1 implementation (STEP 0-5) is complete and validated.
