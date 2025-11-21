#!/usr/bin/env python3
"""
QQQ 1-minute data downloader using yfinance.

NOTE: yfinance 1-minute data limitations:
- Only last ~7 days available
- Max 7 days of 1m data per request
"""

import yfinance as yf
import pandas as pd

print("="*70)
print("Downloading QQQ 1-minute data (last 7 days)")
print("="*70)
print("\nNOTE: yfinance limits 1-minute data to ~7 days")
print("For longer history, you'll need a different data source.\n")

# Use period instead of dates to avoid system clock issues
print("Downloading...")

df = yf.download(
    "QQQ",
    period="7d",      # Last 7 days (max for 1m data)
    interval="1m",    # 1-minute bars
    auto_adjust=True,
    progress=True,
    prepost=False     # Exclude pre/post market
)

if df.empty:
    print("\n✗ No data returned - trying shorter period...")
    df = yf.download("QQQ", period="5d", interval="1m", auto_adjust=True, progress=True)

if df.empty:
    print("\n✗ Still no data - yfinance may be having issues")
    print("\nAlternatives:")
    print("  1. Try again later (yfinance rate limits)")
    print("  2. Use a paid data provider (Alpaca, Interactive Brokers, etc.)")
    print("  3. Use 15-minute data instead: python download_qqq_simple.py")
    exit(1)

print(f"\n✓ Downloaded {len(df)} 1-minute bars")

# Process data
# Flatten multi-level columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})

df = df.reset_index()
df = df.rename(columns={'Datetime': 'timestamp', 'Date': 'timestamp'})
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Keep only required columns
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
df = df.dropna()

# Ensure numeric columns are floats
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()  # Remove any rows that couldn't be converted
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Total bars: {len(df)}")
print(f"  Expected bars per day: ~390 (6.5 hours * 60 minutes)")
print(f"  Price range: ${float(df['close'].min()):.2f} to ${float(df['close'].max()):.2f}")

# Save
output_path = "data/sample_data_template.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Saved to {output_path}")

print("\nFirst 5 bars:")
print(df.head())

print("\nLast 5 bars:")
print(df.tail())

print("\n" + "="*70)
print("SUCCESS! You can now run: python validation_ml.py")
print("="*70)
print("\n⚠️  REMINDER: 1-minute data is limited to ~7 days")
print("   For longer backtests, consider:")
print("   - Using 15-minute data (60+ days available)")
print("   - Switching to a paid data provider")
print("="*70)
