#!/usr/bin/env python3
"""
Simple QQQ downloader using yfinance period (avoids date issues).
"""

import yfinance as yf
import pandas as pd

print("="*70)
print("Downloading QQQ 15-minute data (last 60 days)")
print("="*70)

# Use period instead of dates to avoid system clock issues
print("\nDownloading...")

df = yf.download(
    "QQQ",
    period="60d",  # Last 60 days
    interval="15m",
    auto_adjust=True,
    progress=True
)

if df.empty:
    print("\n✗ No data returned - trying shorter period...")
    df = yf.download("QQQ", period="30d", interval="15m", auto_adjust=True, progress=True)

if df.empty:
    print("\n✗ Still no data - yfinance may be having issues")
    print("Alternative: Use the synthetic data generator for now")
    exit(1)

print(f"\n✓ Downloaded {len(df)} bars")

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
print(f"  Price range: ${float(df['close'].min()):.2f} to ${float(df['close'].max()):.2f}")

# Save
output_path = "data/sample_data_template.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Saved to {output_path}")

print("\nFirst 5 bars:")
print(df.head())

print("\n" + "="*70)
print("SUCCESS! You can now run: python example_step0.py")
print("="*70)
