#!/usr/bin/env python3
"""
QQQ 1-minute data downloader - MAXIMUM available data from yfinance.

yfinance limits:
- 1-minute data: max 7 days per request
- Data must be within last 30 days
- Solution: Make multiple 7-day requests and combine

This script downloads ~30 days of 1-minute data in chunks.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

print("="*80)
print("Downloading MAXIMUM QQQ 1-minute data from yfinance")
print("="*80)
print("\nyfinance limits for 1-minute data:")
print("  - Max 7 days per request")
print("  - Data must be within last 30 days")
print("  - Strategy: Download in 7-day chunks\n")

# Calculate date ranges for last 30 days in 7-day chunks
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Create 7-day chunks (working backwards from today)
chunks = []
current_end = end_date

while current_end > start_date:
    current_start = current_end - timedelta(days=7)
    if current_start < start_date:
        current_start = start_date
    chunks.append((current_start, current_end))
    current_end = current_start

chunks.reverse()  # Process chronologically

print(f"Downloading {len(chunks)} chunks of 7-day data...")
print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")

# Download each chunk
all_data = []
successful_chunks = 0
failed_chunks = 0

for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
    print(f"Chunk {i}/{len(chunks)}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...", end=" ")

    try:
        # Download this chunk
        df_chunk = yf.download(
            "QQQ",
            start=chunk_start,
            end=chunk_end,
            interval="1m",
            auto_adjust=True,
            progress=False
        )

        if not df_chunk.empty:
            print(f"✓ {len(df_chunk)} bars")
            all_data.append(df_chunk)
            successful_chunks += 1
        else:
            print("✗ No data")
            failed_chunks += 1

        # Rate limiting: wait between requests
        if i < len(chunks):
            time.sleep(1)  # 1 second delay between requests

    except Exception as e:
        print(f"✗ Error: {e}")
        failed_chunks += 1

print(f"\n{'='*80}")
print(f"Download complete: {successful_chunks}/{len(chunks)} chunks successful")
print(f"{'='*80}\n")

if not all_data:
    print("✗ No data downloaded!")
    print("\nPossible reasons:")
    print("  1. yfinance rate limiting (try again in a few minutes)")
    print("  2. Market was closed during requested period")
    print("  3. Network issues")
    print("\nAlternatives:")
    print("  - Try download_qqq_simple.py for 15-minute data (60 days)")
    print("  - Use a paid data provider for more reliable 1-minute data")
    exit(1)

# Combine all chunks
print("Combining chunks...")
df = pd.concat(all_data, axis=0)

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

# Remove duplicates (overlapping chunk boundaries)
df = df.drop_duplicates(subset=['timestamp'], keep='first')

# Keep only required columns
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
df = df.dropna()

# Ensure numeric columns are floats
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()  # Remove any rows that couldn't be converted
df = df.sort_values('timestamp').reset_index(drop=True)

# Calculate statistics
num_days = (df['timestamp'].max() - df['timestamp'].min()).days
bars_per_day = len(df) / max(num_days, 1)

print(f"✓ Combined and cleaned data")
print(f"\n{'='*80}")
print("DATA SUMMARY")
print(f"{'='*80}")
print(f"  Total bars: {len(df):,}")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Days covered: {num_days}")
print(f"  Avg bars/day: {bars_per_day:.0f} (expect ~390 for full trading days)")
print(f"  Price range: ${float(df['close'].min()):.2f} to ${float(df['close'].max()):.2f}")
print(f"  Data completeness: {(bars_per_day/390)*100:.1f}% (390 bars = full 6.5hr trading day)")

# Save
output_path = "data/sample_data_template.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Saved to {output_path}")

print("\nFirst 10 bars:")
print(df.head(10))

print("\nLast 10 bars:")
print(df.tail(10))

print(f"\n{'='*80}")
print("SUCCESS! You now have MAXIMUM 1-minute data from yfinance")
print(f"{'='*80}")
print("\nNext steps:")
print("  1. Run: python validation_ml.py")
print("  2. Config is already set for 1-minute bars")
print("\n⚠️  NOTES:")
print(f"  - Got ~{num_days} days of 1-minute data (yfinance max: 30 days)")
print("  - For longer history, use 15-minute data or paid provider")
print(f"{'='*80}")
