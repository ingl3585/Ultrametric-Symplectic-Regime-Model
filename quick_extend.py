#!/usr/bin/env python3
"""Quick extension without pandas dependency."""

from datetime import datetime, timedelta
import random
import math

# Set seed for reproducibility
random.seed(42)

# Read existing data (skip header)
existing_data = """2024-01-02 09:30:00,400.50,401.20,400.30,400.80,1500000
2024-01-02 09:45:00,400.80,401.50,400.70,401.20,1600000
2024-01-02 10:00:00,401.20,401.80,401.00,401.50,1450000
2024-01-02 10:15:00,401.50,402.00,401.30,401.70,1550000
2024-01-02 10:30:00,401.70,402.20,401.60,402.00,1480000"""

# Parse last line
last_line = existing_data.strip().split('\n')[-1]
parts = last_line.split(',')
last_timestamp = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S')
last_close = float(parts[4])
last_volume = int(parts[5])

print(f"Extending from: {last_timestamp}, close=${last_close:.2f}")

# Generate 1555 more bars
total_bars = 1555
bar_vol = 0.20 / math.sqrt(252 * 26)  # 15-min vol from annual vol

output_lines = ["timestamp,open,high,low,close,volume"]
output_lines.extend(existing_data.strip().split('\n'))

current_time = last_timestamp + timedelta(minutes=15)
current_log_price = math.log(last_close)
bars_today = 5

for i in range(total_bars):
    # Generate return with mean reversion
    ret = random.gauss(0, bar_vol)
    mean_rev = -0.01 * (current_log_price - math.log(last_close))
    current_log_price += ret + mean_rev
    close = math.exp(current_log_price)

    # Open (with small gap)
    if i == 0:
        prev_close = last_close
    else:
        prev_close = close

    gap = random.gauss(0, bar_vol * 0.3)
    open_price = prev_close * (1 + gap)

    # High and low
    bar_range = abs(random.gauss(0, bar_vol)) * close
    high = max(open_price, close) + abs(random.uniform(0, bar_range))
    low = min(open_price, close) - abs(random.uniform(0, bar_range))

    # Volume
    volume = int(last_volume * random.gammavariate(2, 0.5))

    # Format line
    line = f"{current_time.strftime('%Y-%m-%d %H:%M:%S')},{open_price:.2f},{high:.2f},{low:.2f},{close:.2f},{volume}"
    output_lines.append(line)

    # Advance time
    current_time += timedelta(minutes=15)
    bars_today += 1

    # Skip to next day after 26 bars
    if bars_today >= 26:
        current_time += timedelta(days=1)
        while current_time.weekday() >= 5:  # Skip weekends
            current_time += timedelta(days=1)
        current_time = current_time.replace(hour=9, minute=30)
        bars_today = 0

# Write to file
with open('data/sample_data_template.csv', 'w') as f:
    f.write('\n'.join(output_lines))

print(f"✓ Extended to {len(output_lines)-1} bars")
print(f"  Date range: {output_lines[1].split(',')[0]} to {output_lines[-1].split(',')[0]}")
