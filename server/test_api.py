#!/usr/bin/env python3
"""
Test script for FastAPI server (STEP 6).

Tests the /signal endpoint with sample bar data.

Usage:
    # Start server first: python server/app.py
    # Then run test: python server/test_api.py
"""

import requests
import json
from datetime import datetime, timedelta

# API endpoint
API_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_signal():
    """Test signal generation endpoint."""
    print("Testing /signal endpoint...")

    # Create sample bar data (10 bars)
    bars = []
    base_time = datetime.now() - timedelta(minutes=150)  # 150 min ago

    for i in range(10):
        bar_time = base_time + timedelta(minutes=15 * i)
        bars.append({
            "timestamp": bar_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": 21000.0 + i * 2,
            "high": 21010.0 + i * 2,
            "low": 20995.0 + i * 2,
            "close": 21005.0 + i * 2,
            "volume": 1500.0 + i * 10
        })

    # Build request
    request_data = {
        "bars": bars,
        "instrument": "NQ 03-25",
        "account": {
            "account_id": "Sim101",
            "cash_value": 100000.0,
            "realized_pnl": 250.0,
            "unrealized_pnl": -50.0,
            "total_buying_power": 200000.0,
            "position_quantity": 0,
            "position_avg_price": 0.0
        }
    }

    # Make request
    response = requests.post(
        f"{API_URL}/signal",
        json=request_data,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()

def test_trade_log():
    """Test trade logging endpoint."""
    print("Testing /trade_log endpoint...")

    trade_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "instrument": "NQ 03-25",
        "side": "Long",
        "quantity": 1,
        "price": 21005.0,
        "realized_pnl": 25.0,
        "strategy": "UltrametricSymplecticStrategy"
    }

    response = requests.post(
        f"{API_URL}/trade_log",
        json=trade_data,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def main():
    """Run all tests."""
    print("=" * 80)
    print("FastAPI Server Test")
    print("=" * 80)
    print()

    try:
        # Test health
        test_health()

        # Test signal generation
        test_signal()

        # Test trade logging
        test_trade_log()

        print("=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server at", API_URL)
        print("Make sure the server is running: python server/app.py")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
