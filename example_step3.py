#!/usr/bin/env python3
"""
STEP 3 Example – Symplectic Global Model vs AR(1)

This script demonstrates:
1. Fitting a symplectic model with global κ (no regimes)
2. Comparing three encodings (A: volume, B: price, C: hybrid)
3. Backtesting vs AR(1) baseline with same cost model
4. Evaluating forecast quality and economic viability

Usage:
    python example_step3.py
"""

import yaml
import numpy as np
from pathlib import Path

from model.data_utils import (
    load_ohlcv_csv,
    resample_to_15m,
    compute_log_price,
    compute_smoothed_volume,
    build_gamma
)
from model.trainer import build_segments
from model.symplectic_model import estimate_global_kappa
from model.signal_api import AR1Model, SymplecticGlobalModel
from model.backtest import run_ar1_backtest, run_symplectic_backtest


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_comparison(results: dict):
    """Pretty print model comparison."""
    print("\n" + "="*80)
    print("Model Performance Comparison (Test Set)")
    print("="*80)
    print(f"{'Model':<30} {'Trades':>8} {'Win%':>8} {'Sharpe':>10} {'Net PnL':>12}")
    print("-"*80)

    for model_name, data in results.items():
        m = data['metrics']
        print(f"{model_name:<30} {m['num_trades']:>8} {m['win_rate']:>7.1%} "
              f"{m['sharpe_ratio']:>10.2f} {m['total_net_pnl']:>12.6f}")

    print("="*80)


def main():
    """Run STEP 3 demonstration."""
    print("\n" + "="*80)
    print("STEP 3: Symplectic Global Model Demonstration")
    print("="*80 + "\n")

    # Load configuration
    print("Loading configuration...")
    config = load_config()
    K = config['segments']['K']
    encoding_default = config['symplectic']['encoding']
    cost_log = config['costs']['cost_log_15m']

    print(f"✓ Configuration loaded")
    print(f"  Segment length (K): {K}")
    print(f"  Default encoding: {encoding_default}")
    print(f"  Cost per trade: {cost_log:.6f} ({cost_log*100:.4f}%)")

    # Load data
    data_path = Path("data/sample_data_template.csv")
    if not data_path.exists():
        print(f"\n⚠ Data not found at {data_path}")
        return

    print(f"\nLoading data from {data_path}...")
    df = load_ohlcv_csv(str(data_path))
    df_15m = resample_to_15m(df)
    print(f"✓ Loaded {len(df_15m)} 15-minute bars")

    # Compute phase-space data
    print("\nComputing phase-space data...")
    p = compute_log_price(df_15m)
    v = compute_smoothed_volume(
        df_15m,
        normalization_window=config['volume']['normalization_window'],
        ema_period=config['volume']['ema_period']
    )
    gamma = build_gamma(p, v)

    # Build segments
    print(f"\nBuilding {K}-bar segments...")
    segments = build_segments(gamma, K)
    print(f"✓ Built {len(segments)} segments")

    # Split data
    split_idx = int(len(p) * 0.7)
    p_train = p[:split_idx]
    p_test = p[split_idx:]

    seg_split_idx = split_idx - K + 1
    segments_train = segments[:seg_split_idx]
    segments_test = segments[seg_split_idx:]

    print(f"\n✓ Data split:")
    print(f"  Train: {len(p_train)} bars, {len(segments_train)} segments")
    print(f"  Test:  {len(p_test)} bars, {len(segments_test)} segments")

    # ========================================================================
    # Baseline: AR(1)
    # ========================================================================
    print("\n" + "-"*80)
    print("1. AR(1) BASELINE")
    print("-"*80)

    ar1_model = AR1Model(config)
    ar1_model.fit(p_train)

    print(f"\nAR(1) parameters:")
    print(f"  Mean return: {ar1_model.mean_ret:.6f}")
    print(f"  AR(1) coeff (phi): {ar1_model.phi:.4f}")

    print(f"\nRunning AR(1) backtest on test set...")
    ar1_results = run_ar1_backtest(ar1_model, p_test, cost_log)

    # ========================================================================
    # Symplectic Global Models (3 encodings)
    # ========================================================================
    print("\n" + "-"*80)
    print("2. SYMPLECTIC GLOBAL MODELS")
    print("-"*80)

    encodings = ['A', 'B', 'C']
    encoding_names = {
        'A': 'Encoding A (volume-based)',
        'B': 'Encoding B (price-only)',
        'C': 'Encoding C (hybrid)'
    }

    symp_results = {}

    for enc in encodings:
        print(f"\n{encoding_names[enc]}:")
        print(f"  Estimating global κ...")

        kappa = estimate_global_kappa(segments_train, encoding=enc)
        print(f"  ✓ Global κ = {kappa:.4f}")

        # Create model
        symp_model = SymplecticGlobalModel(config, kappa, encoding=enc)

        # Backtest
        print(f"  Running backtest on test set...")
        results = run_symplectic_backtest(
            symp_model,
            segments_test,
            p_test,
            K,
            cost_log
        )

        symp_results[enc] = results

    # ========================================================================
    # Comparison
    # ========================================================================
    all_results = {
        'AR(1) Baseline': ar1_results,
        'Symplectic Global (Enc A)': symp_results['A'],
        'Symplectic Global (Enc B)': symp_results['B'],
        'Symplectic Global (Enc C)': symp_results['C'],
    }

    print_comparison(all_results)

    # ========================================================================
    # Detailed Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("Detailed Analysis")
    print("="*80)

    for model_name, data in all_results.items():
        m = data['metrics']
        print(f"\n{model_name}:")
        print(f"  Trades: {m['num_trades']}")
        print(f"  Win rate: {m['win_rate']:.2%}")
        print(f"  Avg gross PnL: {m['avg_gross_pnl']:.6f}")
        print(f"  Avg net PnL: {m['avg_net_pnl']:.6f}")
        print(f"  Total net PnL: {m['total_net_pnl']:.6f}")
        print(f"  Sharpe ratio: {m['sharpe_ratio']:.2f}")
        print(f"  Max drawdown: {m['max_drawdown']:.6f}")
        print(f"  Profit factor: {m['profit_factor']:.2f}")

    # ========================================================================
    # Success Criteria Evaluation
    # ========================================================================
    print("\n" + "="*80)
    print("Success Criteria Evaluation (from CLAUDE.md)")
    print("="*80)

    # Find best symplectic model
    best_enc = max(symp_results.keys(),
                   key=lambda x: symp_results[x]['metrics']['sharpe_ratio'])
    best_symp = symp_results[best_enc]
    best_symp_name = f"Symplectic (Enc {best_enc})"

    print(f"\nBest symplectic model: {best_symp_name}")
    print(f"  Sharpe: {best_symp['metrics']['sharpe_ratio']:.2f}")

    # Compare to AR(1)
    ar1_sharpe = ar1_results['metrics']['sharpe_ratio']
    symp_sharpe = best_symp['metrics']['sharpe_ratio']
    improvement = symp_sharpe - ar1_sharpe

    print(f"\n1. Forecast Quality vs AR(1):")
    print(f"   AR(1) Sharpe: {ar1_sharpe:.2f}")
    print(f"   Symplectic Sharpe: {symp_sharpe:.2f}")
    print(f"   Improvement: {improvement:+.2f} {'✓' if improvement > 0 else '✗'}")

    # Hit rate
    ar1_hit = ar1_results['metrics']['win_rate']
    symp_hit = best_symp['metrics']['win_rate']

    print(f"\n2. Hit Rate:")
    print(f"   AR(1): {ar1_hit:.2%}")
    print(f"   Symplectic: {symp_hit:.2%}")
    print(f"   Target: >52% {'✓' if symp_hit > 0.52 else '✗'}")

    # Economic viability
    symp_pnl = best_symp['metrics']['total_net_pnl']
    symp_trades = best_symp['metrics']['num_trades']
    avg_pnl = symp_pnl / symp_trades if symp_trades > 0 else 0

    print(f"\n3. Economic Viability:")
    print(f"   Post-cost Sharpe: {symp_sharpe:.2f} (target: >1.0) {'✓' if symp_sharpe > 1.0 else '✗'}")
    print(f"   Avg net PnL/trade: {avg_pnl:.6f}")
    print(f"   Cost per trade: {cost_log:.6f}")
    print(f"   Ratio: {abs(avg_pnl/cost_log):.2f}x cost {'✓' if abs(avg_pnl) > abs(2*cost_log) else '✗'}")

    # Summary
    print("\n" + "="*80)
    print("STEP 3 Summary")
    print("="*80)

    if symp_sharpe > ar1_sharpe and symp_hit > 0.52:
        print("\n✓ SUCCESS: Symplectic model shows improvement over AR(1)!")
        print("  Ready to proceed to STEP 4 (Hybrid with regimes)")
    elif symp_sharpe > ar1_sharpe:
        print("\n⚠ PARTIAL: Symplectic beats AR(1) on Sharpe but hit rate marginal")
        print("  Can proceed to STEP 4, may benefit from regime-specific κ")
    else:
        print("\n⚠ MARGINAL: Symplectic not outperforming AR(1) baseline")
        print("  This may be due to synthetic data uniformity")
        print("  Recommend testing on real market data before proceeding")

    print("\nKey Findings:")
    print(f"  - Best encoding: {best_enc}")
    print(f"  - Global κ: {estimate_global_kappa(segments_train, encoding=best_enc):.4f}")
    print(f"  - Sharpe improvement: {improvement:+.2f}")
    print(f"  - Hit rate: {symp_hit:.2%}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
