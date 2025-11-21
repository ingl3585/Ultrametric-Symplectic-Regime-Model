"""
Backtesting harnesses for trading models.

All backtests:
- Replay bar-by-bar on historical data
- Apply realistic costs on position changes
- Track PnL, equity, trades, and performance metrics
- Use same cost model for fair comparison across strategies
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Trade:
    """Record of a single trade."""
    entry_idx: int
    exit_idx: int
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    pnl: float
    net_pnl: float  # After costs


def run_ar1_backtest(
    model,  # AR1Model
    p: np.ndarray,
    cost_log: float
) -> Dict:
    """
    Backtest AR(1) model on log prices.

    Replays prices bar-by-bar, using model.get_signal() to determine position.
    Applies cost_log (in log space) on every position change.

    Args:
        model: Fitted AR1Model instance
        p: Log prices, shape (N,)
        cost_log: Per-round-trip cost in log space (e.g., -0.00048)

    Returns:
        Dictionary with:
            - trades: List of Trade objects
            - equity_curve: Cumulative PnL over time
            - metrics: Dict with win_rate, avg_return_per_trade, sharpe, max_drawdown, etc.
    """
    N = len(p)
    position = 0  # Start flat
    equity = 0.0
    equity_curve = [0.0]

    trades: List[Trade] = []
    current_trade_entry_idx = None
    current_trade_entry_price = None

    for t in range(1, N):
        # Get signal from model
        signal = model.get_signal(p[t], p[t - 1])
        new_position = signal["direction"]

        # Check if position changes
        if new_position != position:
            # Exit current position if any
            if position != 0:
                # Realize PnL on the exit
                price_change = p[t] - current_trade_entry_price
                gross_pnl = position * price_change

                # Apply cost for the round trip
                net_pnl = gross_pnl + cost_log

                # Record trade
                trades.append(Trade(
                    entry_idx=current_trade_entry_idx,
                    exit_idx=t,
                    direction=position,
                    entry_price=current_trade_entry_price,
                    exit_price=p[t],
                    pnl=gross_pnl,
                    net_pnl=net_pnl
                ))

                # Update equity
                equity += net_pnl

            # Enter new position if not flat
            if new_position != 0:
                current_trade_entry_idx = t
                current_trade_entry_price = p[t]

            position = new_position

        equity_curve.append(equity)

    # Close any open position at the end
    if position != 0:
        price_change = p[-1] - current_trade_entry_price
        gross_pnl = position * price_change
        net_pnl = gross_pnl + cost_log

        trades.append(Trade(
            entry_idx=current_trade_entry_idx,
            exit_idx=N - 1,
            direction=position,
            entry_price=current_trade_entry_price,
            exit_price=p[-1],
            pnl=gross_pnl,
            net_pnl=net_pnl
        ))

        equity += net_pnl
        equity_curve[-1] = equity

    # Compute metrics
    metrics = _compute_metrics(trades, equity_curve)

    return {
        "trades": trades,
        "equity_curve": np.array(equity_curve),
        "metrics": metrics
    }


def _compute_metrics(trades: List[Trade], equity_curve: List[float]) -> Dict:
    """
    Compute performance metrics from trades and equity curve.

    Args:
        trades: List of Trade objects
        equity_curve: Cumulative PnL over time

    Returns:
        Dictionary with performance metrics
    """
    if len(trades) == 0:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_gross_pnl": 0.0,
            "avg_net_pnl": 0.0,
            "total_net_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0
        }

    # Extract PnLs
    net_pnls = np.array([t.net_pnl for t in trades])
    gross_pnls = np.array([t.pnl for t in trades])

    # Win/loss analysis
    wins = net_pnls > 0
    losses = net_pnls < 0

    num_wins = np.sum(wins)
    num_losses = np.sum(losses)
    win_rate = num_wins / len(trades) if len(trades) > 0 else 0.0

    avg_win = np.mean(net_pnls[wins]) if num_wins > 0 else 0.0
    avg_loss = np.mean(net_pnls[losses]) if num_losses > 0 else 0.0

    # Profit factor
    total_wins = np.sum(net_pnls[wins]) if num_wins > 0 else 0.0
    total_losses = abs(np.sum(net_pnls[losses])) if num_losses > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

    # Sharpe ratio (simple: mean / std * sqrt(N))
    if len(net_pnls) > 1 and np.std(net_pnls) > 0:
        sharpe_ratio = np.mean(net_pnls) / np.std(net_pnls) * np.sqrt(len(net_pnls))
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = equity_array - running_max
    max_drawdown = np.min(drawdown)

    return {
        "num_trades": len(trades),
        "win_rate": win_rate,
        "num_wins": int(num_wins),
        "num_losses": int(num_losses),
        "avg_gross_pnl": float(np.mean(gross_pnls)),
        "avg_net_pnl": float(np.mean(net_pnls)),
        "total_net_pnl": float(np.sum(net_pnls)),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor)
    }


def run_symplectic_backtest(
    model,  # SymplecticGlobalModel or SymplecticUltrametricModel
    segments: np.ndarray,
    p: np.ndarray,
    K: int,
    cost_log: float
) -> Dict:
    """
    Backtest symplectic model on segments.

    Uses K-bar segments as inputs to model.get_signal().
    Applies cost_log on every position change.

    Args:
        model: SymplecticGlobalModel or SymplecticUltrametricModel
        segments: Segment array, shape (M, K, 2)
        p: Log prices, shape (N,) where N = M + K - 1
        K: Segment length
        cost_log: Per-round-trip cost in log space

    Returns:
        Dictionary with trades, equity_curve, and metrics
    """
    M = len(segments)
    position = 0  # Start flat
    equity = 0.0
    equity_curve = [0.0]

    trades: List[Trade] = []
    current_trade_entry_idx = None
    current_trade_entry_price = None

    # Start from bar K (first segment ends at bar K-1, we trade at bar K)
    for t in range(K, len(p)):
        # Get segment ending at t-1 (bars [t-K:t])
        seg_idx = t - K
        if seg_idx >= M:
            break

        segment = segments[seg_idx]

        # Get signal from model
        signal = model.get_signal(segment)
        new_position = signal["direction"]

        # Check if position changes
        if new_position != position:
            # Exit current position if any
            if position != 0:
                # Realize PnL on the exit
                price_change = p[t] - current_trade_entry_price
                gross_pnl = position * price_change

                # Apply cost for the round trip
                net_pnl = gross_pnl + cost_log

                # Record trade
                trades.append(Trade(
                    entry_idx=current_trade_entry_idx,
                    exit_idx=t,
                    direction=position,
                    entry_price=current_trade_entry_price,
                    exit_price=p[t],
                    pnl=gross_pnl,
                    net_pnl=net_pnl
                ))

                # Update equity
                equity += net_pnl

            # Enter new position if not flat
            if new_position != 0:
                current_trade_entry_idx = t
                current_trade_entry_price = p[t]

            position = new_position

        equity_curve.append(equity)

    # Close any open position at the end
    if position != 0:
        price_change = p[-1] - current_trade_entry_price
        gross_pnl = position * price_change
        net_pnl = gross_pnl + cost_log

        trades.append(Trade(
            entry_idx=current_trade_entry_idx,
            exit_idx=len(p) - 1,
            direction=position,
            entry_price=current_trade_entry_price,
            exit_price=p[-1],
            pnl=gross_pnl,
            net_pnl=net_pnl
        ))

        equity += net_pnl
        equity_curve[-1] = equity

    # Compute metrics
    metrics = _compute_metrics(trades, equity_curve)

    return {
        "trades": trades,
        "equity_curve": np.array(equity_curve),
        "metrics": metrics
    }


def run_hybrid_backtest(
    model,  # SymplecticUltrametricModel
    p: np.ndarray,
    v: np.ndarray,
    K: int,
    cost_log: float
) -> Dict:
    """
    Backtest hybrid ultrametric-symplectic model.

    Similar to symplectic backtest, but builds segments on the fly
    from p and v arrays.

    Args:
        model: SymplecticUltrametricModel
        p: Log prices, shape (N,)
        v: Normalized volumes, shape (N,)
        K: Segment length
        cost_log: Per-round-trip cost in log space

    Returns:
        Dictionary with trades, equity_curve, and metrics
    """
    from .data_utils import build_gamma
    from .trainer import build_segments

    # Build gamma and segments
    gamma = build_gamma(p, v)
    segments = build_segments(gamma, K)
    M = len(segments)

    position = 0  # Start flat
    equity = 0.0
    equity_curve = [0.0]

    trades: List[Trade] = []
    current_trade_entry_idx = None
    current_trade_entry_price = None

    # Start from bar K (first segment ends at bar K-1, we trade at bar K)
    for t in range(K, len(p)):
        # Get segment ending at t-1 (bars [t-K:t])
        seg_idx = t - K
        if seg_idx >= M:
            break

        segment = segments[seg_idx]

        # Get signal from model
        signal = model.get_signal(segment)
        new_position = signal["direction"]

        # Check if position changes
        if new_position != position:
            # Exit current position if any
            if position != 0:
                # Realize PnL on the exit
                price_change = p[t] - current_trade_entry_price
                gross_pnl = position * price_change

                # Apply cost for the round trip
                net_pnl = gross_pnl + cost_log

                # Record trade
                trades.append(Trade(
                    entry_idx=current_trade_entry_idx,
                    exit_idx=t,
                    direction=position,
                    entry_price=current_trade_entry_price,
                    exit_price=p[t],
                    pnl=gross_pnl,
                    net_pnl=net_pnl
                ))

                # Update equity
                equity += net_pnl

            # Enter new position if not flat
            if new_position != 0:
                current_trade_entry_idx = t
                current_trade_entry_price = p[t]

            position = new_position

        equity_curve.append(equity)

    # Close any open position at the end
    if position != 0:
        price_change = p[-1] - current_trade_entry_price
        gross_pnl = position * price_change
        net_pnl = gross_pnl + cost_log

        trades.append(Trade(
            entry_idx=current_trade_entry_idx,
            exit_idx=len(p) - 1,
            direction=position,
            entry_price=current_trade_entry_price,
            exit_price=p[-1],
            pnl=gross_pnl,
            net_pnl=net_pnl
        ))

        equity += net_pnl
        equity_curve[-1] = equity

    # Compute metrics
    metrics = _compute_metrics(trades, equity_curve)

    return {
        "trades": trades,
        "equity_curve": np.array(equity_curve),
        "metrics": metrics
    }
