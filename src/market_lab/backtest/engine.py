"""Minimal but testable backtesting engine.

Supports long and short positions, market orders executed on next-bar open,
fixed-fraction or fixed-quantity sizing, and commission/slippage in bps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from market_lab.utils.config import DEFAULT_COMMISSION_BPS, DEFAULT_SLIPPAGE_BPS
from market_lab.utils.logging import get_logger

log = get_logger("backtest.engine")


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    initial_capital: float = 100_000.0
    commission_bps: float = DEFAULT_COMMISSION_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    position_sizing: Literal["fixed_fraction", "fixed_quantity"] = "fixed_fraction"
    size_value: float = 1.0  # fraction (0-1) or quantity (shares)
    allow_short: bool = True


@dataclass
class Trade:
    """Record of a completed round-trip trade."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    bars_held: int


@dataclass
class BacktestResult:
    """Complete output of a backtest run."""

    config: BacktestConfig
    equity_curve: pd.Series  # indexed by timestamp
    daily_returns: pd.Series
    trades: list[Trade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _apply_slippage(price: float, direction: int, slippage_bps: float) -> float:
    """Adjust price for slippage (worse fill)."""
    slip = price * slippage_bps / 10_000
    return price + slip * direction  # buying costs more, selling gets less


def _commission_cost(price: float, quantity: float, commission_bps: float) -> float:
    """Calculate commission for a trade."""
    return abs(price * quantity) * commission_bps / 10_000


def run_backtest(
    bars: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a backtest.

    Parameters
    ----------
    bars : pd.DataFrame
        Canonical OHLCV bars.
    signals : pd.Series
        Signal series aligned to bars: +1 (long), -1 (short), 0 (flat).
    config : BacktestConfig or None
        Configuration; uses defaults if None.

    Returns
    -------
    BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    bars = bars.copy()
    signals = signals.reindex(bars.index).fillna(0).astype(int)

    if not config.allow_short:
        signals = signals.clip(lower=0)

    n = len(bars)
    equity = np.full(n, config.initial_capital, dtype=float)
    cash = config.initial_capital
    position = 0  # current shares held (neg for short)
    direction = 0
    entry_price = 0.0
    entry_date = bars.index[0]
    entry_idx = 0

    trades: list[Trade] = []

    opens = bars["open"].values
    closes = bars["close"].values

    for i in range(1, n):
        target_dir = int(signals.iloc[i - 1])  # signal from previous bar
        current_open = opens[i]

        # --- Check for position change ---
        if target_dir != direction:
            # Close existing position
            if direction != 0 and position != 0:
                exit_price = _apply_slippage(current_open, -direction, config.slippage_bps)
                close_comm = _commission_cost(exit_price, abs(position), config.commission_bps)
                pnl = direction * (exit_price - entry_price) * abs(position) - close_comm
                cash += position * exit_price - close_comm
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=bars.index[i],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=abs(position),
                    pnl=pnl,
                    commission=close_comm,
                    bars_held=i - entry_idx,
                ))
                position = 0
                direction = 0

            # Open new position
            if target_dir != 0:
                fill_price = _apply_slippage(current_open, target_dir, config.slippage_bps)
                if config.position_sizing == "fixed_fraction":
                    notional = cash * config.size_value
                    qty = abs(notional / fill_price)
                else:
                    qty = config.size_value

                open_comm = _commission_cost(fill_price, qty, config.commission_bps)
                cash -= target_dir * qty * fill_price + open_comm
                position = target_dir * qty
                direction = target_dir
                entry_price = fill_price
                entry_date = bars.index[i]
                entry_idx = i

        # Mark-to-market equity
        mtm = cash + position * closes[i]
        equity[i] = mtm

    # Close any open position at the end
    if direction != 0 and position != 0:
        exit_price = closes[-1]
        close_comm = _commission_cost(exit_price, abs(position), config.commission_bps)
        pnl = direction * (exit_price - entry_price) * abs(position) - close_comm
        cash += position * exit_price - close_comm
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=bars.index[-1],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=abs(position),
            pnl=pnl,
            commission=close_comm,
            bars_held=n - 1 - entry_idx,
        ))
        equity[-1] = cash

    equity_series = pd.Series(equity, index=bars.index, name="equity")
    daily_returns = equity_series.pct_change().fillna(0.0)
    daily_returns.name = "returns"

    result = BacktestResult(
        config=config,
        equity_curve=equity_series,
        daily_returns=daily_returns,
        trades=trades,
    )

    # Compute metrics
    from market_lab.backtest.metrics import compute_metrics
    result.metrics = compute_metrics(result)

    log.info(
        "Backtest complete: %d bars, %d trades, total_return=%.2f%%",
        n, len(trades), result.metrics.get("total_return_pct", 0),
    )
    return result
