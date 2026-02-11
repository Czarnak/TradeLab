"""Standard performance metrics for backtest results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from market_lab.backtest.engine import BacktestResult


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.15 = -15%)."""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def cagr(equity: pd.Series, periods_per_year: float = 252) -> float:
    """Compound annual growth rate."""
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    n_periods = len(equity) - 1
    years = n_periods / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: float = 252) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: float = 252) -> float:
    """Annualised Sortino ratio (downside deviation)."""
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def profit_factor(trades: list) -> float:
    """Gross profit / gross loss. Returns inf if no losses."""
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def win_rate(trades: list) -> float:
    """Fraction of winning trades."""
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t.pnl > 0)
    return winners / len(trades)


def avg_trade_pnl(trades: list) -> float:
    """Average PnL per trade."""
    if not trades:
        return 0.0
    return sum(t.pnl for t in trades) / len(trades)


def exposure(signals: pd.Series) -> float:
    """Fraction of bars with an active position."""
    return float((signals != 0).mean())


def compute_metrics(result: BacktestResult) -> dict:
    """Compute all standard metrics for a backtest result.

    Returns a dict suitable for JSON serialisation.
    """
    eq = result.equity_curve
    rets = result.daily_returns
    trades = result.trades

    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100 if eq.iloc[0] != 0 else 0.0

    return {
        "total_return_pct": round(total_ret, 4),
        "cagr_pct": round(cagr(eq) * 100, 4),
        "max_drawdown_pct": round(max_drawdown(eq) * 100, 4),
        "sharpe": round(sharpe_ratio(rets), 4),
        "sortino": round(sortino_ratio(rets), 4),
        "profit_factor": round(profit_factor(trades), 4),
        "win_rate_pct": round(win_rate(trades) * 100, 2),
        "avg_trade_pnl": round(avg_trade_pnl(trades), 2),
        "total_trades": len(trades),
        "initial_capital": result.config.initial_capital,
        "final_equity": round(eq.iloc[-1], 2),
    }
