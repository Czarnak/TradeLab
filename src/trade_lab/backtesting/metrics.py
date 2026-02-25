import numpy as np
import pandas as pd


def _direction_stats(trade_log: pd.DataFrame, direction: str) -> tuple[float, float, float]:
    """Return (win_rate, avg_win, avg_loss) for a direction."""
    if 'direction' not in trade_log.columns:
        return 0.0, 0.0, 0.0
    trades = trade_log[trade_log['direction'] == direction]
    if len(trades) == 0:
        return 0.0, 0.0, 0.0

    winners = trades[trades['pnl'] > 0]
    losers = trades[trades['pnl'] <= 0]

    win_rate = len(winners) / len(trades)
    avg_win = winners['pnl'].mean() if len(winners) > 0 else 0.0
    avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0.0
    return win_rate, avg_win, avg_loss


def compute_metrics(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
    risk_free_rate: float = 10.0,
) -> dict:
    """Compute performance metrics from backtest results.

    Parameters
    ----------
    equity_curve : pd.Series
        Daily equity values indexed by date.
    trade_log : pd.DataFrame
        Trade log with columns: pnl, bars_held, etc.
    risk_free_rate : float
        Annualised risk-free rate for Sharpe/Sortino calculation.

    Returns
    -------
    dict
        Dictionary of named performance metrics.
    """
    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    n_days = len(equity_curve)

    # --- Return metrics ---
    if np.isfinite(initial) and initial > 0 and np.isfinite(final):
        total_return = (final - initial) / initial
    else:
        total_return = 0.0

    if total_return <= -1.0:
        annualized_return = -1.0
    else:
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    if not np.isfinite(annualized_return):
        annualized_return = 0.0

    # --- Risk metrics ---
    # Equity can hit 0 in edge cases; pct_change from/to 0 can produce +/-inf.
    # Keep only finite returns so volatility metrics don't emit runtime warnings.
    daily_returns = (
        equity_curve
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    annual_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0.0
    if not np.isfinite(annual_vol):
        annual_vol = 0.0
    sharpe = (
        (annualized_return - risk_free_rate) / annual_vol
        if annual_vol > 0 else 0.0
    )

    downside = daily_returns[daily_returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.0
    if not np.isfinite(downside_vol):
        downside_vol = 0.0
    sortino = (
        (annualized_return - risk_free_rate) / downside_vol
        if downside_vol > 0 else 0.0
    )

    # --- Drawdown ---
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()

    # --- Trade statistics ---
    n_trades = len(trade_log)
    if n_trades > 0:
        winners = trade_log[trade_log['pnl'] > 0]
        losers = trade_log[trade_log['pnl'] <= 0]
        win_rate = len(winners) / n_trades
        gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0.0
        gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0.0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0.0
        avg_bars = trade_log['bars_held'].mean()
        total_commission = trade_log['commission'].sum()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        avg_bars = 0.0
        total_commission = 0.0

    long_win_rate, long_avg_win, long_avg_loss = _direction_stats(trade_log, 'long')
    short_win_rate, short_avg_win, short_avg_loss = _direction_stats(trade_log, 'short')

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'annual_volatility': annual_vol,
        'total_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_trade_bars': avg_bars,
        'total_commission': total_commission,
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate,
        'long_avg_win': long_avg_win,
        'long_avg_loss': long_avg_loss,
        'short_avg_win': short_avg_win,
        'short_avg_loss': short_avg_loss,
    }
