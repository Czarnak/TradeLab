from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from trade_lab.strategies.base import BaseStrategy
from trade_lab.backtesting.metrics import compute_metrics


@dataclass
class BacktestResult:
    """Container for backtest output."""
    df: pd.DataFrame
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    metrics: dict


class BacktestEngine:
    """Event-driven backtesting engine.

    Downloads OHLCV data via yfinance, runs the strategy to produce
    a ``signal_strength`` column, then simulates trading bar-by-bar
    using the strategy's entry/exit thresholds.

    Parameters
    ----------
    strategy : BaseStrategy
    ticker : str
        Yahoo Finance ticker symbol.
    start, end : str
        Date strings (e.g. ``'2020-01-01'``).
    initial_capital : float
    commission : float
        Proportional commission rate (default 0.1%).
    slippage : float
        Proportional slippage rate (default 0.05%).
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        ticker: str,
        start: str,
        end: str,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.strategy = strategy
        self.ticker = ticker
        self.start = start
        self.end = end
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(self) -> BacktestResult:
        """Execute the full backtest pipeline."""
        df = self._fetch_data()
        df = self.strategy.generate_signals(df)
        equity_curve, trade_log = self._simulate(df)
        metrics = compute_metrics(equity_curve, trade_log)
        return BacktestResult(
            df=df,
            equity_curve=equity_curve,
            trade_log=trade_log,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _fetch_data(self) -> pd.DataFrame:
        df = yf.download(self.ticker, start=self.start, end=self.end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel('Ticker')
        return df

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Average True Range for position sizing volatility input."""
        high = df['High'].to_numpy()
        low = df['Low'].to_numpy()
        close = df['Close'].to_numpy()

        prev_close = np.empty_like(close)
        prev_close[0] = np.nan
        prev_close[1:] = close[:-1]

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )
        atr = pd.Series(tr).rolling(window=period).mean().to_numpy()
        return atr

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(self, df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        n = len(df)
        signals = df['signal_strength'].to_numpy()
        closes = df['Close'].to_numpy()
        dates = df.index
        has_sizer = self.strategy.position_sizer is not None
        atr = self._compute_atr(df) if has_sizer else None

        # State
        cash = self.initial_capital
        pos = 0.0           # units held (+ long, − short)
        entry_price = 0.0
        entry_date = None
        entry_bar = 0
        entry_comm = 0.0

        equity_arr = np.empty(n)
        trades: list[dict] = []

        et = self.strategy.entry_threshold
        xt = self.strategy.exit_threshold
        allow_long = self.strategy.allow_long
        allow_short = self.strategy.allow_short

        for i in range(n):
            price = closes[i]
            sig = signals[i]
            equity = cash + pos * price

            # Skip bars with no signal or depleted equity
            if np.isnan(sig) or equity <= 0:
                equity_arr[i] = max(equity, 0.0)
                continue

            # ---- Close existing position if triggered ----
            if pos > 0:  # long
                should_exit = abs(sig) < xt
                should_reverse = sig < -et and allow_short
                if should_exit or should_reverse:
                    cash, pos = self._close_long(
                        cash, pos, price, entry_price, entry_comm,
                        entry_date, dates[i], i, entry_bar, trades,
                    )

            elif pos < 0:  # short
                should_exit = abs(sig) < xt
                should_reverse = sig > et and allow_long
                if should_exit or should_reverse:
                    cash, pos = self._close_short(
                        cash, pos, price, entry_price, entry_comm,
                        entry_date, dates[i], i, entry_bar, trades,
                    )

            # ---- Open new position if now flat ----
            if pos == 0.0:
                cur_equity = cash
                atr_val = atr[i] if atr is not None else None

                if sig > et and allow_long:
                    exec_p = price * (1 + self.slippage)
                    units = self._size(sig, cur_equity, exec_p, atr_val)
                    max_afford = cash / (exec_p * (1 + self.commission))
                    units = min(units, max_afford)
                    if units > 0:
                        comm = units * exec_p * self.commission
                        cash -= units * exec_p + comm
                        pos = units
                        entry_price = exec_p
                        entry_date = dates[i]
                        entry_bar = i
                        entry_comm = comm

                elif sig < -et and allow_short:
                    exec_p = price * (1 - self.slippage)
                    units = self._size(sig, cur_equity, exec_p, atr_val)
                    max_units = cur_equity / (exec_p * (1 + self.commission))
                    units = min(units, max_units)
                    if units > 0:
                        comm = units * exec_p * self.commission
                        cash += units * exec_p - comm
                        pos = -units
                        entry_price = exec_p
                        entry_date = dates[i]
                        entry_bar = i
                        entry_comm = comm

            equity_arr[i] = cash + pos * price

        # ---- Force-close any open position at final bar ----
        if pos > 0:
            cash, pos = self._close_long(
                cash, pos, closes[-1], entry_price, entry_comm,
                entry_date, dates[-1], n - 1, entry_bar, trades,
            )
            equity_arr[-1] = cash
        elif pos < 0:
            cash, pos = self._close_short(
                cash, pos, closes[-1], entry_price, entry_comm,
                entry_date, dates[-1], n - 1, entry_bar, trades,
            )
            equity_arr[-1] = cash

        equity_curve = pd.Series(equity_arr, index=dates, name='equity')

        trade_cols = [
            'direction', 'entry_date', 'entry_price', 'exit_date',
            'exit_price', 'units', 'pnl', 'return_pct', 'commission',
            'bars_held',
        ]
        trade_log = pd.DataFrame(trades, columns=trade_cols) if trades else pd.DataFrame(columns=trade_cols)

        return equity_curve, trade_log

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _size(self, signal: float, equity: float, exec_price: float, atr_val: float | None) -> float:
        """Compute position size in units."""
        if self.strategy.position_sizer:
            return self.strategy.position_sizer.compute_size(
                signal_strength=signal,
                equity=equity,
                price=exec_price,
                volatility=atr_val,
            )
        # Default: allocate full equity
        return equity / exec_price

    def _close_long(
        self, cash, pos, price, entry_price, entry_comm,
        entry_date, exit_date, exit_bar, entry_bar, trades,
    ) -> tuple[float, float]:
        exec_p = price * (1 - self.slippage)
        proceeds = pos * exec_p
        comm = proceeds * self.commission
        cash += proceeds - comm
        pnl = pos * (exec_p - entry_price) - entry_comm - comm
        trades.append({
            'direction': 'long',
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exec_p,
            'units': pos,
            'pnl': pnl,
            'return_pct': pnl / (pos * entry_price) if entry_price > 0 else 0.0,
            'commission': entry_comm + comm,
            'bars_held': exit_bar - entry_bar,
        })
        return cash, 0.0

    def _close_short(
        self, cash, pos, price, entry_price, entry_comm,
        entry_date, exit_date, exit_bar, entry_bar, trades,
    ) -> tuple[float, float]:
        units = abs(pos)
        exec_p = price * (1 + self.slippage)
        cost = units * exec_p
        comm = cost * self.commission
        cash -= cost + comm
        pnl = units * (entry_price - exec_p) - entry_comm - comm
        trades.append({
            'direction': 'short',
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exec_p,
            'units': units,
            'pnl': pnl,
            'return_pct': pnl / (units * entry_price) if entry_price > 0 else 0.0,
            'commission': entry_comm + comm,
            'bars_held': exit_bar - entry_bar,
        })
        return cash, 0.0
