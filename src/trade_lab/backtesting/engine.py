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

    ``ticker``, ``start``, and ``end`` are only required when calling
    ``run()`` or ``fetch_data()``. They may be omitted when the engine
    is used exclusively via ``run_on(df)`` — for example inside an
    optimisation loop where data is fetched once and reused across trials.

    Parameters
    ----------
    strategy : BaseStrategy | None
        Trading strategy used by ``run()`` / ``run_on()``.
        May be ``None`` when using the engine only for ``fetch_data()``.
    ticker : str | None
        Yahoo Finance ticker symbol. Required for ``run()`` / ``fetch_data()``.
    start, end : str | None
        Date strings (e.g. ``'2020-01-01'``). Required for ``run()`` / ``fetch_data()``.
    initial_capital : float
    commission : float
        Proportional commission rate (default 0.1%).
    slippage : float
        Proportional slippage rate (default 0.05%).
    """

    def __init__(
        self,
        strategy: BaseStrategy | None = None,
        ticker: str | None = None,
        start: str | None = None,
        end: str | None = None,
        initial_capital: float = 100_000.0,
        leverage: float = 1.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.strategy = strategy
        self.ticker = ticker
        self.start = start
        self.end = end
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission = commission
        self.slippage = slippage

    def run(self) -> BacktestResult:
        """Execute the full backtest pipeline including data download.

        Raises
        ------
        ValueError
            If ``ticker``, ``start``, or ``end`` were not provided at construction.
        """
        if self.strategy is None:
            raise ValueError(
                "BacktestEngine.run()/run_on() requires a strategy. "
                "Set strategy= at construction or assign engine.strategy before calling run."
            )
        if not self.ticker or not self.start or not self.end:
            raise ValueError(
                "BacktestEngine.run() requires ticker, start, and end to be set. "
                "Use run_on(df) if you are supplying data directly."
            )
        df = self._fetch_data()
        return self.run_on(df)

    def run_on(self, df: pd.DataFrame) -> BacktestResult:
        """Execute the backtest pipeline on a pre-built OHLCV DataFrame.

        Skips data download entirely. Useful for Monte Carlo simulations,
        walk-forward testing, or any context where data is sourced externally.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with columns: Open, High, Low, Close, Volume.
            The index should be a DatetimeIndex.

        Returns
        -------
        BacktestResult
        """
        if self.strategy is None:
            raise ValueError(
                "BacktestEngine.run()/run_on() requires a strategy. "
                "Set strategy= at construction or assign engine.strategy before calling run."
            )
        df = self.strategy.generate_signals(df.copy())
        equity_curve, trade_log = self._simulate(df)
        metrics = compute_metrics(equity_curve, trade_log)
        return BacktestResult(
            df=df,
            equity_curve=equity_curve,
            trade_log=trade_log,
            metrics=metrics,
        )

    def fetch_data(self) -> pd.DataFrame:
        """Download and return the OHLCV DataFrame without running the backtest.

        Useful for obtaining the original data to pass into MonteCarloRunner.run().

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame for the configured ticker and date range.

        Raises
        ------
        ValueError
            If ``ticker``, ``start``, or ``end`` were not provided at construction.
        """
        if not self.ticker or not self.start or not self.end:
            raise ValueError(
                "BacktestEngine.fetch_data() requires ticker, start, and end to be set."
            )
        return self._fetch_data()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _fetch_data(self) -> pd.DataFrame:
        df = yf.download(self.ticker, start=self.start, end=self.end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")
        return df

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Average True Range for position sizing volatility input."""
        high = df["High"].to_numpy()
        low = df["Low"].to_numpy()
        close = df["Close"].to_numpy()

        prev_close = np.empty(len(close), dtype=float)
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
        signals = df["signal_strength"].to_numpy()
        closes = df["Close"].to_numpy()
        dates = df.index
        has_sizer = self.strategy.position_sizer is not None
        atr = self._compute_atr(df) if has_sizer else None

        # State
        cash = self.initial_capital
        pos = 0.0  # units held (+ long, − short)
        entry_price = 0.0
        entry_date = None
        entry_bar = 0
        entry_comm = 0.0
        direction = 0
        tp_level: float | None = None
        sl_level: float | None = None
        ts_level: float | None = None
        tp_obj = None
        sl_obj = None
        ts_obj = None

        equity_arr = np.empty(n)
        trades: list[dict] = []

        et = self.strategy.entry_threshold
        xt = self.strategy.exit_threshold
        allow_long = self.strategy.allow_long
        allow_short = self.strategy.allow_short

        def _active_stop_reason(
            stop_loss_level: float | None,
            trailing_stop_level: float | None,
            trade_direction: int,
        ) -> str:
            if stop_loss_level is not None and trailing_stop_level is not None:
                if trade_direction == 1:
                    return "ts" if trailing_stop_level >= stop_loss_level else "sl"
                return "ts" if trailing_stop_level <= stop_loss_level else "sl"
            return "sl" if stop_loss_level is not None else "ts"

        def _reset_risk_state() -> None:
            nonlocal direction, tp_level, sl_level, ts_level, tp_obj, sl_obj, ts_obj
            direction = 0
            tp_level = None
            sl_level = None
            ts_level = None
            tp_obj = None
            sl_obj = None
            ts_obj = None

        for i in range(n):
            price = closes[i]
            sig = signals[i]

            # ---- Close existing position if triggered ----
            if pos > 0:  # long
                bar = df.iloc[i]
                high = df["High"].iloc[i]
                low = df["Low"].iloc[i]

                if ts_obj and ts_level is not None and not np.isnan(sig):
                    ts_level = ts_obj.update(ts_level, direction, sig, bar)

                active_sl = None
                if sl_level is not None and ts_level is not None:
                    active_sl = max(sl_level, ts_level)
                elif sl_level is not None:
                    active_sl = sl_level
                elif ts_level is not None:
                    active_sl = ts_level

                tp_hit = tp_level is not None and high >= tp_level
                sl_hit = active_sl is not None and low <= active_sl

                if tp_hit:
                    exec_price = tp_level
                    comm = abs(pos) * exec_price * self.commission
                    proceeds = pos * exec_price - comm
                    pnl = (proceeds - (pos * entry_price + entry_comm)) / self.leverage
                    cash += pnl
                    trades.append(
                        {
                            "direction": "long",
                            "entry_date": entry_date,
                            "exit_date": dates[i],
                            "entry_price": entry_price,
                            "exit_price": exec_price,
                            "size": pos,
                            "pnl": pnl,
                            "commission": entry_comm + comm,
                            "bars_held": i - entry_bar,
                            "exit_reason": "tp",
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

                elif sl_hit:
                    exec_price = active_sl
                    comm = abs(pos) * exec_price * self.commission
                    proceeds = pos * exec_price - comm
                    pnl = (proceeds - (pos * entry_price + entry_comm)) / self.leverage
                    cash += pnl
                    trades.append(
                        {
                            "direction": "long",
                            "entry_date": entry_date,
                            "exit_date": dates[i],
                            "entry_price": entry_price,
                            "exit_price": exec_price,
                            "size": pos,
                            "pnl": pnl,
                            "commission": entry_comm + comm,
                            "bars_held": i - entry_bar,
                            "exit_reason": _active_stop_reason(
                                sl_level,
                                ts_level,
                                direction,
                            ),
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

                elif sig < xt or (allow_short and sig < -et):
                    exec_price = price * (1 - self.slippage)
                    comm = abs(pos) * exec_price * self.commission
                    proceeds = pos * exec_price - comm
                    pnl = (proceeds - (pos * entry_price + entry_comm)) / self.leverage
                    cash += pnl
                    trades.append(
                        {
                            "direction": "long",
                            "entry_date": entry_date,
                            "exit_date": dates[i],
                            "entry_price": entry_price,
                            "exit_price": exec_price,
                            "size": pos,
                            "pnl": pnl,
                            "commission": entry_comm + comm,
                            "bars_held": i - entry_bar,
                            "exit_reason": "signal",
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

            elif pos < 0:  # short
                bar = df.iloc[i]
                high = df["High"].iloc[i]
                low = df["Low"].iloc[i]

                if ts_obj and ts_level is not None and not np.isnan(sig):
                    ts_level = ts_obj.update(ts_level, direction, sig, bar)

                active_sl = None
                if sl_level is not None and ts_level is not None:
                    active_sl = min(sl_level, ts_level)
                elif sl_level is not None:
                    active_sl = sl_level
                elif ts_level is not None:
                    active_sl = ts_level

                tp_hit = tp_level is not None and low <= tp_level
                sl_hit = active_sl is not None and high >= active_sl

                if tp_hit:
                    exec_price = tp_level
                    comm = abs(pos) * exec_price * self.commission
                    cost = abs(pos) * exec_price + comm
                    pnl = (abs(pos) * entry_price - cost - entry_comm) / self.leverage
                    cash += pnl
                    trades.append(
                        {
                            "direction": "short",
                            "entry_date": entry_date,
                            "exit_date": dates[i],
                            "entry_price": entry_price,
                            "exit_price": exec_price,
                            "size": abs(pos),
                            "pnl": pnl,
                            "commission": entry_comm + comm,
                            "bars_held": i - entry_bar,
                            "exit_reason": "tp",
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

                elif sl_hit:
                    exec_price = active_sl
                    comm = abs(pos) * exec_price * self.commission
                    cost = abs(pos) * exec_price + comm
                    pnl = (abs(pos) * entry_price - cost - entry_comm) / self.leverage
                    cash += pnl
                    trades.append(
                        {
                            "direction": "short",
                            "entry_date": entry_date,
                            "exit_date": dates[i],
                            "entry_price": entry_price,
                            "exit_price": exec_price,
                            "size": abs(pos),
                            "pnl": pnl,
                            "commission": entry_comm + comm,
                            "bars_held": i - entry_bar,
                            "exit_reason": _active_stop_reason(
                                sl_level,
                                ts_level,
                                direction,
                            ),
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

                elif sig > -xt or (allow_long and sig > et):
                    exec_price = price * (1 + self.slippage)
                    comm = abs(pos) * exec_price * self.commission
                    cost = abs(pos) * exec_price + comm
                    pnl = (abs(pos) * entry_price - cost - entry_comm) / self.leverage
                    cash += pnl
                    trades.append(
                        {
                            "direction": "short",
                            "entry_date": entry_date,
                            "exit_date": dates[i],
                            "entry_price": entry_price,
                            "exit_price": exec_price,
                            "size": abs(pos),
                            "pnl": pnl,
                            "commission": entry_comm + comm,
                            "bars_held": i - entry_bar,
                            "exit_reason": "signal",
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

            # Skip new entries on bars with no signal or depleted equity.
            current_equity = cash + pos * (price - entry_price) / self.leverage
            if np.isnan(sig) or current_equity <= 0:
                equity_arr[i] = max(
                    current_equity,
                    0.0,
                )
                continue

            # ---- Open new position if flat ----
            if pos == 0.0:
                if allow_long and sig > et:
                    exec_price = price * (1 + self.slippage)
                    size = (
                        self.strategy.position_sizer.compute_size(
                            sig,
                            cash,
                            exec_price,
                            volatility=atr[i] if atr is not None else None,
                        )
                        if has_sizer
                        else min(round(cash / exec_price / self.leverage, 2), 2250.0)
                    )
                    if size > 0:
                        comm = size * exec_price * self.commission
                        cash -= size * exec_price + comm
                        pos = size
                        entry_price = exec_price
                        entry_date = dates[i]
                        entry_bar = i
                        entry_comm = comm
                        direction = 1
                        bar = df.iloc[i]
                        tp_obj = self.strategy.take_profit
                        sl_obj = self.strategy.stop_loss
                        ts_obj = self.strategy.trailing_stop
                        tp_level = (
                            tp_obj.compute(exec_price, direction, sig, bar)
                            if tp_obj
                            else None
                        )
                        sl_level = (
                            sl_obj.compute(exec_price, direction, sig, bar)
                            if sl_obj
                            else None
                        )
                        ts_level = (
                            ts_obj.compute_initial(exec_price, direction, sig, bar)
                            if ts_obj
                            else None
                        )

                elif allow_short and sig < -et:
                    exec_price = price * (1 - self.slippage)
                    size = (
                        self.strategy.position_sizer.compute_size(
                            sig,
                            cash,
                            exec_price,
                            volatility=atr[i] if atr is not None else None,
                        )
                        if has_sizer
                        else min(round(cash / exec_price / self.leverage, 2), 2250.0)
                    )
                    if size > 0:
                        comm = size * exec_price * self.commission
                        cash += size * exec_price - comm
                        pos = -size
                        entry_price = exec_price
                        entry_date = dates[i]
                        entry_bar = i
                        entry_comm = comm
                        direction = -1
                        bar = df.iloc[i]
                        tp_obj = self.strategy.take_profit
                        sl_obj = self.strategy.stop_loss
                        ts_obj = self.strategy.trailing_stop
                        tp_level = (
                            tp_obj.compute(exec_price, direction, sig, bar)
                            if tp_obj
                            else None
                        )
                        sl_level = (
                            sl_obj.compute(exec_price, direction, sig, bar)
                            if sl_obj
                            else None
                        )
                        ts_level = (
                            ts_obj.compute_initial(exec_price, direction, sig, bar)
                            if ts_obj
                            else None
                        )

            equity_arr[i] = cash + pos * (price - entry_price) / self.leverage

        equity_curve = pd.Series(equity_arr, index=dates, name="equity")
        trade_log = pd.DataFrame(trades)
        return equity_curve, trade_log
