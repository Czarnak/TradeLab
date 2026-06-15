from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_lab.strategies.base import BaseStrategy
from trade_lab.backtesting.metrics import compute_metrics

# Decimal places for default full-equity position sizing.
_SIZE_PRECISION = 2


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
    risk_free_rate : float
        Annual risk-free rate as a plain fraction (e.g. ``0.02`` = 2%),
        used for Sharpe/Sortino. Defaults to ``0.0``.
    leverage : float
        Position-size multiplier, enabling CFD-style margin trading. The size
        produced by the default full-equity sizing or by a ``position_sizer``
        is multiplied by ``leverage`` (e.g. ``2.0`` doubles the exposure for the
        same signal). Defaults to ``1.0`` (no leverage). Must be ``>= 1.0``.
    contract_size : float
        Dollar value of a 1.0 price-point move on a single contract (a.k.a.
        point value / multiplier). For equities this is ``1.0`` (price is the
        dollar value per unit); for index/commodity CFDs it is the contract
        multiplier. The dollar value of a position is ``pos * price *
        contract_size``. Defaults to ``1.0``. Must be ``> 0``.
    maintenance_margin : float
        Stop-out level as a fraction of the initial margin posted at entry. A
        leveraged position is force-liquidated when its equity falls to
        ``maintenance_margin * initial_margin``. Only active when
        ``leverage > 1.0``. Defaults to ``0.5``. Must be in ``[0, 1)``.
    financing_rate : float
        Per-bar financing (swap) rate charged on the full mark-to-market
        notional of a position held across bars. Deducted from cash each bar a
        position is carried and reflected in the per-trade ``financing`` column.
        Defaults to ``0.0`` (no financing).
    execute_on : str
        Execution timing for **signal-driven** entries, exits and flips:
        ``"close"`` (default) fills at the signal bar's close — the legacy,
        backward-compatible path; ``"next_open"`` makes the decision on the prior
        bar's signal and fills at the current bar's open (decide at ``Close[t-1]``,
        fill at ``Open[t]``), removing the same-bar-close lookahead. Protective
        stops (TP/SL/trailing/liquidation) remain level-based intrabar checks and
        equity is always marked-to-market at the close, regardless of this setting.

    Notes
    -----
    Accounting is mark-to-market: equity each bar is
    ``cash + pos * price * contract_size`` (``cash`` holds the remaining
    balance, ``pos`` is signed units — positive long, negative short). Entry
    moves the full notional through ``cash``, exits move the realised
    proceeds/cost, and the per-trade ``pnl`` is kept for the trade log only.

    CFD / leverage: ``leverage`` multiplies the opened position size, so a
    position's notional can exceed available cash — ``cash`` then goes negative
    (borrowed funds) while equity stays correct. ``contract_size`` scales every
    dollar conversion for index/commodity CFDs. When ``leverage > 1.0``, a
    margin-call liquidation level is tracked and force-closes the position if
    equity breaches ``maintenance_margin``; liquidation is disabled at
    ``leverage == 1.0`` (a cash account cannot be margin-called). Overnight
    financing is charged on the full notional of carried positions. All four
    parameters default to a no-op, reproducing the plain-equity behaviour.

    Re-entry is allowed on the **same bar** that a take-profit / stop-loss exit
    fires: after closing, the entry block runs on the same bar, so a position
    can reopen immediately if the signal still exceeds the entry threshold.
    """

    def __init__(
        self,
        strategy: BaseStrategy | None = None,
        ticker: str | None = None,
        start: str | None = None,
        end: str | None = None,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.0,
        leverage: float = 1.0,
        contract_size: float = 1.0,
        maintenance_margin: float = 0.5,
        financing_rate: float = 0.0,
        execute_on: str = "close",
    ):
        if leverage < 1.0:
            raise ValueError("leverage must be >= 1.0")
        if contract_size <= 0:
            raise ValueError("contract_size must be > 0")
        if not 0 <= maintenance_margin < 1:
            raise ValueError("maintenance_margin must be in [0, 1)")
        if not np.isfinite(financing_rate):
            raise ValueError("financing_rate must be finite")
        if execute_on not in ("close", "next_open"):
            raise ValueError("execute_on must be 'close' or 'next_open'")

        self.strategy = strategy
        self.execute_on = execute_on
        self.ticker = ticker
        self.start = start
        self.end = end
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.leverage = leverage
        self.contract_size = contract_size
        self.maintenance_margin = maintenance_margin
        self.financing_rate = financing_rate

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
        metrics = compute_metrics(
            equity_curve, trade_log, risk_free_rate=self.risk_free_rate
        )
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
        import yfinance as yf

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
        highs = df["High"].to_numpy()
        lows = df["Low"].to_numpy()
        dates = df.index
        has_sizer = self.strategy.position_sizer is not None
        atr = self._compute_atr(df) if has_sizer else None

        # Execution timing. In "next_open" mode the decision at bar i uses the
        # prior bar's signal and signal-driven fills happen at this bar's open
        # (decide at close[i-1], fill at open[i]) — no same-bar-close lookahead.
        next_open = self.execute_on == "next_open"
        opens = df["Open"].to_numpy() if next_open else None

        # CFD / leverage parameters (bound locally for the hot loop).
        cs = self.contract_size
        lev = self.leverage
        fin_rate = self.financing_rate
        maint = self.maintenance_margin
        use_liq = lev > 1.0  # margin call only applies to a leveraged account

        # State
        cash = self.initial_capital
        pos = 0.0  # units held (+ long, − short)
        entry_price = 0.0
        entry_date = None
        entry_bar = 0
        entry_comm = 0.0
        financing_accrued = 0.0  # swap charged over the current trade's life
        used_margin = 0.0  # initial margin posted at entry (for liquidation)
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

        def _binding_stop(
            stop_loss_level: float | None,
            trailing_stop_level: float | None,
            liquidation_level: float | None,
            trade_direction: int,
        ) -> tuple[float, str] | None:
            """First protective stop hit on an adverse move: ``(level, reason)``.

            For a long the binding level is the highest present level (price
            falls into it first); for a short it is the lowest. Tie-break
            priority ``ts > sl > liquidation`` (candidate order) preserves the
            legacy sl/ts behaviour when no liquidation level is present.
            """
            candidates: list[tuple[float, str]] = []
            if trailing_stop_level is not None:
                candidates.append((trailing_stop_level, "ts"))
            if stop_loss_level is not None:
                candidates.append((stop_loss_level, "sl"))
            if liquidation_level is not None:
                candidates.append((liquidation_level, "liquidation"))
            if not candidates:
                return None
            if trade_direction == 1:
                return max(candidates, key=lambda c: c[0])
            return min(candidates, key=lambda c: c[0])

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
            price = closes[i]  # mark-to-market / financing / intrabar reference
            if next_open:
                # Act on the prior bar's signal, fill at this bar's open.
                sig = signals[i - 1] if i > 0 else np.nan
                fill = opens[i]
            else:
                sig = signals[i]
                fill = price

            # ---- Overnight financing (swap) on a carried position ----
            # Charged for each night already held (entry bar excluded), on the
            # full mark-to-market notional, before any exit/equity check.
            if pos != 0.0 and i > entry_bar and fin_rate:
                fin = abs(pos) * price * cs * fin_rate
                cash -= fin
                financing_accrued += fin

            # ---- Close existing position if triggered ----
            if pos > 0:  # long
                high = highs[i]
                low = lows[i]

                if ts_obj and ts_level is not None and not np.isnan(sig):
                    bar = df.iloc[i]
                    ts_level = ts_obj.update(ts_level, direction, sig, bar)

                liq_level = (
                    (maint * used_margin - cash) / (pos * cs) if use_liq else None
                )
                binding = _binding_stop(sl_level, ts_level, liq_level, direction)

                tp_hit = tp_level is not None and high >= tp_level
                sl_hit = binding is not None and low <= binding[0]

                if tp_hit:
                    exec_price = tp_level
                    comm = abs(pos) * exec_price * cs * self.commission
                    proceeds = pos * exec_price * cs - comm
                    pnl = (
                        proceeds
                        - (pos * entry_price * cs + entry_comm)
                        - financing_accrued
                    )
                    cash += proceeds
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
                            "financing": financing_accrued,
                            "bars_held": i - entry_bar,
                            "exit_reason": "tp",
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

                elif sl_hit:
                    exec_price = binding[0]
                    comm = abs(pos) * exec_price * cs * self.commission
                    proceeds = pos * exec_price * cs - comm
                    pnl = (
                        proceeds
                        - (pos * entry_price * cs + entry_comm)
                        - financing_accrued
                    )
                    cash += proceeds
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
                            "financing": financing_accrued,
                            "bars_held": i - entry_bar,
                            "exit_reason": binding[1],
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

                elif sig < xt or (allow_short and sig < -et):
                    exec_price = fill * (1 - self.slippage)
                    comm = abs(pos) * exec_price * cs * self.commission
                    proceeds = pos * exec_price * cs - comm
                    pnl = (
                        proceeds
                        - (pos * entry_price * cs + entry_comm)
                        - financing_accrued
                    )
                    cash += proceeds
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
                            "financing": financing_accrued,
                            "bars_held": i - entry_bar,
                            "exit_reason": "signal",
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

            elif pos < 0:  # short
                high = highs[i]
                low = lows[i]

                if ts_obj and ts_level is not None and not np.isnan(sig):
                    bar = df.iloc[i]
                    ts_level = ts_obj.update(ts_level, direction, sig, bar)

                liq_level = (
                    (maint * used_margin - cash) / (pos * cs) if use_liq else None
                )
                binding = _binding_stop(sl_level, ts_level, liq_level, direction)

                tp_hit = tp_level is not None and low <= tp_level
                sl_hit = binding is not None and high >= binding[0]

                if tp_hit:
                    exec_price = tp_level
                    comm = abs(pos) * exec_price * cs * self.commission
                    cost = abs(pos) * exec_price * cs + comm
                    pnl = (
                        abs(pos) * entry_price * cs
                        - cost
                        - entry_comm
                        - financing_accrued
                    )
                    cash -= cost
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
                            "financing": financing_accrued,
                            "bars_held": i - entry_bar,
                            "exit_reason": "tp",
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

                elif sl_hit:
                    exec_price = binding[0]
                    comm = abs(pos) * exec_price * cs * self.commission
                    cost = abs(pos) * exec_price * cs + comm
                    pnl = (
                        abs(pos) * entry_price * cs
                        - cost
                        - entry_comm
                        - financing_accrued
                    )
                    cash -= cost
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
                            "financing": financing_accrued,
                            "bars_held": i - entry_bar,
                            "exit_reason": binding[1],
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

                elif sig > -xt or (allow_long and sig > et):
                    exec_price = fill * (1 + self.slippage)
                    comm = abs(pos) * exec_price * cs * self.commission
                    cost = abs(pos) * exec_price * cs + comm
                    pnl = (
                        abs(pos) * entry_price * cs
                        - cost
                        - entry_comm
                        - financing_accrued
                    )
                    cash -= cost
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
                            "financing": financing_accrued,
                            "bars_held": i - entry_bar,
                            "exit_reason": "signal",
                        }
                    )
                    pos = 0.0
                    _reset_risk_state()

            # Skip new entries on bars with no signal or depleted equity.
            current_equity = cash + pos * price * cs
            if np.isnan(sig) or current_equity <= 0:
                equity_arr[i] = max(
                    current_equity,
                    0.0,
                )
                continue

            # ---- Open new position if flat ----
            if pos == 0.0:
                if allow_long and sig > et:
                    exec_price = fill * (1 + self.slippage)
                    size = (
                        self.strategy.position_sizer.compute_size(
                            sig,
                            cash,
                            exec_price,
                            volatility=atr[i] if atr is not None else None,
                        )
                        * lev
                        / cs
                        if has_sizer
                        else round(cash * lev / (exec_price * cs), _SIZE_PRECISION)
                    )
                    if size > 0:
                        comm = size * exec_price * cs * self.commission
                        cash -= size * exec_price * cs + comm
                        pos = size
                        entry_price = exec_price
                        entry_date = dates[i]
                        entry_bar = i
                        entry_comm = comm
                        financing_accrued = 0.0
                        used_margin = size * exec_price * cs / lev
                        direction = 1
                        tp_obj = self.strategy.take_profit
                        sl_obj = self.strategy.stop_loss
                        ts_obj = self.strategy.trailing_stop
                        bar = df.iloc[i] if (tp_obj or sl_obj or ts_obj) else None
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
                    exec_price = fill * (1 - self.slippage)
                    size = (
                        self.strategy.position_sizer.compute_size(
                            sig,
                            cash,
                            exec_price,
                            volatility=atr[i] if atr is not None else None,
                        )
                        * lev
                        / cs
                        if has_sizer
                        else round(cash * lev / (exec_price * cs), _SIZE_PRECISION)
                    )
                    if size > 0:
                        comm = size * exec_price * cs * self.commission
                        cash += size * exec_price * cs - comm
                        pos = -size
                        entry_price = exec_price
                        entry_date = dates[i]
                        entry_bar = i
                        entry_comm = comm
                        financing_accrued = 0.0
                        used_margin = size * exec_price * cs / lev
                        direction = -1
                        tp_obj = self.strategy.take_profit
                        sl_obj = self.strategy.stop_loss
                        ts_obj = self.strategy.trailing_stop
                        bar = df.iloc[i] if (tp_obj or sl_obj or ts_obj) else None
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

            equity_arr[i] = cash + pos * price * cs

        equity_curve = pd.Series(equity_arr, index=dates, name="equity")
        trade_log = pd.DataFrame(trades)
        return equity_curve, trade_log
