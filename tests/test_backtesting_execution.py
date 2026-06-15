"""Execution-timing of the backtesting engine.

The default ``execute_on="close"`` fills a signal at the same bar's close — the
signal both reads and trades on ``Close[t]``, a same-bar lookahead. ``"next_open"``
makes the decision on the prior bar's signal and fills at the current bar's open,
i.e. decide at ``Close[t-1]`` and execute at ``Open[t]``: the first tradable price
after the decision, with no lookahead.

Open and Close are deliberately distinct so the fill price reveals which bar (and
which price) execution used. Commission/slippage are zeroed for exact values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.strategies.base import BaseStrategy


class _SignalStrategy(BaseStrategy):
    def __init__(self, signals: list[float], **kwargs):
        super().__init__(**kwargs)
        self._signals = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal_strength"] = pd.Series(self._signals, index=out.index)
        return out


def _ohlc(opens: list[float], closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-04-01", periods=len(opens), freq="D")
    highs = [max(o, c) for o, c in zip(opens, closes)]
    lows = [min(o, c) for o, c in zip(opens, closes)]
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1] * len(opens),
        },
        index=idx,
    )


def _long_round_trip() -> tuple[pd.DataFrame, _SignalStrategy]:
    # Signal at bar 0 opens a long; signal flips at bar 2 to close it.
    df = _ohlc(opens=[10.0, 20.0, 30.0, 40.0], closes=[11.0, 21.0, 31.0, 41.0])
    strategy = _SignalStrategy(
        [1.0, 1.0, -1.0, -1.0],
        allow_long=True,
        allow_short=False,
        entry_threshold=0.5,
        exit_threshold=0.5,
    )
    return df, strategy


def test_next_open_fills_entries_and_exits_at_next_bar_open():
    df, strategy = _long_round_trip()

    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        execute_on="next_open",
    ).run_on(df)

    trade = result.trade_log.iloc[0]
    # bar-0 signal -> entry at bar-1 OPEN (20), not bar-0 close (11).
    assert trade["entry_price"] == pytest.approx(20.0)
    assert trade["entry_date"] == df.index[1]
    # bar-2 flip -> exit at bar-3 OPEN (40), not bar-2 close (31).
    assert trade["exit_price"] == pytest.approx(40.0)
    assert trade["exit_date"] == df.index[3]


def test_default_close_mode_fills_at_signal_bar_close():
    df, strategy = _long_round_trip()

    default = BacktestEngine(
        strategy=strategy, initial_capital=10_000.0, commission=0.0, slippage=0.0
    ).run_on(df)
    explicit = BacktestEngine(
        strategy=_long_round_trip()[1],
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        execute_on="close",
    ).run_on(df)

    trade = default.trade_log.iloc[0]
    # Same-bar close fills: entry on bar-0 close (11), exit on bar-2 close (31).
    assert trade["entry_price"] == pytest.approx(11.0)
    assert trade["entry_date"] == df.index[0]
    assert trade["exit_price"] == pytest.approx(31.0)
    assert trade["exit_date"] == df.index[2]
    # Default must equal explicit "close".
    assert explicit.trade_log.iloc[0]["entry_price"] == pytest.approx(11.0)


def test_invalid_execute_on_raises():
    with pytest.raises(ValueError, match="execute_on"):
        BacktestEngine(execute_on="midnight")
