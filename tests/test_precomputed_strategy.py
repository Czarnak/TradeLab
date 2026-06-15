"""PrecomputedSignalStrategy — inject a fixed prediction series as the signal.

Used to score one prediction path through the real ``BacktestEngine`` across a
threshold grid (sizing, financing, leverage, execution timing all live) without
re-running the model. Predictions are aligned to the bar index by date, not by
position, so a misordered or partial series cannot silently mis-map.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.strategies import PrecomputedSignalStrategy


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


def test_injects_signal_series_aligned_by_index_not_position():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    ordered = [0.0, 0.9, 0.0, -0.9]
    # Pass the series out of order; alignment must restore it by date.
    shuffled = pd.Series(ordered, index=idx).iloc[[2, 0, 3, 1]]

    out = PrecomputedSignalStrategy(shuffled).generate_signals(df)

    assert out["signal_strength"].tolist() == ordered


def test_dates_absent_from_the_series_become_nan():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    partial = pd.Series([0.9, -0.9], index=[idx[1], idx[3]])

    out = PrecomputedSignalStrategy(partial).generate_signals(df)
    vals = out["signal_strength"]

    assert math.isnan(vals.iloc[0])
    assert vals.iloc[1] == pytest.approx(0.9)
    assert math.isnan(vals.iloc[2])
    assert vals.iloc[3] == pytest.approx(-0.9)


def test_injected_signals_drive_engine_trades_through_gating_params():
    df = _ohlc(opens=[10.0, 10.0, 11.0, 11.0], closes=[10.0, 10.0, 11.0, 11.0])
    signals = pd.Series([1.0, 0.0, 0.0, -1.0], index=df.index)
    strategy = PrecomputedSignalStrategy(
        signals,
        allow_long=True,
        allow_short=False,
        entry_threshold=0.5,
        exit_threshold=0.0,
    )

    result = BacktestEngine(
        strategy=strategy, initial_capital=10_000.0, commission=0.0, slippage=0.0
    ).run_on(df)

    assert len(result.trade_log) == 1
    assert result.trade_log.iloc[0]["direction"] == "long"
