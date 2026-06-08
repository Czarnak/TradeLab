from __future__ import annotations

import pandas as pd
import pytest

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.risk_management import (
    FixedSL,
    FixedTP,
    FixedTS,
    MovingAverageSL,
    MovingAverageTS,
    ParabolicSARSL,
    ParabolicSARTS,
    SignalStrengthSL,
    SignalStrengthTP,
    SignalStrengthTS,
)
from trade_lab.strategies.base import BaseStrategy


class _SignalStrategy(BaseStrategy):
    def __init__(self, signals: list[float], **kwargs):
        super().__init__(**kwargs)
        self._signals = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal_strength"] = pd.Series(self._signals, index=out.index)
        return out


def _bar(**overrides) -> pd.Series:
    data = {
        "Open": 100.0,
        "High": 103.0,
        "Low": 99.0,
        "Close": 102.0,
        "Volume": 1_000.0,
        "signal_strength": 0.75,
        "ema__ema_20": 101.5,
        "sar": 98.5,
    }
    data.update(overrides)
    return pd.Series(data)


def _ohlcv_with_signals() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 101.0],
            "High": [101.0, 106.0, 104.0, 103.0],
            "Low": [99.0, 100.0, 95.0, 98.0],
            "Close": [100.0, 101.0, 100.0, 99.0],
            "Volume": [1_000, 1_100, 1_200, 1_300],
        },
        index=idx,
    )


def test_take_profit_and_stop_loss_variants_compute_absolute_levels():
    bar = _bar()

    assert FixedTP(5.0).compute(100.0, 1, 0.3, bar) == pytest.approx(105.0)
    assert FixedTP(5.0).compute(100.0, -1, 0.3, bar) == pytest.approx(95.0)
    assert SignalStrengthTP(8.0).compute(100.0, 1, -0.5, bar) == pytest.approx(104.0)

    assert FixedSL(5.0).compute(100.0, 1, 0.3, bar) == pytest.approx(95.0)
    assert FixedSL(5.0).compute(100.0, -1, 0.3, bar) == pytest.approx(105.0)
    assert SignalStrengthSL(8.0).compute(100.0, 1, -0.5, bar) == pytest.approx(96.0)


def test_moving_average_and_parabolic_sar_stop_loss_handle_columns_and_fallbacks():
    bar = _bar()

    assert MovingAverageSL("ema__ema_20").compute(100.0, 1, 0.5, bar) == pytest.approx(
        101.5
    )
    with pytest.raises(KeyError, match="ema__ema_50"):
        MovingAverageSL("ema__ema_50").compute(100.0, 1, 0.5, bar)

    assert ParabolicSARSL().compute(100.0, 1, 0.5, bar) == pytest.approx(98.5)

    no_sar_bar = _bar(sar=pd.NA, High=104.0, Low=99.0)
    assert ParabolicSARSL(af_start=0.02).compute(
        100.0, 1, 0.5, no_sar_bar
    ) == pytest.approx(99.1)


def test_trailing_stop_variants_apply_step_guard_consistently():
    bar = _bar(Close=110.0, ema__ema_20=109.0)

    fixed = FixedTS(base_points=5.0, step_points=2.0)
    assert fixed.compute_initial(100.0, 1, 0.5, bar) == pytest.approx(95.0)
    assert fixed.update(100.0, 1, 0.5, bar) == pytest.approx(105.0)
    assert fixed.update(104.0, 1, 0.5, bar) == pytest.approx(104.0)

    strength = SignalStrengthTS(base_points=10.0, step_points=2.0)
    assert strength.compute_initial(100.0, 1, -0.5, bar) == pytest.approx(95.0)
    assert strength.update(100.0, 1, -0.5, bar) == pytest.approx(105.0)

    ma = MovingAverageTS(column="ema__ema_20", step_points=2.0)
    assert ma.compute_initial(100.0, 1, 0.5, bar) == pytest.approx(109.0)
    assert ma.update(107.0, 1, 0.5, bar) == pytest.approx(109.0)
    assert ma.update(108.5, 1, 0.5, bar) == pytest.approx(108.5)


def test_parabolic_sar_trailing_stop_is_stateful_and_respects_step_guard():
    ts = ParabolicSARTS(af_start=0.02, af_step=0.02, af_max=0.2, step_points=0.1)

    initial = ts.compute_initial(100.0, 1, 0.5, _bar(High=103.0, Low=99.0))
    assert initial == pytest.approx(99.08)

    updated = ts.update(initial, 1, 0.5, _bar(High=105.0, Low=100.0))
    assert updated == pytest.approx(99.3168)

    guarded_ts = ParabolicSARTS(
        af_start=0.02, af_step=0.02, af_max=0.2, step_points=0.5
    )
    guarded_initial = guarded_ts.compute_initial(
        100.0, 1, 0.5, _bar(High=103.0, Low=99.0)
    )
    guarded = guarded_ts.update(guarded_initial, 1, 0.5, _bar(High=105.0, Low=100.0))
    assert guarded == pytest.approx(guarded_initial)


def test_backtest_engine_exits_on_take_profit_before_signal_exit():
    df = _ohlcv_with_signals()
    strategy = _SignalStrategy(
        [0.8, 0.0, 0.0, 0.0],
        entry_threshold=0.5,
        exit_threshold=0.1,
    )
    strategy.take_profit = FixedTP(5.0)

    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
    ).run_on(df)

    assert len(result.trade_log) == 1
    trade = result.trade_log.iloc[0]
    assert trade["exit_reason"] == "tp"
    assert trade["exit_price"] == pytest.approx(105.0)
    assert trade["exit_date"] == df.index[1]


def test_backtest_engine_marks_trailing_stop_as_more_protective_exit():
    df = _ohlcv_with_signals()
    strategy = _SignalStrategy(
        [0.8, 0.9, 0.0, 0.0],
        entry_threshold=0.5,
        exit_threshold=0.1,
    )
    strategy.stop_loss = FixedSL(6.0)
    strategy.trailing_stop = FixedTS(base_points=1.0, step_points=0.0)

    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
    ).run_on(df)

    # Correct mark-to-market equity no longer suppresses same-bar re-entry:
    # after the ts exit on bar 1 the still-strong signal (0.9) reopens a long
    # that is also stopped out (documented same-bar re-entry behaviour).
    assert len(result.trade_log) == 2
    trade = result.trade_log.iloc[0]
    assert trade["exit_reason"] == "ts"
    assert trade["exit_price"] == pytest.approx(100.0)
    assert trade["exit_date"] == df.index[1]


def test_backtest_engine_marks_fixed_stop_loss_exit_for_short_trades():
    idx = pd.date_range("2026-02-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 99.0, 101.0],
            "High": [100.0, 105.0, 102.0],
            "Low": [99.0, 98.0, 99.0],
            "Close": [100.0, 100.0, 101.0],
            "Volume": [1_000, 1_100, 1_200],
        },
        index=idx,
    )
    strategy = _SignalStrategy(
        [-0.8, 0.0, 0.0],
        allow_long=True,
        allow_short=True,
        entry_threshold=0.5,
        exit_threshold=0.1,
    )
    strategy.stop_loss = FixedSL(4.0)

    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
    ).run_on(df)

    assert len(result.trade_log) == 1
    trade = result.trade_log.iloc[0]
    assert trade["direction"] == "short"
    assert trade["exit_reason"] == "sl"
    assert trade["exit_price"] == pytest.approx(104.0)
