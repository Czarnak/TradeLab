"""Leverage / CFD behaviour of the backtesting engine.

Covers the four CFD parameters added to ``BacktestEngine``: ``leverage``
(position-size multiplier), ``contract_size`` (point value), ``financing_rate``
(overnight swap) and ``maintenance_margin`` (margin-call liquidation, gated to
``leverage > 1``). Commission and slippage are zeroed so equity values are exact.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.strategies.base import BaseStrategy


class _SignalStrategy(BaseStrategy):
    def __init__(self, signals: list[float], **kwargs):
        super().__init__(**kwargs)
        self._signals = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal_strength"] = pd.Series(self._signals, index=out.index)
        return out


def _flat_ohlcv(prices: list[float]) -> pd.DataFrame:
    """OHLCV where Open=High=Low=Close=price (no intrabar range)."""
    idx = pd.date_range("2026-04-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [1] * len(prices),
        },
        index=idx,
    )


def _bars(
    closes: list[float],
    highs: list[float],
    lows: list[float],
) -> pd.DataFrame:
    """OHLCV with explicit intrabar High/Low (Open assumed = Close)."""
    idx = pd.date_range("2026-04-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1] * len(closes),
        },
        index=idx,
    )


def _long_then_flat(signals: list[float], **strategy_kwargs) -> _SignalStrategy:
    return _SignalStrategy(
        signals,
        allow_long=True,
        allow_short=False,
        entry_threshold=0.5,
        exit_threshold=0.0,
        **strategy_kwargs,
    )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_leverage_one_reproduces_plain_equity_behaviour():
    # leverage=1.0 / contract_size=1.0 / financing=0 must match the pre-CFD math.
    df = _flat_ohlcv([10.0, 10.0, 11.0, 11.0])
    strategy = _long_then_flat([1.0, 0.0, 0.0, -1.0])
    result = BacktestEngine(
        strategy=strategy, initial_capital=10_000.0, commission=0.0, slippage=0.0
    ).run_on(df)

    assert result.trade_log.iloc[0]["size"] == pytest.approx(1_000.0)
    assert result.metrics["total_return"] == pytest.approx(0.10, abs=1e-9)
    assert result.equity_curve.iloc[-1] == pytest.approx(11_000.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Leverage as a position-size multiplier
# ---------------------------------------------------------------------------


def test_leverage_multiplies_default_sizing_and_amplifies_return():
    # 2x leverage doubles the position; a +10% price move yields +20% equity.
    df = _flat_ohlcv([10.0, 10.0, 11.0, 11.0])
    strategy = _long_then_flat([1.0, 0.0, 0.0, -1.0])
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        leverage=2.0,
    ).run_on(df)

    assert result.trade_log.iloc[0]["size"] == pytest.approx(2_000.0)
    assert result.metrics["total_return"] == pytest.approx(0.20, abs=1e-9)
    assert result.equity_curve.iloc[-1] == pytest.approx(12_000.0, rel=1e-9)


def test_leverage_flat_round_trip_returns_to_initial_capital():
    df = _flat_ohlcv([10.0, 10.0, 10.0, 10.0])
    strategy = _long_then_flat([1.0, 0.0, 0.0, -1.0])
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        leverage=5.0,
    ).run_on(df)

    assert result.equity_curve.iloc[0] == pytest.approx(10_000.0, rel=1e-9)
    assert result.equity_curve.iloc[-1] == pytest.approx(10_000.0, rel=1e-9)


def test_leverage_multiplies_position_sizer_output():
    # FixedPositionSizer(0.5) → 500 units at 1x; leverage=2 lifts it to 1000.
    df = _flat_ohlcv([10.0, 10.0, 11.0, 11.0])
    strategy = _long_then_flat(
        [1.0, 0.0, 0.0, -1.0], position_sizer=FixedPositionSizer(0.5)
    )
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        leverage=2.0,
    ).run_on(df)

    assert result.trade_log.iloc[0]["size"] == pytest.approx(1_000.0)
    assert result.metrics["total_return"] == pytest.approx(0.10, abs=1e-9)


# ---------------------------------------------------------------------------
# Contract / point value
# ---------------------------------------------------------------------------


def test_contract_size_scales_contracts_but_preserves_notional_return():
    # contract_size=10 → each contract is worth 10x; 100 contracts not 1000 units,
    # same $10k notional and same +10% return.
    df = _flat_ohlcv([10.0, 10.0, 11.0, 11.0])
    strategy = _long_then_flat([1.0, 0.0, 0.0, -1.0])
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        contract_size=10.0,
    ).run_on(df)

    trade = result.trade_log.iloc[0]
    assert trade["size"] == pytest.approx(100.0)
    # Entry notional = contracts * price * contract_size = 10_000.
    assert trade["size"] * trade["entry_price"] * 10.0 == pytest.approx(10_000.0)
    assert result.metrics["total_return"] == pytest.approx(0.10, abs=1e-9)


# ---------------------------------------------------------------------------
# Margin-call liquidation (stop-out)
# ---------------------------------------------------------------------------


def test_long_position_is_liquidated_at_maintenance_margin():
    # leverage=10, entry 100 → liq price 95 (maintenance 0.5). A bar with Low=94
    # forces a stop-out filled at 95, flooring equity at 0.5 * initial margin.
    df = _bars(
        closes=[100.0, 96.0, 100.0, 100.0],
        highs=[100.0, 100.0, 100.0, 100.0],
        lows=[100.0, 94.0, 100.0, 100.0],
    )
    strategy = _long_then_flat([1.0, 0.0, 0.0, 0.0])
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        leverage=10.0,
        maintenance_margin=0.5,
    ).run_on(df)

    assert len(result.trade_log) == 1
    trade = result.trade_log.iloc[0]
    assert trade["exit_reason"] == "liquidation"
    assert trade["exit_price"] == pytest.approx(95.0)
    assert trade["pnl"] == pytest.approx(-5_000.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(5_000.0, rel=1e-9)


def test_short_position_is_liquidated_at_maintenance_margin():
    # leverage=10 short entry 100 → liq price 105; a bar with High=106 stops out.
    df = _bars(
        closes=[100.0, 104.0, 100.0, 100.0],
        highs=[100.0, 106.0, 100.0, 100.0],
        lows=[100.0, 100.0, 100.0, 100.0],
    )
    strategy = _SignalStrategy(
        [-1.0, 0.0, 0.0, 0.0],
        allow_long=False,
        allow_short=True,
        entry_threshold=0.5,
        exit_threshold=0.0,
    )
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        leverage=10.0,
        maintenance_margin=0.5,
    ).run_on(df)

    assert len(result.trade_log) == 1
    trade = result.trade_log.iloc[0]
    assert trade["exit_reason"] == "liquidation"
    assert trade["exit_price"] == pytest.approx(105.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(5_000.0, rel=1e-9)


def test_liquidation_is_gated_to_leveraged_accounts():
    # Same deep drawdown: at 1x no liquidation (position simply held, marked down),
    # at 2x the margin call fires.
    df = _flat_ohlcv([100.0, 70.0, 50.0, 40.0])

    unleveraged = BacktestEngine(
        strategy=_long_then_flat([1.0, 0.0, 0.0, 0.0]),
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        leverage=1.0,
    ).run_on(df)
    assert unleveraged.trade_log.empty  # never force-closed
    assert unleveraged.equity_curve.iloc[-1] == pytest.approx(4_000.0, rel=1e-9)

    leveraged = BacktestEngine(
        strategy=_long_then_flat([1.0, 0.0, 0.0, 0.0]),
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        leverage=2.0,
        maintenance_margin=0.5,
    ).run_on(df)
    assert len(leveraged.trade_log) == 1
    assert leveraged.trade_log.iloc[0]["exit_reason"] == "liquidation"


# ---------------------------------------------------------------------------
# Overnight financing (swap)
# ---------------------------------------------------------------------------


def test_financing_is_charged_per_held_bar_and_reduces_pnl():
    # 1000 units * $10 notional * 1% * 3 nights held = $300 financing.
    df = _flat_ohlcv([10.0, 10.0, 10.0, 10.0])
    strategy = _long_then_flat([1.0, 0.0, 0.0, -1.0])
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        financing_rate=0.01,
    ).run_on(df)

    trade = result.trade_log.iloc[0]
    assert trade["financing"] == pytest.approx(300.0)
    assert trade["pnl"] == pytest.approx(-300.0)
    assert result.metrics["total_financing"] == pytest.approx(300.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(9_700.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"leverage": 0.5}, "leverage"),
        ({"contract_size": 0.0}, "contract_size"),
        ({"contract_size": -1.0}, "contract_size"),
        ({"maintenance_margin": 1.0}, "maintenance_margin"),
        ({"maintenance_margin": -0.1}, "maintenance_margin"),
        ({"financing_rate": float("inf")}, "financing_rate"),
        ({"financing_rate": float("nan")}, "financing_rate"),
    ],
)
def test_invalid_cfd_parameters_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BacktestEngine(**kwargs)
