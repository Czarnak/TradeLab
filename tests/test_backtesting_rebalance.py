"""Opt-in position rebalancing (``BacktestEngine(rebalance_band=...)``).

While a position is open and the current bar's signal does not trigger a
close (same direction, no TP/SL/liquidation/exit-signal event), the engine
recomputes the sizer's target size for the current signal and trades the
delta whenever ``abs(target - current) / abs(current)`` exceeds
``rebalance_band``. ``rebalance_band=None`` (the default) reproduces the
legacy fixed-size-until-exit behaviour exactly.

All numeric fixtures use a *flat* OHLC bar (``Open == Close``, constant
``High``/``Low`` range) repeated for every bar, which keeps the ATR-based
volatility input to ``RiskBasedPositionSizer`` exactly constant across the
whole run once the 14-bar rolling window has warmed up (verified directly
against ``pandas.Series.rolling`` — the window needs 14 *consecutive*
non-NaN true-range values, so entries land on bar index 14, the 15th bar).
Keeping price flat isolates rebalancing to being purely signal/equity
driven rather than entangled with mark-to-market price moves, which keeps
every assertion an exact, hand-derivable number.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.sizing.risk_based import RiskBasedPositionSizer
from trade_lab.strategies.base import BaseStrategy

# ATR settles to exactly this value once the 14-bar TR window is filled with
# the constant true range produced by _atr_bars's default High/Low spread.
_ATR = 2.0
_WARMUP_BARS = 15  # index 0..14; bar 14 is the first bar with valid ATR.


class _SignalStrategy(BaseStrategy):
    def __init__(self, signals: list[float], **kwargs):
        super().__init__(**kwargs)
        self._signals = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal_strength"] = pd.Series(self._signals, index=out.index)
        return out


def _atr_bars(
    n: int,
    opens: list[float] | None = None,
    closes: list[float] | None = None,
) -> pd.DataFrame:
    """``n`` bars with a constant High/Low range around a (default flat) price.

    ``High - Low == 2.0`` and ``Close`` constant at ``100.0`` by default,
    which drives ATR(14) to exactly ``2.0`` from bar index 14 onward
    (see module docstring). ``opens``/``closes`` may override the flat
    ``100.0`` default per-bar (e.g. to test ``execute_on="next_open"``).
    High/Low are derived purely from ``Close`` (``Close +/- 1.0``),
    independent of ``Open``, so ATR stays exactly ``2.0`` even on bars
    where ``Open`` is deliberately set far from ``Close`` to pin down
    next-bar-open execution timing.
    """
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    closes = closes if closes is not None else [100.0] * n
    opens = opens if opens is not None else list(closes)
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1] * n,
        },
        index=idx,
    )


def test_atr_is_constant_2_0_from_bar_14_onward():
    # Sanity-check the fixture design itself: ATR(14) on a High-Low=2.0,
    # constant-Close bar series settles to exactly 2.0 forever from index 14.
    df = _atr_bars(20)
    atr = BacktestEngine._compute_atr(df)
    assert all(v == pytest.approx(_ATR) for v in atr[14:])


# ---------------------------------------------------------------------------
# 1. Regression: rebalance_band=None is byte-for-byte the legacy behaviour.
# ---------------------------------------------------------------------------


def test_rebalance_band_none_matches_legacy_full_equity_and_sizer_behaviour():
    from trade_lab.sizing.fixed import FixedPositionSizer

    df = _atr_bars(4, closes=[10.0, 10.0, 11.0, 11.0])
    strategy = _SignalStrategy(
        [1.0, 0.0, 0.0, -1.0],
        allow_long=True,
        allow_short=False,
        entry_threshold=0.5,
        exit_threshold=0.0,
        position_sizer=FixedPositionSizer(0.5),
    )
    default_engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        leverage=2.0,
    )
    explicit_none_engine = BacktestEngine(
        strategy=_SignalStrategy(
            [1.0, 0.0, 0.0, -1.0],
            allow_long=True,
            allow_short=False,
            entry_threshold=0.5,
            exit_threshold=0.0,
            position_sizer=FixedPositionSizer(0.5),
        ),
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        leverage=2.0,
        rebalance_band=None,
    )

    default_result = default_engine.run_on(df)
    explicit_result = explicit_none_engine.run_on(df)

    # Matches the known golden value from test_backtesting_leverage.py's
    # equivalent scenario (FixedPositionSizer(0.5) x leverage=2.0 -> 1000).
    assert default_result.trade_log.iloc[0]["size"] == pytest.approx(1_000.0)
    assert default_result.metrics["total_return"] == pytest.approx(0.10, abs=1e-9)
    assert default_result.metrics["rebalance_count"] == 0

    pd.testing.assert_series_equal(
        default_result.equity_curve, explicit_result.equity_curve
    )
    pd.testing.assert_frame_equal(
        default_result.trade_log.drop(columns=["rebalance_count"]),
        explicit_result.trade_log.drop(columns=["rebalance_count"]),
    )


# ---------------------------------------------------------------------------
# 2. Band respected: signal grows but within band -> no fill.
# ---------------------------------------------------------------------------


def test_signal_change_within_band_does_not_rebalance():
    # sig 0.6 -> entry size 150.0; sig 0.65 at bar 15 -> target 162.5, a
    # ratio of 12.5/150 = 0.0833, below the 0.25 band -> no fill.
    signals = [0.0] * 14 + [0.6, 0.65, 0.0]
    df = _atr_bars(17)
    strategy = _SignalStrategy(
        signals,
        allow_long=True,
        allow_short=False,
        entry_threshold=0.5,
        exit_threshold=0.1,
        position_sizer=RiskBasedPositionSizer(max_fraction=0.1, risk_multiplier=2.0),
    )
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        rebalance_band=0.25,
    ).run_on(df)

    assert result.metrics["rebalance_count"] == 0
    trade = result.trade_log.iloc[0]
    assert trade["size"] == pytest.approx(150.0)
    assert trade["pnl"] == pytest.approx(0.0)
    assert trade["rebalance_count"] == 0


# ---------------------------------------------------------------------------
# 3. Scale up: signal doubles beyond band -> position increases to target.
# ---------------------------------------------------------------------------


def test_signal_doubling_beyond_band_scales_position_up():
    # entry sig=0.4 -> size 100.0 @ entry_price 101.0 (1% buy slippage).
    # bar 15 sig=0.8 (double) -> target 195.98, ratio 0.9598 > 0.25 band.
    # Increase-leg trade_price == entry-style price == 101.0 exactly (flat
    # price + identical slippage direction), so the weighted-average entry
    # price is unchanged at 101.0 despite the size growing.
    signals = [0.0] * 14 + [0.4, 0.8, 0.0]
    df = _atr_bars(17)
    strategy = _SignalStrategy(
        signals,
        allow_long=True,
        allow_short=False,
        entry_threshold=0.3,
        exit_threshold=0.1,
        position_sizer=RiskBasedPositionSizer(max_fraction=0.1, risk_multiplier=2.0),
    )
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.01,
        slippage=0.01,
        rebalance_band=0.25,
    ).run_on(df)

    assert result.metrics["rebalance_count"] == 1
    # Equity right after the rebalance fill (bar index 15).
    assert result.equity_curve.iloc[15] == pytest.approx(9606.0802)

    trade = result.trade_log.iloc[0]
    assert trade["rebalance_count"] == 1
    assert trade["size"] == pytest.approx(195.98)
    assert trade["entry_price"] == pytest.approx(101.0)
    assert trade["pnl"] == pytest.approx(-783.92)
    assert trade["commission"] == pytest.approx(391.96)


# ---------------------------------------------------------------------------
# 4. Scale down: partial close realizes PnL correctly.
# ---------------------------------------------------------------------------


def test_signal_drop_beyond_band_scales_position_down_and_realizes_pnl():
    # entry sig=0.8 -> size 200.0 @ entry_price 101.0.
    # bar 15 sig=0.4 -> target 95.98, ratio 0.5201 > 0.25 band -> reduce.
    # The reduce leg trades at the sell-side price (99.0, opposite the
    # entry's buy-side 101.0) even though the market price never moved,
    # realizing a PnL purely from that bid/ask-style slippage spread.
    signals = [0.0] * 14 + [0.8, 0.4, 0.0]
    df = _atr_bars(17)
    strategy = _SignalStrategy(
        signals,
        allow_long=True,
        allow_short=False,
        entry_threshold=0.3,
        exit_threshold=0.1,
        position_sizer=RiskBasedPositionSizer(max_fraction=0.1, risk_multiplier=2.0),
    )
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.01,
        slippage=0.01,
        rebalance_band=0.25,
    ).run_on(df)

    assert result.metrics["rebalance_count"] == 1
    assert result.equity_curve.iloc[15] == pytest.approx(9391.0002)

    trade = result.trade_log.iloc[0]
    assert trade["rebalance_count"] == 1
    assert trade["size"] == pytest.approx(95.98)
    # Entry price of the remaining units is unchanged by the reduce.
    assert trade["entry_price"] == pytest.approx(101.0)
    assert trade["pnl"] == pytest.approx(-800.0)
    assert trade["commission"] == pytest.approx(400.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(9200.0)


# ---------------------------------------------------------------------------
# 5. total_trades unchanged by rebalances; rebalance_count correct.
# ---------------------------------------------------------------------------


def test_multiple_rebalances_do_not_inflate_total_trades():
    # entry sig=0.4, bar15 sig=0.8 (rebalance #1), bar16 sig=1.0
    # (rebalance #2, ratio ~0.2254 > the 0.1 band used here), bar17 closes.
    signals = [0.0] * 14 + [0.4, 0.8, 1.0, 0.0]
    df = _atr_bars(18)
    strategy = _SignalStrategy(
        signals,
        allow_long=True,
        allow_short=False,
        entry_threshold=0.3,
        exit_threshold=0.1,
        position_sizer=RiskBasedPositionSizer(max_fraction=0.1, risk_multiplier=2.0),
    )
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.01,
        slippage=0.01,
        rebalance_band=0.1,
    ).run_on(df)

    assert result.metrics["total_trades"] == 1
    assert len(result.trade_log) == 1
    assert result.metrics["rebalance_count"] == 2
    assert result.trade_log.iloc[0]["rebalance_count"] == 2


# ---------------------------------------------------------------------------
# 6. Sign flip still closes (and reverses); rebalance does not swallow it.
# ---------------------------------------------------------------------------


def test_sign_flip_closes_and_reverses_instead_of_rebalancing():
    # Long entry (sig=0.6) then a strong opposite signal (sig=-0.8) at bar
    # 15: this must close the long via the plain exit-signal branch (it
    # never reaches the rebalance elif) and reopen as a fresh short — not
    # a partial/rebalanced adjustment of the old long.
    signals = [0.0] * 14 + [0.6, -0.8, 0.0]
    df = _atr_bars(17)
    strategy = _SignalStrategy(
        signals,
        allow_long=True,
        allow_short=True,
        entry_threshold=0.5,
        exit_threshold=0.1,
        position_sizer=RiskBasedPositionSizer(max_fraction=0.1, risk_multiplier=2.0),
    )
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        rebalance_band=0.25,
    ).run_on(df)

    assert result.metrics["rebalance_count"] == 0
    assert len(result.trade_log) == 2

    long_trade = result.trade_log.iloc[0]
    assert long_trade["direction"] == "long"
    assert long_trade["exit_reason"] == "signal"
    assert long_trade["size"] == pytest.approx(150.0)
    assert long_trade["pnl"] == pytest.approx(0.0)

    short_trade = result.trade_log.iloc[1]
    assert short_trade["direction"] == "short"
    assert short_trade["exit_reason"] == "signal"
    assert short_trade["size"] == pytest.approx(200.0)
    assert short_trade["pnl"] == pytest.approx(0.0)

    assert result.equity_curve.iloc[-1] == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# 7. execute_on="next_open": rebalance fills at next bar's open.
# ---------------------------------------------------------------------------


def test_next_open_mode_fills_rebalance_at_next_bar_open_not_current_close():
    # Entry decision at bar 14 (sig=0.6) fires at bar 15's Open (100.0).
    # Rebalance decision at bar 15 (sig=1.0) fires at bar 16's Open (130.0)
    # -- deliberately far from every Close (100.0 throughout) so a wrong
    # (same-bar-close) fill price would be caught immediately.
    signals = [0.0] * 14 + [0.6, 1.0, 0.0, 0.0]
    opens = [100.0] * 16 + [130.0, 100.0]
    closes = [100.0] * 18
    df = _atr_bars(18, opens=opens, closes=closes)
    strategy = _SignalStrategy(
        signals,
        allow_long=True,
        allow_short=False,
        entry_threshold=0.5,
        exit_threshold=0.1,
        position_sizer=RiskBasedPositionSizer(max_fraction=0.1, risk_multiplier=2.0),
    )
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        rebalance_band=0.25,
        execute_on="next_open",
    ).run_on(df)

    assert result.metrics["rebalance_count"] == 1
    # Bought 100 units at Open[16]=130 while marked at Close[16]=100:
    # a $30/unit markdown on 100 units drags equity from 10_000 to 7_000.
    assert result.equity_curve.iloc[16] == pytest.approx(7_000.0)

    trade = result.trade_log.iloc[0]
    assert trade["entry_date"] == df.index[15]
    assert trade["size"] == pytest.approx(250.0)
    # Weighted average: (150 @ 100 + 100 @ 130) / 250 = 112.0.
    assert trade["entry_price"] == pytest.approx(112.0)
    assert trade["exit_date"] == df.index[17]
    assert trade["pnl"] == pytest.approx(-3_000.0)


# ---------------------------------------------------------------------------
# 8. ValueError cases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band", [0.0, -0.1])
def test_rebalance_band_must_be_positive(band):
    with pytest.raises(ValueError, match="rebalance_band"):
        BacktestEngine(rebalance_band=band)


def test_rebalance_band_requires_position_sizer():
    df = _atr_bars(17, closes=[100.0] * 17)
    strategy = _SignalStrategy(
        [0.0] * 14 + [0.6, 0.6, 0.0],
        allow_long=True,
        allow_short=False,
        entry_threshold=0.5,
        exit_threshold=0.1,
        # No position_sizer -> default full-equity sizing, incompatible
        # with rebalancing (there is no target to track).
    )
    engine = BacktestEngine(strategy=strategy, rebalance_band=0.25)

    with pytest.raises(ValueError, match="position_sizer"):
        engine.run_on(df)


# ---------------------------------------------------------------------------
# 9. Financing accrues on post-rebalance notional.
# ---------------------------------------------------------------------------


def test_financing_accrues_on_post_rebalance_size():
    # Entry sig=0.4 -> pos=100 @ 100.0, cash=0.0.
    # Bar 15: financing on pos=100 (pre-rebalance) = 100*100*0.01 = 100.0,
    # then sig=0.8 rebalances pos up to 198.0.
    # Bar 16: financing must use the NEW pos=198 -> 198*100*0.01 = 198.0
    # (a stale pos=100 would instead charge only 100.0 again); sig stays at
    # 0.8 so the ratio vs. the now-larger pos (0.02) stays inside the 0.25
    # band and no further rebalance fires.
    # Bar 17 (sig=0.0, closes the trade): financing is charged for every
    # bar the position is still open *including* the closing bar itself
    # (existing, pre-rebalance engine behaviour) -> one more 198.0 charge
    # before the exit fires. Total financing = 100 + 198 + 198 = 496.0.
    signals = [0.0] * 14 + [0.4, 0.8, 0.8, 0.0]
    df = _atr_bars(18)
    strategy = _SignalStrategy(
        signals,
        allow_long=True,
        allow_short=False,
        entry_threshold=0.3,
        exit_threshold=0.1,
        position_sizer=RiskBasedPositionSizer(max_fraction=0.1, risk_multiplier=2.0),
    )
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        rebalance_band=0.25,
        financing_rate=0.01,
    ).run_on(df)

    assert result.metrics["rebalance_count"] == 1
    # After bar-15 financing (100.0) and rebalance (zero-cost @ flat price),
    # then bar-16 financing on the new pos=198 (198.0): 9900 - 198 = 9702.
    # This alone proves financing used the post-rebalance size: a stale
    # pos=100 would instead have left equity at 9900 - 100 = 9800.
    assert result.equity_curve.iloc[16] == pytest.approx(9702.0)

    trade = result.trade_log.iloc[0]
    assert trade["financing"] == pytest.approx(496.0)
    assert trade["pnl"] == pytest.approx(-496.0)


# ---------------------------------------------------------------------------
# 10. Bonus: margin-call liquidation level is re-derived after a rebalance.
# ---------------------------------------------------------------------------


def test_liquidation_level_reflects_post_rebalance_margin():
    # leverage=4, entry sig=0.9 -> raw sizer units 45.0 -> pos=180 @ 100.0,
    # used_margin=4_500 (raw*price, independent of leverage per the
    # existing engine formula: size*price/lev == raw*price).
    # liq_level there would be (0.5*4_500 + 8_000)/180 = 56.944.
    # Bar 15 sig=1.0 -> target raw=50.0 -> magnitude=200.0, ratio 0.111 >
    # the 0.05 band -> increases pos to 200, used_margin to 5_000.
    # New liq_level = (0.5*5_000 + 10_000)/200 = 62.5 -- strictly between
    # 56.944 and 62.5 is Low=60.0 on bar 16: this breaches the *updated*
    # level but would NOT have breached the stale pre-rebalance level.
    signals = [0.0] * 14 + [0.9, 1.0, 1.0]
    closes = [100.0] * 16 + [100.0]
    opens = list(closes)
    idx = pd.date_range("2026-01-01", periods=17, freq="D")
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    highs[16] = 100.0
    lows[16] = 60.0  # crash wick on the liquidation bar only
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1] * 17,
        },
        index=idx,
    )
    strategy = _SignalStrategy(
        signals,
        allow_long=True,
        allow_short=False,
        entry_threshold=0.5,
        exit_threshold=0.1,
        position_sizer=RiskBasedPositionSizer(max_fraction=0.02, risk_multiplier=2.0),
    )
    result = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.0,
        slippage=0.0,
        rebalance_band=0.05,
        leverage=4.0,
        maintenance_margin=0.5,
    ).run_on(df)

    assert result.metrics["rebalance_count"] == 1
    assert len(result.trade_log) == 1
    trade = result.trade_log.iloc[0]
    assert trade["exit_reason"] == "liquidation"
    assert trade["exit_price"] == pytest.approx(62.5)


# ---------------------------------------------------------------------------
# 14. Zero-target signals must NOT drain the position through the rebalance
#     path (regression: a target of exactly 0 used to close the position
#     silently — no trade row, no _reset_risk_state — and leak the trade's
#     rebalance accumulators into the next trade).
# ---------------------------------------------------------------------------


def test_zero_target_signal_does_not_silently_close_via_rebalance():
    n = _WARMUP_BARS + 3
    # bar 14: enter long (0.6 > et). bar 15: signal exactly 0.0 -> sizer target
    # is exactly 0; with exit_threshold=0.0 no signal exit fires (|0| < 0 is
    # False), so the bar falls through to the rebalance branch, which must
    # treat a zero target as a no-op. bar 16: hold (same signal, delta within
    # band). bar 17: sign flip closes the trade through the normal exit path.
    signals = [0.0] * _WARMUP_BARS + [0.0, 0.6, -0.6]
    signals[_WARMUP_BARS - 1] = 0.6  # bar 14 entry signal
    df = _atr_bars(n)
    strategy = _SignalStrategy(
        signals,
        allow_long=True,
        allow_short=False,
        entry_threshold=0.5,
        exit_threshold=0.0,
        position_sizer=RiskBasedPositionSizer(max_fraction=0.1, risk_multiplier=1.0),
    )
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10_000.0,
        commission=0.001,
        slippage=0.0,
        leverage=1.0,
        financing_rate=0.0,
        rebalance_band=0.25,
    )
    result = engine.run_on(df)

    # Exactly ONE round trip, entered on bar 14 and closed on the sign flip —
    # the buggy path instead drained the position on the 0.0 bar (rebalance
    # fill, no trade row) and re-entered on bar 16, so the logged trade's
    # entry_date moved and rebalance_count picked up the phantom fill.
    assert result.metrics["total_trades"] == 1
    assert result.metrics["rebalance_count"] == 0
    trade = result.trade_log.iloc[0]
    assert trade["entry_date"] == df.index[_WARMUP_BARS - 1]
    assert trade["exit_reason"] == "signal"
    assert trade["rebalance_count"] == 0
