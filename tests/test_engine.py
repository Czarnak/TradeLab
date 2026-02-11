"""Tests for the backtest engine on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_lab.backtest.engine import BacktestConfig, run_backtest


@pytest.fixture
def flat_bars():
    """Bars with flat price (100) — no PnL from holding."""
    idx = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        },
        index=idx,
    )


@pytest.fixture
def trending_up_bars():
    """Bars with monotonically increasing close."""
    idx = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    closes = np.linspace(100, 150, 50)
    return pd.DataFrame(
        {
            "open": closes - 0.5,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": 1_000_000,
        },
        index=idx,
    )


class TestBacktestEngine:
    def test_flat_market_all_cash(self, flat_bars):
        """All-flat signal → no trades, equity stays at initial capital."""
        signals = pd.Series(0, index=flat_bars.index)
        config = BacktestConfig(commission_bps=0, slippage_bps=0)
        result = run_backtest(flat_bars, signals, config)
        assert len(result.trades) == 0
        assert result.equity_curve.iloc[-1] == pytest.approx(100_000)

    def test_long_trending_up(self, trending_up_bars):
        """Constant long signal on uptrend → positive PnL."""
        signals = pd.Series(1, index=trending_up_bars.index)
        config = BacktestConfig(commission_bps=0, slippage_bps=0)
        result = run_backtest(trending_up_bars, signals, config)
        assert result.equity_curve.iloc[-1] > 100_000

    def test_short_trending_up(self, trending_up_bars):
        """Constant short signal on uptrend → negative PnL."""
        signals = pd.Series(-1, index=trending_up_bars.index)
        config = BacktestConfig(commission_bps=0, slippage_bps=0, allow_short=True)
        result = run_backtest(trending_up_bars, signals, config)
        assert result.equity_curve.iloc[-1] < 100_000

    def test_short_disabled(self, trending_up_bars):
        """Short signals become flat when allow_short=False."""
        signals = pd.Series(-1, index=trending_up_bars.index)
        config = BacktestConfig(allow_short=False, commission_bps=0, slippage_bps=0)
        result = run_backtest(trending_up_bars, signals, config)
        assert len(result.trades) == 0
        assert result.equity_curve.iloc[-1] == pytest.approx(100_000)

    def test_commission_reduces_pnl(self, trending_up_bars):
        """Commission should reduce final equity."""
        signals = pd.Series(1, index=trending_up_bars.index)
        no_cost = run_backtest(
            trending_up_bars, signals,
            BacktestConfig(commission_bps=0, slippage_bps=0),
        )
        with_cost = run_backtest(
            trending_up_bars, signals,
            BacktestConfig(commission_bps=10, slippage_bps=0),
        )
        assert with_cost.equity_curve.iloc[-1] < no_cost.equity_curve.iloc[-1]

    def test_slippage_reduces_pnl(self, trending_up_bars):
        """Slippage should reduce final equity."""
        signals = pd.Series(1, index=trending_up_bars.index)
        no_slip = run_backtest(
            trending_up_bars, signals,
            BacktestConfig(commission_bps=0, slippage_bps=0),
        )
        with_slip = run_backtest(
            trending_up_bars, signals,
            BacktestConfig(commission_bps=0, slippage_bps=10),
        )
        assert with_slip.equity_curve.iloc[-1] < no_slip.equity_curve.iloc[-1]

    def test_equity_curve_length(self, flat_bars):
        signals = pd.Series(0, index=flat_bars.index)
        result = run_backtest(flat_bars, signals)
        assert len(result.equity_curve) == len(flat_bars)

    def test_daily_returns_length(self, flat_bars):
        signals = pd.Series(0, index=flat_bars.index)
        result = run_backtest(flat_bars, signals)
        assert len(result.daily_returns) == len(flat_bars)

    def test_metrics_populated(self, trending_up_bars):
        signals = pd.Series(1, index=trending_up_bars.index)
        result = run_backtest(trending_up_bars, signals)
        assert "total_return_pct" in result.metrics
        assert "sharpe" in result.metrics
        assert "max_drawdown_pct" in result.metrics

    def test_signal_switch_creates_trades(self, trending_up_bars):
        """Alternating signals should create multiple trades."""
        n = len(trending_up_bars)
        sig = pd.Series(0, index=trending_up_bars.index)
        sig.iloc[:15] = 1
        sig.iloc[15:30] = -1
        sig.iloc[30:45] = 1
        config = BacktestConfig(commission_bps=0, slippage_bps=0, allow_short=True)
        result = run_backtest(trending_up_bars, sig, config)
        assert len(result.trades) >= 2
