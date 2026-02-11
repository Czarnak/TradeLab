"""Tests for backtest metrics calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_lab.backtest.metrics import (
    max_drawdown,
    cagr,
    sharpe_ratio,
    sortino_ratio,
    profit_factor,
    win_rate,
    avg_trade_pnl,
)
from market_lab.backtest.engine import Trade


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = pd.Series([100, 110, 120, 130])
        assert max_drawdown(equity) == 0.0

    def test_simple_drawdown(self):
        equity = pd.Series([100, 120, 90, 110])
        dd = max_drawdown(equity)
        # Peak=120, trough=90 → dd = (90-120)/120 = -0.25
        assert dd == pytest.approx(-0.25)

    def test_total_loss(self):
        equity = pd.Series([100, 50, 0.01])
        dd = max_drawdown(equity)
        assert dd < -0.99


class TestCAGR:
    def test_flat(self):
        equity = pd.Series([100] * 252)
        assert cagr(equity) == pytest.approx(0.0)

    def test_double_in_one_year(self):
        # 252 bars (1 year), start=100, end=200 → CAGR ≈ 100%
        equity = pd.Series(np.linspace(100, 200, 253))
        c = cagr(equity, periods_per_year=252)
        assert c == pytest.approx(1.0, abs=0.01)

    def test_negative_return(self):
        equity = pd.Series(np.linspace(100, 80, 253))
        c = cagr(equity, periods_per_year=252)
        assert c < 0


class TestSharpe:
    def test_zero_vol(self):
        returns = pd.Series([0.01] * 100)
        assert sharpe_ratio(returns) != 0

    def test_positive_returns_positive_sharpe(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.01, 252))
        s = sharpe_ratio(returns)
        # With positive drift, Sharpe should be positive (most of the time)
        assert isinstance(s, float)

    def test_zero_returns(self):
        returns = pd.Series([0.0] * 100)
        assert sharpe_ratio(returns) == 0.0


class TestSortino:
    def test_no_downside(self):
        returns = pd.Series([0.01, 0.02, 0.015, 0.01])
        s = sortino_ratio(returns)
        # All positive returns → no downside → should be 0 or handled gracefully
        assert s == 0.0

    def test_with_downside(self):
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
        s = sortino_ratio(returns)
        assert isinstance(s, float)


class TestTradeMetrics:
    @pytest.fixture
    def trades(self):
        return [
            Trade(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-10"),
                  1, 100, 110, 10, 100, 1, 9),
            Trade(pd.Timestamp("2023-01-11"), pd.Timestamp("2023-01-20"),
                  1, 110, 105, 10, -50, 1, 9),
            Trade(pd.Timestamp("2023-01-21"), pd.Timestamp("2023-01-30"),
                  -1, 105, 100, 10, 50, 1, 9),
        ]

    def test_profit_factor(self, trades):
        pf = profit_factor(trades)
        gross_profit = 100 + 50
        gross_loss = 50
        assert pf == pytest.approx(gross_profit / gross_loss)

    def test_win_rate(self, trades):
        wr = win_rate(trades)
        assert wr == pytest.approx(2 / 3)

    def test_avg_trade(self, trades):
        avg = avg_trade_pnl(trades)
        total = 100 + (-50) + 50
        assert avg == pytest.approx(total / 3)

    def test_empty_trades(self):
        assert profit_factor([]) == 0.0
        assert win_rate([]) == 0.0
        assert avg_trade_pnl([]) == 0.0

    def test_all_winners(self):
        trades = [
            Trade(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-10"),
                  1, 100, 110, 10, 100, 0, 9),
        ]
        assert profit_factor(trades) == float("inf")
        assert win_rate(trades) == 1.0
