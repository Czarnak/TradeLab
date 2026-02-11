"""Smoke tests for the Optuna-based backtest optimizer."""

from __future__ import annotations

import pytest

from market_lab.data.sample_generator import generate_ohlcv_bars
from market_lab.backtest.engine import BacktestConfig
from market_lab.backtest.optimizer import optimize

# Trigger registration
import market_lab.strategies.ma_crossover  # noqa: F401
import market_lab.strategies.mean_reversion  # noqa: F401

from market_lab.strategies.base import get_strategy


@pytest.fixture
def bars():
    return generate_ohlcv_bars(n_bars=200, seed=42)


class TestOptimizer:
    def test_optimize_ma_sharpe(self, bars):
        """Smoke test: run 5 trials of MA Crossover optimizing Sharpe."""
        strategy = get_strategy("MA Crossover")
        result = optimize(
            strategy=strategy,
            bars=bars,
            objective_metric="sharpe",
            n_trials=5,
            seed=42,
        )
        assert "best_params" in result
        assert "best_value" in result
        assert "trials_df" in result
        assert len(result["trials_df"]) == 5

    def test_optimize_mean_reversion(self, bars):
        """Smoke test: few trials of Mean Reversion optimizing profit_factor."""
        strategy = get_strategy("Mean Reversion")
        result = optimize(
            strategy=strategy,
            bars=bars,
            objective_metric="profit_factor",
            n_trials=3,
            seed=123,
        )
        assert "best_params" in result
        assert len(result["trials_df"]) == 3

    def test_best_params_are_valid(self, bars):
        """Best params should have keys matching the strategy schema."""
        strategy = get_strategy("MA Crossover")
        result = optimize(
            strategy=strategy,
            bars=bars,
            objective_metric="sharpe",
            n_trials=3,
            seed=42,
        )
        schema_keys = set(strategy.parameters_schema().keys())
        best_keys = set(result["best_params"].keys())
        assert best_keys == schema_keys

    def test_progress_callback(self, bars):
        """Progress callback should be called n_trials times."""
        strategy = get_strategy("MA Crossover")
        calls = []

        def callback(current, total):
            calls.append((current, total))

        optimize(
            strategy=strategy,
            bars=bars,
            objective_metric="sharpe",
            n_trials=4,
            seed=42,
            progress_callback=callback,
        )
        assert len(calls) == 4
        assert calls[-1] == (4, 4)
