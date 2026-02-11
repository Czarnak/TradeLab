"""Tests for built-in strategies."""

from __future__ import annotations

import pandas as pd
import pytest

from market_lab.data.sample_generator import generate_ohlcv_bars

# Import to trigger registration
import market_lab.strategies.ma_crossover  # noqa: F401
import market_lab.strategies.mean_reversion  # noqa: F401

from market_lab.strategies.base import (
    get_strategy,
    list_strategies,
    all_strategies,
    StrategySignals,
)


@pytest.fixture
def bars():
    return generate_ohlcv_bars(n_bars=200, seed=42)


class TestStrategyRegistry:
    def test_strategies_registered(self):
        names = list_strategies()
        assert "MA Crossover" in names
        assert "Mean Reversion" in names

    def test_get_strategy(self):
        s = get_strategy("MA Crossover")
        assert s.name == "MA Crossover"

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            get_strategy("NonexistentStrategy")


class TestMACrossover:
    def test_run_default(self, bars):
        s = get_strategy("MA Crossover")
        result = s.run(bars, s.default_params())
        assert isinstance(result, StrategySignals)
        assert len(result.signal) == len(bars)

    def test_signal_values(self, bars):
        s = get_strategy("MA Crossover")
        result = s.run(bars, s.default_params())
        unique_vals = set(result.signal.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_no_short(self, bars):
        s = get_strategy("MA Crossover")
        params = {**s.default_params(), "allow_short": False}
        result = s.run(bars, params)
        assert (result.signal >= 0).all()

    def test_warmup_period_is_flat(self, bars):
        s = get_strategy("MA Crossover")
        params = {"fast_period": 10, "slow_period": 30, "ma_type": "SMA", "allow_short": True}
        result = s.run(bars, params)
        assert (result.signal.iloc[:30] == 0).all()

    def test_metadata_contains_ma(self, bars):
        s = get_strategy("MA Crossover")
        result = s.run(bars, s.default_params())
        assert "fast_ma" in result.metadata
        assert "slow_ma" in result.metadata


class TestMeanReversion:
    def test_run_default(self, bars):
        s = get_strategy("Mean Reversion")
        result = s.run(bars, s.default_params())
        assert isinstance(result, StrategySignals)
        assert len(result.signal) == len(bars)

    def test_signal_values(self, bars):
        s = get_strategy("Mean Reversion")
        result = s.run(bars, s.default_params())
        unique_vals = set(result.signal.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_no_short(self, bars):
        s = get_strategy("Mean Reversion")
        params = {**s.default_params(), "allow_short": False}
        result = s.run(bars, params)
        assert (result.signal >= 0).all()

    def test_parameters_schema(self):
        s = get_strategy("Mean Reversion")
        schema = s.parameters_schema()
        assert "lookback" in schema
        assert "entry_z" in schema
        assert schema["lookback"]["type"] == "int"
