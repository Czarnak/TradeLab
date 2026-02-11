"""Tests for sample data generation and Monte Carlo variations."""

from __future__ import annotations

import pandas as pd
import pytest

from market_lab.data.sample_generator import generate_ohlcv_bars, generate_tick_data
from market_lab.data.monte_carlo import (
    bootstrap_resample,
    gbm_fitted,
    noise_injection,
    generate_variations,
    MCMethod,
)


class TestSampleGenerator:
    def test_ohlcv_shape(self):
        df = generate_ohlcv_bars(n_bars=100, seed=1)
        assert len(df) == 100
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_ohlcv_consistency(self):
        df = generate_ohlcv_bars(n_bars=200, seed=2)
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_ohlcv_index(self):
        df = generate_ohlcv_bars(n_bars=50, seed=3)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"
        assert df.index.is_monotonic_increasing

    def test_ohlcv_reproducible(self):
        df1 = generate_ohlcv_bars(seed=42)
        df2 = generate_ohlcv_bars(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_tick_shape(self):
        df = generate_tick_data(n_ticks=500, seed=1)
        assert len(df) == 500
        assert "bid" in df.columns
        assert "ask" in df.columns
        assert "mid" in df.columns

    def test_tick_spread(self):
        df = generate_tick_data(n_ticks=1000, seed=2)
        assert (df["ask"] > df["bid"]).all()

    def test_tick_mid(self):
        df = generate_tick_data(n_ticks=100, seed=3)
        expected = (df["bid"] + df["ask"]) / 2
        pd.testing.assert_series_equal(df["mid"], expected, check_names=False)


class TestMonteCarlo:
    @pytest.fixture
    def bars(self):
        return generate_ohlcv_bars(n_bars=200, seed=10)

    def test_bootstrap_count(self, bars):
        results = bootstrap_resample(bars, n_simulations=5, seed=1)
        assert len(results) == 5
        for df in results:
            assert len(df) == len(bars)
            assert list(df.columns) == list(bars.columns)

    def test_gbm_fitted_count(self, bars):
        results = gbm_fitted(bars, n_simulations=3, seed=2)
        assert len(results) == 3

    def test_noise_injection_count(self, bars):
        results = noise_injection(bars, n_simulations=4, noise_scale=0.2, seed=3)
        assert len(results) == 4

    def test_generate_variations_all(self, bars):
        results = generate_variations(bars, methods=None, n_simulations=2, seed=5)
        assert set(results.keys()) == {"bootstrap", "gbm_fitted", "noise_injection"}
        for paths in results.values():
            assert len(paths) == 2

    def test_generate_variations_single(self, bars):
        results = generate_variations(
            bars, methods=[MCMethod.BOOTSTRAP], n_simulations=3, seed=6,
        )
        assert "bootstrap" in results
        assert len(results) == 1

    def test_ohlcv_consistency_preserved(self, bars):
        results = bootstrap_resample(bars, n_simulations=1, seed=7)
        df = results[0]
        assert (df["high"] >= df["low"]).all()
