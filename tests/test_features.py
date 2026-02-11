"""Tests for ML feature builders in input_definitions.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_lab.data.sample_generator import generate_ohlcv_bars
from market_lab.ml.input_definitions import (
    CloseLags,
    OpenLags,
    ReturnLags,
    HighLowRangeLags,
    VolumeLags,
    get_feature_builder,
    list_feature_builders,
    all_feature_builders,
)


@pytest.fixture
def bars():
    return generate_ohlcv_bars(n_bars=100, seed=42)


class TestRegistry:
    def test_all_registered(self):
        names = list_feature_builders()
        expected = {"close_lags", "open_lags", "return_lags", "hl_range_lags", "volume_lags"}
        assert expected == set(names)

    def test_get_existing(self):
        builder = get_feature_builder("close_lags")
        assert builder.name == "close_lags"

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            get_feature_builder("nonexistent_feature")

    def test_all_feature_builders_dict(self):
        d = all_feature_builders()
        assert isinstance(d, dict)
        assert len(d) == 5


class TestCloseLags:
    def test_shape(self, bars):
        builder = CloseLags()
        df = builder.build_features(bars, n_lags=5)
        assert df.shape == (len(bars), 5)
        assert list(df.columns) == [f"close_lag_{i}" for i in range(1, 6)]

    def test_values(self, bars):
        builder = CloseLags()
        df = builder.build_features(bars, n_lags=3)
        # lag_1 should equal close shifted by 1
        expected = bars["close"].shift(1)
        pd.testing.assert_series_equal(df["close_lag_1"], expected, check_names=False)

    def test_output_dim(self):
        builder = CloseLags()
        assert builder.output_dim(10) == 10

    def test_nan_in_first_rows(self, bars):
        builder = CloseLags()
        df = builder.build_features(bars, n_lags=5)
        assert df.iloc[0].isna().all()
        assert df.iloc[4].isna().any()
        assert df.iloc[5].notna().all()


class TestOpenLags:
    def test_shape(self, bars):
        builder = OpenLags()
        df = builder.build_features(bars, n_lags=3)
        assert df.shape == (len(bars), 3)

    def test_values(self, bars):
        builder = OpenLags()
        df = builder.build_features(bars, n_lags=2)
        expected = bars["open"].shift(1)
        pd.testing.assert_series_equal(df["open_lag_1"], expected, check_names=False)


class TestReturnLags:
    def test_shape(self, bars):
        builder = ReturnLags()
        df = builder.build_features(bars, n_lags=5)
        assert df.shape == (len(bars), 5)

    def test_returns_are_log(self, bars):
        builder = ReturnLags()
        df = builder.build_features(bars, n_lags=1)
        manual_log_ret = np.log(bars["close"] / bars["close"].shift(1))
        expected_lag1 = manual_log_ret.shift(1)
        pd.testing.assert_series_equal(
            df["return_lag_1"], expected_lag1, check_names=False,
        )


class TestHighLowRangeLags:
    def test_shape(self, bars):
        builder = HighLowRangeLags()
        df = builder.build_features(bars, n_lags=4)
        assert df.shape == (len(bars), 4)

    def test_positive_range(self, bars):
        builder = HighLowRangeLags()
        df = builder.build_features(bars, n_lags=2)
        valid = df.dropna()
        assert (valid >= 0).all().all()


class TestVolumeLags:
    def test_shape(self, bars):
        builder = VolumeLags()
        df = builder.build_features(bars, n_lags=3)
        assert df.shape == (len(bars), 3)

    def test_values(self, bars):
        builder = VolumeLags()
        df = builder.build_features(bars, n_lags=1)
        expected = bars["volume"].shift(1)
        pd.testing.assert_series_equal(df["volume_lag_1"], expected, check_names=False)
