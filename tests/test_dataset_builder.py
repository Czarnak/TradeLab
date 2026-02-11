"""Tests for ML dataset builder."""

from __future__ import annotations

import numpy as np
import pytest

from market_lab.data.sample_generator import generate_ohlcv_bars
from market_lab.ml.dataset_builder import (
    FeatureSelection,
    DatasetConfig,
    BuiltDataset,
    build_dataset,
    build_features_df,
    build_target,
    compute_input_dim,
)


@pytest.fixture
def bars():
    return generate_ohlcv_bars(n_bars=200, seed=42)


@pytest.fixture
def default_config():
    return DatasetConfig(
        feature_selections=[
            FeatureSelection("close_lags", n_lags=5),
            FeatureSelection("return_lags", n_lags=3),
        ],
        task_type="classification",
        split_ratio=0.8,
        shuffle=False,
    )


class TestComputeInputDim:
    def test_single_feature(self):
        sels = [FeatureSelection("close_lags", 5)]
        assert compute_input_dim(sels) == 5

    def test_multiple_features(self):
        sels = [
            FeatureSelection("close_lags", 5),
            FeatureSelection("return_lags", 3),
            FeatureSelection("volume_lags", 2),
        ]
        assert compute_input_dim(sels) == 10


class TestBuildFeaturesDF:
    def test_shape(self, bars):
        sels = [
            FeatureSelection("close_lags", 5),
            FeatureSelection("return_lags", 3),
        ]
        df = build_features_df(bars, sels)
        assert df.shape[0] == len(bars)
        assert df.shape[1] == 8  # 5 + 3

    def test_column_names(self, bars):
        sels = [FeatureSelection("close_lags", 2)]
        df = build_features_df(bars, sels)
        assert list(df.columns) == ["close_lag_1", "close_lag_2"]

    def test_empty_selections_raises(self, bars):
        with pytest.raises(ValueError, match="No feature selections"):
            build_features_df(bars, [])


class TestBuildTarget:
    def test_classification(self, bars):
        target = build_target(bars, "classification")
        assert len(target) == len(bars)
        valid = target.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_regression(self, bars):
        target = build_target(bars, "regression")
        assert len(target) == len(bars)
        valid = target.dropna()
        # Log returns should be small numbers, not binary
        assert valid.dtype == np.float64

    def test_last_row_is_nan(self, bars):
        target = build_target(bars, "classification")
        assert np.isnan(target.iloc[-1])

    def test_invalid_type_raises(self, bars):
        with pytest.raises(ValueError, match="Unknown task type"):
            build_target(bars, "invalid_task")


class TestBuildDataset:
    def test_basic_build(self, bars, default_config):
        ds = build_dataset([bars], default_config)
        assert isinstance(ds, BuiltDataset)
        assert ds.input_dim == 8  # 5 + 3
        assert ds.X_train.shape[1] == 8
        assert ds.X_val.shape[1] == 8

    def test_split_ratio(self, bars, default_config):
        ds = build_dataset([bars], default_config)
        total = len(ds.X_train) + len(ds.X_val)
        ratio = len(ds.X_train) / total
        assert abs(ratio - 0.8) < 0.02

    def test_no_nan(self, bars, default_config):
        ds = build_dataset([bars], default_config)
        assert not np.isnan(ds.X_train).any()
        assert not np.isnan(ds.X_val).any()
        assert not np.isnan(ds.y_train).any()
        assert not np.isnan(ds.y_val).any()

    def test_float32_dtype(self, bars, default_config):
        ds = build_dataset([bars], default_config)
        assert ds.X_train.dtype == np.float32
        assert ds.y_train.dtype == np.float32

    def test_feature_names(self, bars, default_config):
        ds = build_dataset([bars], default_config)
        assert len(ds.feature_names) == 8
        assert "close_lag_1" in ds.feature_names
        assert "return_lag_1" in ds.feature_names

    def test_multi_dataset(self, bars):
        bars2 = generate_ohlcv_bars(n_bars=150, seed=99)
        config = DatasetConfig(
            feature_selections=[FeatureSelection("close_lags", 3)],
            task_type="classification",
        )
        ds = build_dataset([bars, bars2], config)
        # Total rows should be close to 200 + 150 minus NaN rows
        assert ds.X_train.shape[0] + ds.X_val.shape[0] > 300

    def test_shuffle_changes_order(self, bars):
        config_no_shuffle = DatasetConfig(
            feature_selections=[FeatureSelection("close_lags", 3)],
            shuffle=False,
        )
        config_shuffle = DatasetConfig(
            feature_selections=[FeatureSelection("close_lags", 3)],
            shuffle=True,
            random_seed=42,
        )
        ds1 = build_dataset([bars], config_no_shuffle)
        ds2 = build_dataset([bars], config_shuffle)
        # Shuffled data should differ (with very high probability)
        assert not np.array_equal(ds1.X_train, ds2.X_train)

    def test_regression_task(self, bars):
        config = DatasetConfig(
            feature_selections=[FeatureSelection("return_lags", 5)],
            task_type="regression",
        )
        ds = build_dataset([bars], config)
        # Regression targets should not be binary
        unique = np.unique(ds.y_train)
        assert len(unique) > 2

    def test_empty_bars_raises(self, default_config):
        with pytest.raises(ValueError, match="at least one"):
            build_dataset([], default_config)

    def test_no_features_raises(self, bars):
        config = DatasetConfig(feature_selections=[])
        with pytest.raises(ValueError, match="At least one"):
            build_dataset([bars], config)
