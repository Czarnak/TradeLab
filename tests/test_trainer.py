"""Tests for ML trainer — integration tests with small models and few epochs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_lab.data.sample_generator import generate_ohlcv_bars
from market_lab.ml.dataset_builder import (
    FeatureSelection,
    DatasetConfig,
    build_dataset,
)
from market_lab.ml.model_builder import (
    LayerConfig,
    ModelConfig,
    build_model,
)
from market_lab.ml.trainer import (
    train_model,
    compute_equity_curves,
    TrainingResult,
)


@pytest.fixture
def bars():
    return generate_ohlcv_bars(n_bars=200, seed=42)


@pytest.fixture
def dataset(bars):
    config = DatasetConfig(
        feature_selections=[
            FeatureSelection("return_lags", 5),
            FeatureSelection("volume_lags", 3),
        ],
        task_type="classification",
        split_ratio=0.8,
    )
    return build_dataset([bars], config)


@pytest.fixture
def small_config(dataset):
    return ModelConfig(
        input_dim=dataset.input_dim,
        layers=[LayerConfig(16, "relu")],
        optimizer="adam",
        learning_rate=0.01,
        epochs=3,
        batch_size=32,
        loss="binary_crossentropy",
        task_type="classification",
    )


class TestTrainModel:
    def test_basic_training(self, dataset, small_config):
        model = build_model(small_config)
        result = train_model(model, dataset, small_config, verbose=0)

        assert isinstance(result, TrainingResult)
        assert "loss" in result.history
        assert "val_loss" in result.history
        assert len(result.history["loss"]) == 3

    def test_predictions_shape(self, dataset, small_config):
        model = build_model(small_config)
        result = train_model(model, dataset, small_config, verbose=0)

        assert result.train_probs.shape == (len(dataset.X_train),)
        assert result.val_probs.shape == (len(dataset.X_val),)

    def test_predictions_bounded(self, dataset, small_config):
        model = build_model(small_config)
        result = train_model(model, dataset, small_config, verbose=0)

        # Sigmoid output: should be in [0, 1]
        assert result.train_probs.min() >= 0.0
        assert result.train_probs.max() <= 1.0
        assert result.val_probs.min() >= 0.0
        assert result.val_probs.max() <= 1.0

    def test_final_metrics_populated(self, dataset, small_config):
        model = build_model(small_config)
        result = train_model(model, dataset, small_config, verbose=0)

        assert result.final_train_loss > 0
        assert result.final_val_loss > 0
        assert 0 <= result.final_train_metric <= 1
        assert 0 <= result.final_val_metric <= 1

    def test_epoch_callback(self, dataset, small_config):
        model = build_model(small_config)
        calls = []

        def callback(epoch, logs):
            calls.append((epoch, logs))

        train_model(model, dataset, small_config, verbose=0, epoch_callback=callback)
        assert len(calls) == 3
        assert calls[0][0] == 1
        assert calls[-1][0] == 3

    def test_regression_training(self, bars):
        config = DatasetConfig(
            feature_selections=[FeatureSelection("return_lags", 5)],
            task_type="regression",
        )
        ds = build_dataset([bars], config)
        model_config = ModelConfig(
            input_dim=ds.input_dim,
            layers=[LayerConfig(16, "relu")],
            optimizer="adam",
            learning_rate=0.01,
            epochs=2,
            batch_size=32,
            loss="mse",
            task_type="regression",
        )
        model = build_model(model_config)
        result = train_model(model, ds, model_config, verbose=0)

        assert "loss" in result.history
        assert result.train_probs.shape[0] > 0


class TestEquityCurves:
    def test_compute_equity(self, bars, dataset, small_config):
        model = build_model(small_config)
        result = train_model(model, dataset, small_config, verbose=0)
        compute_equity_curves(result, [bars], threshold=0.5)

        assert result.train_equity is not None
        assert result.val_equity is not None
        assert result.train_benchmark is not None
        assert result.val_benchmark is not None

    def test_equity_starts_at_one(self, bars, dataset, small_config):
        model = build_model(small_config)
        result = train_model(model, dataset, small_config, verbose=0)
        compute_equity_curves(result, [bars])

        assert result.train_equity.iloc[0] == pytest.approx(1.0)
        assert result.val_equity.iloc[0] == pytest.approx(1.0)

    def test_equity_positive(self, bars, dataset, small_config):
        model = build_model(small_config)
        result = train_model(model, dataset, small_config, verbose=0)
        compute_equity_curves(result, [bars])

        assert (result.train_equity > 0).all()
        assert (result.val_equity > 0).all()

    def test_val_sharpe_computed(self, bars, dataset, small_config):
        model = build_model(small_config)
        result = train_model(model, dataset, small_config, verbose=0)
        compute_equity_curves(result, [bars])

        assert isinstance(result.val_sharpe, float)
