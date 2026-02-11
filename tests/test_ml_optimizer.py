"""Smoke tests for the ML Optuna hyperparameter optimizer."""

from __future__ import annotations

import pytest

from market_lab.data.sample_generator import generate_ohlcv_bars
from market_lab.ml.dataset_builder import (
    FeatureSelection,
    DatasetConfig,
    build_dataset,
)
from market_lab.ml.optimizer import optimize_ml


@pytest.fixture
def bars():
    return generate_ohlcv_bars(n_bars=200, seed=42)


@pytest.fixture
def dataset(bars):
    config = DatasetConfig(
        feature_selections=[
            FeatureSelection("return_lags", 5),
        ],
        task_type="classification",
        split_ratio=0.8,
    )
    return build_dataset([bars], config)


class TestMLOptimizer:
    def test_optimize_accuracy(self, bars, dataset):
        """Smoke test: 3 trials optimizing val_accuracy."""
        result = optimize_ml(
            dataset=dataset,
            bars_list=[bars],
            objective="val_accuracy",
            n_trials=3,
            min_layers=1,
            max_layers=2,
            min_units=8,
            max_units=32,
            min_epochs=2,
            max_epochs=5,
            seed=42,
        )

        assert "best_config" in result
        assert "best_value" in result
        assert "trials_df" in result
        assert len(result["trials_df"]) == 3

    def test_optimize_sharpe(self, bars, dataset):
        """Smoke test: 2 trials optimizing val_sharpe."""
        result = optimize_ml(
            dataset=dataset,
            bars_list=[bars],
            objective="val_sharpe",
            n_trials=2,
            min_layers=1,
            max_layers=2,
            min_units=8,
            max_units=32,
            min_epochs=2,
            max_epochs=5,
            seed=42,
        )

        assert "best_value" in result
        assert isinstance(result["best_value"], float)

    def test_best_config_valid(self, bars, dataset):
        """Best config should pass validation."""
        result = optimize_ml(
            dataset=dataset,
            bars_list=[bars],
            objective="val_accuracy",
            n_trials=2,
            min_layers=1,
            max_layers=2,
            min_units=8,
            max_units=32,
            min_epochs=2,
            max_epochs=5,
            seed=42,
        )

        config = result["best_config"]
        errors = config.validate()
        assert errors == []

    def test_progress_callback(self, bars, dataset):
        calls = []

        def callback(current, total):
            calls.append((current, total))

        optimize_ml(
            dataset=dataset,
            bars_list=[bars],
            objective="val_accuracy",
            n_trials=3,
            min_layers=1,
            max_layers=1,
            min_units=8,
            max_units=16,
            min_epochs=2,
            max_epochs=3,
            seed=42,
            progress_callback=callback,
        )

        assert len(calls) == 3
        assert calls[-1] == (3, 3)
