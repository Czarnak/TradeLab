"""Tests for ML model builder."""

from __future__ import annotations

import pytest

from market_lab.ml.model_builder import (
    LayerConfig,
    ModelConfig,
    build_model,
    default_classification_config,
    default_regression_config,
    VALID_ACTIVATIONS,
    VALID_OPTIMIZERS,
)


class TestModelConfig:
    def test_default_valid(self):
        config = ModelConfig()
        errors = config.validate()
        assert errors == []

    def test_invalid_activation(self):
        config = ModelConfig(layers=[LayerConfig(32, "invalid_act")])
        errors = config.validate()
        assert any("activation" in e for e in errors)

    def test_invalid_optimizer(self):
        config = ModelConfig(optimizer="invalid_opt")
        errors = config.validate()
        assert any("optimizer" in e for e in errors)

    def test_zero_units(self):
        config = ModelConfig(layers=[LayerConfig(0, "relu")])
        errors = config.validate()
        assert any("units" in e for e in errors)

    def test_negative_lr(self):
        config = ModelConfig(learning_rate=-0.01)
        errors = config.validate()
        assert any("learning_rate" in e for e in errors)

    def test_to_dict_and_back(self):
        config = ModelConfig(
            input_dim=20,
            layers=[LayerConfig(64, "relu"), LayerConfig(32, "tanh")],
            optimizer="rmsprop",
            learning_rate=0.01,
        )
        d = config.to_dict()
        restored = ModelConfig.from_dict(d)
        assert restored.input_dim == 20
        assert len(restored.layers) == 2
        assert restored.layers[0].units == 64
        assert restored.layers[1].activation == "tanh"
        assert restored.optimizer == "rmsprop"

    def test_serialisation_roundtrip(self):
        config = default_classification_config(15)
        d = config.to_dict()
        restored = ModelConfig.from_dict(d)
        assert restored.input_dim == config.input_dim
        assert restored.task_type == config.task_type
        assert len(restored.layers) == len(config.layers)


class TestBuildModel:
    def test_classification_model(self):
        config = default_classification_config(10)
        model = build_model(config)
        assert model is not None
        # Output should be 1 unit (sigmoid)
        assert model.output_shape == (None, 1)

    def test_regression_model(self):
        config = default_regression_config(10)
        model = build_model(config)
        assert model is not None
        assert model.output_shape == (None, 1)

    def test_custom_architecture(self):
        config = ModelConfig(
            input_dim=20,
            layers=[
                LayerConfig(128, "relu"),
                LayerConfig(64, "elu"),
                LayerConfig(32, "tanh"),
            ],
            optimizer="sgd",
            learning_rate=0.1,
        )
        model = build_model(config)
        # 3 hidden layers + 1 output = model should work
        assert model is not None

    def test_invalid_config_raises(self):
        config = ModelConfig(layers=[LayerConfig(0, "relu")])
        with pytest.raises(ValueError, match="Invalid model config"):
            build_model(config)

    def test_single_layer(self):
        config = ModelConfig(
            input_dim=5,
            layers=[LayerConfig(16, "relu")],
        )
        model = build_model(config)
        assert model is not None

    def test_all_activations(self):
        """All valid activations should build without error."""
        for act in VALID_ACTIVATIONS:
            if act in ("sigmoid", "linear"):
                continue  # these are output activations
            config = ModelConfig(
                input_dim=5,
                layers=[LayerConfig(8, act)],
                epochs=1,
            )
            model = build_model(config)
            assert model is not None

    def test_all_optimizers(self):
        """All valid optimizers should build without error."""
        for opt in VALID_OPTIMIZERS:
            config = ModelConfig(
                input_dim=5,
                layers=[LayerConfig(8, "relu")],
                optimizer=opt,
            )
            model = build_model(config)
            assert model is not None
