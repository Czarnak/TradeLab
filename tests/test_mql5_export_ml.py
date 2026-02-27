from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest

from trade_lab.ml.models import KerasModelWrapper
from trade_lab.mql5_export.ml_introspector import (
    MLStrategyIntrospector,
    _extract_sizing,
    _normalise_activation,
)
from trade_lab.mql5_export.ml_validator import validate_ml_strategy
from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.sizing.risk_based import RiskBasedPositionSizer
from trade_lab.strategies.ml_strategy import MLStrategy


def _install_fake_keras(monkeypatch):
    keras = types.ModuleType("keras")

    class Dense:
        def __init__(
            self,
            units=1,
            activation="linear",
            name="dense",
            kernel=None,
            bias=None,
        ):
            self.units = units
            self.activation = activation
            self.name = name
            self._kernel = np.array(
                kernel if kernel is not None else np.zeros((1, units), dtype=float),
                dtype=float,
            )
            self._bias = None if bias is None else np.array(bias, dtype=float)

        def get_config(self):
            return {
                "units": self.units,
                "activation": self.activation,
            }

        def get_weights(self):
            if self._bias is None:
                return [self._kernel.copy()]
            return [self._kernel.copy(), self._bias.copy()]

    class Dropout:
        pass

    class InputLayer:
        pass

    class Concatenate:
        pass

    keras.layers = types.SimpleNamespace(
        Dense=Dense,
        Dropout=Dropout,
        InputLayer=InputLayer,
        Concatenate=Concatenate,
    )

    monkeypatch.setitem(sys.modules, "keras", keras)
    return keras


def _strategy_with_wrapper(
    keras_model,
    input_names: list[str],
    *,
    position_sizer=None,
    allow_long=True,
    allow_short=True,
    entry_threshold=0.5,
    exit_threshold=0.1,
):
    wrapper = KerasModelWrapper(keras_model, input_names)
    return MLStrategy(
        model=wrapper,
        indicators=[],
        position_sizer=position_sizer,
        allow_long=allow_long,
        allow_short=allow_short,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )


def test_normalise_activation_handles_string_and_dict():
    assert _normalise_activation("ReLU") == "relu"
    assert _normalise_activation({"class_name": "Sigmoid"}) == "sigmoid"
    assert _normalise_activation({}) == "linear"


def test_extract_sizing_variants():
    model = types.SimpleNamespace(input_names=["f1"], predict=lambda _: np.array([0.0]))

    none_strategy = MLStrategy(model=model, indicators=[], position_sizer=None)
    fixed_strategy = MLStrategy(
        model=model, indicators=[], position_sizer=FixedPositionSizer(0.2),
    )
    risk_strategy = MLStrategy(
        model=model,
        indicators=[],
        position_sizer=RiskBasedPositionSizer(max_fraction=0.03, risk_multiplier=3.0),
    )
    unknown_strategy = MLStrategy(model=model, indicators=[], position_sizer=object())

    assert _extract_sizing(none_strategy).sizer_type == "none"
    assert _extract_sizing(fixed_strategy).params["fraction"] == pytest.approx(0.2)
    risk_cfg = _extract_sizing(risk_strategy)
    assert risk_cfg.sizer_type == "risk_based"
    assert risk_cfg.params["max_fraction"] == pytest.approx(0.03)
    assert risk_cfg.params["risk_multiplier"] == pytest.approx(3.0)
    unknown_cfg = _extract_sizing(unknown_strategy)
    assert unknown_cfg.sizer_type == "unknown"
    assert unknown_cfg.params["class"] == "object"


def test_ml_introspector_raises_when_keras_unavailable(monkeypatch):
    real_import = builtins.__import__

    def _fail_keras(name, *args, **kwargs):
        if name == "keras":
            raise ImportError("forced keras import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_keras)

    strategy = MLStrategy(
        model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[]), input_names=["f1"]),
        indicators=[],
    )
    with pytest.raises(ImportError, match="Keras is required for ML introspection"):
        MLStrategyIntrospector().introspect(strategy)


def test_ml_introspector_builds_dense_config_and_skips_non_dense(monkeypatch):
    keras = _install_fake_keras(monkeypatch)

    dense_1 = keras.layers.Dense(
        units=3,
        activation={"class_name": "ReLU"},
        name="dense_1",
        kernel=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        bias=np.array([0.1, 0.2, 0.3]),
    )
    dense_2 = keras.layers.Dense(
        units=1,
        activation="tanh",
        name="dense_2",
        kernel=np.array([[0.5], [0.6], [0.7]]),
        bias=None,
    )
    keras_model = types.SimpleNamespace(
        layers=[
            keras.layers.InputLayer(),
            dense_1,
            keras.layers.Dropout(),
            keras.layers.Concatenate(),
            dense_2,
        ]
    )

    strategy = MLStrategy(
        model=types.SimpleNamespace(model=keras_model, input_names=["feat_a", "feat_b"]),
        indicators=[],
        position_sizer=RiskBasedPositionSizer(max_fraction=0.03, risk_multiplier=2.5),
        entry_threshold=0.4,
        exit_threshold=0.15,
        allow_long=False,
        allow_short=True,
    )

    cfg = MLStrategyIntrospector().introspect(strategy)

    assert cfg.feature_names == ["feat_a", "feat_b"]
    assert cfg.n_features == 2
    assert cfg.entry_threshold == pytest.approx(0.4)
    assert cfg.exit_threshold == pytest.approx(0.15)
    assert cfg.allow_long is False
    assert cfg.allow_short is True
    assert cfg.sizing.sizer_type == "risk_based"
    assert cfg.sizing.params["max_fraction"] == pytest.approx(0.03)
    assert len(cfg.layers) == 2

    l0 = cfg.layers[0]
    assert l0.index == 0
    assert l0.units_in == 2
    assert l0.units_out == 3
    assert l0.activation == "relu"
    assert l0.weights == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert l0.biases == [0.1, 0.2, 0.3]

    l1 = cfg.layers[1]
    assert l1.index == 1
    assert l1.units_in == 3
    assert l1.units_out == 1
    assert l1.activation == "tanh"
    assert l1.biases == [0.0]


def test_validate_ml_strategy_rejects_non_ml_strategy():
    result = validate_ml_strategy(object())
    assert result.is_valid is False
    assert any("Strategy must be MLStrategy" in e for e in result.errors)


def test_validate_ml_strategy_rejects_non_wrapper_model():
    strategy = MLStrategy(
        model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[]), input_names=["f1"]),
        indicators=[],
    )
    result = validate_ml_strategy(strategy)
    assert result.is_valid is False
    assert any("KerasModelWrapper" in e for e in result.errors)


def test_validate_ml_strategy_reports_keras_missing(monkeypatch):
    real_import = builtins.__import__

    def _fail_keras(name, *args, **kwargs):
        if name == "keras":
            raise ImportError("forced keras import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_keras)

    strategy = _strategy_with_wrapper(
        keras_model=types.SimpleNamespace(layers=[]),
        input_names=["f1"],
    )
    result = validate_ml_strategy(strategy)

    assert result.is_valid is False
    assert any("Keras is required for ML export" in e for e in result.errors)


def test_validate_ml_strategy_collects_multiple_errors(monkeypatch):
    keras = _install_fake_keras(monkeypatch)

    class Conv1D:
        pass

    bad_dense = keras.layers.Dense(
        units=2,
        activation="softmax",
        name="bad_out",
        kernel=np.array([[1.0, 2.0], [3.0, 4.0]]),
        bias=np.array([0.1, 0.2]),
    )
    keras_model = types.SimpleNamespace(layers=[Conv1D(), bad_dense])

    strategy = _strategy_with_wrapper(
        keras_model=keras_model,
        input_names=[],
        position_sizer=object(),
    )
    result = validate_ml_strategy(strategy)

    assert result.is_valid is False
    assert any("Unsupported layer type 'Conv1D'" in e for e in result.errors)
    assert any("Unsupported activation 'softmax'" in e for e in result.errors)
    assert any("Final Dense layer must have exactly 1 unit (got 2)" in e for e in result.errors)
    assert any("Unsupported position sizer 'object'" in e for e in result.errors)
    assert any("input_names is empty" in e for e in result.errors)


def test_validate_ml_strategy_requires_at_least_one_dense(monkeypatch):
    keras = _install_fake_keras(monkeypatch)
    keras_model = types.SimpleNamespace(layers=[keras.layers.InputLayer(), keras.layers.Dropout()])

    strategy = _strategy_with_wrapper(keras_model=keras_model, input_names=["f1"])
    result = validate_ml_strategy(strategy)

    assert result.is_valid is False
    assert any("No Dense layers found" in e for e in result.errors)


def test_validate_ml_strategy_emits_sigmoid_and_large_model_warnings(monkeypatch):
    keras = _install_fake_keras(monkeypatch)

    hidden = keras.layers.Dense(
        units=100,
        activation="relu",
        name="hidden",
        kernel=np.ones((101, 100)),
        bias=np.zeros(100),
    )
    out = keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="out",
        kernel=np.ones((100, 1)),
        bias=np.zeros(1),
    )
    keras_model = types.SimpleNamespace(layers=[hidden, out])

    strategy = _strategy_with_wrapper(
        keras_model=keras_model,
        input_names=[f"f{i}" for i in range(101)],
        position_sizer=FixedPositionSizer(0.02),
    )
    result = validate_ml_strategy(strategy)

    assert result.is_valid is True
    assert result.errors == []
    assert any("Output layer uses sigmoid activation" in w for w in result.warnings)
    assert any("Model has 10,301 parameters" in w for w in result.warnings)
