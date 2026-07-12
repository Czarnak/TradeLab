"""Tests for the ONNX ML validator against seed-ensemble Keras graphs.

Seed-ensemble exports build a single functional graph:

    inputs = keras.Input(...)
    outs = [member(inputs) for member in members]
    avg = keras.layers.Average()(outs)
    model = keras.Model(inputs, avg)

Iterating ``model.layers`` on this graph yields ``InputLayer``, N nested
``Functional`` model-layers (one per member), and an ``Average`` layer — no
top-level ``Dense``. ``validate_ml_strategy`` (the hardcoded-weights export
path) must keep rejecting this shape, but ``validate_ml_strategy_onnx`` (the
ONNX export path, which hands the whole graph to MT5's ``OnnxRun`` and never
translates individual layers) must accept it.

The ensemble fixture here is built with raw ``keras`` calls only — it does
NOT import ``ensemble_average_model`` from ``trade_lab.ml.models`` — to keep
this test module decoupled from that (separately developed) helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trade_lab.ml.models import KerasModelWrapper
from trade_lab.mql5_export.code_generator import (
    _format_ml_summary,
    export_ml_to_mql5_onnx,
)
from trade_lab.mql5_export.ml_introspector import MLStrategyIntrospector
from trade_lab.mql5_export.ml_validator import (
    validate_ml_strategy,
    validate_ml_strategy_onnx,
)
from trade_lab.strategies.ml_strategy import MLStrategy

keras = pytest.importorskip("keras")


# ---------------------------------------------------------------------------
# Fixtures — built with raw keras, no trade_lab.ml coupling
# ---------------------------------------------------------------------------


def _build_member(input_dim: int, seed: int):
    """A tiny functional tanh-head model — one ensemble member."""
    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(4, activation="relu")(inputs)
    output = keras.layers.Dense(1, activation="tanh")(x)
    return keras.Model(inputs, output)


def _build_ensemble(input_dim: int = 3, n_members: int = 3):
    """Average ``n_members`` independently-trained members in one graph."""
    members = [_build_member(input_dim, seed=i) for i in range(n_members)]
    inputs = keras.Input(shape=(input_dim,))
    outputs = [member(inputs) for member in members]
    averaged = keras.layers.Average()(outputs)
    return keras.Model(inputs, averaged)


def _strategy_with_wrapper(keras_model, input_names: list[str]) -> MLStrategy:
    wrapper = KerasModelWrapper(keras_model, input_names)
    return MLStrategy(model=wrapper, indicators=[])


# ---------------------------------------------------------------------------
# validate_ml_strategy_onnx — accepts the ensemble graph
# ---------------------------------------------------------------------------


def test_validate_ml_strategy_onnx_accepts_ensemble():
    ensemble = _build_ensemble()
    strategy = _strategy_with_wrapper(ensemble, ["feat_a", "feat_b", "feat_c"])

    result = validate_ml_strategy_onnx(strategy)

    assert result.errors == []
    assert result.is_valid is True


def test_validate_ml_strategy_onnx_ensemble_has_no_layer_errors_even_with_more_members():
    ensemble = _build_ensemble(input_dim=2, n_members=5)
    strategy = _strategy_with_wrapper(ensemble, ["feat_a", "feat_b"])

    result = validate_ml_strategy_onnx(strategy)

    assert not any("Unsupported layer type" in e for e in result.errors)
    assert result.is_valid is True


# ---------------------------------------------------------------------------
# validate_ml_strategy — the hardcoded-weights path must still reject it
# ---------------------------------------------------------------------------


def test_validate_ml_strategy_still_rejects_ensemble():
    ensemble = _build_ensemble()
    strategy = _strategy_with_wrapper(ensemble, ["feat_a", "feat_b", "feat_c"])

    result = validate_ml_strategy(strategy)

    assert result.is_valid is False
    assert any("Unsupported layer type 'Functional'" in e for e in result.errors)
    assert any("Unsupported layer type 'Average'" in e for e in result.errors)
    assert any("No Dense layers found" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_ml_strategy_onnx — still rejects genuinely unsupported layers
# ---------------------------------------------------------------------------


def test_validate_ml_strategy_onnx_rejects_unsupported_layer_type():
    inputs = keras.Input(shape=(5, 2))
    x = keras.layers.LSTM(4)(inputs)
    output = keras.layers.Dense(1, activation="tanh")(x)
    model = keras.Model(inputs, output)
    strategy = _strategy_with_wrapper(model, ["feat_a", "feat_b"])

    result = validate_ml_strategy_onnx(strategy)

    assert result.is_valid is False
    assert any("Unsupported layer type 'LSTM'" in e for e in result.errors)


def test_validate_ml_strategy_onnx_rejects_unsupported_layer_inside_nested_member():
    # A nested member itself contains an unsupported layer (Conv1D) — the
    # recursive walk must catch it even though the top-level container
    # (Functional) is whitelisted.
    inputs = keras.Input(shape=(4, 2))
    x = keras.layers.Conv1D(3, kernel_size=2)(inputs)
    x = keras.layers.Flatten()(x)
    output = keras.layers.Dense(1, activation="tanh")(x)
    bad_member = keras.Model(inputs, output)

    top_inputs = keras.Input(shape=(4, 2))
    wrapped_output = bad_member(top_inputs)
    ensemble = keras.Model(top_inputs, wrapped_output)

    strategy = _strategy_with_wrapper(ensemble, ["feat_a", "feat_b"])

    result = validate_ml_strategy_onnx(strategy)

    assert result.is_valid is False
    assert any("Unsupported layer type 'Conv1D'" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_ml_strategy_onnx — scalar output-shape check
# ---------------------------------------------------------------------------


def test_validate_ml_strategy_onnx_rejects_non_scalar_output():
    inputs = keras.Input(shape=(3,))
    output = keras.layers.Dense(3, activation="linear")(inputs)
    model = keras.Model(inputs, output)
    strategy = _strategy_with_wrapper(model, ["feat_a", "feat_b", "feat_c"])

    result = validate_ml_strategy_onnx(strategy)

    assert result.is_valid is False
    assert any("output shape" in e.lower() for e in result.errors)
    assert any("(None, 3)" in e for e in result.errors)


def test_validate_ml_strategy_onnx_accepts_linear_scalar_head_without_warning():
    # A linear-head model is a legitimate export now — no sigmoid-style
    # warning should fire for it.
    inputs = keras.Input(shape=(2,))
    output = keras.layers.Dense(1, activation="linear")(inputs)
    model = keras.Model(inputs, output)
    strategy = _strategy_with_wrapper(model, ["feat_a", "feat_b"])

    result = validate_ml_strategy_onnx(strategy)

    assert result.is_valid is True
    assert not any("sigmoid" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Introspector / summary — must not raise for a nested-ensemble model
# ---------------------------------------------------------------------------


def test_ml_strategy_introspector_handles_ensemble_without_raising():
    ensemble = _build_ensemble()
    strategy = _strategy_with_wrapper(ensemble, ["feat_a", "feat_b", "feat_c"])

    cfg = MLStrategyIntrospector().introspect(strategy)

    assert cfg.layers == []
    assert cfg.feature_names == ["feat_a", "feat_b", "feat_c"]
    assert cfg.nested_summary is not None
    assert "3 sub-models" in cfg.nested_summary


def test_format_ml_summary_is_non_crashing_for_ensemble():
    ensemble = _build_ensemble()
    strategy = _strategy_with_wrapper(ensemble, ["feat_a", "feat_b", "feat_c"])
    cfg = MLStrategyIntrospector().introspect(strategy)

    summary = _format_ml_summary(cfg)

    assert summary[0] == f"Features: {cfg.feature_names}"
    assert any("nested model" in line for line in summary)


# ---------------------------------------------------------------------------
# End-to-end ONNX export smoke test (skips cleanly if tf2onnx/onnx missing)
# ---------------------------------------------------------------------------


def test_export_ml_to_mql5_onnx_ensemble_end_to_end(tmp_path):
    pytest.importorskip("tf2onnx")
    pytest.importorskip("onnx")

    ensemble = _build_ensemble(input_dim=3, n_members=2)
    strategy = _strategy_with_wrapper(ensemble, ["feat_a", "feat_b", "feat_c"])

    out_file = tmp_path / "ensemble_ea.mq5"
    result = export_ml_to_mql5_onnx(strategy=strategy, output_path=str(out_file))

    assert result.validation.is_valid is True
    assert Path(result.filepath).exists()
    assert result.onnx_filepath is not None
    assert Path(result.onnx_filepath).exists()
    assert "OnnxRun" in result.code
