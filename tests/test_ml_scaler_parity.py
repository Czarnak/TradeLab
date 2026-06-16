"""Regression tests for feature-scaling train/serve parity.

A model trained on scaled features must produce identical predictions when the
*raw* features are fed at inference — otherwise backtests, the Optuna objective,
and the exported ONNX/MQL5 EA all silently see out-of-distribution inputs.

The fix folds the scaler's affine transform ``(x - offset) / scale`` into the
first Dense layer's weights, so the deployed model is raw-feature-native and
parity holds everywhere (Python predict, ONNX graph, hardcoded MQL5 forward
pass) with no separate scaling step to keep in sync.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.indicators.base import BaseIndicator
from trade_lab.ml.models import (
    KerasModelWrapper,
    fold_scaler_into_first_dense,
    scaler_affine,
    wrap_with_scaling,
)
from trade_lab.ml.preprocessing import FeatureScaler
from trade_lab.ml.targets import BaseTarget
from trade_lab.ml.trainer import MLTrainer

# These tests exercise the real Keras forward pass; skip cleanly if unavailable.
keras = pytest.importorskip("keras")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_input_model(n_features: int, units: int = 4):
    inp = keras.Input(shape=(n_features,))
    x = keras.layers.Dense(units, activation="relu")(inp)
    out = keras.layers.Dense(1, activation="tanh")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


class _FeatureIndicator(BaseIndicator):
    """Indicator that injects a fixed column of values — deterministic features."""

    def __init__(self, name: str, values: list[float]):
        super().__init__()
        self._name = name
        self._values = np.asarray(values, dtype=float)

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.get_data(df.copy())
        out[self._raw_output_columns[0]] = self._values[: len(out)]
        return out

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.zeros(len(df)), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None):
        return None

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__{self._name}"]


class _NextReturnTarget(BaseTarget):
    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            df["Close"].pct_change().shift(-1), index=df.index, name="target"
        )


# ---------------------------------------------------------------------------
# scaler_affine
# ---------------------------------------------------------------------------


def test_scaler_affine_from_feature_scaler_standard():
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    scaler = FeatureScaler("standard")
    scaler.fit(X, ["a", "b"])
    offset, scale = scaler_affine(scaler)
    np.testing.assert_allclose(offset, X.mean(axis=0))
    np.testing.assert_allclose(scale, X.std(axis=0))


def test_scaler_affine_from_sklearn_like_attrs():
    scaler = type(
        "S", (), {"mean_": np.array([1.0, 2.0]), "scale_": np.array([3.0, 4.0])}
    )()
    offset, scale = scaler_affine(scaler)
    np.testing.assert_allclose(offset, [1.0, 2.0])
    np.testing.assert_allclose(scale, [3.0, 4.0])


# ---------------------------------------------------------------------------
# fold_scaler_into_first_dense
# ---------------------------------------------------------------------------


def test_fold_standard_scaler_matches_manual_scaling():
    keras.utils.set_random_seed(0)
    n = 5
    model = _single_input_model(n)
    rng = np.random.RandomState(1)
    X = rng.uniform(-3.0, 7.0, size=(12, n)).astype("float32")
    offset = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0

    expected = model.predict((X - offset) / scale, verbose=0)
    folded = fold_scaler_into_first_dense(model, offset, scale)
    got = folded.predict(X, verbose=0)

    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_fold_minmax_scaler_matches_manual_scaling():
    keras.utils.set_random_seed(0)
    n = 4
    model = _single_input_model(n)
    rng = np.random.RandomState(2)
    X = rng.uniform(0.0, 50.0, size=(10, n)).astype("float32")
    offset = X.min(axis=0)
    rng_span = X.max(axis=0) - offset
    rng_span[rng_span == 0] = 1.0

    expected = model.predict((X - offset) / rng_span, verbose=0)
    folded = fold_scaler_into_first_dense(model, offset, rng_span)
    got = folded.predict(X, verbose=0)

    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_fold_raises_when_no_dense_layer():
    model = type("M", (), {"layers": []})()
    with pytest.raises(ValueError, match="no Dense layer"):
        fold_scaler_into_first_dense(model, np.array([0.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# KerasModelWrapper carries / applies the scaler (fallback path)
# ---------------------------------------------------------------------------


def test_wrapper_applies_scaler_in_predict():
    class _SumModel:
        def predict(self, X, verbose=0):
            return X.sum(axis=1, keepdims=True)

    scaler = FeatureScaler("standard")
    scaler.fit(np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]]), ["a", "b"])
    wrapper = KerasModelWrapper(_SumModel(), ["a", "b"], scaler=scaler)

    # Row equal to the column means scales to all-zeros → sum 0.
    out = wrapper.predict(pd.DataFrame({"a": [10.0], "b": [20.0]}))
    assert out[0] == pytest.approx(0.0)


def test_wrapper_save_load_roundtrips_scaler(tmp_path):
    model = _single_input_model(2)
    scaler = FeatureScaler("standard")
    scaler.fit(np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]), ["a", "b"])
    wrapper = KerasModelWrapper(model, ["a", "b"], scaler=scaler)

    df = pd.DataFrame({"a": [2.0, 4.0], "b": [4.0, 8.0]})
    before = wrapper.predict(df)

    wrapper.save(str(tmp_path / "m"))
    loaded = KerasModelWrapper.load(str(tmp_path / "m"))
    after = loaded.predict(df)

    assert loaded.input_names == ["a", "b"]
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end: MLTrainer with a scaler is raw-native after training
# ---------------------------------------------------------------------------


def test_mltrainer_folds_scaler_so_predict_uses_raw_features():
    idx = pd.date_range("2026-01-01", periods=16, freq="D")
    df = pd.DataFrame({"Close": np.linspace(100.0, 130.0, 16)}, index=idx)
    f1 = _FeatureIndicator("f1", list(np.linspace(0.0, 3.0, 16)))
    f2 = _FeatureIndicator("f2", list(np.linspace(50.0, 80.0, 16)))

    W = np.array([[0.5], [-0.25]])
    b = np.array([0.1])

    def builder(input_dim: int):
        inp = keras.Input(shape=(input_dim,))
        dense = keras.layers.Dense(1, activation="linear")
        out = dense(inp)
        model = keras.Model(inp, out)
        dense.set_weights([W.copy(), b.copy()])
        model.compile(optimizer="adam", loss="mse")
        return model

    scaler = FeatureScaler("standard")
    trainer = MLTrainer(
        indicators=[f1, f2],
        target=_NextReturnTarget(),
        model_builder=builder,
        scaler=scaler,
    )

    # Reconstruct the exact raw feature matrix the trainer trains on.
    X_raw, _, feature_cols, _ = trainer.build_dataset(df)

    wrapper = trainer.train(df, epochs=0, validation_split=0.3, verbose=0)

    # Scaler is fitted on the full raw X during train().
    offset = X_raw.mean(axis=0)
    scale = X_raw.std(axis=0)
    expected = (((X_raw - offset) / scale) @ W + b).flatten()

    got = wrapper.predict(pd.DataFrame(X_raw, columns=feature_cols))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_wrap_with_scaling_falls_back_for_non_keras_model():
    class _SumModel:
        def predict(self, X, verbose=0):
            return X.sum(axis=1, keepdims=True)

    scaler = FeatureScaler("standard")
    scaler.fit(np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]]), ["a", "b"])

    wrapper = wrap_with_scaling(_SumModel(), ["a", "b"], scaler)
    # No Dense layer to fold into → scaler is attached and applied at predict.
    assert wrapper.scaler is scaler
    out = wrapper.predict(pd.DataFrame({"a": [10.0], "b": [20.0]}))
    assert out[0] == pytest.approx(0.0)
