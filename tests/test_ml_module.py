from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trade_lab.indicators.base import BaseIndicator
from trade_lab.ml.models import KerasModelWrapper, dense_model
from trade_lab.ml.preprocessing import FeatureScaler, prepare_features
from trade_lab.ml.targets import BaseTarget, DirectionalTarget, FutureReturn
from trade_lab.ml.trainer import MLTrainer
from trade_lab.ml.validation import WalkForwardSplit


class _SimpleTarget(BaseTarget):
    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            df["Close"].pct_change().shift(-1),
            index=df.index,
            name="target",
        )


class _FeatureIndicator(BaseIndicator):
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


def _install_fake_keras_for_builders(monkeypatch):
    keras = types.ModuleType("keras")
    calls: dict[str, object] = {
        "input_shapes": [],
        "dense_inits": [],
        "dropout_rates": [],
        "lstm_units": [],
        "compile": [],
        "adam_lrs": [],
    }

    class _Tensor:
        def __init__(self, name: str, payload=None):
            self.name = name
            self.payload = payload

    def Input(shape):
        calls["input_shapes"].append(shape)
        return _Tensor("input", shape)

    class Dense:
        def __init__(self, units, activation="linear"):
            calls["dense_inits"].append({"units": units, "activation": activation})
            self.units = units
            self.activation = activation

        def __call__(self, x):
            return _Tensor(
                "dense", {"prev": x, "units": self.units, "activation": self.activation}
            )

    class Dropout:
        def __init__(self, rate):
            calls["dropout_rates"].append(rate)
            self.rate = rate

        def __call__(self, x):
            return _Tensor("dropout", {"prev": x, "rate": self.rate})

    class LSTM:
        def __init__(self, units):
            calls["lstm_units"].append(units)
            self.units = units

        def __call__(self, x):
            return _Tensor("lstm", {"prev": x, "units": self.units})

    class Adam:
        def __init__(self, learning_rate):
            calls["adam_lrs"].append(learning_rate)
            self.learning_rate = learning_rate

    class Model:
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs
            self.compiled = None

        def compile(self, optimizer, loss):
            self.compiled = {"optimizer": optimizer, "loss": loss}
            calls["compile"].append(self.compiled)

    keras.Input = Input
    keras.Model = Model
    keras.layers = types.SimpleNamespace(Dense=Dense, Dropout=Dropout, LSTM=LSTM)
    keras.optimizers = types.SimpleNamespace(Adam=Adam)
    monkeypatch.setitem(sys.modules, "keras", keras)
    return calls


def test_prepare_features_drops_nan_rows_and_aligns_index():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, np.nan, 4.0],
            "f2": [10.0, 11.0, 12.0, 13.0],
        },
        index=idx,
    )
    target = pd.Series([0.1, np.nan, 0.3, 0.4], index=idx)

    X, y, out_idx = prepare_features(df, ["f1", "f2"], target)

    np.testing.assert_allclose(X, np.array([[1.0, 10.0], [4.0, 13.0]]))
    np.testing.assert_allclose(y, np.array([0.1, 0.4]))
    assert list(out_idx) == [idx[0], idx[3]]


def test_feature_scaler_standard_and_minmax_with_constant_feature():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])

    standard = FeatureScaler("standard")
    X_std = standard.fit_transform(X, ["a", "b"])
    assert X_std.shape == (3, 2)
    np.testing.assert_allclose(X_std[:, 1], np.zeros(3))
    assert float(np.mean(X_std[:, 0])) == pytest.approx(0.0)

    minmax = FeatureScaler("minmax")
    X_mm = minmax.fit_transform(X, ["a", "b"])
    np.testing.assert_allclose(X_mm[:, 0], np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(X_mm[:, 1], np.zeros(3))


def test_feature_scaler_raises_for_invalid_method_and_transform_before_fit():
    with pytest.raises(ValueError, match="method must be 'standard' or 'minmax'"):
        FeatureScaler("robust")

    scaler = FeatureScaler("standard")
    with pytest.raises(RuntimeError, match="Scaler has not been fitted yet"):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_targets_generate_expected_values():
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    df = pd.DataFrame({"Close": [100.0, 110.0, 99.0]}, index=idx)

    future_target = FutureReturn(periods=1, scale=1.0).generate(df)
    directional_target = DirectionalTarget(periods=1).generate(df)

    expected_future = np.array(
        [np.tanh(np.log(110.0 / 100.0)), np.tanh(np.log(99.0 / 110.0))]
    )
    np.testing.assert_allclose(future_target.iloc[:2].to_numpy(), expected_future)
    assert np.isnan(future_target.iloc[2])
    np.testing.assert_allclose(
        directional_target.iloc[:2].to_numpy(), np.array([1.0, -1.0])
    )
    assert np.isnan(directional_target.iloc[2])


def test_walk_forward_split_expanding_and_sliding_and_errors():
    expanding = WalkForwardSplit(
        n_splits=2, initial_train_ratio=0.6, expanding=True
    ).split(10)
    np.testing.assert_array_equal(expanding[0].train_idx, np.arange(0, 6))
    np.testing.assert_array_equal(expanding[0].test_idx, np.arange(6, 8))
    np.testing.assert_array_equal(expanding[1].train_idx, np.arange(0, 8))
    np.testing.assert_array_equal(expanding[1].test_idx, np.arange(8, 10))

    sliding = WalkForwardSplit(
        n_splits=2, initial_train_ratio=0.6, expanding=False
    ).split(10)
    np.testing.assert_array_equal(sliding[0].train_idx, np.arange(0, 6))
    np.testing.assert_array_equal(sliding[1].train_idx, np.arange(2, 8))

    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        WalkForwardSplit(n_splits=0)
    with pytest.raises(ValueError, match="initial_train_ratio must be in"):
        WalkForwardSplit(initial_train_ratio=1.0)
    with pytest.raises(ValueError, match="Not enough samples"):
        WalkForwardSplit(n_splits=10, initial_train_ratio=0.8).split(5)


def test_keras_wrapper_predict_save_and_load(monkeypatch, tmp_path: Path):
    class _Model:
        def __init__(self):
            self.saved_to = None
            self.last_X = None

        def predict(self, X, verbose=0):
            self.last_X = X
            return np.array([[0.25], [-0.5]])

        def save(self, path):
            self.saved_to = Path(path)
            self.saved_to.write_text("dummy-model", encoding="utf-8")

    model = _Model()
    wrapper = KerasModelWrapper(model, ["f1", "f2"])

    preds = wrapper.predict(pd.DataFrame({"f1": [1, 2], "f2": [3, 4]}))
    np.testing.assert_allclose(preds, np.array([0.25, -0.5]))
    assert model.last_X.shape == (2, 2)

    out_dir = tmp_path / "saved_model"
    wrapper.save(str(out_dir))
    assert (out_dir / "model.keras").exists()
    assert json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))[
        "input_names"
    ] == ["f1", "f2"]

    fake_loaded_model = object()
    fake_keras = types.ModuleType("keras")
    fake_keras.models = types.SimpleNamespace(load_model=lambda _: fake_loaded_model)
    monkeypatch.setitem(sys.modules, "keras", fake_keras)

    loaded = KerasModelWrapper.load(str(out_dir))
    assert loaded.model is fake_loaded_model
    assert loaded.input_names == ["f1", "f2"]


def test_dense_model_builder_uses_expected_keras_calls(monkeypatch):
    calls = _install_fake_keras_for_builders(monkeypatch)

    builder = dense_model(layers=[4, 2], dropout=0.1, learning_rate=0.005)
    model = builder(input_dim=3)

    assert calls["input_shapes"] == [(3,)]
    assert [d["units"] for d in calls["dense_inits"]] == [4, 2, 1]
    assert [d["activation"] for d in calls["dense_inits"]] == ["relu", "relu", "tanh"]
    assert calls["dropout_rates"] == [0.1, 0.1]
    assert calls["adam_lrs"] == [0.005]
    assert callable(calls["compile"][0]["loss"])
    assert calls["compile"][0]["loss"].__name__ == "directional_loss"
    assert model.compiled is not None


def test_ml_trainer_build_dataset_and_collect_feature_columns():
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    df = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105]}, index=idx)
    indicator = _FeatureIndicator("alpha", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    trainer = MLTrainer(
        indicators=[indicator],
        target=_SimpleTarget(),
        model_builder=lambda input_dim: None,
        scaler=None,
    )

    X, y, feature_columns, out_idx = trainer.build_dataset(df)

    assert feature_columns == ["indicator__alpha"]
    assert X.shape[1] == 1
    assert len(X) == len(y) == len(out_idx)


def test_ml_trainer_train_uses_chronological_validation_split(monkeypatch):
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    feature_columns = ["f1", "f2"]
    index = pd.date_range("2026-01-01", periods=10, freq="D")

    class _Scaler:
        def __init__(self):
            self.calls = []

        def fit_transform(self, X_in, feature_names):
            self.calls.append((X_in.copy(), list(feature_names)))
            return X_in + 100.0

    class _Model:
        def __init__(self):
            self.fit_args = None

        def fit(
            self,
            X_train,
            y_train,
            validation_data=None,
            epochs=1,
            batch_size=32,
            verbose=0,
        ):
            self.fit_args = {
                "X_train": X_train,
                "y_train": y_train,
                "validation_data": validation_data,
                "epochs": epochs,
                "batch_size": batch_size,
                "verbose": verbose,
            }

    built_models = []

    def _builder(input_dim):
        model = _Model()
        model.input_dim = input_dim
        built_models.append(model)
        return model

    trainer = MLTrainer(
        indicators=[],
        target=_SimpleTarget(),
        model_builder=_builder,
        scaler=_Scaler(),
    )
    monkeypatch.setattr(
        trainer,
        "build_dataset",
        lambda _: (X.copy(), y.copy(), feature_columns, index),
    )

    wrapped = trainer.train(
        df=pd.DataFrame(),
        epochs=3,
        batch_size=4,
        validation_split=0.3,
        verbose=0,
    )

    model = built_models[0]
    assert model.input_dim == 2
    assert model.fit_args["X_train"].shape[0] == 7
    assert model.fit_args["validation_data"][0].shape[0] == 3
    assert model.fit_args["epochs"] == 3
    assert model.fit_args["batch_size"] == 4
    assert wrapped.input_names == feature_columns


def test_ml_trainer_walk_forward_returns_folded_results(monkeypatch):
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.linspace(-1.0, 1.0, 12)
    feature_columns = ["f1", "f2"]
    index = pd.date_range("2026-01-01", periods=12, freq="D")

    class _History:
        history = {"loss": [0.123]}

    class _Model:
        def fit(self, X_train, y_train, epochs=1, batch_size=32, verbose=0):
            self.last_fit = (X_train, y_train, epochs, batch_size, verbose)
            return _History()

        def predict(self, X_test, verbose=0):
            return np.full((len(X_test), 1), 0.5)

    def _builder(input_dim):
        model = _Model()
        model.input_dim = input_dim
        return model

    trainer = MLTrainer(
        indicators=[],
        target=_SimpleTarget(),
        model_builder=_builder,
        scaler=FeatureScaler("standard"),
    )
    monkeypatch.setattr(
        trainer,
        "build_dataset",
        lambda _: (X.copy(), y.copy(), feature_columns, index),
    )

    results = trainer.walk_forward(
        df=pd.DataFrame(),
        n_splits=2,
        initial_train_ratio=0.6,
        expanding=False,
        epochs=2,
        batch_size=5,
        verbose=0,
    )

    assert len(results) == 2
    assert [r.fold for r in results] == [0, 1]
    assert all(float(r.train_loss) == pytest.approx(0.123) for r in results)
    assert list(results[0].test_predictions.index) == [index[7], index[8]]
    assert list(results[1].test_predictions.index) == [index[9], index[10], index[11]]
    assert all(r.model.input_names == feature_columns for r in results)
