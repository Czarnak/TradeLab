from __future__ import annotations

import copy
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import trade_lab.ml_optimization as mlopt_api
import trade_lab.ml_optimization.objective as ml_objective
import trade_lab.ml_optimization.optimizer as ml_optimizer
import trade_lab.ml_optimization.pruning as ml_pruning
from trade_lab.indicators.base import BaseIndicator
from trade_lab.ml_optimization.feature_builder import FeatureMatrix
from trade_lab.ml_optimization.objective import (
    MLObjective,
    _deserialize_specs,
    _serialize_specs,
)
from trade_lab.ml_optimization.optimizer import MLOptimizer, _infer_direction
from trade_lab.ml_optimization.result import MLOptimizationResult
from trade_lab.ml_optimization.search_space import IndicatorSpec


def _sample_df(n: int = 8) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": np.linspace(10, 10 + n - 1, n),
            "High": np.linspace(11, 11 + n - 1, n),
            "Low": np.linspace(9, 9 + n - 1, n),
            "Close": np.linspace(10, 12, n),
            "Volume": np.linspace(100, 100 + n - 1, n),
        },
        index=idx,
    )


class _DummyIndicator(BaseIndicator):
    def __init__(self, period: int = 2, lag: int = 0):
        super().__init__(lag=lag)
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.get_data(df.copy())
        out[self._raw_output_columns[0]] = out["Close"].rolling(self.period).mean()
        return out

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.zeros(len(df)), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None):
        return None

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__dummy_{self.period}"]


class _FakeTrial:
    def __init__(self):
        self._attrs = {}

    def suggest_categorical(self, name, choices):
        if name.endswith("__include"):
            return choices[0]
        return choices[0]

    def suggest_int(self, name, low, high, step=1):
        return low

    def set_user_attr(self, key, value):
        self._attrs[key] = value

    @property
    def user_attrs(self):
        return self._attrs


class _FakeModel:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.fit_calls = 0

    def fit(self, X, y, validation_data=None, epochs=1, callbacks=None, verbose=0):
        self.fit_calls += 1
        return None


def _fake_model_factory(n_features: int):
    return _FakeModel(n_features)


def _stub_optuna_integration(monkeypatch):
    import sys
    import types

    integration = types.ModuleType("optuna.integration")

    class _KerasPruningCallback:
        def __init__(self, *args, **kwargs):
            pass

    integration.KerasPruningCallback = _KerasPruningCallback
    monkeypatch.setitem(sys.modules, "optuna.integration", integration)


def _install_fake_keras(monkeypatch):
    import sys
    import types

    keras = types.ModuleType("keras")

    class Dense:
        def __init__(self, units=2, name="dense"):
            self.units = units
            self.name = name
            self._config = {"units": units, "name": name}
            self._weights = [np.array([[0.0]], dtype=float), np.array([0.0], dtype=float)]

        def get_config(self):
            return dict(self._config)

        def get_weights(self):
            return [w.copy() for w in self._weights]

        def set_weights(self, weights):
            self._weights = [np.array(w, dtype=float) for w in weights]

        @classmethod
        def from_config(cls, config):
            return cls(units=config.get("units", 2), name=config.get("name", "dense"))

        def __call__(self, x):
            return x

    class Concatenate:
        def __call__(self, inputs):
            return inputs

    class _Input:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    def Input(name, shape):
        return _Input(name=name, shape=shape)

    class Model:
        def __init__(self, inputs=None, outputs=None, layers=None):
            self.inputs = inputs or []
            self.outputs = outputs
            self.layers = layers or []
            self.input_names = [inp.name for inp in self.inputs] if self.inputs else []

        def get_weights(self):
            out = []
            for layer in self.layers:
                out.extend(layer.get_weights())
            return out

        def set_weights(self, weights):
            if not self.layers:
                return None
            i = 0
            for layer in self.layers:
                ws = layer.get_weights()
                n = len(ws)
                layer.set_weights(weights[i: i + n])
                i += n
            return None

        def compile(self, optimizer="adam", loss="mse"):
            return None

        def fit(self, X, y, epochs=1, verbose=0, validation_data=None, callbacks=None):
            return None

        def predict(self, X):
            return np.zeros(len(X))

    def clone_model(model):
        return copy.deepcopy(model)

    keras.layers = types.SimpleNamespace(Dense=Dense, Concatenate=Concatenate)
    keras.Input = Input
    keras.Model = Model
    keras.models = types.SimpleNamespace(clone_model=clone_model)

    monkeypatch.setitem(sys.modules, "keras", keras)
    return keras


def test_ml_optimization_public_api_exports():
    assert "MLOptimizer" in mlopt_api.__all__
    assert "ModelPruner" in mlopt_api.__all__
    assert "IndicatorSpec" in mlopt_api.__all__
    assert hasattr(mlopt_api, "FeatureMatrix")


def test_indicator_spec_validation_rules():
    with pytest.raises(ValueError, match="valid Python identifier"):
        IndicatorSpec("bad-name", _DummyIndicator, 2, 5, [0, 3])
    with pytest.raises(ValueError, match="period_low"):
        IndicatorSpec("x", _DummyIndicator, 5, 5, [0, 3])
    with pytest.raises(ValueError, match="lag_values must contain at least one value"):
        IndicatorSpec("x", _DummyIndicator, 2, 5, [])
    with pytest.raises(ValueError, match="All lag_values must be >= 0"):
        IndicatorSpec("x", _DummyIndicator, 2, 5, [0, -1])

    spec = IndicatorSpec("ema", _DummyIndicator, 2, 5, [0, 3], optional=False)
    assert spec.optional is False


def test_indicator_lag_and_feature_matrix_scaling_flow():
    indicator = _DummyIndicator(period=2, lag=2)
    out = indicator.compute(_sample_df())
    for col in indicator.output_columns:
        assert col in out.columns

    fm = FeatureMatrix([indicator])
    X_train, y_train = fm.build(_sample_df(10), fit_scaler=True)
    X_val, y_val = fm.build(_sample_df(10), fit_scaler=False)

    assert X_train.shape[0] > 0 and X_train.shape[1] == len(fm.feature_names)
    assert y_train.shape[0] == X_train.shape[0]
    assert y_val.shape[0] == X_val.shape[0]
    assert fm.scaler is not None


def test_serialize_and_deserialize_specs_round_trip():
    raw = _serialize_specs([(_DummyIndicator, 5, 2)])
    decoded = _deserialize_specs(raw)
    cls, period, lag = decoded[0]
    assert cls is _DummyIndicator
    assert period == 5
    assert lag == 2


def test_ml_objective_prunes_when_no_indicator_selected(monkeypatch):
    _stub_optuna_integration(monkeypatch)
    spec = IndicatorSpec("opt_ind", _DummyIndicator, 2, 5, [0, 1, 2], optional=True)
    obj = MLObjective(
        indicator_specs=[spec],
        model_factory=_fake_model_factory,
        train_df=_sample_df(),
        val_df=_sample_df(),
        metric="sharpe_ratio",
        n_epochs=1,
        engine_kwargs={},
    )

    class TrialNone(_FakeTrial):
        def suggest_categorical(self, name, choices):
            if name.endswith("__include"):
                return False
            return super().suggest_categorical(name, choices)

    with pytest.raises(ml_objective.optuna.TrialPruned, match="No indicators selected"):
        obj(TrialNone())


def test_ml_objective_prunes_when_no_training_samples(monkeypatch):
    _stub_optuna_integration(monkeypatch)
    spec = IndicatorSpec("req_ind", _DummyIndicator, 2, 5, [0, 1, 2], optional=False)
    obj = MLObjective(
        indicator_specs=[spec],
        model_factory=_fake_model_factory,
        train_df=_sample_df(),
        val_df=_sample_df(),
        metric="sharpe_ratio",
        n_epochs=1,
        engine_kwargs={},
    )

    class EmptyFeatureMatrix:
        def __init__(self, lagged_indicators):
            self.feature_names = ["f1"]

        def build(self, df, fit_scaler=False):
            return np.empty((0, 1)), np.empty((0,))

    monkeypatch.setattr(ml_objective, "FeatureMatrix", EmptyFeatureMatrix)
    with pytest.raises(ml_objective.optuna.TrialPruned, match="No training samples"):
        obj(_FakeTrial())


def test_ml_objective_happy_path_and_metric_nan_prune(monkeypatch):
    _stub_optuna_integration(monkeypatch)
    spec = IndicatorSpec("req_ind", _DummyIndicator, 2, 5, [0, 1, 2], optional=False)
    obj = MLObjective(
        indicator_specs=[spec],
        model_factory=_fake_model_factory,
        train_df=_sample_df(),
        val_df=_sample_df(),
        metric="sharpe_ratio",
        n_epochs=1,
        engine_kwargs={},
    )

    class FixedFeatureMatrix:
        def __init__(self, lagged_indicators):
            self.feature_names = ["a", "b"]

        def build(self, df, fit_scaler=False):
            return np.ones((4, 2)), np.ones((4,))

    monkeypatch.setattr(ml_objective, "FeatureMatrix", FixedFeatureMatrix)
    monkeypatch.setattr(ml_objective, "_wrap_model", lambda model, names: SimpleNamespace(input_names=names))

    class FakeEngine:
        def __init__(self, strategy, **kwargs):
            self.strategy = strategy

        def run_on(self, df):
            return SimpleNamespace(metrics={"sharpe_ratio": 1.5})

    monkeypatch.setattr(ml_objective, "BacktestEngine", FakeEngine)

    value = obj(_FakeTrial())
    assert value == 1.5

    class FakeEngineNan(FakeEngine):
        def run_on(self, df):
            return SimpleNamespace(metrics={"sharpe_ratio": np.nan})

    monkeypatch.setattr(ml_objective, "BacktestEngine", FakeEngineNan)
    with pytest.raises(ml_objective.optuna.TrialPruned, match="None or NaN"):
        obj(_FakeTrial())


def test_wrap_model_transfers_dense_weights(monkeypatch):
    keras = _install_fake_keras(monkeypatch)

    d1 = keras.layers.Dense(units=3, name="d1")
    d1.set_weights([np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.1, 0.2])])
    d2 = keras.layers.Dense(units=1, name="d2")
    d2.set_weights([np.array([[5.0], [6.0]]), np.array([0.3])])
    model = SimpleNamespace(layers=[d1, d2])

    wrapped = ml_objective._wrap_model(model, ["f1", "f2"])

    assert wrapped.input_names == ["f1", "f2"]


def test_ml_optimizer_direction_storage_and_optimize_wiring(monkeypatch, tmp_path):
    assert _infer_direction("annual_volatility") == "minimize"
    assert _infer_direction("sharpe_ratio") == "maximize"

    opt = MLOptimizer(
        indicator_specs=[IndicatorSpec("i", _DummyIndicator, 2, 5, [0, 1, 2], optional=False)],
        model_factory=_fake_model_factory,
        train_df=_sample_df(),
        val_df=_sample_df(),
        metric="sharpe_ratio",
        n_jobs=1,
        n_trials=2,
    )
    assert opt._resolve_storage() is None

    opt2 = MLOptimizer(
        indicator_specs=[IndicatorSpec("i", _DummyIndicator, 2, 5, [0, 1, 2], optional=False)],
        model_factory=_fake_model_factory,
        train_df=_sample_df(),
        val_df=_sample_df(),
        metric="sharpe_ratio",
        n_jobs=2,
        storage_path="ml_study.db",
    )
    monkeypatch.chdir(tmp_path)
    storage = opt2._resolve_storage()
    assert storage == "sqlite:///ml_study.db"

    captured = {}

    class FakeStudy:
        def optimize(self, objective, n_trials, n_jobs, show_progress_bar):
            captured["objective"] = objective
            captured["n_trials"] = n_trials
            captured["n_jobs"] = n_jobs
            captured["show_progress_bar"] = show_progress_bar

    def fake_create_study(**kwargs):
        captured["study_kwargs"] = kwargs
        return FakeStudy()

    monkeypatch.setattr(ml_optimizer.optuna, "create_study", fake_create_study)
    monkeypatch.setattr(MLOptimizer, "_build_result", lambda self, study: "DONE")

    out = opt.optimize()
    assert out == "DONE"
    assert captured["n_trials"] == 2
    assert captured["n_jobs"] == 1
    assert captured["show_progress_bar"] is False
    assert isinstance(captured["objective"], MLObjective)


def test_ml_optimizer_build_result_and_trials_df(monkeypatch):
    spec = IndicatorSpec("i", _DummyIndicator, 2, 5, [0, 1, 2], optional=False)
    optimizer = MLOptimizer(
        indicator_specs=[spec],
        model_factory=_fake_model_factory,
        train_df=_sample_df(),
        val_df=_sample_df(),
        metric="sharpe_ratio",
        test_df=_sample_df(),
        n_epochs=1,
    )

    monkeypatch.setattr(
        ml_optimizer,
        "_deserialize_specs",
        lambda raw: [(_DummyIndicator, 2, 1)],
    )
    monkeypatch.setattr(ml_optimizer, "_wrap_model", lambda model, names: SimpleNamespace(input_names=names, predict=lambda X: np.zeros(len(X))))

    class FakeEngine:
        def __init__(self, strategy, **kwargs):
            self.strategy = strategy

        def run_on(self, df):
            return SimpleNamespace(metrics={"sharpe_ratio": 0.9, "total_return": 0.1})

    monkeypatch.setattr(ml_optimizer, "BacktestEngine", FakeEngine)

    t0_start = datetime(2026, 1, 1, 0, 0, 0)
    t0_end = t0_start + timedelta(seconds=2)
    trial_complete = SimpleNamespace(
        number=0,
        value=1.23,
        state=ml_optimizer.optuna.trial.TrialState.COMPLETE,
        datetime_start=t0_start,
        datetime_complete=t0_end,
        params={"p": 1},
        user_attrs={"feature_names": ["a", "b"], "indicator_specs": "raw"},
    )
    trial_fail = SimpleNamespace(
        number=1,
        value=None,
        state=ml_optimizer.optuna.trial.TrialState.FAIL,
        datetime_start=None,
        datetime_complete=None,
        params={},
        user_attrs={},
    )
    study = SimpleNamespace(
        best_trial=trial_complete,
        trials=[trial_complete, trial_fail],
    )

    result = optimizer._build_result(study)
    assert isinstance(result, MLOptimizationResult)
    assert result.best_value == 1.23
    assert result.test_metrics is not None
    assert result.n_trials_completed == 1
    assert result.n_trials_failed == 1
    assert list(result.trials_df["trial"]) == [0, 1]

    empty = MLOptimizer._build_trials_df(SimpleNamespace(trials=[]))
    assert empty.empty


def test_ml_optimization_result_summary_contains_key_sections():
    ind = _DummyIndicator(period=2, lag=1)
    result = MLOptimizationResult(
        best_params={"x": 1},
        best_value=0.5,
        metric="sharpe_ratio",
        direction="maximize",
        trials_df=pd.DataFrame(),
        study=SimpleNamespace(),
        best_model=SimpleNamespace(),
        best_feature_spec=[ind],
        best_strategy=SimpleNamespace(),
        feature_names=["f1", "f2"],
        scaler=SimpleNamespace(),
        val_metrics={"sharpe_ratio": 0.5},
        test_metrics={"sharpe_ratio": 0.4},
        train_df=_sample_df(),
        n_trials_completed=3,
        n_trials_failed=1,
    )
    text = result.summary()
    assert "MLOptimizationResult" in text
    assert "Best indicators" in text
    assert "Validation metrics" in text
    assert "Test metrics" in text


def test_model_pruner_init_and_prune_model_threshold_percentile(monkeypatch):
    keras = _install_fake_keras(monkeypatch)

    with pytest.raises(ValueError, match="percentile must be in"):
        ml_pruning.ModelPruner(percentile=-0.1)
    with pytest.raises(ValueError, match="percentile must be in"):
        ml_pruning.ModelPruner(percentile=100.1)

    dense = keras.layers.Dense(name="dense_1")
    dense.set_weights([np.array([[0.01, 0.02], [0.5, 0.8]]), np.array([0.0, 0.0])])
    model = keras.Model(layers=[dense])

    pruner = ml_pruning.ModelPruner(percentile=50)
    pruned, report = pruner.prune_model(model, feature_names=["f1", "f2"])
    assert pruned is model
    assert report["zero_fraction"] > 0
    assert "f1" in report["dead_features"]
    assert "f2" in report["surviving_features"]

    pruner_pct = ml_pruning.ModelPruner(percentile=50, per_layer=True)
    _, report_pct = pruner_pct.prune_model(model, feature_names=["f1", "f2"])
    assert report_pct["zero_fraction"] > 0


def test_model_pruner_prune_result_updates_result(monkeypatch):
    keras = _install_fake_keras(monkeypatch)
    monkeypatch.setattr(ml_pruning, "_wrap_model", lambda model, names: SimpleNamespace(input_names=names, predict=lambda X: np.zeros(len(X))), raising=False)

    dense = keras.layers.Dense(name="dense_1")
    dense.set_weights([np.array([[0.0, 0.0], [0.2, 0.3]]), np.array([0.0, 0.0])])
    base_model = keras.Model(layers=[dense])

    li_a = _DummyIndicator(period=2)
    li_b = _DummyIndicator(period=3)
    result = MLOptimizationResult(
        best_params={"x": 1},
        best_value=0.5,
        metric="sharpe_ratio",
        direction="maximize",
        trials_df=pd.DataFrame(),
        study=SimpleNamespace(),
        best_model=base_model,
        best_feature_spec=[li_a, li_b],
        best_strategy=SimpleNamespace(),
        feature_names=[li_a.output_columns[0], li_b.output_columns[0]],
        scaler=SimpleNamespace(),
        val_metrics={"sharpe_ratio": 0.5},
        train_df=_sample_df(12),
    )

    pruner = ml_pruning.ModelPruner(percentile=10)

    def fake_prune_model(model, feature_names=None):
        report = {
            "total_weights": 4,
            "zeroed_weights": 2,
            "zero_fraction": 0.5,
            "layers": {"dense_1": {"total": 4, "zeroed": 2, "fraction": 0.5}},
            "dead_features": [li_a.output_columns[0]],
        }
        return model, report

    monkeypatch.setattr(pruner, "prune_model", fake_prune_model)

    updated, report = pruner.prune_result(result, fine_tune_epochs=1)
    assert report["zero_fraction"] == 0.5
    assert len(updated.best_feature_spec) == 1
    assert updated.feature_names == [li_b.output_columns[0]]
