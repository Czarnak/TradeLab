from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import trade_lab.optimization as opt_api
import trade_lab.optimization.objective as opt_objective
import trade_lab.optimization.optimizer as opt_optimizer
from trade_lab.optimization.objective import Objective
from trade_lab.optimization.optimizer import OptunaOptimizer, _infer_direction
from trade_lab.optimization.param_space import (
    CategoricalParam,
    FloatParam,
    IntParam,
)
from trade_lab.optimization.result import OptimizationResult


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "Open": [10, 11, 12, 13, 14, 15],
            "High": [11, 12, 13, 14, 15, 16],
            "Low": [9, 10, 11, 12, 13, 14],
            "Close": [10, 11, 12, 13, 14, 15],
            "Volume": [100, 110, 120, 130, 140, 150],
        },
        index=idx,
    )


class _FakeTrial:
    def __init__(self):
        self.calls: list[tuple] = []

    def suggest_int(self, name, low, high, step=1):
        self.calls.append(("int", name, low, high, step))
        return low

    def suggest_float(self, name, low, high, log=False):
        self.calls.append(("float", name, low, high, log))
        return low

    def suggest_categorical(self, name, choices):
        self.calls.append(("cat", name, tuple(choices)))
        return choices[0]


class _FakeStrategy:
    pass


class _FakeEngine:
    def __init__(self, strategy, **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs

    def run_on(self, df):
        return SimpleNamespace(metrics={"score": 1.25, "sharpe_ratio": 2.0})


class _FakeEngineNoneMetric(_FakeEngine):
    def run_on(self, df):
        return SimpleNamespace(metrics={"score": None})


class _FakeEngineNanMetric(_FakeEngine):
    def run_on(self, df):
        return SimpleNamespace(metrics={"score": np.nan})


def _factory(params):
    return _FakeStrategy()


def test_optimization_package_exports_expected_api():
    assert "OptunaOptimizer" in opt_api.__all__
    assert "Objective" in opt_api.__all__
    assert "OptimizationResult" in opt_api.__all__
    assert hasattr(opt_api, "IntParam")
    assert hasattr(opt_api, "FloatParam")
    assert hasattr(opt_api, "CategoricalParam")


def test_categorical_param_requires_at_least_two_choices():
    with pytest.raises(ValueError, match="must have at least 2 choices"):
        CategoricalParam("mode", choices=[True])

    ok = CategoricalParam("mode", choices=[True, False])
    assert ok.choices == [True, False]


def test_objective_samples_params_for_all_descriptor_types():
    objective = Objective(
        strategy_factory=_factory,
        param_space=[
            IntParam("i", 1, 5, step=2),
            FloatParam("f", 0.1, 1.5, log=True),
            CategoricalParam("c", ["a", "b"]),
        ],
        train_df=_sample_df(),
        metric="score",
        engine_kwargs={"initial_capital": 10_000},
    )
    trial = _FakeTrial()

    params = objective._sample_params(trial)

    assert params == {"i": 1, "f": 0.1, "c": "a"}
    assert trial.calls == [
        ("int", "i", 1, 5, 2),
        ("float", "f", 0.1, 1.5, True),
        ("cat", "c", ("a", "b")),
    ]


def test_objective_sample_params_rejects_unknown_descriptor_type():
    @dataclass
    class UnknownParam:
        name: str = "x"

    objective = Objective(
        strategy_factory=_factory,
        param_space=[UnknownParam()],  # type: ignore[list-item]
        train_df=_sample_df(),
        metric="score",
        engine_kwargs={},
    )

    with pytest.raises(TypeError, match="Unknown param descriptor type"):
        objective._sample_params(_FakeTrial())


def test_objective_call_runs_backtest_and_returns_metric(monkeypatch):
    monkeypatch.setattr(opt_objective, "BacktestEngine", _FakeEngine)
    objective = Objective(
        strategy_factory=_factory,
        param_space=[IntParam("x", 1, 2)],
        train_df=_sample_df(),
        metric="score",
        engine_kwargs={"commission": 0.001},
    )

    value = objective(_FakeTrial())

    assert value == 1.25


def test_objective_call_prunes_when_metric_is_none(monkeypatch):
    monkeypatch.setattr(opt_objective, "BacktestEngine", _FakeEngineNoneMetric)
    objective = Objective(
        strategy_factory=_factory,
        param_space=[IntParam("x", 1, 2)],
        train_df=_sample_df(),
        metric="score",
        engine_kwargs={},
    )

    with pytest.raises(opt_objective.optuna.exceptions.TrialPruned, match="None or NaN"):
        objective(_FakeTrial())


def test_objective_call_prunes_when_metric_is_nan(monkeypatch):
    monkeypatch.setattr(opt_objective, "BacktestEngine", _FakeEngineNanMetric)
    objective = Objective(
        strategy_factory=_factory,
        param_space=[IntParam("x", 1, 2)],
        train_df=_sample_df(),
        metric="score",
        engine_kwargs={},
    )

    with pytest.raises(opt_objective.optuna.exceptions.TrialPruned, match="None or NaN"):
        objective(_FakeTrial())


def test_infer_direction_covers_minimize_and_maximize():
    assert _infer_direction("max_drawdown") == "minimize"
    assert _infer_direction("sharpe_ratio") == "maximize"


def test_optimizer_engine_kwargs_and_study_name(monkeypatch):
    monkeypatch.setattr(opt_optimizer.time, "time", lambda: 1234567890)
    optimizer = OptunaOptimizer(
        strategy_factory=_factory,
        param_space=[],
        train_df=_sample_df(),
        metric="sharpe_ratio",
        initial_capital=5_000,
        commission=0.002,
        slippage=0.001,
    )

    assert optimizer.study_name == "tradelab_sharpe_ratio_1234567890"
    assert optimizer._engine_kwargs() == {
        "initial_capital": 5_000,
        "commission": 0.002,
        "slippage": 0.001,
    }


def test_optimizer_build_storage_paths(monkeypatch, tmp_path):
    optimizer_single = OptunaOptimizer(
        strategy_factory=_factory,
        param_space=[],
        train_df=_sample_df(),
        n_jobs=1,
    )
    assert optimizer_single._build_storage() is None

    optimizer_custom = OptunaOptimizer(
        strategy_factory=_factory,
        param_space=[],
        train_df=_sample_df(),
        n_jobs=2,
        storage_path="custom.db",
    )
    assert optimizer_custom._build_storage() == "sqlite:///custom.db"

    monkeypatch.chdir(tmp_path)
    optimizer_default = OptunaOptimizer(
        strategy_factory=_factory,
        param_space=[],
        train_df=_sample_df(),
        n_jobs=2,
        study_name="study_x",
    )
    storage = optimizer_default._build_storage()
    assert storage is not None
    assert storage.startswith("sqlite:///")
    assert "optuna_studies" in storage
    assert storage.endswith("study_x.db")
    assert (tmp_path / "optuna_studies").exists()


def test_optimizer_optimize_wires_study_objective_and_returns_result(monkeypatch):
    captured = {}

    class FakeStudy:
        def optimize(self, objective, n_trials, n_jobs, catch, show_progress_bar):
            captured["objective"] = objective
            captured["n_trials"] = n_trials
            captured["n_jobs"] = n_jobs
            captured["catch"] = catch
            captured["show_progress_bar"] = show_progress_bar

    fake_study = FakeStudy()

    def fake_create_study(**kwargs):
        captured["create_study_kwargs"] = kwargs
        return fake_study

    monkeypatch.setattr(opt_optimizer.optuna, "create_study", fake_create_study)
    monkeypatch.setattr(
        OptunaOptimizer,
        "_build_result",
        lambda self, study: "RESULT",
    )

    optimizer = OptunaOptimizer(
        strategy_factory=_factory,
        param_space=[IntParam("x", 1, 2)],
        train_df=_sample_df(),
        metric="sharpe_ratio",
        n_trials=7,
        n_jobs=3,
        study_name="my_study",
        storage_path="db.sqlite",
    )

    result = optimizer.optimize()

    assert result == "RESULT"
    assert captured["create_study_kwargs"]["study_name"] == "my_study"
    assert captured["create_study_kwargs"]["direction"] == "maximize"
    assert captured["create_study_kwargs"]["storage"] == "sqlite:///db.sqlite"
    assert isinstance(captured["create_study_kwargs"]["sampler"], opt_optimizer.optuna.samplers.TPESampler)
    assert captured["create_study_kwargs"]["load_if_exists"] is True
    assert captured["n_trials"] == 7
    assert captured["n_jobs"] == 3
    assert captured["catch"] == (Exception,)
    assert captured["show_progress_bar"] is True
    assert isinstance(captured["objective"], Objective)


def test_optimizer_evaluate_uses_backtest_engine(monkeypatch):
    monkeypatch.setattr(opt_optimizer, "BacktestEngine", _FakeEngine)
    optimizer = OptunaOptimizer(
        strategy_factory=_factory,
        param_space=[],
        train_df=_sample_df(),
    )

    metrics = optimizer._evaluate({"x": 1}, _sample_df())

    assert metrics["score"] == 1.25
    assert metrics["sharpe_ratio"] == 2.0


def test_optimizer_build_trials_df_handles_empty_and_filled_trials():
    empty_study = SimpleNamespace(trials=[])
    empty = OptunaOptimizer._build_trials_df(empty_study)
    assert empty.empty

    trial_complete = SimpleNamespace(
        number=1,
        value=1.5,
        state=opt_optimizer.optuna.trial.TrialState.COMPLETE,
        duration=timedelta(seconds=3),
        params={"fast": 10},
    )
    trial_fail = SimpleNamespace(
        number=0,
        value=None,
        state=opt_optimizer.optuna.trial.TrialState.FAIL,
        duration=None,
        params={"fast": 5},
    )
    study = SimpleNamespace(trials=[trial_complete, trial_fail])

    df = OptunaOptimizer._build_trials_df(study)

    assert df["trial_number"].tolist() == [0, 1]
    assert df["state"].tolist() == ["FAIL", "COMPLETE"]
    assert np.isnan(df["duration_s"].iloc[0])
    assert df["duration_s"].iloc[1] == 3.0
    assert df["fast"].tolist() == [5, 10]


def test_optimizer_build_result_includes_validation_and_trial_counts(monkeypatch):
    optimizer = OptunaOptimizer(
        strategy_factory=_factory,
        param_space=[],
        train_df=_sample_df(),
        val_df=_sample_df(),
        metric="sharpe_ratio",
    )

    monkeypatch.setattr(
        optimizer,
        "_evaluate",
        lambda params, df: {"sharpe_ratio": 2.2, "total_return": 0.1},
    )
    monkeypatch.setattr(
        optimizer,
        "_build_trials_df",
        lambda study: pd.DataFrame({"trial_number": [0, 1, 2]}),
    )

    trials = [
        SimpleNamespace(state=opt_optimizer.optuna.trial.TrialState.COMPLETE),
        SimpleNamespace(state=opt_optimizer.optuna.trial.TrialState.FAIL),
        SimpleNamespace(state=opt_optimizer.optuna.trial.TrialState.PRUNED),
    ]
    study = SimpleNamespace(
        best_params={"a": 1},
        best_value=2.2,
        trials=trials,
    )

    result = optimizer._build_result(study)

    assert result.best_params == {"a": 1}
    assert result.best_value == 2.2
    assert result.train_metrics["sharpe_ratio"] == 2.2
    assert result.val_metrics is not None
    assert result.n_trials_completed == 1
    assert result.n_trials_failed == 2
    assert result.metric == "sharpe_ratio"


def test_optimizer_build_result_without_validation(monkeypatch):
    optimizer = OptunaOptimizer(
        strategy_factory=_factory,
        param_space=[],
        train_df=_sample_df(),
        val_df=None,
    )
    monkeypatch.setattr(optimizer, "_evaluate", lambda params, df: {"x": 1.0})
    monkeypatch.setattr(optimizer, "_build_trials_df", lambda study: pd.DataFrame())

    study = SimpleNamespace(
        best_params={"p": 1},
        best_value=1.0,
        trials=[],
    )

    result = optimizer._build_result(study)
    assert result.val_metrics is None


def test_optimization_result_summary_with_and_without_validation():
    study = SimpleNamespace()
    base = OptimizationResult(
        best_params={"fast": 10, "slow": 30},
        best_value=1.2345,
        metric="sharpe_ratio",
        direction="maximize",
        trials_df=pd.DataFrame(),
        study=study,
        train_metrics={"sharpe_ratio": 1.2345},
        val_metrics=None,
        n_trials_completed=8,
        n_trials_failed=2,
    )
    text = base.summary()
    assert "Optimisation Result" in text
    assert "Metric     : sharpe_ratio (maximize)" in text
    assert "Trials     : 8 completed, 2 failed" in text
    assert "fast" in text and "slow" in text
    assert "Validation sharpe_ratio" not in text

    with_val = OptimizationResult(
        best_params={"x": 1},
        best_value=0.5,
        metric="max_drawdown",
        direction="minimize",
        trials_df=pd.DataFrame(),
        study=study,
        train_metrics={"max_drawdown": -0.2},
        val_metrics={"max_drawdown": -0.3},
    )
    val_text = with_val.summary()
    assert "Validation max_drawdown" in val_text
