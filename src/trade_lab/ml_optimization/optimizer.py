"""Optuna-based ML indicator and model optimiser.

``MLOptimizer`` orchestrates the full optimisation workflow: Optuna study
creation, trial execution via ``MLObjective``, best-trial reconstruction
(retrain from scratch), and result compilation.

Parallelism and storage
-----------------------
Same pattern as ``OptunaOptimizer``: in-memory storage for single-process
runs, SQLite for ``n_jobs > 1``. The SQLite file is created automatically
under ``./optuna_studies/`` unless ``storage_path`` is provided.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import optuna
import pandas as pd

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.indicators.base import BaseIndicator
from trade_lab.ml_optimization.feature_builder import FeatureMatrix
from trade_lab.ml_optimization.objective import (
    MLObjective,
    ModelFactory,
    _deserialize_specs,
    _wrap_model,
)
from trade_lab.ml_optimization.search_space import IndicatorSpec
from trade_lab.strategies.ml_strategy import MLStrategy

if TYPE_CHECKING:
    from trade_lab.ml_optimization.result import MLOptimizationResult

optuna.logging.set_verbosity(optuna.logging.WARNING)

_MINIMIZE_METRICS = {
    "annual_volatility",
    "total_commission",
}


def _infer_direction(metric: str) -> str:
    return "minimize" if metric in _MINIMIZE_METRICS else "maximize"


class MLOptimizer:
    """Optimise indicator selection and Keras model via Optuna.

    Samples indicator configurations (period + lag per indicator) from the
    provided ``IndicatorSpec`` list, trains a Keras model per trial, evaluates
    via full backtest, and returns the best configuration with a retrained
    model.

    Parameters
    ----------
    indicator_specs : list[IndicatorSpec]
        Candidate indicators and their hyperparameter ranges.
    model_factory : ModelFactory
        Callable ``(n_features: int) -> keras.Model``. Must return a compiled
        Keras model.
    train_df : pd.DataFrame
        Training OHLCV data.
    val_df : pd.DataFrame
        Validation OHLCV data used for backtest evaluation during search.
    metric : str
        Backtest metric to optimise.
    test_df : pd.DataFrame | None
        Optional held-out test set. Evaluated post-search only.
    n_trials : int
        Number of Optuna trials.
    n_epochs : int
        Training epochs per trial (and for final retraining).
    n_jobs : int
        Parallel workers. Uses SQLite storage when > 1.
    initial_capital : float
        Starting capital for backtest engine.
    commission : float
        Proportional commission rate.
    slippage : float
        Proportional slippage rate.
    storage_path : str | None
        Path for SQLite storage. Auto-generated when ``None`` and
        ``n_jobs > 1``.
    """

    def __init__(
        self,
        indicator_specs: list[IndicatorSpec],
        model_factory: ModelFactory,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        metric: str,
        test_df: pd.DataFrame | None = None,
        n_trials: int = 50,
        n_epochs: int = 20,
        n_jobs: int = 1,
        initial_capital: float = 10_000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
        storage_path: str | None = None,
    ) -> None:
        self.indicator_specs = indicator_specs
        self.model_factory = model_factory
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.metric = metric
        self.n_trials = n_trials
        self.n_epochs = n_epochs
        self.n_jobs = n_jobs
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.storage_path = storage_path
        self.direction = _infer_direction(metric)

    def optimize(self) -> MLOptimizationResult:
        """Run the full optimisation and return the best result.

        Returns
        -------
        MLOptimizationResult
        """
        storage = self._resolve_storage()

        study = optuna.create_study(
            direction=self.direction,
            pruner=optuna.pruners.MedianPruner(),
            storage=storage,
            load_if_exists=True,
        )

        objective = MLObjective(
            indicator_specs=self.indicator_specs,
            model_factory=self.model_factory,
            train_df=self.train_df,
            val_df=self.val_df,
            metric=self.metric,
            n_epochs=self.n_epochs,
            engine_kwargs=self._engine_kwargs(),
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=False,
        )

        return self._build_result(study)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_storage(self) -> str | None:
        if self.n_jobs <= 1:
            return None
        if self.storage_path:
            return f"sqlite:///{self.storage_path}"
        os.makedirs("optuna_studies", exist_ok=True)
        ts = int(time.time())
        return f"sqlite:///optuna_studies/ml_study_{ts}.db"

    def _engine_kwargs(self) -> dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "commission": self.commission,
            "slippage": self.slippage,
        }

    def _build_result(self, study: optuna.Study) -> MLOptimizationResult:
        """Reconstruct and retrain the best trial from scratch.

        Retrains for the full ``n_epochs`` to ensure optimal weights (trial
        models may have been pruned early by ``KerasPruningCallback``).
        """
        from trade_lab.ml_optimization.result import MLOptimizationResult

        best_trial = study.best_trial
        best_value = best_trial.value

        # Retrieve stored spec
        raw_specs = best_trial.user_attrs["indicator_specs"]
        spec_tuples = _deserialize_specs(raw_specs)

        # Rebuild indicators with lag baked in
        indicators: list[BaseIndicator] = [
            cls(period=period, lag=lag) for cls, period, lag in spec_tuples
        ]

        # Rebuild FeatureMatrix and refit scaler on training data
        feature_matrix = FeatureMatrix(indicators)
        X_train, y_train = feature_matrix.build(self.train_df, fit_scaler=True)
        X_val, y_val = feature_matrix.build(self.val_df, fit_scaler=False)

        # Retrain from scratch with full epochs (no pruning callback)
        n_features = X_train.shape[1]
        model = self.model_factory(n_features)
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.n_epochs,
            verbose=0,
        )

        # Wrap model and build strategy. Fold the refitted scaler into the model
        # so the reported val/test metrics — and deployment — use raw features
        # consistently with how the model was trained.
        wrapped_model = _wrap_model(
            model, feature_matrix.feature_names, scaler=feature_matrix.scaler
        )
        strategy = MLStrategy(
            model=wrapped_model,
            indicators=indicators,
            allow_long=True,
            allow_short=True,
        )

        # Evaluate
        engine = BacktestEngine(strategy=strategy, **self._engine_kwargs())
        val_metrics = engine.run_on(self.val_df).metrics
        test_metrics = (
            engine.run_on(self.test_df).metrics if self.test_df is not None else None
        )

        trials_df = self._build_trials_df(study)
        completed = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        failed = sum(
            1
            for t in study.trials
            if t.state
            in (
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.PRUNED,
            )
        )

        return MLOptimizationResult(
            best_params=best_trial.params,
            best_value=best_value,
            metric=self.metric,
            direction=self.direction,
            trials_df=trials_df,
            study=study,
            best_model=wrapped_model,
            best_feature_spec=indicators,
            best_strategy=strategy,
            feature_names=feature_matrix.feature_names,
            scaler=feature_matrix.scaler,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            train_df=self.train_df,
            n_trials_completed=completed,
            n_trials_failed=failed,
        )

    @staticmethod
    def _build_trials_df(study: optuna.Study) -> pd.DataFrame:
        rows = []
        for t in study.trials:
            row = {"trial": t.number, "value": t.value, "state": t.state.name}
            row.update(t.params)
            if t.datetime_start and t.datetime_complete:
                row["duration_s"] = (
                    t.datetime_complete - t.datetime_start
                ).total_seconds()
            rows.append(row)
        return pd.DataFrame(rows)
