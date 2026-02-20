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

import numpy as np
import optuna
import pandas as pd

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.ml_optimization.feature_builder import FeatureMatrix, LaggedIndicator
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

# Suppress per-trial INFO logs — keep WARNING and above.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Metrics where a lower value is better. All others default to 'maximize'.
_MINIMIZE_METRICS = {
    'annual_volatility',
    'total_commission',
}


def _infer_direction(metric: str) -> str:
    """Infer optimisation direction from the metric name.

    Parameters
    ----------
    metric : str
        Metric key as returned by ``compute_metrics``.

    Returns
    -------
    str
        ``'minimize'`` or ``'maximize'``.
    """
    return 'minimize' if metric in _MINIMIZE_METRICS else 'maximize'


class MLOptimizer:
    """Optimise indicator selection and Keras model via Optuna.

    Samples indicator configurations (periods, lags, inclusion) from the
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
    study_name : str | None
        Optuna study name. Auto-generated if not provided.
    storage_path : str | None
        Path to SQLite ``.db`` file. Auto-generated if not provided.
    n_warmup_trials : int
        Number of warmup steps for the ``MedianPruner``.
    """

    def __init__(
        self,
        indicator_specs: list[IndicatorSpec],
        model_factory: ModelFactory,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        metric: str = 'sharpe_ratio',
        test_df: pd.DataFrame | None = None,
        n_trials: int = 100,
        n_epochs: int = 30,
        n_jobs: int = 1,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        study_name: str | None = None,
        storage_path: str | None = None,
        n_warmup_trials: int = 10,
    ) -> None:
        self.indicator_specs = indicator_specs
        self.model_factory = model_factory
        self.train_df = train_df
        self.val_df = val_df
        self.metric = metric
        self.test_df = test_df
        self.n_trials = n_trials
        self.n_epochs = n_epochs
        self.n_jobs = n_jobs
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.n_warmup_trials = n_warmup_trials

        self.direction = _infer_direction(metric)
        self.study_name = study_name or f"tradelab_ml_{metric}_{int(time.time())}"
        self.storage_path = storage_path

    def optimize(self) -> MLOptimizationResult:
        """Run the optimisation and return results.

        Creates an Optuna study, runs ``n_trials`` evaluations, then
        retrains the best configuration from scratch for the full
        ``n_epochs`` to obtain optimal weights.

        Returns
        -------
        MLOptimizationResult
        """
        from trade_lab.ml_optimization.result import MLOptimizationResult

        storage = self._build_storage()
        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=self.n_warmup_trials,
        )
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(),
            pruner=pruner,
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
            catch=(Exception,),
            show_progress_bar=True,
        )

        return self._build_result(study)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _engine_kwargs(self) -> dict[str, Any]:
        """Engine configuration shared across all trials."""
        return {
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            'slippage': self.slippage,
        }

    def _build_storage(self) -> str | None:
        """Return an Optuna storage URL or ``None`` (in-memory).

        Returns
        -------
        str | None
        """
        if self.n_jobs <= 1:
            return None

        if self.storage_path is not None:
            db_path = self.storage_path
        else:
            os.makedirs('optuna_studies', exist_ok=True)
            db_path = os.path.join('optuna_studies', f'{self.study_name}.db')

        return f'sqlite:///{db_path}'

    def _build_result(self, study: optuna.Study) -> MLOptimizationResult:
        """Reconstruct the best trial's full context and compile results.

        Retrains the best model from scratch for the full ``n_epochs`` to
        ensure optimal weights (the trial model may have been pruned early).

        Parameters
        ----------
        study : optuna.Study
            Completed Optuna study.

        Returns
        -------
        MLOptimizationResult
        """
        from trade_lab.ml_optimization.result import MLOptimizationResult

        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        # Retrieve stored artifacts
        feature_names = best_trial.user_attrs['feature_names']
        raw_specs = best_trial.user_attrs['lagged_indicator_specs']
        spec_tuples = _deserialize_specs(raw_specs)

        # Rebuild LaggedIndicator list
        lagged_indicators = [
            LaggedIndicator(cls(period=period), lags)
            for cls, period, lags in spec_tuples
        ]

        # Rebuild FeatureMatrix and refit scaler on training data
        feature_matrix = FeatureMatrix(lagged_indicators)
        X_train, y_train = feature_matrix.build(
            self.train_df, fit_scaler=True,
        )
        X_val, y_val = feature_matrix.build(self.val_df, fit_scaler=False)

        # Retrain model from scratch with full epochs (no pruning)
        n_features = X_train.shape[1]
        model = self.model_factory(n_features)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.n_epochs,
            verbose=0,
        )

        # Wrap model and build strategy
        wrapped_model = _wrap_model(model, feature_matrix.feature_names)
        strategy = MLStrategy(
            model=wrapped_model,
            indicators=[li.indicator for li in lagged_indicators],
            allow_long=True,
            allow_short=True,
        )

        # Evaluate on validation data
        engine = BacktestEngine(strategy=strategy, **self._engine_kwargs())
        val_result = engine.run_on(self.val_df)
        val_metrics = val_result.metrics

        # Evaluate on test data (if provided)
        test_metrics = None
        if self.test_df is not None:
            test_result = engine.run_on(self.test_df)
            test_metrics = test_result.metrics

        # Trials DataFrame
        trials_df = self._build_trials_df(study)

        # Trial counts
        completed = sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        failed = sum(
            1 for t in study.trials
            if t.state in (
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.PRUNED,
            )
        )

        return MLOptimizationResult(
            best_params=best_params,
            best_value=best_value,
            metric=self.metric,
            direction=self.direction,
            trials_df=trials_df,
            study=study,
            best_model=wrapped_model,
            best_feature_spec=lagged_indicators,
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
        """Flatten Optuna trial objects into a tidy DataFrame.

        Parameters
        ----------
        study : optuna.Study

        Returns
        -------
        pd.DataFrame
            One row per trial, sorted by trial number.
        """
        rows: list[dict[str, Any]] = []
        for trial in study.trials:
            row: dict[str, Any] = {
                'trial_number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                'duration_s': (
                    trial.duration.total_seconds()
                    if trial.duration is not None
                    else None
                ),
            }
            row.update(trial.params)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .sort_values('trial_number')
            .reset_index(drop=True)
        )
