"""Optuna-based strategy parameter optimiser.

``OptunaOptimizer`` is the main entry point for the optimisation module. It
wires together the param space, strategy factory, backtest engine, and Optuna
study into a single ``optimize()`` call.

Parallelism and storage
-----------------------
Optuna's default in-memory storage is not safe to share across processes.
When ``n_jobs > 1``, this class transparently switches to a SQLite-backed
``RDBStorage``. SQLite handles concurrent writes via file-level locking,
which is sufficient for local multi-core parallelism without any external
infrastructure.

The SQLite file path defaults to ``./optuna_studies/<study_name>.db`` and is
created automatically. Users can override the path via ``storage_path``.
The study name is derived from the metric and a timestamp by default, or
supplied explicitly — the name is needed so that multiple worker processes
can locate and write to the same shared study.

Optuna logging
--------------
Optuna emits INFO-level log lines for every trial by default, which is noisy
when running hundreds of trials. This class sets the Optuna log level to
WARNING at construction time. Users who want trial-level logs can call
``optuna.logging.set_verbosity(optuna.logging.INFO)`` after construction.
"""

from __future__ import annotations

import os
import time
from typing import Any

import optuna
import pandas as pd

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.optimization.objective import Objective, StrategyFactory
from trade_lab.optimization.param_space import ParamSpace
from trade_lab.optimization.result import OptimizationResult

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


class OptunaOptimizer:
    """Optimises strategy parameters using Optuna's TPE sampler.

    Parameters
    ----------
    strategy_factory : StrategyFactory
        Callable ``(params: dict) -> BaseStrategy``. Maps a sampled parameter
        dict to a fully configured strategy. The dict will contain exactly the
        keys defined in ``param_space``.

        Example::

            def factory(params):
                return StandardStrategy(
                    indicators=[
                        (EMA(period=params['fast']), params['weight_fast']),
                        (EMA(period=params['slow']), params['weight_slow']),
                    ],
                    entry_threshold=params['entry_threshold'],
                )

    param_space : ParamSpace
        List of ``IntParam``, ``FloatParam``, ``CategoricalParam`` descriptors.
    train_df : pd.DataFrame
        OHLCV DataFrame used for every trial. Typically a date-sliced portion
        of the full historical data (e.g. the first 70%).
    metric : str
        Metric to optimise. Must be a key returned by ``compute_metrics``.
        Defaults to ``'sharpe_ratio'``. Direction (maximise vs minimise) is
        inferred automatically from the metric name.
    val_df : pd.DataFrame | None
        Optional out-of-sample OHLCV DataFrame. If provided, the best
        strategy found on training data is also evaluated on this DataFrame
        after optimisation completes. Results are stored in
        ``OptimizationResult.val_metrics``. The optimizer never uses
        ``val_df`` during the search — it is strictly post-hoc.
    n_trials : int
        Number of Optuna trials to run. Each trial is one complete backtest.
        More trials → better coverage of the parameter space but longer
        runtime. 100–500 is a reasonable starting range for simple spaces;
        complex spaces may need 1000+.
    n_jobs : int
        Number of parallel worker processes. Defaults to 1 (sequential).
        When ``n_jobs > 1``, a SQLite-backed storage is used automatically.
        Each worker runs in its own process and pickles the ``Objective``;
        ensure your strategy factory and all objects it references are
        picklable.
    initial_capital : float
        Starting capital for every trial's backtest engine.
    commission : float
        Proportional commission rate for every trial's backtest engine.
    slippage : float
        Proportional slippage rate for every trial's backtest engine.
    study_name : str | None
        Name of the Optuna study. Auto-generated from metric and timestamp
        if not provided. Required for resuming a study from an existing
        SQLite file.
    storage_path : str | None
        Path to the SQLite ``.db`` file used when ``n_jobs > 1``.
        Defaults to ``./optuna_studies/<study_name>.db``.

    Examples
    --------
    >>> optimizer = OptunaOptimizer(
    ...     strategy_factory=factory,
    ...     param_space=[
    ...         IntParam('fast', 5, 50),
    ...         IntParam('slow', 20, 200, step=5),
    ...         FloatParam('weight_fast', 0.1, 3.0),
    ...         FloatParam('weight_slow', 0.1, 3.0),
    ...     ],
    ...     train_df=train_df,
    ...     val_df=val_df,
    ...     metric='sharpe_ratio',
    ...     n_trials=200,
    ...     n_jobs=4,
    ... )
    >>> result = optimizer.optimize()
    >>> print(result.summary())
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        param_space: ParamSpace,
        train_df: pd.DataFrame,
        metric: str = 'sharpe_ratio',
        val_df: pd.DataFrame | None = None,
        n_trials: int = 100,
        n_jobs: int = 1,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        study_name: str | None = None,
        storage_path: str | None = None,
    ) -> None:
        self.strategy_factory = strategy_factory
        self.param_space = param_space
        self.train_df = train_df
        self.metric = metric
        self.val_df = val_df
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        self.direction = _infer_direction(metric)

        # Study identity
        self.study_name = study_name or f"tradelab_{metric}_{int(time.time())}"
        self.storage_path = storage_path

    def optimize(self) -> OptimizationResult:
        """Run the optimisation and return results.

        Creates (or resumes) an Optuna study, runs ``n_trials`` evaluations,
        and returns a fully populated ``OptimizationResult``.

        Returns
        -------
        OptimizationResult
        """
        storage = self._build_storage()
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=storage,
            # load_if_exists allows resuming a study that was interrupted.
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(),
        )

        objective = Objective(
            strategy_factory=self.strategy_factory,
            param_space=self.param_space,
            train_df=self.train_df,
            metric=self.metric,
            engine_kwargs=self._engine_kwargs(),
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            # Catch all exceptions: failed trials are logged by Optuna and
            # the search continues. Without this, one bad param combination
            # (e.g. fast > slow period) would abort the whole run.
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
        """Return an Optuna storage URL or None (in-memory).

        In-memory storage (None) is used for single-process runs. It is fast
        and leaves no files on disk, but is not shareable across processes.

        SQLite storage is used when n_jobs > 1. The .db file is created in
        ``./optuna_studies/`` by default. The directory is created if absent.
        The storage URL format Optuna expects is ``sqlite:///path/to/file.db``.

        Returns
        -------
        str | None
            Optuna storage URL, or None for in-memory.
        """
        if self.n_jobs <= 1:
            return None

        if self.storage_path is not None:
            db_path = self.storage_path
        else:
            os.makedirs('optuna_studies', exist_ok=True)
            db_path = os.path.join('optuna_studies', f'{self.study_name}.db')

        return f'sqlite:///{db_path}'

    def _build_result(self, study: optuna.Study) -> OptimizationResult:
        """Compile the final OptimizationResult from a completed study.

        Steps:
        1. Extract best params and best value from the study.
        2. Re-run a clean backtest with best params on train_df to get all
           metrics (not just the optimised one).
        3. If val_df was provided, run the same backtest on val_df.
        4. Build the trials DataFrame from study.trials.
        5. Count completed vs failed trials.

        Parameters
        ----------
        study : optuna.Study
            Completed Optuna study.

        Returns
        -------
        OptimizationResult
        """
        best_params = study.best_params
        best_value = study.best_value

        # --- Full metrics on training data ---
        train_metrics = self._evaluate(best_params, self.train_df)

        # --- Full metrics on validation data (post-hoc, not used in search) ---
        val_metrics = (
            self._evaluate(best_params, self.val_df)
            if self.val_df is not None
            else None
        )

        # --- Trials DataFrame ---
        trials_df = self._build_trials_df(study)

        # --- Trial counts ---
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

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            metric=self.metric,
            direction=self.direction,
            trials_df=trials_df,
            study=study,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            n_trials_completed=completed,
            n_trials_failed=failed,
        )

    def _evaluate(self, params: dict[str, Any], df: pd.DataFrame) -> dict[str, float]:
        """Run one clean backtest and return all metrics.

        This is a separate call from the trial evaluations — those used the
        Objective wrapper which returns only a scalar. Here we want the full
        metrics dict for the final result report.

        Parameters
        ----------
        params : dict[str, Any]
            Parameter dict to pass to the strategy factory.
        df : pd.DataFrame
            Data to run the backtest on.

        Returns
        -------
        dict[str, float]
        """
        strategy = self.strategy_factory(params)
        engine = BacktestEngine(strategy=strategy, **self._engine_kwargs())
        result = engine.run_on(df)
        return result.metrics

    @staticmethod
    def _build_trials_df(study: optuna.Study) -> pd.DataFrame:
        """Flatten Optuna trial objects into a tidy DataFrame.

        Each completed or failed trial becomes one row. Columns are:
        - ``trial_number``  : Optuna trial index (0-based)
        - ``value``         : objective metric value (NaN for failed/pruned)
        - ``state``         : 'COMPLETE', 'FAIL', or 'PRUNED'
        - ``duration_s``    : wall-clock seconds for the trial
        - one column per parameter name

        Parameters
        ----------
        study : optuna.Study

        Returns
        -------
        pd.DataFrame
            One row per trial, sorted by trial number.
        """
        rows = []
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

        return pd.DataFrame(rows).sort_values('trial_number').reset_index(drop=True)
