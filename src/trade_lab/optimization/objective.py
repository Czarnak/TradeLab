"""Optuna objective function for strategy optimisation.

The ``Objective`` class is a callable that Optuna passes an ``optuna.Trial``
object to on each evaluation. It samples a parameter dict from the trial,
calls the user-supplied strategy factory, runs a backtest, and returns the
chosen metric value as the scalar Optuna minimises or maximises.

Design note — why a class rather than a closure?
-------------------------------------------------
A callable class (``__call__``) is the Optuna convention and has two
advantages over a plain closure here:

1. Optuna can pickle and ship it to worker processes (n_jobs > 1). Closures
   that capture local variables are often not picklable, especially when they
   reference complex objects. A class with explicit ``__init__`` attributes is.

2. It keeps the objective's internal state (engine kwargs, param space, etc.)
   clearly visible rather than hidden in a closure's ``__closure__`` cells,
   which helps debugging.
"""

from __future__ import annotations

from typing import Callable, Any

import pandas as pd

import optuna

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.strategies.base import BaseStrategy
from trade_lab.optimization.param_space import (
    ParamSpace,
    IntParam,
    FloatParam,
    CategoricalParam,
)


# Type alias: a factory that accepts a sampled params dict and returns a strategy.
StrategyFactory = Callable[[dict[str, Any]], BaseStrategy]


class Objective:
    """Optuna-compatible objective that wraps a backtest.

    Parameters
    ----------
    strategy_factory : StrategyFactory
        Callable ``(params: dict) -> BaseStrategy``. Receives the sampled
        parameter dict for the current trial and must return a fully
        configured strategy instance. The factory is responsible for mapping
        param names to indicator constructor arguments, weights, thresholds,
        etc.
    param_space : ParamSpace
        List of ``IntParam``, ``FloatParam``, or ``CategoricalParam``
        descriptors defining the search space.
    train_df : pd.DataFrame
        OHLCV DataFrame used for every trial's backtest.
    metric : str
        Key of the metric to optimise (must be present in the dict returned
        by ``compute_metrics``).
    engine_kwargs : dict
        Additional keyword arguments forwarded to ``BacktestEngine``
        (``initial_capital``, ``commission``, ``slippage``).
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        param_space: ParamSpace,
        train_df: pd.DataFrame,
        metric: str,
        engine_kwargs: dict[str, Any],
    ) -> None:
        self.strategy_factory = strategy_factory
        self.param_space = param_space
        self.train_df = train_df
        self.metric = metric
        self.engine_kwargs = engine_kwargs

    def __call__(self, trial: optuna.Trial) -> float:
        """Sample params, build strategy, run backtest, return metric.

        Optuna expects the objective to:
        - Return a finite float when the trial succeeds.
        - Raise ``optuna.TrialPruned`` to signal early stopping (not used
          here at the strategy level — pruning is relevant for iterative ML
          training, not single-shot backtests).
        - Allow any other exception to propagate; Optuna will mark the trial
          as FAILED and continue.

        Parameters
        ----------
        trial : optuna.Trial
            Provided by Optuna on each call.

        Returns
        -------
        float
            The value of ``self.metric`` for the current parameter set.
        """
        params = self._sample_params(trial)

        strategy = self.strategy_factory(params)

        engine = BacktestEngine(
            strategy=strategy,
            **self.engine_kwargs,
        )

        result = engine.run_on(self.train_df)
        value = result.metrics.get(self.metric)

        if value is None or value != value:  # None or NaN
            # Return a very bad value rather than crashing; Optuna will
            # see this trial as a valid but poor result and steer away from it.
            raise optuna.exceptions.TrialPruned(
                f"Metric '{self.metric}' was None or NaN — trial skipped."
            )

        return float(value)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Translate the param space descriptors into Optuna suggestions.

        Each descriptor type maps to one Optuna ``trial.suggest_*`` call.
        The resulting dict uses the descriptor's ``name`` as the key —
        exactly the same key the strategy factory expects.

        Parameters
        ----------
        trial : optuna.Trial

        Returns
        -------
        dict[str, Any]
            Sampled parameter values keyed by param name.
        """
        params: dict[str, Any] = {}

        for param in self.param_space:
            if isinstance(param, IntParam):
                params[param.name] = trial.suggest_int(
                    param.name,
                    param.low,
                    param.high,
                    step=param.step,
                )
            elif isinstance(param, FloatParam):
                params[param.name] = trial.suggest_float(
                    param.name,
                    param.low,
                    param.high,
                    log=param.log,
                )
            elif isinstance(param, CategoricalParam):
                params[param.name] = trial.suggest_categorical(
                    param.name,
                    param.choices,
                )
            else:
                raise TypeError(
                    f"Unknown param descriptor type: {type(param).__name__}. "
                    "Expected IntParam, FloatParam, or CategoricalParam."
                )

        return params