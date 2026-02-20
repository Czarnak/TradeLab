"""Strategy parameter optimisation module for trade_lab.

Uses Optuna (TPE sampler) to search for optimal indicator parameters and
indicator weights within a strategy. Supports single-process and parallel
multi-process runs (SQLite-backed storage).

Typical workflow
----------------
1. Define a **strategy factory** — a callable ``(params: dict) -> BaseStrategy``
   that builds a fresh strategy from a sampled parameter dict.

2. Define a **param space** — a list of ``IntParam``, ``FloatParam``, and/or
   ``CategoricalParam`` descriptors specifying what Optuna is allowed to tune.

3. Optionally prepare a **validation DataFrame** (out-of-sample data) to
   evaluate the best strategy after the search without influencing it.

4. Create an ``OptunaOptimizer`` and call ``optimize()``.

5. Inspect the returned ``OptimizationResult``: ``best_params``, ``best_value``,
   ``train_metrics``, ``val_metrics``, and the full ``trials_df``.

Example
-------
>>> import pandas as pd
>>> from trade_lab.backtesting import BacktestEngine
>>> from trade_lab.indicators import EMA, RSI
>>> from trade_lab.strategies import StandardStrategy
>>> from trade_lab.optimization import (
...     OptunaOptimizer, IntParam, FloatParam, CategoricalParam
... )
>>>
>>> # Fetch data once; split into train / validation
>>> engine = BacktestEngine(strategy=None, ticker='SPY',
...                         start='2015-01-01', end='2024-01-01')
>>> full_df = engine.fetch_data()
>>> train_df = full_df[:'2021-12-31']
>>> val_df   = full_df['2022-01-01':]
>>>
>>> def factory(params):
...     return StandardStrategy(
...         indicators=[
...             (EMA(period=params['fast']), params['w_fast']),
...             (EMA(period=params['slow']), params['w_slow']),
...             (RSI(period=params['rsi']),  params['w_rsi']),
...         ],
...         entry_threshold=params['entry_thr'],
...         exit_threshold=0.05,
...     )
>>>
>>> result = OptunaOptimizer(
...     strategy_factory=factory,
...     param_space=[
...         IntParam('fast',      5,   50),
...         IntParam('slow',      20,  200, step=5),
...         IntParam('rsi',       7,   28),
...         FloatParam('w_fast',  0.1, 3.0),
...         FloatParam('w_slow',  0.1, 3.0),
...         FloatParam('w_rsi',   0.1, 3.0),
...         FloatParam('entry_thr', 0.1, 0.8),
...     ],
...     train_df=train_df,
...     val_df=val_df,
...     metric='sharpe_ratio',
...     n_trials=300,
...     n_jobs=4,
... ).optimize()
>>>
>>> print(result.summary())
>>> print(result.trials_df.sort_values('value', ascending=False).head(10))
"""

from trade_lab.optimization.param_space import (
    IntParam,
    FloatParam,
    CategoricalParam,
    ParamSpace,
)
from trade_lab.optimization.objective import Objective, StrategyFactory
from trade_lab.optimization.optimizer import OptunaOptimizer
from trade_lab.optimization.result import OptimizationResult

__all__ = [
    'IntParam',
    'FloatParam',
    'CategoricalParam',
    'ParamSpace',
    'Objective',
    'StrategyFactory',
    'OptunaOptimizer',
    'OptimizationResult',
]