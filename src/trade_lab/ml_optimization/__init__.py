"""ML-based indicator selection and model optimisation.

This module provides Optuna-driven hyperparameter search over indicator
configurations (periods, lags, inclusion) combined with Keras model
training and backtest evaluation.

Example workflow
----------------
>>> from trade_lab.indicators.moving_averages import EMA, SMA
>>> from trade_lab.ml_optimization import (
...     IndicatorSpec, MLOptimizer, ModelPruner,
... )
>>>
>>> # 1. Define indicator search space
>>> specs = [
...     IndicatorSpec('ema', EMA, period_low=5, period_high=50,
...                   lag_low=0, lag_high=10, max_lags=3, optional=False),
...     IndicatorSpec('sma', SMA, period_low=10, period_high=100,
...                   lag_low=0, lag_high=20, max_lags=5, optional=True),
... ]
>>>
>>> # 2. Define model factory
>>> def build_model(n_features: int):
...     import keras
...     model = keras.Sequential([
...         keras.layers.Dense(32, activation='relu',
...                            input_shape=(n_features,)),
...         keras.layers.Dense(16, activation='relu'),
...         keras.layers.Dense(1, activation='tanh'),
...     ])
...     model.compile(optimizer='adam', loss='mse')
...     return model
>>>
>>> # 3. Optimise
>>> optimizer = MLOptimizer(
...     indicator_specs=specs,
...     model_factory=build_model,
...     train_df=train_df,
...     val_df=val_df,
...     test_df=test_df,
...     metric='sharpe_ratio',
...     n_trials=50,
...     n_epochs=20,
... )
>>> result = optimizer.optimize()
>>> print(result.summary())
>>>
>>> # 4. Prune
>>> pruner = ModelPruner(percentile=20)
>>> pruned_result, report = pruner.prune_result(result)
>>> print(f"Zeroed {report['zero_fraction']:.1%} of weights")
>>> print(f"Dead features: {report['dead_features']}")
>>>
>>> # 5. Backtest pruned result
>>> from trade_lab.backtesting.engine import BacktestEngine
>>> engine = BacktestEngine(strategy=pruned_result.best_strategy)
>>> bt = engine.run_on(test_df)
>>> print(bt.metrics)
"""

from trade_lab.ml_optimization.feature_builder import FeatureMatrix, LaggedIndicator
from trade_lab.ml_optimization.optimizer import MLOptimizer
from trade_lab.ml_optimization.pruning import ModelPruner
from trade_lab.ml_optimization.result import MLOptimizationResult
from trade_lab.ml_optimization.search_space import IndicatorSpec

__all__ = [
    'LaggedIndicator',
    'FeatureMatrix',
    'IndicatorSpec',
    'MLOptimizer',
    'MLOptimizationResult',
    'ModelPruner',
]
