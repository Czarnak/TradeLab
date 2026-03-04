"""ML-based indicator selection and model optimisation.

This module provides Optuna-driven hyperparameter search over indicator
configurations (periods, lags, inclusion) combined with Keras model
training and backtest evaluation.

Lag is now a first-class property of every indicator and signal — the former
``LaggedIndicator`` wrapper has been removed.  Pass ``lag=k`` directly to any
indicator constructor instead.

Example workflow
----------------
>>> from trade_lab.indicators.moving_averages import EMA, SMA
>>> from trade_lab.ml_optimization import IndicatorSpec, MLOptimizer
>>>
>>> # 1. Define indicator search space
>>> specs = [
...     IndicatorSpec('ema', EMA, period_low=5, period_high=50,
...                   lag_values=[0, 1, 2, 5], optional=False),
...     IndicatorSpec('sma', SMA, period_low=10, period_high=100,
...                   lag_values=[0, 1, 3], optional=True),
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
>>> from trade_lab.ml_optimization import ModelPruner
>>> pruner = ModelPruner(percentile=20)
>>> pruned_result, report = pruner.prune_result(result)
>>> print(f"Zeroed {report['zero_fraction']:.1%} of weights")
>>> print(f"Dead features: {report['dead_features']}")
"""

from trade_lab.ml_optimization.feature_builder import FeatureMatrix
from trade_lab.ml_optimization.optimizer import MLOptimizer
from trade_lab.ml_optimization.pruning import ModelPruner
from trade_lab.ml_optimization.result import MLOptimizationResult
from trade_lab.ml_optimization.search_space import IndicatorSpec

__all__ = [
    "FeatureMatrix",
    "IndicatorSpec",
    "MLOptimizer",
    "MLOptimizationResult",
    "ModelPruner",
]
