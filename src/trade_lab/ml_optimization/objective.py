"""Optuna objective function for ML-based indicator optimisation.

``MLObjective`` is a picklable callable that samples indicator configurations,
builds feature matrices, trains a Keras model, and evaluates the result via a
full backtest — all within a single Optuna trial.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.indicators.base import BaseIndicator
from trade_lab.ml_optimization.feature_builder import FeatureMatrix
from trade_lab.ml_optimization.search_space import IndicatorSpec
from trade_lab.strategies.ml_strategy import MLStrategy

#: Type alias for the user-supplied model constructor.
#: Takes ``n_features: int`` and returns a compiled ``keras.Model``.
ModelFactory = Callable[[int], Any]


def _wrap_model(model: Any, feature_names: list[str]) -> Any:
    """Rebuild a trained Keras model with named ``Input`` layers.

    ``MLStrategy`` resolves feature columns via ``model.input_names``. A
    standard Sequential/Functional model built by the user typically lacks
    named inputs that match DataFrame columns. This helper:

    1. Extracts trained weights from each Dense layer.
    2. Builds a new Functional API model with one ``keras.Input(name=…,
       shape=(1,))`` per feature.
    3. Concatenates inputs and replays the original Dense stack, transferring
       weights.

    Parameters
    ----------
    model : keras.Model
        Trained model whose weights will be transferred.
    feature_names : list[str]
        Feature names that become the ``Input`` layer names.

    Returns
    -------
    keras.Model
        New Functional model with ``model.input_names == feature_names``.
    """
    import keras

    dense_info: list[tuple[dict, list[np.ndarray]]] = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            config = layer.get_config()
            dense_info.append((config, layer.get_weights()))

    inputs = [keras.Input(name=name, shape=(1,)) for name in feature_names]
    x = inputs[0] if len(inputs) == 1 else keras.layers.Concatenate()(inputs)

    for config, weights in dense_info:
        layer = keras.layers.Dense.from_config(config)
        x = layer(x)
        layer.set_weights(weights)

    return keras.Model(inputs=inputs, outputs=x)


def _serialize_specs(specs: list[tuple[type, int, int]]) -> str:
    """Serialize indicator specs for storage in Optuna user attrs.

    Parameters
    ----------
    specs : list[tuple[type, int, int]]
        Each entry is ``(indicator_class, period, lag)``.

    Returns
    -------
    str
        JSON string safe for ``trial.set_user_attr``.
    """
    serializable = [
        {
            "class_module": cls.__module__,
            "class_name": cls.__name__,
            "period": period,
            "lag": lag,
        }
        for cls, period, lag in specs
    ]
    return json.dumps(serializable)


def _deserialize_specs(raw: str) -> list[tuple[type, int, int]]:
    """Reconstruct indicator specs from Optuna user attrs.

    Parameters
    ----------
    raw : str
        JSON string produced by ``_serialize_specs``.

    Returns
    -------
    list[tuple[type, int, int]]
        Each entry is ``(indicator_class, period, lag)``.
    """
    import importlib

    entries = json.loads(raw)
    result: list[tuple[type, int, int]] = []
    for entry in entries:
        module = importlib.import_module(entry["class_module"])
        cls = getattr(module, entry["class_name"])
        result.append((cls, entry["period"], entry["lag"]))
    return result


class MLObjective:
    """Optuna objective for ML indicator optimisation.

    Each ``__call__`` invocation corresponds to one Optuna trial: sample
    indicator configuration (period + lag per indicator), build features,
    train a Keras model, run a backtest, and return the chosen metric.

    The instance is fully picklable for ``n_jobs > 1`` parallel studies.

    Parameters
    ----------
    indicator_specs : list[IndicatorSpec]
        Search space of candidate indicators.
    model_factory : ModelFactory
        Callable ``(n_features: int) -> keras.Model``. Must return a
        compiled model ready for ``.fit()``.
    train_df : pd.DataFrame
        Training OHLCV data.
    val_df : pd.DataFrame
        Validation OHLCV data for backtest evaluation.
    metric : str
        Backtest metric to optimise (e.g. ``'sharpe_ratio'``).
    n_epochs : int
        Number of training epochs per trial.
    engine_kwargs : dict[str, Any]
        Keyword arguments forwarded to ``BacktestEngine``.
    """

    def __init__(
        self,
        indicator_specs: list[IndicatorSpec],
        model_factory: ModelFactory,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        metric: str,
        n_epochs: int,
        engine_kwargs: dict[str, Any],
    ) -> None:
        self.indicator_specs = indicator_specs
        self.model_factory = model_factory
        self.train_df = train_df
        self.val_df = val_df
        self.metric = metric
        self.n_epochs = n_epochs
        self.engine_kwargs = engine_kwargs

    def __call__(self, trial: optuna.Trial) -> float:
        """Execute a single optimisation trial.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object for parameter sampling.

        Returns
        -------
        float
            Value of the target metric on the validation backtest.

        Raises
        ------
        optuna.TrialPruned
            If no indicators are selected, training data is empty after NaN
            dropping, or the backtest metric is unavailable.
        """
        from optuna.integration import KerasPruningCallback

        # Step 1 — Sample indicator configuration
        indicators: list[BaseIndicator] = []
        included_specs: list[tuple[type, int, int]] = []

        for spec in self.indicator_specs:
            if spec.optional:
                include = trial.suggest_categorical(
                    f"{spec.name}__include",
                    [True, False],
                )
            else:
                include = True

            if include:
                period = trial.suggest_int(
                    f"{spec.name}__period",
                    spec.period_low,
                    spec.period_high,
                )
                # Single lag sampled from the candidate list
                lag = trial.suggest_categorical(
                    f"{spec.name}__lag",
                    spec.lag_values,
                )
                indicator = spec.indicator_class(period=period, lag=lag)
                indicators.append(indicator)
                included_specs.append((spec.indicator_class, period, lag))

        # Step 2 — Guard: at least one indicator required
        if not indicators:
            raise optuna.TrialPruned("No indicators selected.")

        # Step 3 — Build feature matrices
        feature_matrix = FeatureMatrix(indicators)
        X_train, y_train = feature_matrix.build(self.train_df, fit_scaler=True)
        X_val, y_val_arr = feature_matrix.build(self.val_df, fit_scaler=False)

        if X_train.shape[0] == 0:
            raise optuna.TrialPruned("No training samples after NaN dropping.")

        # Step 4 — Build and train model
        n_features = X_train.shape[1]
        model = self.model_factory(n_features)

        callbacks = [KerasPruningCallback(trial, monitor="val_loss")]
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val_arr),
            epochs=self.n_epochs,
            callbacks=callbacks,
            verbose=0,
        )

        # Step 5 — Evaluate via backtest
        wrapped_model = _wrap_model(model, feature_matrix.feature_names)
        strategy = MLStrategy(
            model=wrapped_model,
            indicators=indicators,
            allow_long=True,
            allow_short=True,
        )
        engine = BacktestEngine(strategy=strategy, **self.engine_kwargs)
        result = engine.run_on(self.val_df)
        value = result.metrics.get(self.metric)

        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise optuna.TrialPruned(f"Metric {self.metric!r} is None or NaN.")

        # Store artifacts for best-trial reconstruction
        trial.set_user_attr("feature_names", feature_matrix.feature_names)
        trial.set_user_attr("indicator_specs", _serialize_specs(included_specs))

        return float(value)
