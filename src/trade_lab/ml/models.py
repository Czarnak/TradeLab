from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


class KerasModelWrapper:
    """Bridges a Keras model with the MLStrategy interface.

    MLStrategy expects ``model.input_names`` (list of DataFrame column
    names) and ``model.predict(features_df) → 1D array``.  Raw Keras
    models don't satisfy this — their ``input_names`` returns layer
    names like ``'input_1'``.  This wrapper stores the actual feature
    column names and handles DataFrame→numpy conversion.
    """

    def __init__(self, model, input_names: list[str]):
        self.model = model
        self.input_names = input_names

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict signal strength from a feature DataFrame.

        Returns a 1-D array of values in [-1, 1] (tanh output).
        """
        X = features.to_numpy(dtype=np.float64)
        preds = self.model.predict(X, verbose=0)
        return preds.flatten()

    def save(self, path: str) -> None:
        """Persist model weights and feature metadata to a directory."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(out / "model.keras")
        with open(out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"input_names": self.input_names}, f)

    @classmethod
    def load(cls, path: str) -> KerasModelWrapper:
        """Load a previously saved model + metadata."""
        import keras  # deferred to keep module importable without tensorflow

        p = Path(path)
        model = keras.models.load_model(p / "model.keras")
        with open(p / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(model, meta["input_names"])


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------


def directional_loss(magnitude_weight: float = 0.1) -> Callable:
    """Return a Keras loss function that prioritises directional correctness.

    The loss has two components:

    1. **Directional component** — penalises predictions whose sign disagrees
       with the target.  A confident wrong prediction (e.g. predicting +0.9
       when target is negative) incurs higher penalty than a weak wrong
       prediction (e.g. predicting +0.1).  Correct-direction predictions
       contribute negative loss, pulling the total down.

       Formula:  -mean( sign(y_true) * y_pred )

    2. **MSE regularisation** — a small mean-squared-error term that keeps
       gradients smooth during early training when predictions are random,
       preventing the directional component from producing vanishing or
       exploding gradients.

       Formula:  mean( (y_true - y_pred)^2 )

    Combined:  directional_component + magnitude_weight * mse_component

    Parameters
    ----------
    magnitude_weight : float
        Weight of the MSE regularisation term.  Default 0.1 means MSE
        contributes 10% relative to the directional component.
        Set to 0.0 for pure directional loss (less stable).
        Set to 1.0 to weight both equally.

    Returns
    -------
    Callable
        A Keras-compatible loss function ``loss_fn(y_true, y_pred) -> Tensor``.

    Notes
    -----
    Compatible with both continuous targets (log returns) and tanh-scaled
    targets — direction is extracted via ``tf.sign(y_true)`` which works
    correctly for any non-zero real value.

    The returned loss function is named ``'directional_loss'`` for
    Keras history tracking.
    """

    def loss_fn(y_true, y_pred):
        import tensorflow as tf

        directional = -tf.reduce_mean(tf.sign(y_true) * y_pred)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return directional + magnitude_weight * mse

    loss_fn.__name__ = "directional_loss"
    return loss_fn


# ------------------------------------------------------------------
# Model builder factories
# ------------------------------------------------------------------
# Each factory returns a callable  (input_dim: int) → compiled Model.
# Walk-forward validation calls the builder once per fold so that
# each fold starts with freshly initialised weights.
# ------------------------------------------------------------------


def dense_model(
    layers: list[int] | None = None,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    magnitude_weight: float = 0.1,
) -> Callable:
    """Return a builder for a feed-forward Dense network.

    Architecture:
        Input → [Dense(units, relu) → Dropout] × N → Dense(1, tanh)

    Parameters
    ----------
    layers : list[int]
        Hidden layer sizes (default ``[64, 32]``).
    dropout : float
        Dropout rate between hidden layers.
    learning_rate : float
        Adam optimiser learning rate.
    magnitude_weight : float
        MSE regularisation weight passed to ``directional_loss``.
        Default 0.1 gives 10% MSE influence for gradient stability.
        Pass ``0.0`` for pure directional loss or a higher value for
        stronger MSE influence.
    """
    if layers is None:
        layers = [64, 32]

    def builder(input_dim: int):
        import keras

        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for units in layers:
            x = keras.layers.Dense(units, activation="relu")(x)
            x = keras.layers.Dropout(dropout)(x)
        output = keras.layers.Dense(1, activation="tanh")(x)

        model = keras.Model(inputs, output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=directional_loss(magnitude_weight=magnitude_weight),
        )
        return model

    return builder


def lstm_model(
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    magnitude_weight: float = 0.1,
) -> Callable:
    """Return a builder for an LSTM network.

    Architecture:
        Input(sequence_length, n_features) → LSTM → Dropout → Dense(1, tanh)

    The ``input_dim`` passed to the builder is the number of features.
    Sequence length is inferred from the training data shape.

    Parameters
    ----------
    units : int
        Number of LSTM units.
    dropout : float
        Dropout rate after LSTM layer.
    learning_rate : float
        Adam optimiser learning rate.
    magnitude_weight : float
        MSE regularisation weight passed to ``directional_loss``.
        Default 0.1 gives 10% MSE influence for gradient stability.
        Pass ``0.0`` for pure directional loss or a higher value for
        stronger MSE influence.
    """

    def builder(input_dim: int, sequence_length: int = 10):
        import keras

        inputs = keras.Input(shape=(sequence_length, input_dim))
        x = keras.layers.LSTM(units)(inputs)
        x = keras.layers.Dropout(dropout)(x)
        output = keras.layers.Dense(1, activation="tanh")(x)

        model = keras.Model(inputs, output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=directional_loss(magnitude_weight=magnitude_weight),
        )
        return model

    return builder
