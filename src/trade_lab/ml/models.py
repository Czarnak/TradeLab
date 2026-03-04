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
            loss="mse",
        )
        return model

    return builder


def lstm_model(
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
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
            loss="mse",
        )
        return model

    return builder
