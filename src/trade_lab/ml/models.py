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

    def __init__(self, model, input_names: list[str], scaler: object | None = None):
        self.model = model
        self.input_names = input_names
        # Optional feature scaler applied at inference. Normally ``None`` because
        # scaling is folded into the model's first Dense layer at training time
        # (see ``wrap_with_scaling``); the attribute is the fallback path for
        # models that cannot be folded into (e.g. non-Dense architectures).
        self.scaler = scaler

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict signal strength from a feature DataFrame.

        Returns a 1-D array of values in [-1, 1] (tanh output).
        """
        X = features.to_numpy(dtype=np.float64)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        preds = self.model.predict(X, verbose=0)
        return preds.flatten()

    def save(self, path: str) -> None:
        """Persist model weights and feature metadata to a directory."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(out / "model.keras")
        meta: dict = {"input_names": self.input_names}
        if self.scaler is not None:
            offset, scale = scaler_affine(self.scaler)
            meta["scaler"] = {"offset": offset.tolist(), "scale": scale.tolist()}
        with open(out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> KerasModelWrapper:
        """Load a previously saved model + metadata."""
        import keras  # deferred to keep module importable without tensorflow

        p = Path(path)
        model = keras.models.load_model(p / "model.keras")
        with open(p / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        scaler = None
        scaler_meta = meta.get("scaler")
        if scaler_meta is not None:
            scaler = AffineScaler(
                np.asarray(scaler_meta["offset"], dtype=np.float64),
                np.asarray(scaler_meta["scale"], dtype=np.float64),
            )
        return cls(model, meta["input_names"], scaler=scaler)


# ------------------------------------------------------------------
# Feature scaling — parity helpers
# ------------------------------------------------------------------
# Training fits a scaler on the features, but the deployed model (Python
# inference, the exported ONNX graph, and the hardcoded MQL5 forward pass) is
# fed *raw* features. To keep the two consistent we fold the scaler's per-feature
# affine map ``(x - offset) / scale`` directly into the first Dense layer's
# weights, making the model raw-feature-native. A Dense layer computes
# ``z = Wᵀx + b``; substituting the scaled input gives the exact equivalent
# ``W' = W / scale`` (row-wise) and ``b' = b - offsetᵀW'``.
# ------------------------------------------------------------------


class AffineScaler:
    """Minimal per-feature affine scaler: ``transform(X) == (X - offset) / scale``.

    Used to reconstruct a saved scaler on load without depending on the original
    scaler class (``FeatureScaler`` or sklearn ``StandardScaler``).
    """

    def __init__(self, offset: np.ndarray, scale: np.ndarray):
        self.offset = np.asarray(offset, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.offset) / self.scale

    def affine(self) -> tuple[np.ndarray, np.ndarray]:
        return self.offset, self.scale


def scaler_affine(scaler: object) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(offset, scale)`` from any supported scaler.

    Supports anything exposing ``affine()`` (``FeatureScaler``, ``AffineScaler``)
    and sklearn-style scalers exposing ``mean_`` / ``scale_``.
    """
    affine = getattr(scaler, "affine", None)
    if callable(affine):
        offset, scale = affine()
        return np.asarray(offset, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is not None and scale is not None:
        return np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    raise TypeError(
        f"Cannot extract affine parameters from scaler of type {type(scaler).__name__}"
    )


def fold_scaler_into_first_dense(model, offset: np.ndarray, scale: np.ndarray):
    """Fold ``(x - offset) / scale`` into ``model``'s first Dense layer, in place.

    After folding, ``model(raw_x)`` equals the original model evaluated on
    ``(raw_x - offset) / scale``. The model object is mutated and returned.

    Raises
    ------
    ValueError
        If the model has no Dense layer, the scaler dimension does not match the
        layer's input width, or the layer has no bias to absorb a non-zero
        offset.
    """
    import keras

    offset = np.asarray(offset, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)

    dense = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            dense = layer
            break
    if dense is None:
        raise ValueError("fold_scaler_into_first_dense: model has no Dense layer")

    weights = dense.get_weights()
    kernel = np.asarray(weights[0], dtype=np.float64)  # (units_in, units_out)
    if kernel.shape[0] != offset.shape[0]:
        raise ValueError(
            f"scaler dimension {offset.shape[0]} != first Dense input {kernel.shape[0]}"
        )

    kernel_folded = kernel / scale[:, None]
    if len(weights) > 1:
        bias = np.asarray(weights[1], dtype=np.float64)
        bias_folded = bias - offset @ kernel_folded
        new_weights = [
            kernel_folded.astype(weights[0].dtype),
            bias_folded.astype(weights[1].dtype),
        ]
    else:
        if np.any(offset != 0.0):
            raise ValueError(
                "Cannot fold a non-zero offset into a Dense layer without a bias "
                "(use_bias=False)."
            )
        new_weights = [kernel_folded.astype(weights[0].dtype)]

    dense.set_weights(new_weights)
    return model


def wrap_with_scaling(
    model, input_names: list[str], scaler: object | None
) -> KerasModelWrapper:
    """Wrap a trained model, baking the scaler into it for inference parity.

    When a ``scaler`` is given, its affine map is folded into the model's first
    Dense layer so the returned wrapper feeds *raw* features and the scaling
    rides along into ONNX / MQL5 export for free. If folding is not possible
    (e.g. the model exposes no foldable Dense layer, as in lightweight test
    doubles), the scaler is attached to the wrapper and applied at ``predict``
    time instead — both routes honour the scaler at inference.
    """
    if scaler is None:
        return KerasModelWrapper(model, input_names)
    try:
        offset, scale = scaler_affine(scaler)
        folded = fold_scaler_into_first_dense(model, offset, scale)
        return KerasModelWrapper(folded, input_names)
    except (AttributeError, TypeError, ValueError):
        return KerasModelWrapper(model, input_names, scaler=scaler)


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------


def directional_loss(magnitude_weight: float = 0.1) -> Callable:
    """Return a Keras loss function that prioritises directional correctness.

    The loss has two components:

    1. **Directional hinge** — penalises predictions whose sign *disagrees* with
       the target, by the magnitude of the wrong-side prediction.  A confident
       wrong prediction (e.g. predicting +0.9 when the target is negative)
       incurs a larger penalty than a weak wrong prediction (e.g. +0.1).
       Correctly-signed predictions contribute **zero** — the term neither
       rewards nor pushes them further, so it cannot drag the prediction toward
       the ``±1`` saturation boundary.

       Formula:  mean( relu( -sign(y_true) * y_pred ) )

    2. **MSE term** — drives magnitude: once the sign is right, this is the only
       active term, so the prediction is pulled toward the target *value* rather
       than toward saturation.

       Formula:  mean( (y_true - y_pred)^2 )

    Combined:  directional_hinge + magnitude_weight * mse

    Why a hinge?  The earlier formulation used a linear reward
    ``-mean(sign(y) * y_pred)`` whose gradient w.r.t. ``y_pred`` is a constant
    ``±1`` regardless of how correct the prediction already is.  That constant
    push drove the tanh head to ``±1`` for every sample (observed holdout range
    ``[0.995, 1.000]``), collapsing the graded ``signal_strength`` that position
    sizing and entry/exit thresholds depend on.  The hinge removes the reward
    once the sign is correct, so magnitude is preserved.

    Parameters
    ----------
    magnitude_weight : float
        Weight of the MSE term relative to the directional hinge.  Default
        ``0.1`` keeps the directional hinge dominant for wrong-sign samples
        while still anchoring magnitude for correctly-signed ones.  Raise it
        (e.g. ``1.0``) to weight magnitude fidelity more strongly.  Values
        ``<= 0`` leave magnitude unconstrained for correctly-signed predictions
        and are not recommended.

    Returns
    -------
    Callable
        A Keras-compatible loss function ``loss_fn(y_true, y_pred) -> Tensor``.

    Notes
    -----
    Compatible with both continuous targets (log returns) and tanh-scaled
    targets — direction is extracted via ``tf.sign(y_true)`` which works
    correctly for any non-zero real value.  For hard directional targets
    (``y in {-1, +1}``) the MSE term legitimately pulls the prediction to the
    matching ``±1``; that is the target value, not loss-induced saturation.

    The returned loss function is named ``'directional_loss'`` for
    Keras history tracking.
    """

    def loss_fn(y_true, y_pred):
        import tensorflow as tf

        directional = tf.reduce_mean(tf.nn.relu(-tf.sign(y_true) * y_pred))
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
        Weight of the MSE term relative to the directional hinge, passed to
        ``directional_loss``.  Default 0.1 keeps direction dominant while still
        anchoring magnitude; raise it (e.g. 1.0) to weight magnitude fidelity
        more strongly.  Values ``<= 0`` leave magnitude unconstrained for
        correctly-signed predictions and are not recommended.
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
