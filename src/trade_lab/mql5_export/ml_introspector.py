"""ML strategy introspector — walks an MLStrategy object tree.

Produces an ``MLStrategyConfig`` dataclass that captures everything the Jinja2
ML template needs to render a complete MQL5 Expert Advisor forward pass without
any further Python-side logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trade_lab.mql5_export.introspector import (
    RiskConfig,
    SizingConfig,
    extract_risk_config,
)
from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.sizing.risk_based import RiskBasedPositionSizer
from trade_lab.strategies.ml_strategy import MLStrategy

# Keras container classes wrapping a nested sub-graph. Seed-ensemble ONNX
# exports call each trained member as a layer inside a shared functional
# graph, so members show up as one of these at the top level instead of a
# Dense layer — see ``ensemble_average_model`` in ``trade_lab.ml.models``.
_NESTED_MODEL_TYPES = ("Functional", "Sequential")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MLLayerConfig:
    """Structured representation of one Dense layer in the Keras model.

    Parameters
    ----------
    index : int
        Zero-based position in the Dense-only layer sequence (Dropout and
        InputLayer layers are excluded — they are inference no-ops or
        transparent to the forward pass).
    units_in : int
        Number of input features / previous-layer units.
    units_out : int
        Number of units (neurons) in this layer.
    activation : str
        Normalised activation name: ``'relu'``, ``'tanh'``, ``'linear'``,
        or ``'sigmoid'``.
    weights : list[list[float]]
        Kernel matrix as a Python list of shape ``[units_in][units_out]``.
        Stored row-major so the template can write
        ``LAYER{i}_W[units_in][units_out]`` directly.
    biases : list[float]
        Bias vector of length ``units_out``.
    """

    index: int
    units_in: int
    units_out: int
    activation: str
    weights: list[list[float]]
    biases: list[float]


@dataclass
class MLStrategyConfig:
    """Full structured configuration of an ``MLStrategy`` for MQL5 export.

    Deliberately mirrors the attribute names used by the standard
    ``StrategyConfig`` for fields that are shared with ``trade_logic.mq5.j2``
    (``allow_long``, ``allow_short``, ``entry_threshold``, ``exit_threshold``,
    ``sizing``).  Jinja2 accesses attributes by name, so the existing
    trade-logic sub-template reuses without modification.

    Parameters
    ----------
    layers : list[MLLayerConfig]
        Dense layers in forward-pass order (Dropout / InputLayer excluded).
    feature_names : list[str]
        Ordered list of input feature column names from
        ``KerasModelWrapper.input_names``.
    n_features : int
        ``len(feature_names)`` — convenience field for the template.
    entry_threshold : float
        From ``BaseStrategy.entry_threshold``.
    exit_threshold : float
        From ``BaseStrategy.exit_threshold``.
    allow_long : bool
        From ``BaseStrategy.allow_long``.
    allow_short : bool
        From ``BaseStrategy.allow_short``.
    sizing : SizingConfig
        Position sizer configuration (reuses existing dataclass).
    nested_summary : str | None
        One-line description of the model's nested/ensemble structure (e.g.
        ``"nested model: 3 sub-models, Average output"``), populated only
        when ``layers`` is empty because the model has no top-level Dense
        layer (a seed-ensemble ONNX export). ``None`` for ordinary flat
        Dense-layer models.
    """

    layers: list[MLLayerConfig]
    feature_names: list[str]
    n_features: int
    entry_threshold: float
    exit_threshold: float
    allow_long: bool
    allow_short: bool
    sizing: SizingConfig
    risk: RiskConfig = field(default_factory=RiskConfig)
    nested_summary: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_activation(raw: str | dict) -> str:
    """Return a lower-case activation name string from a Keras config value.

    Keras stores activations as plain strings (``'relu'``) in simple cases and
    as dicts (``{'class_name': 'ReLU', 'config': {...}}`` ) for the object
    form. We only care about the class name.
    """
    if isinstance(raw, dict):
        return raw.get("class_name", "linear").lower()
    return str(raw).lower()


def _describe_nested_model(keras_model: object) -> str | None:
    """Return a one-line description of a model with no top-level Dense layer.

    Used for seed-ensemble ONNX exports (``ensemble_average_model``):
    ``inputs -> [member(inputs) for member in members] -> Average``. The
    top-level layer list then holds an ``InputLayer``, one nested
    ``Functional``/``Sequential`` layer per member, and an ``Average`` layer
    — no top-level Dense to describe, so this reports the container count
    and whether the outputs are averaged instead.

    Returns ``None`` if the model has no nested sub-model layers either (an
    unusual, effectively empty model) — callers treat that as "nothing to
    report" rather than raising.
    """
    layers = list(getattr(keras_model, "layers", []))
    nested_containers = [
        layer for layer in layers if type(layer).__name__ in _NESTED_MODEL_TYPES
    ]
    if not nested_containers:
        return None

    has_average = any(type(layer).__name__ == "Average" for layer in layers)
    suffix = ", Average output" if has_average else ""
    return f"nested model: {len(nested_containers)} sub-models{suffix}"


def _extract_sizing(strategy: MLStrategy) -> SizingConfig:
    """Extract a ``SizingConfig`` from the strategy's position sizer."""
    sizer = strategy.position_sizer
    if sizer is None:
        return SizingConfig(sizer_type="none", params={})
    if isinstance(sizer, FixedPositionSizer):
        return SizingConfig(sizer_type="fixed", params={"fraction": sizer.fraction})
    if isinstance(sizer, RiskBasedPositionSizer):
        return SizingConfig(
            sizer_type="risk_based",
            params={
                "max_fraction": sizer.max_fraction,
                "risk_multiplier": sizer.risk_multiplier,
            },
        )
    # Fallback — validator should have caught unsupported types
    return SizingConfig(sizer_type="unknown", params={"class": type(sizer).__name__})


# ---------------------------------------------------------------------------
# Public introspector
# ---------------------------------------------------------------------------


class MLStrategyIntrospector:
    """Walks an ``MLStrategy`` object tree and returns an ``MLStrategyConfig``.

    Only Dense layers contribute to the forward pass. Dropout layers are
    ignored (they are identity at inference time). InputLayer and Concatenate
    layers are also skipped — they carry no learnable weights.
    """

    def introspect(self, strategy: MLStrategy) -> MLStrategyConfig:
        """Introspect ``strategy`` and return a fully populated ``MLStrategyConfig``.

        Parameters
        ----------
        strategy : MLStrategy
            Strategy to introspect. Must have passed ``validate_ml_strategy``
            (i.e. ``strategy.model`` is a ``KerasModelWrapper`` containing a
            Dense-only Keras model).

        Returns
        -------
        MLStrategyConfig
        """
        try:
            import keras
        except ImportError as exc:
            raise ImportError(
                "Keras is required for ML introspection. "
                "Install it with: pip install 'TradeLab[ml]'"
            ) from exc

        wrapper = strategy.model
        keras_model = wrapper.model
        feature_names: list[str] = list(wrapper.input_names)

        dense_layers: list[MLLayerConfig] = []
        dense_index = 0
        for layer in keras_model.layers:
            if not isinstance(layer, keras.layers.Dense):
                continue  # skip InputLayer, Dropout, Concatenate

            cfg = layer.get_config()
            units_out: int = cfg["units"]
            activation = _normalise_activation(cfg.get("activation", "linear"))

            ws = layer.get_weights()
            # ws[0] is the kernel of shape (units_in, units_out)
            # ws[1] is the bias of shape (units_out,)  — may be absent if use_bias=False
            kernel: np.ndarray = ws[0]
            bias: np.ndarray = ws[1] if len(ws) > 1 else np.zeros(units_out)

            units_in = kernel.shape[0]  # ground truth from actual weights

            dense_layers.append(
                MLLayerConfig(
                    index=dense_index,
                    units_in=units_in,
                    units_out=units_out,
                    activation=activation,
                    # .tolist() converts numpy arrays to plain Python lists,
                    # which Jinja2 can iterate without numpy being available
                    # in the template context.
                    weights=kernel.tolist(),
                    biases=bias.tolist(),
                )
            )

            dense_index += 1

        nested_summary = None
        if not dense_layers:
            nested_summary = _describe_nested_model(keras_model)

        return MLStrategyConfig(
            layers=dense_layers,
            feature_names=feature_names,
            n_features=len(feature_names),
            nested_summary=nested_summary,
            entry_threshold=strategy.entry_threshold,
            exit_threshold=strategy.exit_threshold,
            allow_long=strategy.allow_long,
            allow_short=strategy.allow_short,
            sizing=_extract_sizing(strategy),
            risk=extract_risk_config(strategy),
        )
