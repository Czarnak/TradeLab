"""Pre-export validation for ML-based MQL5 code generation.

Checks an MLStrategy object for compatibility with the ML MQL5 exporter
before any code is generated. Reuses ``ValidationResult`` from the standard
validator — the schema is identical, only the checks differ.

Two validators live here:

``validate_ml_strategy``
    The *hardcoded-weights* export path (``export_ml_to_mql5``). The
    generated EA translates every layer into MQL5 source by hand, so the
    Keras model must be a flat Dense/Dropout/InputLayer/Concatenate stack
    with a single-unit final Dense layer.

``validate_ml_strategy_onnx``
    The *ONNX* export path (``export_ml_to_mql5_onnx``). The generated EA
    never inspects individual layers — it hands the whole graph to MT5's
    ``OnnxRun`` — so this validator accepts any architecture tf2onnx can
    convert, including seed-ensemble graphs built from nested Functional/
    Sequential sub-models averaged with ``keras.layers.Average``.  It walks
    the layer tree recursively instead of inheriting the strict, flat
    whitelist used by ``validate_ml_strategy``.
"""

from __future__ import annotations

from trade_lab.mql5_export.validators import ValidationResult, collect_risk_warnings
from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.sizing.risk_based import RiskBasedPositionSizer
from trade_lab.strategies.ml_strategy import MLStrategy

_SUPPORTED_SIZERS = (type(None), FixedPositionSizer, RiskBasedPositionSizer)

# Keras layer class names that are safe to skip at inference time (Dropout
# is identity at inference) or to translate into MQL5 (Dense). Used by the
# strict, hardcoded-weights validator only.
_SUPPORTED_LAYER_TYPES = ("Dense", "Dropout", "InputLayer", "Concatenate")

# Activations we can render as MQL5 expressions.
_SUPPORTED_ACTIVATIONS = ("relu", "tanh", "linear", "sigmoid")

# Keras container classes that wrap a nested sub-graph (``.layers`` holds the
# children). Seed-ensemble exports call each trained member as a layer inside
# a shared functional graph, so each member shows up as one of these.
_ONNX_NESTED_MODEL_TYPES = ("Functional", "Sequential")

# Layer types accepted anywhere in the (recursively flattened) ONNX export
# graph. tf2onnx — not this validator — is responsible for turning them into
# ONNX ops; we only need to reject architectures we know are unsupported.
_ONNX_SUPPORTED_LAYER_TYPES = (
    "Dense",
    "Dropout",
    "InputLayer",
    "Concatenate",
    "Average",
) + _ONNX_NESTED_MODEL_TYPES


# ---------------------------------------------------------------------------
# Shared helpers — used by both validators so the messages stay in sync.
# ---------------------------------------------------------------------------


def _check_strategy_type(strategy: object) -> str | None:
    """Return an error message if ``strategy`` is not an ``MLStrategy``."""
    if isinstance(strategy, MLStrategy):
        return None
    return (
        f"Strategy must be MLStrategy, got {type(strategy).__name__}. "
        "StandardStrategy and other subclasses use export_to_mql5() instead."
    )


def _check_model_wrapper(strategy: MLStrategy) -> str | None:
    """Return an error message if ``strategy.model`` is not a ``KerasModelWrapper``."""
    from trade_lab.ml.models import KerasModelWrapper

    if isinstance(strategy.model, KerasModelWrapper):
        return None
    return (
        f"strategy.model must be a KerasModelWrapper, got "
        f"{type(strategy.model).__name__}. "
        "Wrap your Keras model with KerasModelWrapper(model, input_names) "
        "before exporting."
    )


def _check_position_sizer(strategy: MLStrategy) -> str | None:
    """Return an error message if the position sizer class is unsupported."""
    sizer = strategy.position_sizer
    if isinstance(sizer, _SUPPORTED_SIZERS):
        return None
    return (
        f"Unsupported position sizer '{type(sizer).__name__}'. "
        f"Supported: None, FixedPositionSizer, RiskBasedPositionSizer."
    )


def _check_input_names(wrapper: object) -> str | None:
    """Return an error message if ``wrapper.input_names`` is empty."""
    if wrapper.input_names:
        return None
    return (
        "KerasModelWrapper.input_names is empty. "
        "The exporter needs feature column names to generate EA inputs."
    )


def _normalise_dense_activation(layer: object) -> str:
    """Return the lower-cased activation name configured on a Dense layer."""
    cfg = layer.get_config()
    activation = cfg.get("activation", "linear")
    if isinstance(activation, dict):
        return activation.get("class_name", "linear").lower()
    return str(activation).lower()


def _check_dense_activation(layer: object) -> tuple[str, str | None]:
    """Return ``(activation, error_or_None)`` for one Dense layer.

    ``error_or_None`` is a fatal-error message if the activation is not one
    of ``_SUPPORTED_ACTIVATIONS``, else ``None``.
    """
    activation = _normalise_dense_activation(layer)
    if activation in _SUPPORTED_ACTIVATIONS:
        return activation, None
    return activation, (
        f"Unsupported activation '{activation}' in layer "
        f"'{layer.name}'. Supported: {', '.join(_SUPPORTED_ACTIVATIONS)}."
    )


def _dense_param_count(layer: object) -> int:
    """Return the total kernel + bias element count for one Dense layer."""
    weights = layer.get_weights()
    if not weights:
        return 0
    total = weights[0].size  # kernel
    if len(weights) > 1:
        total += weights[1].size  # bias
    return total


# ---------------------------------------------------------------------------
# Strict validator — hardcoded-weights export path
# ---------------------------------------------------------------------------


def validate_ml_strategy(strategy: object) -> ValidationResult:
    """Validate an ``MLStrategy`` for MQL5 export compatibility.

    Parameters
    ----------
    strategy : object
        The strategy to validate. Must be an ``MLStrategy`` instance whose
        ``model`` attribute is a ``KerasModelWrapper``.

    Returns
    -------
    ValidationResult
        Contains ``is_valid``, ``errors`` (fatal), and ``warnings`` (non-fatal).

    Notes
    -----
    Validation order:

    1. Strategy class check — must be ``MLStrategy``.
    2. Model wrapper check — ``strategy.model`` must be a ``KerasModelWrapper``.
    3. Layer type check — all layers must be Dense, Dropout, InputLayer, or
       Concatenate (Dropout and InputLayer are inference no-ops; Concatenate
       is handled transparently by Keras in Functional models).
    4. Activation check — Dense activations must be relu, tanh, linear, or
       sigmoid.
    5. Output unit check — final Dense layer must have exactly one unit.
    6. Position sizer class check — must be None, FixedPositionSizer, or
       RiskBasedPositionSizer.
    7. Feature name check — ``model.input_names`` must be non-empty.
    8. Warnings for sigmoid output (convention expects tanh) and for large
       weight arrays (>10k parameters total, which inflates the .mq5 file).

    This is the strict, flat-layer validator for the hardcoded-weights export
    path (``export_ml_to_mql5``), which translates every layer into MQL5
    source by hand. Nested sub-models (e.g. seed-ensemble graphs) are
    rejected here — use ``validate_ml_strategy_onnx`` for those.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Strategy type
    strategy_type_error = _check_strategy_type(strategy)
    if strategy_type_error is not None:
        errors.append(strategy_type_error)
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 2. Model wrapper presence and type
    wrapper_error = _check_model_wrapper(strategy)
    if wrapper_error is not None:
        errors.append(wrapper_error)
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    wrapper = strategy.model
    keras_model = wrapper.model

    # 3 & 4. Layer and activation checks
    try:
        import keras
    except ImportError:
        errors.append(
            "Keras is required for ML export. "
            "Install it with: pip install 'TradeLab[ml]'"
        )
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    total_params = 0
    dense_count = 0
    last_dense_units: int | None = None
    last_dense_activation: str | None = None

    for layer in keras_model.layers:
        layer_type = type(layer).__name__
        if layer_type not in _SUPPORTED_LAYER_TYPES:
            errors.append(
                f"Unsupported layer type '{layer_type}'. "
                f"Only Dense layers are translated to MQL5. "
                f"Supported types: {', '.join(_SUPPORTED_LAYER_TYPES)}."
            )
            continue

        if not isinstance(layer, keras.layers.Dense):
            continue  # InputLayer / Dropout / Concatenate — skip, no checks needed

        dense_count += 1
        activation, activation_error = _check_dense_activation(layer)
        if activation_error is not None:
            errors.append(activation_error)

        total_params += _dense_param_count(layer)

        last_dense_units = layer.get_config().get("units")
        last_dense_activation = activation

    if dense_count == 0:
        errors.append(
            "No Dense layers found in the Keras model. "
            "The ML exporter requires at least one Dense layer."
        )

    # 5. Output unit check
    if last_dense_units is not None and last_dense_units != 1:
        errors.append(
            f"Final Dense layer must have exactly 1 unit (got {last_dense_units}). "
            "The EA expects a scalar signal_strength in [-1, 1]."
        )

    # 6. Position sizer check
    sizer_error = _check_position_sizer(strategy)
    if sizer_error is not None:
        errors.append(sizer_error)

    # 7. Feature names
    input_names_error = _check_input_names(wrapper)
    if input_names_error is not None:
        errors.append(input_names_error)

    # Warnings
    if last_dense_activation == "sigmoid":
        warnings.append(
            "Output layer uses sigmoid activation (range [0, 1]). "
            "TradeLab convention expects tanh (range [-1, 1]). "
            "Entry/exit threshold comparisons in the EA assume the signal "
            "is centred around 0. Consider retraining with tanh output."
        )

    if total_params > 10_000:
        warnings.append(
            f"Model has {total_params:,} parameters. Large weight arrays "
            "may produce very large .mq5 files and slow MetaEditor compilation. "
            "Consider pruning the model before export."
        )

    warnings.extend(collect_risk_warnings(strategy))

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Recursive layer walk — ONNX export path
# ---------------------------------------------------------------------------


def _flatten_onnx_layers(keras_model: object) -> tuple[list[str], list]:
    """Recursively walk ``keras_model.layers``, expanding nested sub-models.

    Seed-ensemble exports call each trained member as a layer inside a shared
    functional graph (``inputs -> [member(inputs) for member in members] ->
    Average``); each member shows up as a nested ``Functional`` (or
    ``Sequential``) layer whose own ``.layers`` holds its real Dense/Dropout
    stack. tf2onnx traces the whole graph directly, so unlike the strict
    validator we need to look *inside* those containers rather than reject
    them outright.

    Parameters
    ----------
    keras_model : keras.Model
        The top-level model to walk.

    Returns
    -------
    tuple[list[str], list]
        ``(errors, flat_layers)`` — one fatal "unsupported layer type" error
        per rejected layer, and every layer encountered during the walk
        (containers included) that *is* in the ONNX whitelist, in traversal
        order.
    """
    errors: list[str] = []
    flat_layers: list = []

    def walk(layers) -> None:
        for layer in layers:
            layer_type = type(layer).__name__
            if layer_type not in _ONNX_SUPPORTED_LAYER_TYPES:
                errors.append(
                    f"Unsupported layer type '{layer_type}' for ONNX export. "
                    f"Supported types: {', '.join(_ONNX_SUPPORTED_LAYER_TYPES)}."
                )
                continue
            flat_layers.append(layer)
            if layer_type in _ONNX_NESTED_MODEL_TYPES:
                walk(layer.layers)

    walk(keras_model.layers)
    return errors, flat_layers


def _final_dense_activation(keras_model: object, keras_module: object) -> str | None:
    """Return the normalised activation of the model's last top-level layer.

    Returns ``None`` if the model has no layers, or if its last top-level
    layer is not a Dense layer (e.g. a seed-ensemble graph terminating in
    ``Average`` — there the "final activation" is not a single well-defined
    value, so the sigmoid-output warning is simply dropped for such models).
    """
    layers = list(keras_model.layers)
    if not layers:
        return None
    last_layer = layers[-1]
    if not isinstance(last_layer, keras_module.layers.Dense):
        return None
    activation, _ = _check_dense_activation(last_layer)
    return activation


def validate_ml_strategy_onnx(strategy: object) -> ValidationResult:
    """Validate an ``MLStrategy`` for ONNX-based MQL5 export.

    Parameters
    ----------
    strategy : object
        The strategy to validate.

    Returns
    -------
    ValidationResult
        Contains ``is_valid``, ``errors`` (fatal), and ``warnings`` (non-fatal).

    Notes
    -----
    The generated ONNX EA runs the whole graph via ``OnnxRun`` — it never
    translates individual layers into MQL5 — so this validator does **not**
    inherit ``validate_ml_strategy``'s flat, hardcoded-weights layer
    whitelist. Instead:

    1. Strategy class check — must be ``MLStrategy``.
    2. Model wrapper check — ``strategy.model`` must be a ``KerasModelWrapper``.
    3. Keras availability check.
    4. Recursive layer walk — nested ``Functional``/``Sequential`` sub-models
       (seed-ensemble members) are expanded and their children validated
       too. Every layer encountered (containers included) must be one of
       Dense, Dropout, InputLayer, Concatenate, Average, Functional, or
       Sequential; anything else (e.g. LSTM, Conv1D) is a fatal error.
    5. Activation check — every flattened Dense layer's activation must be
       relu, tanh, linear, or sigmoid.
    6. Output-shape check — ``keras_model.output_shape`` must be
       ``(None, 1)``. This replaces the strict validator's "final Dense
       layer has 1 unit" check, which does not apply once the final layer
       may be ``Average`` rather than ``Dense``. The EA expects a single
       scalar ``signal_strength`` per prediction.
    7. Position sizer class check — must be None, FixedPositionSizer, or
       RiskBasedPositionSizer.
    8. Feature name check — ``model.input_names`` must be non-empty.
    9. ``tf2onnx`` package present — fatal if missing.
    10. ``onnx`` package present — fatal if missing.
    11. Warnings: sigmoid final activation (only when determinable — dropped
        for nested/ensemble models whose final layer isn't Dense), >10k
        parameters across the flattened Dense layers, Concatenate layers
        (tf2onnx may need an explicit input_signature), and
        ``collect_risk_warnings``. A linear final activation is a legitimate
        export and never warns.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Strategy type
    strategy_type_error = _check_strategy_type(strategy)
    if strategy_type_error is not None:
        errors.append(strategy_type_error)
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 2. Model wrapper presence and type
    wrapper_error = _check_model_wrapper(strategy)
    if wrapper_error is not None:
        errors.append(wrapper_error)
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    wrapper = strategy.model
    keras_model = wrapper.model

    # 3. Keras availability
    try:
        import keras
    except ImportError:
        errors.append(
            "Keras is required for ML export. "
            "Install it with: pip install 'TradeLab[ml]'"
        )
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 4 & 5. Recursive layer walk + Dense activation checks
    layer_errors, flat_layers = _flatten_onnx_layers(keras_model)
    errors.extend(layer_errors)

    dense_layers = [
        layer for layer in flat_layers if isinstance(layer, keras.layers.Dense)
    ]
    total_params = 0
    for dense in dense_layers:
        _, activation_error = _check_dense_activation(dense)
        if activation_error is not None:
            errors.append(activation_error)
        total_params += _dense_param_count(dense)

    # 6. Output-shape check — replaces the strict validator's "final Dense
    # layer has 1 unit" check, since the final layer may be Average.
    output_shape = getattr(keras_model, "output_shape", None)
    if output_shape != (None, 1):
        errors.append(
            f"Model output shape must be (None, 1) for a scalar signal_strength "
            f"(got {output_shape}). The EA expects a single scalar signal in "
            "[-1, 1] per prediction."
        )

    # 7. Position sizer check
    sizer_error = _check_position_sizer(strategy)
    if sizer_error is not None:
        errors.append(sizer_error)

    # 8. Feature names
    input_names_error = _check_input_names(wrapper)
    if input_names_error is not None:
        errors.append(input_names_error)

    # 9. tf2onnx available?
    try:
        import tf2onnx  # noqa: F401
    except ImportError:
        errors.append(
            "tf2onnx is not installed. Install it with:  pip install 'TradeLab[onnx]' "
            "(Python 3.10-3.12)."
        )

    # 10. onnx available?
    try:
        import onnx  # noqa: F401
    except ImportError:
        errors.append(
            "onnx is not installed. Install it with:  pip install 'TradeLab[onnx]' "
            "(Python 3.10-3.12)."
        )

    # 11. Warnings
    final_activation = _final_dense_activation(keras_model, keras)
    if final_activation == "sigmoid":
        warnings.append(
            "Output layer uses sigmoid activation (range [0, 1]). "
            "TradeLab convention expects tanh (range [-1, 1]). "
            "Entry/exit threshold comparisons in the EA assume the signal "
            "is centred around 0. Consider retraining with tanh output."
        )

    if total_params > 10_000:
        warnings.append(
            f"Model has {total_params:,} parameters across its Dense layers. "
            "Large weight arrays increase the .onnx file size and inference "
            "latency. Consider pruning the model before export."
        )

    if not errors:
        has_concatenate = any(
            type(layer).__name__ == "Concatenate" for layer in flat_layers
        )
        if has_concatenate:
            warnings.append(
                "Model contains Concatenate layers (Functional API). "
                "tf2onnx may require an explicit input_signature for "
                "correct conversion. If export fails, try rebuilding the "
                "model as a Sequential architecture."
            )

    warnings.extend(collect_risk_warnings(strategy))

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
