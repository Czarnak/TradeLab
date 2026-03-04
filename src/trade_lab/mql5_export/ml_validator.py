"""Pre-export validation for ML-based MQL5 code generation.

Checks an MLStrategy object for compatibility with the ML MQL5 exporter
before any code is generated. Reuses ``ValidationResult`` from the standard
validator — the schema is identical, only the checks differ.
"""

from __future__ import annotations

from trade_lab.mql5_export.validators import ValidationResult
from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.sizing.risk_based import RiskBasedPositionSizer
from trade_lab.strategies.ml_strategy import MLStrategy

_SUPPORTED_SIZERS = (type(None), FixedPositionSizer, RiskBasedPositionSizer)

# Keras layer class names that are safe to skip at inference time (Dropout
# is identity at inference) or to translate into MQL5 (Dense).
_SUPPORTED_LAYER_TYPES = ("Dense", "Dropout", "InputLayer", "Concatenate")

# Activations we can render as MQL5 expressions.
_SUPPORTED_ACTIVATIONS = ("relu", "tanh", "linear", "sigmoid")


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
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Strategy type
    if not isinstance(strategy, MLStrategy):
        errors.append(
            f"Strategy must be MLStrategy, got {type(strategy).__name__}. "
            "StandardStrategy and other subclasses use export_to_mql5() instead."
        )
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 2. Model wrapper presence and type
    from trade_lab.ml.models import KerasModelWrapper

    if not isinstance(strategy.model, KerasModelWrapper):
        errors.append(
            f"strategy.model must be a KerasModelWrapper, got "
            f"{type(strategy.model).__name__}. "
            "Wrap your Keras model with KerasModelWrapper(model, input_names) "
            "before exporting."
        )
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
        cfg = layer.get_config()

        # Normalise activation — Keras stores it as a string or a dict
        activation = cfg.get("activation", "linear")
        if isinstance(activation, dict):
            activation = activation.get("class_name", "linear").lower()
        else:
            activation = str(activation).lower()

        if activation not in _SUPPORTED_ACTIVATIONS:
            errors.append(
                f"Unsupported activation '{activation}' in layer "
                f"'{layer.name}'. Supported: {', '.join(_SUPPORTED_ACTIVATIONS)}."
            )

        ws = layer.get_weights()
        if ws:
            total_params += ws[0].size  # kernel
            if len(ws) > 1:
                total_params += ws[1].size  # bias

        last_dense_units = cfg.get("units")
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
    sizer = strategy.position_sizer
    if not isinstance(sizer, _SUPPORTED_SIZERS):
        errors.append(
            f"Unsupported position sizer '{type(sizer).__name__}'. "
            f"Supported: None, FixedPositionSizer, RiskBasedPositionSizer."
        )

    # 7. Feature names
    if not wrapper.input_names:
        errors.append(
            "KerasModelWrapper.input_names is empty. "
            "The exporter needs feature column names to generate EA inputs."
        )

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

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_ml_strategy_onnx(strategy: object) -> "ValidationResult":
    """Validate an ``MLStrategy`` for ONNX-based MQL5 export.

    Runs all checks from ``validate_ml_strategy`` first, then adds
    ONNX-specific checks: whether ``tf2onnx`` and ``onnx`` are installed,
    and whether the model architecture is known to be tf2onnx-compatible.

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
    Additional ONNX checks (on top of standard ML checks):

    9.  ``tf2onnx`` package present — fatal if missing.
    10. ``onnx`` package present — fatal if missing.
    11. Concatenate layers — warn that Functional models with Concatenate may
        require explicit input signatures; export may still succeed.
    """
    # Run the standard ML checks first — they already cover strategy type,
    # model wrapper, layer types, activations, output unit, sizer, and
    # feature names.  We inherit and extend.
    base_result = validate_ml_strategy(strategy)
    errors: list[str] = list(base_result.errors)
    warnings: list[str] = list(base_result.warnings)

    # If base validation already failed on structure, no point adding more.
    # We still check deps so the user gets all actionable errors at once.

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

    # 11. Warn about Concatenate layers (Functional API models)
    if base_result.is_valid:
        from trade_lab.ml.models import KerasModelWrapper

        if isinstance(strategy.model, KerasModelWrapper):
            has_concatenate = any(
                type(layer).__name__ == "Concatenate"
                for layer in strategy.model.model.layers
            )
            if has_concatenate:
                warnings.append(
                    "Model contains Concatenate layers (Functional API). "
                    "tf2onnx may require an explicit input_signature for "
                    "correct conversion. If export fails, try rebuilding the "
                    "model as a Sequential architecture."
                )

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
