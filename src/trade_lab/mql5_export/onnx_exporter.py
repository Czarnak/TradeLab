"""ONNX model exporter for TradeLab MLStrategy.

Converts a trained Keras model (wrapped in ``KerasModelWrapper``) to the
ONNX format and writes the resulting ``.onnx`` binary file to disk.

This module is an internal implementation detail of ``export_ml_to_mql5_onnx``.
It is intentionally thin — all orchestration and validation live in
``code_generator.py``; this module only handles the tf2onnx conversion step.

Why tf2onnx?
------------
``tf2onnx`` is the canonical converter for TensorFlow/Keras → ONNX.  It
supports the ``keras.Sequential`` and ``keras.Model`` (Functional API) model
types that TradeLab uses, and produces ONNX opsets compatible with the MT5
built-in ONNX runtime (opset 12+).

Dependency note
---------------
``tf2onnx`` and ``onnx`` are *optional* dependencies included in the
``[onnx]`` extra:

    pip install "TradeLab[onnx]"

They are imported lazily inside the function so the rest of the package
remains importable without them.
"""
from __future__ import annotations

from pathlib import Path


def export_keras_to_onnx(
    keras_model: object,
    output_path: str,
    opset: int = 12,
) -> str:
    """Convert a Keras model to ONNX and save it to ``output_path``.

    Parameters
    ----------
    keras_model : keras.Model
        A compiled / trained Keras model.  Must be a ``Sequential`` or
        Functional ``Model`` — subclassed models are not supported by tf2onnx.
    output_path : str
        Destination file path for the ``.onnx`` file.
    opset : int
        ONNX opset version to target.  Default 12 is the minimum supported
        by the MetaTrader 5 built-in runtime (build 3490+).  Use a higher
        value only if you know your MT5 installation is recent enough.

    Returns
    -------
    str
        Absolute path of the written ``.onnx`` file.

    Raises
    ------
    ImportError
        If ``tf2onnx`` or ``onnx`` are not installed.
    RuntimeError
        If the conversion itself fails.

    Notes
    -----
    ``tf2onnx.convert.from_keras`` works via the SavedModel intermediate
    format under the hood.  The function writes a temporary SavedModel to a
    system temp directory, converts it, then discards the temporary files.
    This is transparent to the caller.
    """
    # --- optional dependency guard -------------------------------------------
    try:
        import tf2onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires tf2onnx. "
            "Install it with:  pip install 'TradeLab[onnx]'"
        ) from exc

    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires the onnx package. "
            "Install it with:  pip install 'TradeLab[onnx]'"
        ) from exc

    # -------------------------------------------------------------------------
    # tf2onnx.convert.from_keras is the high-level API.
    # It returns (onnx_model_proto, external_tensor_storage).
    # We only need the proto — write it directly with onnx.save_model.
    # -------------------------------------------------------------------------
    import tf2onnx.convert as tf2onnx_convert
    import onnx as onnx_lib

    try:
        # input_signature can be None for Sequential models — tf2onnx infers
        # it from the model's built input spec.
        onnx_model, _ = tf2onnx_convert.from_keras(
            keras_model,
            opset=opset,
            output_path=None,  # we write manually for explicit control
        )
    except Exception as exc:
        raise RuntimeError(
            f"tf2onnx conversion failed: {exc}\n"
            "Check that all layers are supported by tf2onnx for your opset."
        ) from exc

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    onnx_lib.save_model(onnx_model, str(output_file))

    return str(output_file.resolve())
