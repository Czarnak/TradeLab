from __future__ import annotations

import builtins
import sys
import types

import pytest

from trade_lab.mql5_export.onnx_exporter import export_keras_to_onnx


def test_export_keras_to_onnx_raises_when_tf2onnx_missing(monkeypatch):
    real_import = builtins.__import__

    def _fail_tf2onnx(name, *args, **kwargs):
        if name == "tf2onnx":
            raise ImportError("forced tf2onnx import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_tf2onnx)

    with pytest.raises(ImportError, match="requires tf2onnx"):
        export_keras_to_onnx(keras_model=object(), output_path="dummy.onnx")


def test_export_keras_to_onnx_raises_when_onnx_missing(monkeypatch):
    tf2onnx_pkg = types.ModuleType("tf2onnx")
    tf2onnx_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "tf2onnx", tf2onnx_pkg)

    real_import = builtins.__import__

    def _fail_onnx(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("forced onnx import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_onnx)

    with pytest.raises(ImportError, match="requires the onnx package"):
        export_keras_to_onnx(keras_model=object(), output_path="dummy.onnx")


def test_export_keras_to_onnx_wraps_tf2onnx_conversion_errors(monkeypatch, tmp_path):
    tf2onnx_pkg = types.ModuleType("tf2onnx")
    tf2onnx_pkg.__path__ = []
    tf2onnx_convert = types.ModuleType("tf2onnx.convert")

    def _raise_from_keras(*args, **kwargs):
        raise ValueError("broken graph")

    tf2onnx_convert.from_keras = _raise_from_keras

    onnx_mod = types.ModuleType("onnx")
    onnx_mod.save_model = lambda *_args, **_kwargs: None

    monkeypatch.setitem(sys.modules, "tf2onnx", tf2onnx_pkg)
    monkeypatch.setitem(sys.modules, "tf2onnx.convert", tf2onnx_convert)
    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)

    with pytest.raises(RuntimeError, match="tf2onnx conversion failed: broken graph"):
        export_keras_to_onnx(
            keras_model=object(),
            output_path=str(tmp_path / "model.onnx"),
            opset=13,
        )


def test_export_keras_to_onnx_writes_model_and_returns_resolved_path(
    monkeypatch, tmp_path
):
    tf2onnx_pkg = types.ModuleType("tf2onnx")
    tf2onnx_pkg.__path__ = []
    tf2onnx_convert = types.ModuleType("tf2onnx.convert")
    onnx_mod = types.ModuleType("onnx")

    call_log: dict[str, object] = {}
    onnx_model = object()

    def _from_keras(model, opset, output_path):
        call_log["model"] = model
        call_log["opset"] = opset
        call_log["output_path"] = output_path
        return onnx_model, {"external_storage": None}

    def _save_model(model, out_path):
        call_log["saved_model"] = model
        call_log["saved_path"] = out_path

    tf2onnx_convert.from_keras = _from_keras
    onnx_mod.save_model = _save_model

    monkeypatch.setitem(sys.modules, "tf2onnx", tf2onnx_pkg)
    monkeypatch.setitem(sys.modules, "tf2onnx.convert", tf2onnx_convert)
    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)

    output_file = tmp_path / "nested" / "model.onnx"
    result_path = export_keras_to_onnx(
        keras_model="keras-model",
        output_path=str(output_file),
        opset=17,
    )

    assert call_log["model"] == "keras-model"
    assert call_log["opset"] == 17
    assert call_log["output_path"] is None
    assert call_log["saved_model"] is onnx_model
    assert call_log["saved_path"] == str(output_file)
    assert output_file.parent.exists()
    assert result_path == str(output_file.resolve())
