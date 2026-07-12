"""Tests for the extended model-builder factories in ``trade_lab.ml.models``.

Covers three additions:

1. ``ensemble_average_model`` — averages already-trained member models into a
   single exportable functional model (seed-ensemble export, no distillation).
2. ``head`` / ``loss`` options on ``dense_model`` — a linear head + MSE loss
   alternative to the default tanh head + directional hinge.
3. ``quantile_model`` / ``pinball_loss`` — a multi-quantile regression head.

These tests exercise the real Keras forward pass; the module skips cleanly if
``keras`` (and its ``tensorflow`` backend) is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from trade_lab.ml.models import (
    dense_model,
    ensemble_average_model,
    pinball_loss,
    quantile_model,
)

keras = pytest.importorskip("keras")
tf = pytest.importorskip("tensorflow")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_small_model(input_dim: int, seed: int):
    """A tiny functional tanh-head model with seed-dependent random weights."""
    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(4, activation="relu")(inputs)
    output = keras.layers.Dense(1, activation="tanh")(x)
    model = keras.Model(inputs, output)
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------------------------------------------------------------------------
# ensemble_average_model
# ---------------------------------------------------------------------------


def test_ensemble_average_model_matches_mean_of_member_predictions():
    members = [_build_small_model(3, seed=i) for i in range(3)]
    rng = np.random.RandomState(0)
    X = rng.uniform(-1.0, 1.0, size=(10, 3)).astype("float32")

    member_preds = np.stack([m.predict(X, verbose=0) for m in members], axis=0)
    expected = member_preds.mean(axis=0)

    ensemble = ensemble_average_model(members)
    got = ensemble.predict(X, verbose=0)

    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_ensemble_average_model_works_with_dense_model_builder_members():
    rng = np.random.RandomState(1)
    X = rng.uniform(-1.0, 1.0, size=(20, 4)).astype("float32")
    y = rng.uniform(-1.0, 1.0, size=(20,)).astype("float32")

    members = []
    for seed in range(2):
        keras.utils.set_random_seed(seed)
        builder = dense_model(layers=[8, 4])
        model = builder(input_dim=4)
        model.fit(X, y, epochs=1, batch_size=5, verbose=0)
        members.append(model)

    member_preds = np.stack([m.predict(X, verbose=0) for m in members], axis=0)
    expected = member_preds.mean(axis=0)

    ensemble = ensemble_average_model(members, input_dim=4)
    got = ensemble.predict(X, verbose=0)

    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_ensemble_average_model_raises_on_empty_members():
    with pytest.raises(ValueError, match="members"):
        ensemble_average_model([])


def test_ensemble_average_model_raises_on_mismatched_member_input_dims():
    m1 = _build_small_model(3, seed=0)
    m2 = _build_small_model(4, seed=1)

    with pytest.raises(ValueError, match="input dim"):
        ensemble_average_model([m1, m2])


def test_ensemble_average_model_single_member_is_passthrough():
    member = _build_small_model(3, seed=0)
    rng = np.random.RandomState(2)
    X = rng.uniform(-1.0, 1.0, size=(6, 3)).astype("float32")
    expected = member.predict(X, verbose=0)

    ensemble = ensemble_average_model([member])
    got = ensemble.predict(X, verbose=0)

    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert not any(isinstance(layer, keras.layers.Average) for layer in ensemble.layers)


def test_ensemble_average_model_save_load_roundtrip_preserves_predictions(tmp_path):
    members = [_build_small_model(3, seed=i) for i in range(2)]
    ensemble = ensemble_average_model(members)

    rng = np.random.RandomState(3)
    X = rng.uniform(-1.0, 1.0, size=(8, 3)).astype("float32")
    before = ensemble.predict(X, verbose=0)

    save_path = tmp_path / "ensemble.keras"
    ensemble.save(save_path)
    loaded = keras.models.load_model(save_path)
    after = loaded.predict(X, verbose=0)

    np.testing.assert_allclose(after, before, atol=1e-6)


def test_ensemble_average_model_onnx_parity(tmp_path):
    pytest.importorskip("tf2onnx")
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    from trade_lab.mql5_export.onnx_exporter import export_keras_to_onnx

    members = [_build_small_model(3, seed=i) for i in range(2)]
    ensemble = ensemble_average_model(members)

    rng = np.random.RandomState(4)
    X = rng.uniform(-1.0, 1.0, size=(5, 3)).astype("float32")
    keras_out = ensemble.predict(X, verbose=0)

    onnx_path = export_keras_to_onnx(
        ensemble, str(tmp_path / "ensemble.onnx"), opset=13
    )

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: X})[0]

    np.testing.assert_allclose(onnx_out, keras_out, atol=1e-4)


# ---------------------------------------------------------------------------
# dense_model head / loss options
# ---------------------------------------------------------------------------


def test_dense_model_default_head_and_loss_are_unchanged():
    builder = dense_model(layers=[4, 2])
    model = builder(input_dim=3)

    assert model.layers[-1].activation.__name__ == "tanh"
    assert model.loss.__name__ == "directional_loss"


def test_dense_model_linear_head_and_mse_loss_has_no_final_activation():
    builder = dense_model(layers=[4, 2], head="linear", loss="mse")
    model = builder(input_dim=3)

    assert model.layers[-1].activation.__name__ == "linear"
    assert model.loss == "mse"


def test_dense_model_linear_head_and_mse_loss_trains_one_epoch():
    builder = dense_model(layers=[4, 2], head="linear", loss="mse")
    model = builder(input_dim=3)

    rng = np.random.RandomState(5)
    X = rng.uniform(-1.0, 1.0, size=(10, 3)).astype("float32")
    y = rng.uniform(-5.0, 5.0, size=(10,)).astype("float32")

    history = model.fit(X, y, epochs=1, batch_size=5, verbose=0)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


def test_dense_model_raises_on_unknown_head():
    with pytest.raises(ValueError, match="head"):
        dense_model(head="sigmoid")


def test_dense_model_raises_on_unknown_loss():
    with pytest.raises(ValueError, match="loss"):
        dense_model(loss="huber")


# ---------------------------------------------------------------------------
# pinball_loss
# ---------------------------------------------------------------------------


def test_pinball_loss_matches_hand_computed_value():
    # e = y_true - y_pred = [1.0, -0.5, -2.0] for q in (0.1, 0.5, 0.9):
    #   q=0.1: max(0.1*1.0, -0.9*1.0)   = max(0.1, -0.9)  = 0.1
    #   q=0.5: max(0.5*-0.5, -0.5*-0.5) = max(-0.25, 0.25) = 0.25
    #   q=0.9: max(0.9*-2.0, -0.1*-2.0) = max(-1.8, 0.2)   = 0.2
    # mean = (0.1 + 0.25 + 0.2) / 3 = 0.18333...
    fn = pinball_loss((0.1, 0.5, 0.9))
    y_true = tf.constant([[0.0]], dtype=tf.float32)
    y_pred = tf.constant([[-1.0, 0.5, 2.0]], dtype=tf.float32)

    loss = float(fn(y_true, y_pred).numpy())

    assert loss == pytest.approx(0.1833333, abs=1e-5)


def test_pinball_loss_accepts_1d_y_true():
    fn = pinball_loss((0.1, 0.5, 0.9))
    y_true_1d = tf.constant([0.0], dtype=tf.float32)
    y_true_2d = tf.constant([[0.0]], dtype=tf.float32)
    y_pred = tf.constant([[-1.0, 0.5, 2.0]], dtype=tf.float32)

    loss_1d = float(fn(y_true_1d, y_pred).numpy())
    loss_2d = float(fn(y_true_2d, y_pred).numpy())

    assert loss_1d == pytest.approx(loss_2d)


def test_pinball_loss_keeps_its_name():
    assert pinball_loss((0.5,)).__name__ == "pinball_loss"


def test_pinball_loss_is_zero_for_perfect_median_prediction_all_quantiles():
    # When every quantile prediction equals y_true, every per-quantile error is
    # zero, so max(q*0, (q-1)*0) == 0 for all q.
    fn = pinball_loss((0.1, 0.5, 0.9))
    y_true = tf.constant([[1.0]], dtype=tf.float32)
    y_pred = tf.constant([[1.0, 1.0, 1.0]], dtype=tf.float32)

    assert float(fn(y_true, y_pred).numpy()) == pytest.approx(0.0, abs=1e-7)


def test_pinball_loss_raises_on_empty_quantiles():
    with pytest.raises(ValueError, match="quantiles"):
        pinball_loss(())


@pytest.mark.parametrize("bad_quantiles", [(0.0, 0.5), (0.5, 1.0), (0.5, -0.1)])
def test_pinball_loss_raises_on_out_of_range_quantiles(bad_quantiles):
    with pytest.raises(ValueError, match="quantiles"):
        pinball_loss(bad_quantiles)


# ---------------------------------------------------------------------------
# quantile_model
# ---------------------------------------------------------------------------


def test_quantile_model_builder_output_shape_and_loss_name():
    builder = quantile_model(layers=[8, 4], quantiles=(0.1, 0.5, 0.9))
    model = builder(input_dim=3)

    rng = np.random.RandomState(6)
    X = rng.uniform(-1.0, 1.0, size=(5, 3)).astype("float32")
    preds = model.predict(X, verbose=0)

    assert preds.shape == (5, 3)
    assert model.loss.__name__ == "pinball_loss"
    assert model.layers[-1].activation.__name__ == "linear"


def test_quantile_model_trains_one_epoch():
    builder = quantile_model(layers=[8, 4], quantiles=(0.1, 0.5, 0.9))
    model = builder(input_dim=3)

    rng = np.random.RandomState(7)
    X = rng.uniform(-1.0, 1.0, size=(10, 3)).astype("float32")
    y = rng.normal(size=(10,)).astype("float32")

    history = model.fit(X, y, epochs=1, batch_size=5, verbose=0)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


def test_quantile_model_default_quantiles_are_low_median_high():
    builder = quantile_model()
    model = builder(input_dim=2)

    assert model.output_shape == (None, 3)


def test_quantile_model_raises_on_invalid_quantiles():
    with pytest.raises(ValueError, match="quantiles"):
        quantile_model(quantiles=())

    with pytest.raises(ValueError, match="quantiles"):
        quantile_model(quantiles=(0.0, 0.5))
