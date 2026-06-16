"""Behavioural tests for ``directional_loss``.

The original loss rewarded directional correctness *linearly*
(``-mean(sign(y)·ŷ)``), whose gradient is a constant ``±1`` — it always pushed
the tanh head to the ``±1`` boundary, collapsing the graded signal strength that
position sizing and entry/exit thresholds rely on.

The fixed loss penalises only *wrong-sign* predictions (a margin-0 hinge) and
lets the MSE term own magnitude, so a correctly-signed prediction is pulled
toward the target value instead of toward saturation.
"""

from __future__ import annotations

import numpy as np
import pytest

from trade_lab.ml.models import directional_loss

tf = pytest.importorskip("tensorflow")


def _loss(magnitude_weight: float, y: list[float], p: list[float]) -> float:
    fn = directional_loss(magnitude_weight=magnitude_weight)
    y_t = tf.constant(y, dtype=tf.float32)
    p_t = tf.constant(p, dtype=tf.float32)
    return float(fn(y_t, p_t).numpy())


def test_loss_function_keeps_its_name():
    assert directional_loss().__name__ == "directional_loss"


def test_correct_sign_prediction_minimised_at_target_not_at_saturation():
    # For a correctly-signed target the loss must bottom out near p == y,
    # NOT at the saturated boundary p == +1 (the regression we are fixing).
    y = [0.3, 0.3, 0.3]
    grid = np.linspace(-1.0, 1.0, 41)
    losses = [_loss(0.1, y, [float(p)] * 3) for p in grid]
    best = grid[int(np.argmin(losses))]

    assert best == pytest.approx(0.3, abs=0.1)
    # Predicting the target must beat saturating to +1.
    assert _loss(0.1, y, [0.3, 0.3, 0.3]) < _loss(0.1, y, [1.0, 1.0, 1.0])


def test_perfect_prediction_has_zero_loss():
    y = [0.4, -0.2, 0.7]
    assert _loss(0.1, y, y) == pytest.approx(0.0, abs=1e-6)


def test_loss_is_non_negative():
    # Hinge ≥ 0 and MSE ≥ 0 → the loss can never be pulled negative (the old
    # reward term could, which is what drove saturation).
    rng = np.random.RandomState(0)
    for _ in range(20):
        y = rng.uniform(-1.0, 1.0, size=8).tolist()
        p = rng.uniform(-1.0, 1.0, size=8).tolist()
        assert _loss(0.1, y, p) >= -1e-7


def test_confident_wrong_penalised_more_than_weak_wrong():
    y = [-0.5]
    assert _loss(0.1, y, [0.9]) > _loss(0.1, y, [0.1])


def test_higher_magnitude_weight_tightens_magnitude_tracking():
    # With a correctly-signed target, a larger MSE weight makes an over-confident
    # (but correctly-signed) prediction relatively more costly.
    y = [0.2, 0.2]
    over_confident = [1.0, 1.0]
    assert _loss(1.0, y, over_confident) > _loss(0.1, y, over_confident)
