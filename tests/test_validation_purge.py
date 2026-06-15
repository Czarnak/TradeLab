"""Purging / embargo on the time-series splitters.

Both the outer ``WalkForwardSplit`` and the inner early-stopping split must drop
the last ``embargo`` training rows before the evaluation window so that a forward
-looking target (e.g. a 5-bar return) cannot leak its label into validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from trade_lab.ml.validation import WalkForwardSplit, purged_chronological_split


# ---------------------------------------------------------------------------
# WalkForwardSplit embargo
# ---------------------------------------------------------------------------


def test_embargo_drops_last_train_rows_of_each_expanding_fold():
    # 10 samples, ratio 0.6, 2 folds: train ends at test_start; embargo=1 purges
    # the single row adjacent to each test window.
    folds = WalkForwardSplit(
        n_splits=2, initial_train_ratio=0.6, expanding=True, embargo=1
    ).split(10)

    np.testing.assert_array_equal(folds[0].train_idx, np.arange(0, 5))
    np.testing.assert_array_equal(folds[0].test_idx, np.arange(6, 8))
    np.testing.assert_array_equal(folds[1].train_idx, np.arange(0, 7))
    np.testing.assert_array_equal(folds[1].test_idx, np.arange(8, 10))


def test_embargo_leaves_a_gap_of_exactly_embargo_rows_before_each_test():
    embargo = 3
    folds = WalkForwardSplit(
        n_splits=3, initial_train_ratio=0.5, expanding=True, embargo=embargo
    ).split(40)

    for fold in folds:
        last_train = fold.train_idx[-1]
        first_test = fold.test_idx[0]
        assert first_test - last_train - 1 == embargo


def test_embargo_zero_matches_unpurged_behaviour():
    purged = WalkForwardSplit(n_splits=2, initial_train_ratio=0.6, embargo=0).split(10)
    plain = WalkForwardSplit(n_splits=2, initial_train_ratio=0.6).split(10)
    for a, b in zip(purged, plain):
        np.testing.assert_array_equal(a.train_idx, b.train_idx)
        np.testing.assert_array_equal(a.test_idx, b.test_idx)


def test_negative_embargo_raises():
    with pytest.raises(ValueError, match="embargo"):
        WalkForwardSplit(embargo=-1)


# ---------------------------------------------------------------------------
# purged_chronological_split (inner early-stopping split)
# ---------------------------------------------------------------------------


def test_purged_chronological_split_inserts_embargo_gap():
    train_sl, val_sl = purged_chronological_split(100, val_fraction=0.15, embargo=5)

    assert train_sl == slice(0, 80)
    assert val_sl == slice(85, 100)
    # The gap between the last train row and the first val row equals the embargo.
    assert val_sl.start - train_sl.stop == 5


def test_purged_chronological_split_respects_min_val_floor():
    train_sl, val_sl = purged_chronological_split(
        100, val_fraction=0.15, embargo=5, min_val=60
    )

    # min_val forces 60 validation rows regardless of the 15% fraction.
    assert val_sl == slice(40, 100)
    assert train_sl == slice(0, 35)


def test_purged_chronological_split_zero_embargo_is_contiguous():
    train_sl, val_sl = purged_chronological_split(100, val_fraction=0.2, embargo=0)

    assert train_sl == slice(0, 80)
    assert val_sl == slice(80, 100)
    assert train_sl.stop == val_sl.start


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_samples": 100, "val_fraction": 0.0}, "val_fraction"),
        ({"n_samples": 100, "val_fraction": 1.0}, "val_fraction"),
        ({"n_samples": 100, "val_fraction": 0.15, "embargo": -1}, "embargo"),
        ({"n_samples": 10, "val_fraction": 0.5, "embargo": 9}, "leaves no training"),
    ],
)
def test_purged_chronological_split_validates_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        purged_chronological_split(**kwargs)
