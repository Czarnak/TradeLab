from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WalkForwardFold:
    """Index ranges for a single walk-forward fold."""

    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class WalkForwardResult:
    """Outcome of training and evaluating one fold."""

    fold: int
    train_loss: float
    test_predictions: pd.Series
    test_targets: pd.Series
    model: object  # KerasModelWrapper (typed loosely to avoid circular import)


class WalkForwardSplit:
    """Time-series walk-forward cross-validation splitter.

    Divides ``n_samples`` into ``n_splits`` sequential test chunks.
    Training data is everything *before* the current test chunk.

    Two modes:
        **expanding** (default) — training window starts at index 0
        and grows with each fold.

        **sliding** — training window has a fixed size equal to
        ``n_samples × initial_train_ratio`` and slides forward.

    Example with ``n_splits=3`` on 100 samples, expanding:

        Fold 0:  train [ 0:60]  test [60:73]
        Fold 1:  train [ 0:73]  test [73:87]
        Fold 2:  train [ 0:87]  test [87:100]

    Parameters
    ----------
    n_splits : int
        Number of test folds.
    initial_train_ratio : float
        Fraction of data used for training in the first fold.
    expanding : bool
        If True, training window grows.  If False, it slides.
    embargo : int
        Number of rows dropped from the **end** of every fold's training window
        before its test window. With a forward-looking target spanning ``H`` bars,
        set ``embargo=H`` so the last training labels cannot overlap (leak into)
        the test period. Defaults to ``0`` (no purge).
    """

    def __init__(
        self,
        n_splits: int = 5,
        initial_train_ratio: float = 0.6,
        expanding: bool = True,
        embargo: int = 0,
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not 0 < initial_train_ratio < 1:
            raise ValueError("initial_train_ratio must be in (0, 1)")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.n_splits = n_splits
        self.initial_train_ratio = initial_train_ratio
        self.expanding = expanding
        self.embargo = embargo

    def split(self, n_samples: int) -> list[WalkForwardFold]:
        """Generate train/test index arrays for each fold."""
        train_size = int(n_samples * self.initial_train_ratio)
        remaining = n_samples - train_size
        chunk = remaining // self.n_splits

        if chunk < 1:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} folds "
                f"with initial_train_ratio={self.initial_train_ratio}"
            )

        folds: list[WalkForwardFold] = []
        for i in range(self.n_splits):
            test_start = train_size + i * chunk
            test_end = test_start + chunk if i < self.n_splits - 1 else n_samples

            if self.expanding:
                fold_train_start = 0
            else:
                fold_train_start = max(0, test_start - train_size)

            train_idx = np.arange(fold_train_start, test_start)
            if self.embargo > 0:
                train_idx = train_idx[: -self.embargo]

            folds.append(
                WalkForwardFold(
                    train_idx=train_idx,
                    test_idx=np.arange(test_start, test_end),
                )
            )

        return folds


def purged_chronological_split(
    n_samples: int,
    val_fraction: float,
    embargo: int = 0,
    min_val: int = 0,
) -> tuple[slice, slice]:
    """Chronological train/validation split with a purge gap, for early stopping.

    The last ``val_fraction`` of the data (floored at ``min_val`` rows) becomes the
    validation tail; the ``embargo`` rows immediately preceding it are dropped from
    training so a forward-looking target cannot leak its label across the boundary.

    Parameters
    ----------
    n_samples : int
        Total number of (already chronologically ordered) rows.
    val_fraction : float
        Fraction of rows reserved for validation, in ``(0, 1)``.
    embargo : int
        Rows purged between the training and validation slices. Defaults to ``0``.
    min_val : int
        Minimum validation rows; overrides ``val_fraction`` when larger.

    Returns
    -------
    tuple[slice, slice]
        ``(train_slice, val_slice)`` to index the row axis of ``X``/``y``.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    n_val = min(max(int(n_samples * val_fraction), min_val), n_samples)
    val_start = n_samples - n_val
    train_end = val_start - embargo
    if train_end <= 0:
        raise ValueError(
            "embargo/val_fraction leaves no training rows "
            f"(n_samples={n_samples}, n_val={n_val}, embargo={embargo})"
        )

    return slice(0, train_end), slice(val_start, n_samples)
