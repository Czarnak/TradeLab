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
    """

    def __init__(
        self,
        n_splits: int = 5,
        initial_train_ratio: float = 0.6,
        expanding: bool = True,
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not 0 < initial_train_ratio < 1:
            raise ValueError("initial_train_ratio must be in (0, 1)")
        self.n_splits = n_splits
        self.initial_train_ratio = initial_train_ratio
        self.expanding = expanding

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

            folds.append(WalkForwardFold(
                train_idx=np.arange(fold_train_start, test_start),
                test_idx=np.arange(test_start, test_end),
            ))

        return folds
