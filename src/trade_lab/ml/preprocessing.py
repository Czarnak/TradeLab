from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Align features and target, dropping rows with any NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing computed signal/indicator columns.
    feature_columns : list[str]
        Column names to use as model inputs.
    target : pd.Series
        Target values aligned with ``df.index``.

    Returns
    -------
    X : np.ndarray, shape (n_clean, n_features)
    y : np.ndarray, shape (n_clean,)
    index : pd.DatetimeIndex
        Surviving index after NaN removal (needed for walk-forward alignment).
    """
    combined = df[feature_columns].copy()
    combined['_target'] = target
    combined = combined.dropna()

    X = combined[feature_columns].to_numpy(dtype=np.float64)
    y = combined['_target'].to_numpy(dtype=np.float64)
    return X, y, combined.index


class FeatureScaler:
    """Per-column feature scaling.

    Supports ``'standard'`` (zero mean, unit variance) and
    ``'minmax'`` (scale to [0, 1]).

    Must be ``fit()`` on training data only, then ``transform()``
    applied to both train and test sets to prevent data leakage.
    """

    def __init__(self, method: str = 'standard'):
        if method not in ('standard', 'minmax'):
            raise ValueError(f"method must be 'standard' or 'minmax', got '{method}'")
        self.method = method
        self._params: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, feature_names: list[str]) -> None:
        """Compute scaling parameters from training data."""
        self._feature_names = feature_names
        if self.method == 'standard':
            self._params['mean'] = X.mean(axis=0)
            self._params['std'] = X.std(axis=0)
            # Prevent division by zero for constant features
            self._params['std'][self._params['std'] == 0] = 1.0
        else:  # minmax
            self._params['min'] = X.min(axis=0)
            self._params['max'] = X.max(axis=0)
            rng = self._params['max'] - self._params['min']
            rng[rng == 0] = 1.0
            self._params['range'] = rng

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaling to data."""
        if not self._params:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        if self.method == 'standard':
            return (X - self._params['mean']) / self._params['std']
        else:  # minmax
            return (X - self._params['min']) / self._params['range']

    def fit_transform(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """Fit on data and return transformed result."""
        self.fit(X, feature_names)
        return self.transform(X)
