"""Feature matrix construction from indicators with built-in lag support.

``FeatureMatrix`` assembles a list of ``BaseIndicator`` instances (which may
carry non-zero ``lag`` values themselves) into a training-ready feature matrix
with optional scaling.

The ``LaggedIndicator`` wrapper that previously existed here has been removed.
Lag is now a first-class property of every indicator (and signal) via the
``lag`` constructor parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    from trade_lab.indicators.base import BaseIndicator


class FeatureMatrix:
    """Assemble indicators into a feature matrix with optional scaling.

    The instance is **stateful**: after calling ``build(df, fit_scaler=True)``
    the fitted ``StandardScaler`` persists and is applied automatically on
    subsequent ``build`` calls.  Use the same ``FeatureMatrix`` instance for
    both training and validation/test data.

    Parameters
    ----------
    indicators : list[BaseIndicator]
        Ordered list of indicators that define the feature set.  Each
        indicator's ``lag`` attribute determines column naming and shift
        behaviour — no external wrapper is needed.
    """

    def __init__(self, indicators: list[BaseIndicator]) -> None:
        self.indicators = indicators
        self._scaler: _StandardScaler | None = None

    def build(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix ``X`` and target ``y`` from raw OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with at least a ``Close`` column.
        fit_scaler : bool
            If ``True``, fit a ``StandardScaler`` on ``X`` and transform it.
            If ``False``, apply the previously fitted scaler (if any) without
            refitting.  If no scaler exists, ``X`` is returned unscaled.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(X, y)`` with NaN rows dropped.  ``X`` has shape
            ``(n_samples, n_features)`` and ``y`` has shape ``(n_samples,)``.
        """
        from sklearn.preprocessing import StandardScaler

        df = df.copy()

        # Step 1 — compute all indicators (each applies its own lag)
        for indicator in self.indicators:
            df = indicator.compute(df)

        # Step 2 — collect feature columns
        feature_cols = self.feature_names

        # Step 3 — compute target: log forward return
        y_series = np.log(df["Close"]).diff().shift(-1)

        # Step 4 — drop NaN rows (indicator warmup, lags, final row)
        mask = df[feature_cols].notna().all(axis=1) & y_series.notna()
        X = df.loc[mask, feature_cols].to_numpy(dtype=np.float64)
        y = y_series[mask].to_numpy(dtype=np.float64)

        # Step 5 — scaling
        if fit_scaler:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        elif self._scaler is not None:
            X = self._scaler.transform(X)

        return X, y

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature column names in ``X``.

        Derived from ``indicator.output_columns`` for each indicator,
        which already incorporates the lag suffix when ``lag > 0``.
        """
        return [col for ind in self.indicators for col in ind.output_columns]

    @property
    def scaler(self) -> _StandardScaler | None:
        """The fitted ``StandardScaler``, or ``None`` if not yet fitted."""
        return self._scaler
