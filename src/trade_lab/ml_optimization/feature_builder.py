"""Feature matrix construction with lagged indicator support.

``LaggedIndicator`` composes a ``BaseIndicator`` with configurable time lags,
and ``FeatureMatrix`` assembles multiple lagged indicators into a training-ready
feature matrix with optional scaling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    from trade_lab.indicators.base import BaseIndicator


class LaggedIndicator:
    """Compose a ``BaseIndicator`` with shifted copies of its output columns.

    Lag 0 (the unshifted column) is always included, even if not explicitly
    listed. All lags are deduplicated and sorted in ascending order.

    Parameters
    ----------
    indicator : BaseIndicator
        The indicator instance to compute before applying lags.
    lags : list[int]
        Lag values to apply. Lag 0 = unshifted. Lag *k* > 0 produces a column
        shifted by *k* bars (i.e. the value from *k* bars ago).
    """

    def __init__(self, indicator: BaseIndicator, lags: list[int]) -> None:
        self.indicator = indicator
        # Always include lag 0, deduplicate, and sort.
        lags_set = set(lags) | {0}
        self.lags: list[int] = sorted(lags_set)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicator values and append lagged columns.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame. Modified in place and returned.

        Returns
        -------
        pd.DataFrame
            DataFrame with base indicator columns and lagged copies appended.
        """
        df = self.indicator.compute(df)
        base_cols = self.indicator.output_columns
        for lag in self.lags:
            if lag == 0:
                continue
            for col in base_cols:
                df[f"{col}__lag_{lag}"] = df[col].shift(lag)
        return df

    @property
    def output_columns(self) -> list[str]:
        """All columns produced by this lagged indicator.

        Returns
        -------
        list[str]
            Base columns (lag 0) followed by lagged columns in ascending
            lag order.
        """
        base_cols = self.indicator.output_columns
        cols: list[str] = []
        for lag in self.lags:
            if lag == 0:
                cols.extend(base_cols)
            else:
                cols.extend(f"{col}__lag_{lag}" for col in base_cols)
        return cols


class FeatureMatrix:
    """Assemble lagged indicators into a feature matrix with optional scaling.

    The instance is **stateful**: after calling ``build(df, fit_scaler=True)``
    the fitted ``StandardScaler`` persists and is applied automatically on
    subsequent ``build`` calls. Use the same ``FeatureMatrix`` instance for
    both training and validation/test data.

    Parameters
    ----------
    lagged_indicators : list[LaggedIndicator]
        Ordered list of lagged indicators that define the feature set.
    """

    def __init__(self, lagged_indicators: list[LaggedIndicator]) -> None:
        self.lagged_indicators = lagged_indicators
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
            refitting. If no scaler exists, ``X`` is returned as-is.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(X, y)`` with NaN rows dropped. ``X`` has shape
            ``(n_samples, n_features)`` and ``y`` has shape ``(n_samples,)``.
        """
        from sklearn.preprocessing import StandardScaler

        df = df.copy()

        # Step 1: compute all lagged indicators
        for li in self.lagged_indicators:
            df = li.compute(df)

        # Step 2: collect feature columns
        feature_cols = self.feature_names

        # Step 3: compute target — log forward return
        y_series = np.log(df['Close']).diff().shift(-1)

        # Step 4: drop NaN rows (indicator warmup, lags, final row)
        mask = df[feature_cols].notna().all(axis=1) & y_series.notna()
        X = df.loc[mask, feature_cols].to_numpy(dtype=np.float64)
        y = y_series[mask].to_numpy(dtype=np.float64)

        # Step 5: scaling
        if fit_scaler:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        elif self._scaler is not None:
            X = self._scaler.transform(X)

        return X, y

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature column names in ``X``.

        Returns
        -------
        list[str]
        """
        cols: list[str] = []
        for li in self.lagged_indicators:
            cols.extend(li.output_columns)
        return cols

    @property
    def scaler(self) -> _StandardScaler | None:
        """The fitted ``StandardScaler``, or ``None`` if not yet fitted.

        Returns
        -------
        StandardScaler | None
        """
        return self._scaler
