from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class LaggableColumnProducer(ABC):
    """Shared scaffold for objects that append lagged output columns to a DataFrame.

    Both :class:`~trade_lab.signals.base.BaseSignal` and
    :class:`~trade_lab.indicators.base.BaseIndicator` produce one or more named
    columns and support an optional ``lag`` that shifts those columns backward
    in time. This base holds that common machinery — the ``lag`` validation,
    the ``compute()`` orchestration, and the ``output_columns`` naming — so the
    two hierarchies do not duplicate it.

    Subclasses implement ``_compute`` (write raw values into the
    ``_raw_output_columns``) and ``_raw_output_columns`` (the unlagged names),
    plus their own upstream-data resolution layer (``get_data``).

    Parameters
    ----------
    lag : int
        Number of bars to shift output columns backward in time. Lag 0
        (default) means no shift. Lag k > 0 gives the value from k bars ago,
        matching pandas ``df.shift(k)``. When lag > 0 the column names gain a
        ``_lag{k}`` suffix so that two instances with different lags can coexist
        in the same DataFrame without overwriting each other.
    """

    def __init__(self, lag: int = 0) -> None:
        if lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")
        self.lag = lag

    # ------------------------------------------------------------------
    # Public orchestration layer
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute values, apply the lag shift, and return df.

        Subclasses must implement ``_compute()`` — this method is the public
        entry point and must not be overridden.

        Steps
        -----
        1. Call ``_compute(df)`` which writes to ``_raw_output_columns``.
        2. If ``lag > 0``: rename each raw column to its final (suffixed)
           name and shift the values by ``lag`` bars.
        3. Return df.
        """
        df = self._compute(df)
        if self.lag > 0:
            for raw, final in zip(self._raw_output_columns, self.output_columns):
                df[final] = df[raw].shift(self.lag)
                df.drop(columns=[raw], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_columns(self) -> list[str]:
        """Final column names after lag is applied.

        When ``lag == 0`` this is identical to ``_raw_output_columns``.
        When ``lag > 0`` each name gains a ``_lag{lag}`` suffix.
        """
        if self.lag == 0:
            return self._raw_output_columns
        return [f"{col}_lag{self.lag}" for col in self._raw_output_columns]

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute values and write them to ``_raw_output_columns``.

        Must resolve upstream data first (via ``get_data()``). Column names
        written here must match ``_raw_output_columns`` exactly — lag renaming
        is handled by ``compute()``.
        """
        ...

    @property
    @abstractmethod
    def _raw_output_columns(self) -> list[str]:
        """Column names written by ``_compute()`` — no lag suffix.

        Used by ``compute()`` to know which columns to rename and shift.
        """
        ...
