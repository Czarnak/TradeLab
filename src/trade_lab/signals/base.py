from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseSignal(ABC):
    """
    Input:  OHLCV DataFrame or output of another Signal.
    Output: DataFrame with new columns appended.

    Signals may be chained via the source parameter.
    CyclicalTemporalSignal is an exception — it reads from df.index
    and never accepts a source.

    Parameters
    ----------
    source : BaseSignal | None
        Upstream signal to compute before this one.
    lag : int
        Number of bars to shift output columns backward in time.
        Lag 0 (default) means no shift — behaviour is identical to
        the pre-lag version. Lag k > 0 gives the value from k bars ago,
        matching pandas ``df.shift(k)``.
        When lag > 0 the column names gain a ``_lag{k}`` suffix so that
        two instances of the same signal with different lags can coexist
        in the same DataFrame without overwriting each other.
    """

    def __init__(self, source: BaseSignal | None = None, lag: int = 0) -> None:
        if lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")
        self.source = source
        self.lag = lag

    # ------------------------------------------------------------------
    # Public orchestration layer
    # ------------------------------------------------------------------

    def get_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolve upstream signal chain before computing."""
        if self.source is not None:
            return self.source.compute(df)
        return df

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute signal, apply lag shift, and return df.

        Subclasses must implement ``_compute()`` — this method is the
        public entry point and must not be overridden.

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
        """Compute signal values and write them to ``_raw_output_columns``.

        Must call ``get_data()`` to resolve the upstream chain first.
        Column names written here must match ``_raw_output_columns``
        exactly — lag renaming is handled by the base ``compute()``.
        """
        ...

    @abstractmethod
    def plot(self, df: pd.DataFrame) -> None:
        """Visualise this signal's output columns."""
        ...

    @property
    @abstractmethod
    def _raw_output_columns(self) -> list[str]:
        """Column names written by ``_compute()`` — no lag suffix.

        Used by the base ``compute()`` to know which columns to rename
        and shift. Convention: ``'signal__<n>'``.
        """
        ...