from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from trade_lab.signals.base import BaseSignal


class BaseIndicator(ABC):
    """
    Input:  One or more Signals (computed sequentially before indicator runs).
    Output: DataFrame with new indicator columns appended.

    Separation of concerns:
        compute()            → raw indicator values (e.g. RSI 0–100), with lag applied
        to_signal_strength() → interpretation mapped to [-1.0, 1.0]

    This separation means the same indicator can be interpreted
    differently by different strategies.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals computed before this indicator.
    lag : int
        Number of bars to shift output columns backward in time.
        Lag 0 (default) means no shift — behaviour is identical to the
        pre-lag version.  Lag k > 0 gives the value from k bars ago,
        matching pandas ``df.shift(k)``.
        When lag > 0 the column names gain a ``_lag{k}`` suffix so that
        two instances of the same indicator with different lags can coexist
        in the same DataFrame without overwriting each other.
    """

    def __init__(self, *signals: BaseSignal, lag: int = 0) -> None:
        if lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")
        self.signals = signals
        self.lag = lag

    # ------------------------------------------------------------------
    # Public orchestration layer
    # ------------------------------------------------------------------

    def get_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all upstream signals sequentially."""
        for signal in self.signals:
            df = signal.compute(df)
        return df

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicator, apply lag shift, and return df.

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
        """Compute indicator values and write them to ``_raw_output_columns``.

        Must call ``get_data()`` first to run upstream signals.
        Column names written here must match ``_raw_output_columns``
        exactly — lag renaming is handled by the base ``compute()``.
        """
        ...

    @abstractmethod
    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """Map indicator values to [-1.0, 1.0].

        Reads from ``self.output_columns`` (the final, possibly lagged names).

        -1.0 = maximum bearish conviction
         0.0 = neutral
        +1.0 = maximum bullish conviction
        """
        ...

    @abstractmethod
    def plot(self, df: pd.DataFrame, ax=None) -> None:
        """Visualise indicator output on given axes."""
        ...

    @property
    @abstractmethod
    def _raw_output_columns(self) -> list[str]:
        """Column names written by ``_compute()`` — no lag suffix.

        Used by the base ``compute()`` to know which columns to rename
        and shift. Convention: ``'indicator__<name>_<params>'``.
        """
        ...
