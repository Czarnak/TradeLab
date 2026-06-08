from __future__ import annotations

from abc import abstractmethod

import pandas as pd

from trade_lab.signals.base import BaseSignal
from trade_lab.lag_base import LaggableColumnProducer


class BaseIndicator(LaggableColumnProducer):
    """
    Input:  One or more Signals (computed sequentially before indicator runs).
    Output: DataFrame with new indicator columns appended.

    Separation of concerns:
        compute()            → raw indicator values (e.g. RSI 0–100), with lag applied
        to_signal_strength() → interpretation mapped to [-1.0, 1.0]

    This separation means the same indicator can be interpreted
    differently by different strategies.

    The lag/compute scaffold (``compute()``, ``output_columns``, lag
    validation) is inherited from ``LaggableColumnProducer``.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals computed before this indicator.
    lag : int
        Number of bars to shift output columns backward in time. See
        ``LaggableColumnProducer`` for the full lag semantics; when lag > 0 the
        column names gain a ``_lag{k}`` suffix.
    """

    def __init__(self, *signals: BaseSignal, lag: int = 0) -> None:
        super().__init__(lag=lag)
        self.signals = signals

    # ------------------------------------------------------------------
    # Public orchestration layer
    # ------------------------------------------------------------------

    def get_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all upstream signals sequentially."""
        for signal in self.signals:
            df = signal.compute(df)
        return df

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicator values and write them to ``_raw_output_columns``.

        Must call ``get_data()`` first to run upstream signals.
        Column names written here must match ``_raw_output_columns``
        exactly — lag renaming is handled by ``compute()``.
        Convention: ``'indicator__<name>_<params>'``.
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
