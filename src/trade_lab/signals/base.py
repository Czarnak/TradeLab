from __future__ import annotations

from abc import abstractmethod
from enum import Enum
import pandas as pd

from trade_lab.lag_base import LaggableColumnProducer


class BaseSignal(LaggableColumnProducer):
    """
    Input:  OHLCV DataFrame or output of another Signal.
    Output: DataFrame with new columns appended.

    Signals may be chained via the source parameter.
    CyclicalTemporalSignal is an exception — it reads from df.index
    and never accepts a source.

    The lag/compute scaffold (``compute()``, ``output_columns``, lag
    validation) is inherited from ``LaggableColumnProducer``.

    Parameters
    ----------
    source : BaseSignal | None
        Upstream signal to compute before this one.
    lag : int
        Number of bars to shift output columns backward in time. See
        ``LaggableColumnProducer`` for the full lag semantics; when lag > 0 the
        column names gain a ``_lag{k}`` suffix.
    """

    def __init__(self, source: BaseSignal | None = None, lag: int = 0) -> None:
        super().__init__(lag=lag)
        self.source = source

    # ------------------------------------------------------------------
    # Public orchestration layer
    # ------------------------------------------------------------------

    def get_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolve upstream signal chain before computing."""
        if self.source is not None:
            return self.source.compute(df)
        return df

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute signal values and write them to ``_raw_output_columns``.

        Must call ``get_data()`` to resolve the upstream chain first.
        Column names written here must match ``_raw_output_columns``
        exactly — lag renaming is handled by ``compute()``.
        Convention: ``'signal__<n>'``.
        """
        ...

    @abstractmethod
    def plot(self, df: pd.DataFrame) -> None:
        """Visualise this signal's output columns."""
        ...


class PriceSource(Enum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
