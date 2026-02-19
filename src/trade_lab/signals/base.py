from abc import ABC, abstractmethod
import pandas as pd


class BaseSignal(ABC):
    """
    Input:  OHLCV DataFrame or output of another Signal.
    Output: DataFrame with new columns appended.

    Signals may be chained via the source parameter.
    CyclicalTemporalSignal is an exception — it reads from df.index
    and never accepts a source.
    """

    def __init__(self, source: 'BaseSignal | None' = None):
        self.source = source

    def get_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolve upstream signal chain before computing."""
        if self.source is not None:
            return self.source.compute(df)
        return df

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute signal and append output columns to df.
        Must call get_data() internally to respect chaining.
        """
        ...

    @abstractmethod
    def plot(self, df: pd.DataFrame):
        """Visualise this signal's output columns."""
        ...

    @property
    @abstractmethod
    def output_columns(self) -> list[str]:
        """
        Declare which columns this signal appends.
        Used for validation and ML feature resolution.
        Convention: 'signal__<name>'
        """
        ...