from abc import ABC, abstractmethod
import pandas as pd
from trade_lab.signals.base import BaseSignal


class BaseIndicator(ABC):
    """
    Input:  One or more Signals (computed sequentially before indicator runs).
    Output: DataFrame with new indicator columns appended.

    Separation of concerns:
        compute()           → raw indicator values (e.g. RSI 0-100)
        to_signal_strength() → interpretation mapped to [-1.0, 1.0]

    This separation means the same indicator can be interpreted
    differently by different strategies.
    """

    def __init__(self, *signals: BaseSignal):
        self.signals = signals

    def get_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all upstream signals sequentially."""
        for signal in self.signals:
            df = signal.compute(df)
        return df

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute indicator values and append to df.
        Must call get_data() internally.
        Convention: 'indicator__<name>'
        """
        ...

    @abstractmethod
    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Map indicator values to [-1.0, 1.0].
        -1.0 = maximum bearish conviction
         0.0 = neutral
        +1.0 = maximum bullish conviction
        """
        ...

    @abstractmethod
    def plot(self, df: pd.DataFrame, ax=None):
        """Visualise indicator output on given axes."""
        ...

    @property
    @abstractmethod
    def output_columns(self) -> list[str]:
        """Declare which columns this indicator appends."""
        ...