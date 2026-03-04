from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseTarget(ABC):
    """Generates training labels from OHLCV data.

    Labels are aligned with the DataFrame index — rows where
    the target cannot be computed (e.g. last N bars for future
    returns) will contain NaN and are dropped during preprocessing.
    """

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of target values aligned with ``df.index``."""
        ...


class FutureReturn(BaseTarget):
    """Target = tanh(log_return(t+n) * scale).

    Squashes future log-returns into [-1, 1] to match the
    tanh-activated model output.  The ``scale`` parameter controls
    sensitivity — higher values push small returns closer to ±1.

    With the default ``scale=10.0``:
        ±1%  move → ±0.10  target
        ±5%  move → ±0.46  target
        ±10% move → ±0.76  target
    """

    def __init__(self, periods: int = 5, column: str = "Close", scale: float = 10.0):
        self.periods = periods
        self.column = column
        self.scale = scale

    def generate(self, df: pd.DataFrame) -> pd.Series:
        future = df[self.column].shift(-self.periods)
        log_return = np.log(future / df[self.column])
        return pd.Series(
            np.tanh(log_return * self.scale), index=df.index, name="target"
        )


class DirectionalTarget(BaseTarget):
    """Target = sign of future price change.

    Values in {-1, 0, +1}:
        +1  price went up
        -1  price went down
         0  unchanged (rare for real data)
    """

    def __init__(self, periods: int = 5, column: str = "Close"):
        self.periods = periods
        self.column = column

    def generate(self, df: pd.DataFrame) -> pd.Series:
        future = df[self.column].shift(-self.periods)
        return pd.Series(
            np.sign(future - df[self.column]), index=df.index, name="target"
        )
