import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trade_lab.indicators.base import BaseIndicator
from trade_lab.signals.base import BaseSignal

class BaseMA(BaseIndicator):
    """Base class for Moving Average indicator.

    Signal strength: normalised distance between price and SMA,
    squashed to [-1, 1] via tanh.  Price above MA → bullish,
    below → bearish.  Magnitude reflects conviction.
    """

    def __init__(self, *signals: BaseSignal, column: str = 'Close', period: int = 20):
        super().__init__(*signals)
        self.column = column
        self.period = period
        self.plot_title = f'MA({self.period})'

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        ma = df[self.output_columns[0]]
        price = df[self.column]
        std = price.rolling(self.period).std().replace(0, np.nan)
        normalized = (price - ma) / std
        return pd.Series(np.tanh(normalized.fillna(0)), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None):
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[self.column],
            mode='lines', name=self.column,
        ))
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[col],
            mode='lines', name=col,
        ))
        fig.update_layout(
            title=self.plot_title,
            xaxis_title='Date',
            yaxis_title='Price',
        )
        fig.show()

    @property
    def output_columns(self) -> list[str]:
        return [f'indicator__ma_{self.period}']

class SMA(BaseMA):
    """Simple Moving Average indicator.

    Computes the arithmetic mean of a column over a rolling window.

    Signal strength: normalised distance between price and SMA,
    squashed to [-1, 1] via tanh.  Price above SMA → bullish,
    below → bearish.  Magnitude reflects conviction.
    """

    def __init__(self, *signals: BaseSignal, column: str = 'Close', period: int = 20):
        super().__init__(*signals, column = column, period = period)
        self.plot_title = f'SMA({self.period})'

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        col = self.output_columns[0]
        df[col] = df[self.column].rolling(window=self.period).mean()
        return df

    @property
    def output_columns(self) -> list[str]:
        return [f'indicator__sma_{self.period}']


class EMA(BaseMA):
    """Exponential Moving Average indicator.

    Computes the exponentially weighted moving average of a column.
    Recent prices receive more weight than older prices.

    Signal strength: same approach as SMA — normalised distance
    between price and EMA, squashed via tanh.
    """

    def __init__(self, *signals: BaseSignal, column: str = 'Close', period: int = 20):
        super().__init__(*signals, column = column, period = period)
        self.plot_title = f'EMA({self.period})'

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        col = self.output_columns[0]
        df[col] = df[self.column].ewm(span=self.period, adjust=False).mean()
        return df

    @property
    def output_columns(self) -> list[str]:
        return [f'indicator__ema_{self.period}']

class WMA(BaseMA):
    """Weighted Moving Average indicator.

    Computes the weighted moving average of a column.
    Recent prices receive more weight than older prices.

    Signal strength: same approach as SMA — normalised distance
    between price and WMA, squashed via tanh.
    """

    def __init__(self, *signals: BaseSignal, column: str = 'Close', period: int = 20):
        super().__init__(*signals, column = column, period = period)
        self.plot_title = f'WMA({self.period})'

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        col = self.output_columns[0]
        weights = np.arange(1, self.period + 1) / self.period
        df[col] = df[self.column].rolling(self.period).apply(lambda x: np.dot(x, weights), raw=True)
        return df

    @property
    def output_columns(self) -> list[str]:
        return [f'indicator__wma_{self.period}']


class CMA(BaseMA):
    """Cumulative Moving Average indicator.

    Computes the cumulative moving average of a column.
    Recent prices receive more weight than older prices.

    Signal strength: same approach as SMA — normalised distance
    between price and CMA, squashed via tanh.
    """

    def __init__(self, *signals: BaseSignal, column: str = 'Close', period: int = 20):
        super().__init__(*signals, column = column, period = period)
        self.plot_title = f'CMA({self.period})'

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        col = self.output_columns[0]
        df[col] = df[self.column].expanding(min_periods=self.period).mean()
        return df

    @property
    def output_columns(self) -> list[str]:
        return [f'indicator__cma_{self.period}']