import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trade_lab.indicators.base import BaseIndicator
from trade_lab.signals.base import BaseSignal


class SMA(BaseIndicator):
    """Simple Moving Average indicator.

    Computes the arithmetic mean of a column over a rolling window.

    Signal strength: normalised distance between price and SMA,
    squashed to [-1, 1] via tanh.  Price above SMA → bullish,
    below → bearish.  Magnitude reflects conviction.
    """

    def __init__(self, *signals: BaseSignal, column: str = 'Close', period: int = 20):
        super().__init__(*signals)
        self.column = column
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        col = self.output_columns[0]
        df[col] = df[self.column].rolling(window=self.period).mean()
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        sma = df[self.output_columns[0]]
        price = df[self.column]
        std = price.rolling(self.period).std().replace(0, np.nan)
        normalized = (price - sma) / std
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
            title=f'SMA({self.period})',
            xaxis_title='Date',
            yaxis_title='Price',
        )
        fig.show()

    @property
    def output_columns(self) -> list[str]:
        return [f'indicator__sma_{self.period}']


class EMA(BaseIndicator):
    """Exponential Moving Average indicator.

    Computes the exponentially weighted moving average of a column.
    Recent prices receive more weight than older prices.

    Signal strength: same approach as SMA — normalised distance
    between price and EMA, squashed via tanh.
    """

    def __init__(self, *signals: BaseSignal, column: str = 'Close', period: int = 20):
        super().__init__(*signals)
        self.column = column
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        col = self.output_columns[0]
        df[col] = df[self.column].ewm(span=self.period, adjust=False).mean()
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        ema = df[self.output_columns[0]]
        price = df[self.column]
        std = price.rolling(self.period).std().replace(0, np.nan)
        normalized = (price - ema) / std
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
            title=f'EMA({self.period})',
            xaxis_title='Date',
            yaxis_title='Price',
        )
        fig.show()

    @property
    def output_columns(self) -> list[str]:
        return [f'indicator__ema_{self.period}']
