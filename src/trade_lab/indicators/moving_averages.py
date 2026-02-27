from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trade_lab.indicators.base import BaseIndicator
from trade_lab.signals.base import BaseSignal


class BaseMA(BaseIndicator):
    """Base class for Moving Average indicators.

    Signal strength: normalised distance between price and MA,
    squashed to [-1, 1] via tanh.  Price above MA → bullish,
    below → bearish.  Magnitude reflects conviction.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        OHLCV column to compute the MA on.
    period : int
        Lookback window.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        period: int = 20,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.column = column
        self.period = period
        self.plot_title = f'MA({self.period})'

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implemented by concrete subclasses.
        ...

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        ma = df[self.output_columns[0]]
        price = df[self.column]
        std = price.rolling(self.period).std().replace(0, np.nan)
        normalized = (price - ma) / std
        return pd.Series(np.tanh(normalized.fillna(0)), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
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
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__ma_{self.period}']


class SMA(BaseMA):
    """Simple Moving Average indicator.

    Computes the arithmetic mean of a column over a rolling window.

    Signal strength: normalised distance between price and SMA,
    squashed to [-1, 1] via tanh.  Price above SMA → bullish,
    below → bearish.  Magnitude reflects conviction.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        period: int = 20,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, column=column, period=period, lag=lag)
        self.plot_title = f'SMA({self.period})'

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        col = self._raw_output_columns[0]
        df[col] = df[self.column].rolling(window=self.period).mean()
        return df

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__sma_{self.period}']


class EMA(BaseMA):
    """Exponential Moving Average indicator.

    Computes the exponentially weighted moving average of a column.
    Recent prices receive more weight than older prices.

    Signal strength: same approach as SMA — normalised distance
    between price and EMA, squashed via tanh.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        period: int = 20,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, column=column, period=period, lag=lag)
        self.plot_title = f'EMA({self.period})'

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        col = self._raw_output_columns[0]
        df[col] = df[self.column].ewm(span=self.period, adjust=False).mean()
        return df

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__ema_{self.period}']


class WMA(BaseMA):
    """Weighted Moving Average indicator.

    Computes the linearly weighted moving average of a column.
    Weights are proportional to position: oldest bar gets weight 1,
    newest gets weight ``period``, then normalised to sum to 1.

    Signal strength: same approach as SMA — normalised distance
    between price and WMA, squashed via tanh.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        period: int = 20,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, column=column, period=period, lag=lag)
        self.plot_title = f'WMA({self.period})'

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        col = self._raw_output_columns[0]
        # Weights [1, 2, ..., period] normalised by their sum period*(period+1)/2
        weights = np.arange(1, self.period + 1) / (self.period * (self.period + 1) / 2)
        df[col] = df[self.column].rolling(self.period).apply(
            lambda x: np.dot(x, weights), raw=True,
        )
        return df

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__wma_{self.period}']


class CMA(BaseMA):
    """Cumulative Moving Average indicator.

    Computes the expanding mean of a column from the first available bar.
    The ``period`` parameter controls ``min_periods`` (the minimum number
    of bars required before a value is returned).

    Signal strength: same approach as SMA — normalised distance
    between price and CMA, squashed via tanh.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        period: int = 20,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, column=column, period=period, lag=lag)
        self.plot_title = f'CMA({self.period})'

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        col = self._raw_output_columns[0]
        df[col] = df[self.column].expanding(min_periods=self.period).mean()
        return df

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__cma_{self.period}']
    

class DEMA(BaseMA):
    """Double Exponential Moving Average indicator.

    Reduces the lag of a standard EMA by applying a correction term:

        DEMA = 2 * EMA(period) - EMA(EMA(period))

    The second EMA term is subtracted to cancel out the delay introduced by
    single smoothing, resulting in a moving average that tracks price changes
    faster than EMA for the same period.

    Signal strength: same as all MA indicators — normalised distance between
    price and DEMA, squashed to [-1, 1] via tanh.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        OHLCV column to compute DEMA on.
    period : int
        EMA lookback window applied at both smoothing stages.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        period: int = 20,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, column=column, period=period, lag=lag)
        self.plot_title = f'DEMA({self.period})'

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        ema = df[self.column].ewm(span=self.period, adjust=False).mean()
        ema_of_ema = ema.ewm(span=self.period, adjust=False).mean()
        df[self._raw_output_columns[0]] = 2.0 * ema - ema_of_ema
        return df

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__dema_{self.period}']


class TEMA(BaseMA):
    """Triple Exponential Moving Average indicator.

    Further reduces EMA lag by applying a third smoothing correction:

        TEMA = 3 * EMA(period) - 3 * EMA(EMA(period)) + EMA(EMA(EMA(period)))

    This makes TEMA the most responsive of the EMA family, at the cost of
    increased sensitivity to noise. Typically used on shorter periods.

    Signal strength: same as all MA indicators — normalised distance between
    price and TEMA, squashed to [-1, 1] via tanh.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        OHLCV column to compute TEMA on.
    period : int
        EMA lookback window applied at all three smoothing stages.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        period: int = 20,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, column=column, period=period, lag=lag)
        self.plot_title = f'TEMA({self.period})'

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        ema1 = df[self.column].ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()
        df[self._raw_output_columns[0]] = 3.0 * ema1 - 3.0 * ema2 + ema3
        return df

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__tema_{self.period}']