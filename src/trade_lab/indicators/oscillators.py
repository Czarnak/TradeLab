from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trade_lab.indicators.base import BaseIndicator
from trade_lab.signals.base import BaseSignal


class RSI(BaseIndicator):
    """Relative Strength Index oscillator.

    Measures the speed and magnitude of recent price changes
    on a scale of 0 to 100.

    Signal strength uses a mean-reversion interpretation:
        RSI > 70  → overbought  → bearish (negative strength)
        RSI < 30  → oversold    → bullish (positive strength)
        RSI ≈ 50  → neutral     → ~0

    Mapping: tanh((50 - RSI) / 20)

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        Price column to compute RSI on.
    period : int
        Lookback window (Wilder smoothing).
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        period: int = 14,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.column = column
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        delta = df[self.column].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / self.period, min_periods=self.period, adjust=False,
        ).mean()
        avg_loss = loss.ewm(
            alpha=1 / self.period, min_periods=self.period, adjust=False,
        ).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        df[self._raw_output_columns[0]] = rsi
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        rsi = df[self.output_columns[0]]
        return pd.Series(np.tanh((50 - rsi) / 20), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[col], mode='lines', name=col,
        ))
        fig.add_hline(y=70, line_dash='dash', line_color='red',
                      annotation_text='Overbought')
        fig.add_hline(y=30, line_dash='dash', line_color='green',
                      annotation_text='Oversold')
        fig.add_hline(y=50, line_dash='dot', line_color='gray')
        fig.update_layout(
            title=f'RSI({self.period})',
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis=dict(range=[0, 100]),
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__rsi_{self.period}']


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence oscillator.

    Computes the MACD line (fast EMA − slow EMA) and a 9-bar signal
    line (EMA of the MACD line).

    Signal strength returns the raw signal line value — not bounded to
    [-1, 1].  Use a small weight in ``StandardStrategy`` or normalise
    downstream.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        Price column to compute on.
    fast_period : int
        Fast EMA span.
    slow_period : int
        Slow EMA span.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        fast_period: int = 12,
        slow_period: int = 26,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.column = column
        self.fast_period = fast_period
        self.slow_period = slow_period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        macd_col, signal_col = self._raw_output_columns
        df[macd_col] = (
            df[self.column].ewm(span=self.fast_period, adjust=False).mean()
            - df[self.column].ewm(span=self.slow_period, adjust=False).mean()
        )
        df[signal_col] = df[macd_col].ewm(span=9, adjust=False).mean()
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        # Raw signal line — magnitude depends on price scale.
        return df[self.output_columns[1]].fillna(0)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        macd_col, signal_col = self.output_columns
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[macd_col], mode='lines', name='MACD',
        ))
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[signal_col], mode='lines', name='Signal',
        ))
        fig.update_layout(title='MACD')
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [
            f'indicator__macd_{self.fast_period}_{self.slow_period}',
            f'indicator__macd_signal_{self.fast_period}_{self.slow_period}',
        ]


class Momentum(BaseIndicator):
    """Momentum oscillator.

    Computes the price difference over ``period`` bars.
    Signal strength is the sign of the momentum: +1 (up), -1 (down), 0 (flat).

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        Price column to compute on.
    period : int
        Lookback window for the difference.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        period: int = 14,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.column = column
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        df[self._raw_output_columns[0]] = df[self.column].diff(self.period)
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.sign(df[self.output_columns[0]]), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[col], mode='lines', name='Momentum',
        ))
        fig.update_layout(title=f'Momentum({self.period})')
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__momentum_{self.period}']


class LarryWilliams(BaseIndicator):
    """Larry Williams %R oscillator.

    Measures overbought/oversold conditions on a scale of -100 to 0.

        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Signal strength uses a mean-reversion interpretation:
        %R near   0  → overbought  → bearish (negative strength)
        %R near -100 → oversold    → bullish (positive strength)

    Mapping: tanh((-50 - %R) / 20)

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        Price column used as Close in the %R formula.
    period : int
        Lookback window for highest high and lowest low.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = 'Close',
        period: int = 14,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.column = column
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        highest_high = df['High'].rolling(self.period).max()
        lowest_low = df['Low'].rolling(self.period).min()
        denominator = (highest_high - lowest_low).replace(0, np.nan)
        df[self._raw_output_columns[0]] = (
            (highest_high - df[self.column]) / denominator * -100
        )
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        lwr = df[self.output_columns[0]]
        return pd.Series(np.tanh((-50 - lwr) / 20), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[col], mode='lines', name='LW%R',
        ))
        fig.add_hline(y=-20, line_dash='dash', line_color='red',
                      annotation_text='Overbought')
        fig.add_hline(y=-80, line_dash='dash', line_color='green',
                      annotation_text='Oversold')
        fig.update_layout(
            title=f'Larry Williams %R({self.period})',
            xaxis_title='Date',
            yaxis_title='%R',
            yaxis=dict(range=[-100, 0]),
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__lw_{self.period}']