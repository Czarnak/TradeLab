"""Volume-based indicators for TradeLab.

All indicators in this module require a ``Volume`` column in the input
DataFrame. Yahoo Finance tick volume is an acceptable source; the
indicators do not distinguish between tick volume and real volume.

Indicators
----------
OBV          On-Balance Volume (rate-of-change, normalised)
ForceIndex   Elder's Force Index: volume × price change, smoothed
CHO          Chaikin Oscillator: fast vs slow EMA of the AD line
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trade_lab.indicators.base import BaseIndicator
from trade_lab.signals.base import BaseSignal


class OBV(BaseIndicator):
    """On-Balance Volume indicator (rate-of-change variant).

    Raw OBV is a cumulative sum and therefore non-stationary, which makes it
    unsuitable as a direct ML feature or signal strength input.  TradeLab
    exposes the *rate of change* of OBV over a rolling window, normalised by
    its own standard deviation:

        OBV[i] += volume[i]  if close[i] > close[i-1]
        OBV[i] -= volume[i]  if close[i] < close[i-1]
        OBV[i]  = OBV[i-1]  if close[i] == close[i-1]

        OBV_ROC[i] = (OBV[i] - OBV[i - period]) / std(OBV, period)

    A positive OBV_ROC means volume is flowing into the asset (accumulation);
    negative means volume is flowing out (distribution).

    Signal strength: tanh(OBV_ROC)
    Already normalised by rolling std, so tanh saturation is appropriate.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    period : int
        Lookback window for the ROC and standard-deviation normalisation.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        period: int = 14,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        close = df["Close"]
        volume = df["Volume"]
        # Direction of each bar (+1, -1, or 0)
        direction = np.sign(close.diff())
        # Cumulative OBV
        obv = (direction * volume).cumsum()
        # Rate of change, normalised by rolling std
        obv_past = obv.shift(self.period)
        std = obv.rolling(self.period).std().replace(0, np.nan)
        df[self._raw_output_columns[0]] = (obv - obv_past) / std
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        obv_roc = df[self.output_columns[0]]
        # Already std-normalised — tanh squashes to [-1, 1]
        return pd.Series(np.tanh(obv_roc.fillna(0)), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name="OBV ROC (normalised)",
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"OBV ROC({self.period})",
            xaxis_title="Date",
            yaxis_title="Normalised ROC",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__obv_roc_{self.period}"]


class ForceIndex(BaseIndicator):
    """Elder's Force Index oscillator.

    Combines price change and volume to measure the force behind a price move:

        raw_force[i] = volume[i] * (close[i] - close[i-1])

    The raw series is then smoothed with an EMA over ``period`` bars to reduce
    noise. A positive Force Index means bulls are in control; negative means
    bears dominate.

    Signal strength: tanh(force / rolling_std(force, period))
    Normalised by the force index's own rolling standard deviation to make
    the signal comparable across assets with different price and volume scales.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    period : int
        EMA smoothing window applied to raw force values.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        period: int = 13,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        raw_force = df["Volume"] * df["Close"].diff()
        df[self._raw_output_columns[0]] = raw_force.ewm(
            span=self.period,
            adjust=False,
        ).mean()
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        fi = df[self.output_columns[0]]
        std = fi.rolling(self.period).std().replace(0, np.nan)
        normalised = (fi / std).fillna(0)
        return pd.Series(np.tanh(normalised), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name="Force Index",
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"Force Index({self.period})",
            xaxis_title="Date",
            yaxis_title="Force",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__force_{self.period}"]


class CHO(BaseIndicator):
    """Chaikin Oscillator.

    Measures momentum in the Accumulation/Distribution (AD) line by comparing
    a fast EMA to a slow EMA of the cumulative AD line:

        AD[i] = ((close - low) - (high - close)) / (high - low) * volume[i]
        AD_line = cumsum(AD)
        CHO = EMA(AD_line, fast_period) - EMA(AD_line, slow_period)

    The AD line itself measures money flow: positive AD means price closed
    in the upper part of its range (bullish pressure). CHO being positive
    indicates accumulation momentum is accelerating.

    Signal strength: tanh(CHO / rolling_std(CHO, slow_period))
    Normalised by rolling standard deviation over the slow period window.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    fast_period : int
        EMA window for the fast smoothing of the AD line. Default 3.
    slow_period : int
        EMA window for the slow smoothing of the AD line. Default 10.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        fast_period: int = 3,
        slow_period: int = 10,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        volume = df["Volume"]

        # Money Flow Multiplier: position of close within the high-low range
        hl_range = (high - low).replace(0, np.nan)
        mfm = ((close - low) - (high - close)) / hl_range

        # Accumulation/Distribution line (cumulative)
        ad_line = (mfm * volume).cumsum()

        fast_ema = ad_line.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = ad_line.ewm(span=self.slow_period, adjust=False).mean()
        df[self._raw_output_columns[0]] = fast_ema - slow_ema
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        cho = df[self.output_columns[0]]
        std = cho.rolling(self.slow_period).std().replace(0, np.nan)
        normalised = (cho / std).fillna(0)
        return pd.Series(np.tanh(normalised), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name="CHO",
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"Chaikin Oscillator({self.fast_period},{self.slow_period})",
            xaxis_title="Date",
            yaxis_title="CHO",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__cho_{self.fast_period}_{self.slow_period}"]
