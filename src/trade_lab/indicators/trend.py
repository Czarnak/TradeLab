"""Trend and volatility indicators for TradeLab.

Indicators
----------
ADX          Average Directional Index — directional trend strength
ATR          Average True Range — volatility (utility, no signal strength)
MassIndex    Mass Index — volatility expansion / reversal alert
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trade_lab.indicators.base import BaseIndicator
from trade_lab.signals.base import BaseSignal


class ADX(BaseIndicator):
    """Average Directional Index indicator.

    Computes three series using Wilder smoothing (alpha = 1/period):

    * **+DI** — positive directional indicator (upward pressure, 0–100)
    * **−DI** — negative directional indicator (downward pressure, 0–100)
    * **ADX** — trend strength, irrespective of direction (0–100)

    Signal strength combines direction and strength into a single value:

        directional_ratio = (+DI - -DI) / (+DI + -DI)   ∈ [-1, +1]
        strength          = tanh(ADX / 25)               ≈ 0 flat, ≈ 1 trending
        signal            = directional_ratio × strength

    This means the signal approaches zero in ranging markets (low ADX) even
    if the DI lines are crossed, which avoids false directional calls in
    choppy conditions.  ``ADX / 25`` is calibrated so that ADX = 25 (the
    traditional "trend confirmed" threshold) produces ``tanh(1) ≈ 0.76``
    strength.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    period : int
        Wilder smoothing window for True Range, +DM, −DM, and ADX.
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
        high = df['High']
        low = df['Low']
        close = df['Close']

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # --- True Range ---
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # --- Directional Movement ---
        # +DM: today's high exceeded yesterday's high and by more than -DM
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm_s = pd.Series(plus_dm, index=df.index)
        minus_dm_s = pd.Series(minus_dm, index=df.index)

        # --- Wilder smoothing (EWM with alpha = 1/period) ---
        # Using adjust=False replicates Wilder's recursive formula exactly:
        # Smoothed[i] = Smoothed[i-1] * (period-1)/period + raw[i] * 1/period
        alpha = 1.0 / self.period
        atr_wilder = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = plus_dm_s.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm_s.ewm(alpha=alpha, adjust=False).mean()

        # --- Directional Indicators (scaled 0–100) ---
        plus_di = 100.0 * plus_dm_smooth / atr_wilder.replace(0, np.nan)
        minus_di = 100.0 * minus_dm_smooth / atr_wilder.replace(0, np.nan)

        # --- DX and ADX ---
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        adx_col, plus_di_col, minus_di_col = self._raw_output_columns
        df[adx_col] = adx
        df[plus_di_col] = plus_di
        df[minus_di_col] = minus_di
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        adx_col, plus_di_col, minus_di_col = self.output_columns
        adx = df[adx_col]
        plus_di = df[plus_di_col]
        minus_di = df[minus_di_col]

        di_sum = (plus_di + minus_di).replace(0, np.nan)
        directional_ratio = (plus_di - minus_di) / di_sum
        strength = np.tanh(adx / 25.0)

        signal = (directional_ratio * strength).fillna(0)
        return pd.Series(signal.values, index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        adx_col, plus_di_col, minus_di_col = self.output_columns
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[adx_col],
            mode='lines', name='ADX',
        ))
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[plus_di_col],
            mode='lines', name='+DI', line=dict(dash='dash', color='green'),
        ))
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[minus_di_col],
            mode='lines', name='-DI', line=dict(dash='dash', color='red'),
        ))
        fig.add_hline(y=25, line_dash='dot', line_color='gray',
                      annotation_text='Trend threshold')
        fig.update_layout(
            title=f'ADX({self.period})',
            xaxis_title='Date',
            yaxis_title='Value',
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [
            f'indicator__adx_{self.period}',
            f'indicator__adx_plus_di_{self.period}',
            f'indicator__adx_minus_di_{self.period}',
        ]


class ATR(BaseIndicator):
    """Average True Range — volatility utility indicator.

    Measures market volatility as the Wilder-smoothed average of the True
    Range over ``period`` bars:

        TR[i]  = max(High - Low, |High - Close[-1]|, |Low - Close[-1]|)
        ATR[i] = EWM(TR, alpha=1/period)

    ATR is **purely a volatility measure** and carries no directional
    information.  It is most useful as:

    * an ML feature to encode market regime (high / low volatility),
    * a denominator for normalising other indicators,
    * an input to risk-based position sizing (already used internally by
      ``RiskBasedPositionSizer``).

    .. note::
        This indicator intentionally does **not** implement
        ``to_signal_strength``.  Calling it will raise ``NotImplementedError``.
        Use ATR as a feature column in ``MLStrategy`` or as a scaling input —
        not as a standalone signal in ``StandardStrategy``.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    period : int
        Wilder smoothing window. Default 14.
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
        high = df['High']
        low = df['Low']
        prev_close = df['Close'].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Wilder smoothing: alpha = 1/period
        df[self._raw_output_columns[0]] = tr.ewm(
            alpha=1.0 / self.period, adjust=False,
        ).mean()
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError(
            "ATR is a volatility utility indicator and does not produce a "
            "directional signal strength.  Use the raw ATR column as an ML "
            "feature or as a scaling input.  Do not include ATR in a "
            "StandardStrategy indicator list."
        )

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[col],
            mode='lines', name='ATR',
        ))
        fig.update_layout(
            title=f'ATR({self.period})',
            xaxis_title='Date',
            yaxis_title='ATR',
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__atr_{self.period}']


class MassIndex(BaseIndicator):
    """Mass Index — volatility expansion and reversal alert.

    Identifies potential price reversals by detecting when the high-low range
    expands significantly (the "bulge") and then contracts:

        EMA1[i]  = EMA(High - Low, ema_period)
        EMA2[i]  = EMA(EMA1, ema_period)
        ratio[i] = EMA1[i] / EMA2[i]
        MI[i]    = sum(ratio, sum_period)

    A "bulge reversal" signal fires when MI rises above 27 and then crosses
    back below 26.5.  The direction of the reversal is determined by other
    indicators.

    **Signal strength approximation**: rather than reproducing the threshold
    logic exactly (which is context-dependent), TradeLab maps MI to a
    continuous signal that peaks negative near the 27 bulge and rises as MI
    compresses below 26.5:

        signal = tanh((26.75 - MI) / 1.5)

    This means:
    * MI > 27  → signal ≈ −0.84  (market stretched, reversal likely)
    * MI = 26.75 → signal = 0      (neutral)
    * MI < 26.5 → signal ≈ +0.16  (compressed, no alert)

    Think of it as a "be cautious" indicator: high MI warns of instability,
    low MI indicates a quiet market.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    ema_period : int
        Period for both EMA stages of the high-low range.  Default 9.
    sum_period : int
        Summation window for the EMA ratio.  Default 25.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        ema_period: int = 9,
        sum_period: int = 25,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.ema_period = ema_period
        self.sum_period = sum_period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        hl = df['High'] - df['Low']
        ema1 = hl.ewm(span=self.ema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.ema_period, adjust=False).mean()
        ratio = ema1 / ema2.replace(0, np.nan)
        df[self._raw_output_columns[0]] = ratio.rolling(self.sum_period).sum()
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        mi = df[self.output_columns[0]]
        # Centred at 26.75 (midpoint of the 26.5–27 bulge zone)
        # Negative = stretched/reversal warning, positive = compressed/quiet
        signal = np.tanh((26.75 - mi.fillna(26.75)) / 1.5)
        return pd.Series(signal.values, index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index.to_numpy(), y=df[col],
            mode='lines', name='Mass Index',
        ))
        fig.add_hline(y=27.0, line_dash='dash', line_color='red',
                      annotation_text='Bulge upper (27)')
        fig.add_hline(y=26.5, line_dash='dash', line_color='orange',
                      annotation_text='Bulge lower (26.5)')
        fig.update_layout(
            title=f'Mass Index({self.ema_period},{self.sum_period})',
            xaxis_title='Date',
            yaxis_title='Mass Index',
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f'indicator__mi_{self.ema_period}_{self.sum_period}']
