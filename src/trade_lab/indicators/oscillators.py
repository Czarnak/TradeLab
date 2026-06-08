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
        column: str = "Close",
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
            alpha=1 / self.period,
            min_periods=self.period,
            adjust=False,
        ).mean()
        avg_loss = loss.ewm(
            alpha=1 / self.period,
            min_periods=self.period,
            adjust=False,
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
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name=col,
            )
        )
        fig.add_hline(
            y=70, line_dash="dash", line_color="red", annotation_text="Overbought"
        )
        fig.add_hline(
            y=30, line_dash="dash", line_color="green", annotation_text="Oversold"
        )
        fig.add_hline(y=50, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"RSI({self.period})",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__rsi_{self.period}"]


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
        column: str = "Close",
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
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[macd_col],
                mode="lines",
                name="MACD",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[signal_col],
                mode="lines",
                name="Signal",
            )
        )
        fig.update_layout(title="MACD")
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [
            f"indicator__macd_{self.fast_period}_{self.slow_period}",
            f"indicator__macd_signal_{self.fast_period}_{self.slow_period}",
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
        column: str = "Close",
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
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name="Momentum",
            )
        )
        fig.update_layout(title=f"Momentum({self.period})")
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__momentum_{self.period}"]


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
        column: str = "Close",
        period: int = 14,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.column = column
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        highest_high = df["High"].rolling(self.period).max()
        lowest_low = df["Low"].rolling(self.period).min()
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
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name="LW%R",
            )
        )
        fig.add_hline(
            y=-20, line_dash="dash", line_color="red", annotation_text="Overbought"
        )
        fig.add_hline(
            y=-80, line_dash="dash", line_color="green", annotation_text="Oversold"
        )
        fig.update_layout(
            title=f"Larry Williams %R({self.period})",
            xaxis_title="Date",
            yaxis_title="%R",
            yaxis=dict(range=[-100, 0]),
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__lw_{self.period}"]


class BollingerBands(BaseIndicator):
    """Bollinger Bands indicator.

    Computes an SMA middle band and upper/lower bands placed ``num_deviations``
    standard deviations above and below it.  Signal strength is derived from
    the %B position of price within the bands:

        %B = (price - lower) / (upper - lower)

    %B = 1.0 means price is at the upper band (overbought → bearish).
    %B = 0.0 means price is at the lower band (oversold → bullish).
    %B = 0.5 means price is at the middle band (neutral).

    Signal strength mapping: tanh((0.5 - %B) * 4)

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        Price column to base the bands on.
    period : int
        SMA lookback window and rolling standard-deviation window.
    num_deviations : float
        Number of standard deviations for band width. Default 2.0.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = "Close",
        period: int = 20,
        num_deviations: float = 2.0,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.column = column
        self.period = period
        self.num_deviations = num_deviations

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        price = df[self.column]
        middle = price.rolling(self.period).mean()
        std = price.rolling(self.period).std()
        upper_col, middle_col, lower_col = self._raw_output_columns
        df[middle_col] = middle
        df[upper_col] = middle + self.num_deviations * std
        df[lower_col] = middle - self.num_deviations * std
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        upper_col, _, lower_col = self.output_columns
        upper = df[upper_col]
        lower = df[lower_col]
        price = df[self.column]
        band_width = (upper - lower).replace(0, np.nan)
        pct_b = (price - lower) / band_width
        # tanh((0.5 - %B) * 4): %B=1 → tanh(-2) ≈ -0.96, %B=0 → tanh(2) ≈ 0.96
        return pd.Series(np.tanh((0.5 - pct_b.fillna(0.5)) * 4), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        upper_col, middle_col, lower_col = self.output_columns
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[self.column],
                mode="lines",
                name=self.column,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[upper_col],
                mode="lines",
                name="Upper",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[middle_col],
                mode="lines",
                name="Middle",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[lower_col],
                mode="lines",
                name="Lower",
                line=dict(dash="dash"),
                fill="tonexty",
                fillcolor="rgba(100,149,237,0.1)",
            )
        )
        fig.update_layout(
            title=f"Bollinger Bands({self.period}, {self.num_deviations})",
            xaxis_title="Date",
            yaxis_title="Price",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [
            f"indicator__bb_upper_{self.period}",
            f"indicator__bb_middle_{self.period}",
            f"indicator__bb_lower_{self.period}",
        ]


class CCI(BaseIndicator):
    """Commodity Channel Index oscillator.

    Measures how far price has deviated from its recent statistical mean,
    normalised by mean absolute deviation:

        typical = (High + Low + Close) / 3
        CCI = (typical - SMA(typical)) / (0.015 * mean_abs_dev(typical))

    The 0.015 constant scales so that ~70–80% of CCI values fall within
    ±100 under normal conditions. Values outside ±100 indicate trend
    extremes.

    Signal strength: tanh(CCI / 100)
        CCI = +100 → tanh(1.0) ≈ +0.76  (overbought → bearish)
        CCI = -100 → tanh(-1.0) ≈ -0.76 (oversold → bullish)

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    period : int
        Lookback window for SMA and mean absolute deviation.
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
        typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
        sma = typical.rolling(self.period).mean()
        # Mean absolute deviation of typical price from its SMA
        mean_dev = typical.rolling(self.period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        denom = (0.015 * mean_dev).replace(0, np.nan)
        df[self._raw_output_columns[0]] = (typical - sma) / denom
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        cci = df[self.output_columns[0]]
        # Mean-reversion: high CCI = overbought = bearish
        return pd.Series(np.tanh(-cci / 100), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name="CCI",
            )
        )
        fig.add_hline(
            y=100, line_dash="dash", line_color="red", annotation_text="Overbought"
        )
        fig.add_hline(
            y=-100, line_dash="dash", line_color="green", annotation_text="Oversold"
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"CCI({self.period})",
            xaxis_title="Date",
            yaxis_title="CCI",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__cci_{self.period}"]


class Stochastic(BaseIndicator):
    """Stochastic Oscillator.

    Measures the position of the closing price relative to the high–low
    range over a lookback window, optionally smoothed:

        %K = 100 * (close - lowest_low(k)) / (highest_high(k) - lowest_low(k))

    A ``slowing`` parameter averages %K before applying the final D-period
    SMA to produce the signal line.

    Signal strength: tanh((50 - %K) / 20)
        %K = 80 → tanh(-1.5) ≈ -0.91 (overbought → bearish)
        %K = 20 → tanh(+1.5) ≈ +0.91 (oversold → bullish)

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    k_period : int
        Lookback window for highest high / lowest low.
    d_period : int
        SMA period applied to %K to produce the %D signal line.
    slowing : int
        Internal SMA smoothing applied to raw %K before computing %D.
        ``slowing=1`` returns the raw fast stochastic.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        k_period: int = 14,
        d_period: int = 3,
        slowing: int = 3,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        highest_high = df["High"].rolling(self.k_period).max()
        lowest_low = df["Low"].rolling(self.k_period).min()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        # Raw %K
        raw_k = 100.0 * (df["Close"] - lowest_low) / denom
        # Apply slowing: SMA of raw_k over `slowing` bars
        k_col, d_col = self._raw_output_columns
        df[k_col] = raw_k.rolling(self.slowing).mean()
        df[d_col] = df[k_col].rolling(self.d_period).mean()
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        k_col = self.output_columns[0]
        k = df[k_col]
        # Mean-reversion: high %K = overbought = bearish (same mapping as RSI)
        return pd.Series(np.tanh((50.0 - k.fillna(50)) / 20.0), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        k_col, d_col = self.output_columns
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[k_col],
                mode="lines",
                name="%K",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[d_col],
                mode="lines",
                name="%D",
                line=dict(dash="dot"),
            )
        )
        fig.add_hline(
            y=80, line_dash="dash", line_color="red", annotation_text="Overbought"
        )
        fig.add_hline(
            y=20, line_dash="dash", line_color="green", annotation_text="Oversold"
        )
        fig.update_layout(
            title=f"Stochastic({self.k_period},{self.d_period},{self.slowing})",
            xaxis_title="Date",
            yaxis_title="%",
            yaxis=dict(range=[0, 100]),
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [
            f"indicator__stoch_k_{self.k_period}_{self.d_period}_{self.slowing}",
            f"indicator__stoch_d_{self.k_period}_{self.d_period}_{self.slowing}",
        ]


class ROC(BaseIndicator):
    """Rate of Change oscillator.

    Measures the percentage price change over a fixed lookback window:

        ROC = (price - price[period]) / price[period] * 100

    ROC oscillates around zero. Positive values indicate upward momentum,
    negative values downward momentum.

    Signal strength: tanh(ROC / rolling_std(ROC, period))
    Normalising by its own rolling standard deviation makes the signal
    comparable across different assets and market regimes.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        Price column to compute ROC on.
    period : int
        Lookback window in bars.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = "Close",
        period: int = 12,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.column = column
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        price = df[self.column]
        past_price = price.shift(self.period).replace(0, np.nan)
        df[self._raw_output_columns[0]] = (price - past_price) / past_price * 100.0
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        roc = df[self.output_columns[0]]
        std = roc.rolling(self.period).std().replace(0, np.nan)
        normalised = (roc / std).fillna(0)
        return pd.Series(np.tanh(normalised), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name="ROC",
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"ROC({self.period})",
            xaxis_title="Date",
            yaxis_title="ROC (%)",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__roc_{self.period}"]


class TRIX(BaseIndicator):
    """Triple Exponential Average oscillator.

    Applies EMA smoothing three times, then measures the percentage rate of
    change of the resulting triple-smoothed series:

        EMA1 = EMA(price, period)
        EMA2 = EMA(EMA1, period)
        EMA3 = EMA(EMA2, period)
        TRIX = (EMA3 - EMA3[1]) / EMA3[1] * 100

    The triple smoothing makes TRIX very resistant to noise and short-term
    price fluctuations. A positive TRIX value indicates rising long-term
    momentum; negative indicates falling.

    Signal strength: tanh(TRIX / rolling_std(TRIX, period))
    Normalised by its own rolling standard deviation for cross-asset
    comparability.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        Price column to compute TRIX on.
    period : int
        EMA lookback applied at all three smoothing stages.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = "Close",
        period: int = 14,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.column = column
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        ema1 = df[self.column].ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()
        prev_ema3 = ema3.shift(1).replace(0, np.nan)
        df[self._raw_output_columns[0]] = (ema3 - prev_ema3) / prev_ema3 * 100.0
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        trix = df[self.output_columns[0]]
        std = trix.rolling(self.period).std().replace(0, np.nan)
        normalised = (trix / std).fillna(0)
        return pd.Series(np.tanh(normalised), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name="TRIX",
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"TRIX({self.period})",
            xaxis_title="Date",
            yaxis_title="TRIX (%)",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__trix_{self.period}"]


class DPO(BaseIndicator):
    """Detrended Price Oscillator.

    Removes the dominant trend from price by comparing the current price to a
    shorter moving average, isolating shorter-term price cycles:

        half = period // 2 + 1
        DPO = price - SMA(price, half)

    Note: this is the non-displaced variant. The classic DPO additionally
    shifts the SMA back by ``period // 2 + 1`` bars; this implementation uses
    the shorter SMA window directly without that backward displacement.

    Signal strength: tanh(DPO / rolling_std(price, period))
    Normalised by the price rolling standard deviation to make the signal
    scale-independent.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    column : str
        Price column to compute DPO on.
    period : int
        Lookback window. The internal SMA uses ``period // 2 + 1`` bars.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        column: str = "Close",
        period: int = 20,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.column = column
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        price = df[self.column]
        half_period = self.period // 2 + 1
        sma = price.rolling(half_period).mean()
        df[self._raw_output_columns[0]] = price - sma
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        dpo = df[self.output_columns[0]]
        price_std = df[self.column].rolling(self.period).std().replace(0, np.nan)
        normalised = (dpo / price_std).fillna(0)
        return pd.Series(np.tanh(normalised), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name="DPO",
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"DPO({self.period})",
            xaxis_title="Date",
            yaxis_title="DPO",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__dpo_{self.period}"]


class RVI(BaseIndicator):
    """Relative Vigor Index oscillator.

    Measures the tendency of prices to close higher than they open (vigor) in
    a rising market versus falling in a declining one. Uses triangular weights
    (1, 2, 2, 1) over 4-bar windows to smooth the close-open and high-low
    numerator and denominator before summing over ``period`` bars:

        co_smooth = (CO + 2*CO[-1] + 2*CO[-2] + CO[-3]) / 6
        hl_smooth = (HL + 2*HL[-1] + 2*HL[-2] + HL[-3]) / 6
        RVI = sum(co_smooth, period) / sum(hl_smooth, period)

    Where CO = Close - Open and HL = High - Low.

    A signal line (4-bar triangular average of RVI) is also produced.

    Signal strength: tanh(RVI * 5)
    RVI is naturally centred at 0 and typically bounded in [-1, 1];
    the ×5 factor ensures the tanh is well-saturated at the extremes.

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    period : int
        Summation window for the RVI numerator and denominator.
    lag : int
        Bars to shift output backward. See ``BaseIndicator`` for details.
    """

    def __init__(
        self,
        *signals: BaseSignal,
        period: int = 10,
        lag: int = 0,
    ) -> None:
        super().__init__(*signals, lag=lag)
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        co = df["Close"] - df["Open"]
        hl = df["High"] - df["Low"]
        # Triangular 4-bar smoothing (weights 1,2,2,1 → sum=6)
        co_smooth = (co + 2 * co.shift(1) + 2 * co.shift(2) + co.shift(3)) / 6.0
        hl_smooth = (hl + 2 * hl.shift(1) + 2 * hl.shift(2) + hl.shift(3)) / 6.0
        numerator = co_smooth.rolling(self.period).sum()
        denominator = hl_smooth.rolling(self.period).sum().replace(0, np.nan)
        rvi = numerator / denominator
        # Signal line: same triangular smoothing applied to RVI
        signal = (rvi + 2 * rvi.shift(1) + 2 * rvi.shift(2) + rvi.shift(3)) / 6.0
        rvi_col, signal_col = self._raw_output_columns
        df[rvi_col] = rvi
        df[signal_col] = signal
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        rvi_col = self.output_columns[0]
        rvi = df[rvi_col]
        # RVI is centred at 0 and naturally bounded; ×5 saturates tanh
        return pd.Series(np.tanh(rvi.fillna(0) * 5.0), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        rvi_col, signal_col = self.output_columns
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[rvi_col],
                mode="lines",
                name="RVI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[signal_col],
                mode="lines",
                name="Signal",
                line=dict(dash="dot"),
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"RVI({self.period})",
            xaxis_title="Date",
            yaxis_title="RVI",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [
            f"indicator__rvi_{self.period}",
            f"indicator__rvi_signal_{self.period}",
        ]


class DeMarker(BaseIndicator):
    """DeMarker oscillator.

    Compares the current bar's high/low to the previous bar's to assess
    demand pressure. Values above 0.7 indicate overbought conditions; values
    below 0.3 indicate oversold:

        DeMax[i] = max(High[i] - High[i-1], 0)
        DeMin[i] = max(Low[i-1] - Low[i], 0)
        DeMarker  = SMA(DeMax) / (SMA(DeMax) + SMA(DeMin))

    Signal strength: tanh((0.5 - DeMarker) * 6)
        DeMarker = 0.7 → tanh(-1.2) ≈ -0.83 (overbought → bearish)
        DeMarker = 0.3 → tanh(+1.2) ≈ +0.83 (oversold → bullish)

    Parameters
    ----------
    *signals : BaseSignal
        Upstream signals.
    period : int
        SMA window applied to both DeMax and DeMin series.
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
        de_max = (df["High"] - df["High"].shift(1)).clip(lower=0)
        de_min = (df["Low"].shift(1) - df["Low"]).clip(lower=0)
        avg_de_max = de_max.rolling(self.period).mean()
        avg_de_min = de_min.rolling(self.period).mean()
        denom = (avg_de_max + avg_de_min).replace(0, np.nan)
        df[self._raw_output_columns[0]] = avg_de_max / denom
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        dem = df[self.output_columns[0]]
        return pd.Series(np.tanh((0.5 - dem.fillna(0.5)) * 6.0), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[col],
                mode="lines",
                name="DeMarker",
            )
        )
        fig.add_hline(
            y=0.7, line_dash="dash", line_color="red", annotation_text="Overbought"
        )
        fig.add_hline(
            y=0.3, line_dash="dash", line_color="green", annotation_text="Oversold"
        )
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"DeMarker({self.period})",
            xaxis_title="Date",
            yaxis_title="DeMarker",
            yaxis=dict(range=[0, 1]),
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__demarker_{self.period}"]
