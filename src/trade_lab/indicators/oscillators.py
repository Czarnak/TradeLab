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
    """

    def __init__(self, *signals: BaseSignal, column: str = 'Close', period: int = 14):
        super().__init__(*signals)
        self.column = column
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        delta = df[self.column].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        df[self.output_columns[0]] = rsi
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        rsi = df[self.output_columns[0]]
        return pd.Series(np.tanh((50 - rsi) / 20), index=df.index)

    def plot(self, df: pd.DataFrame):
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
    def output_columns(self) -> list[str]:
        return [f'indicator__rsi_{self.period}']
