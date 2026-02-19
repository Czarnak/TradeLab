import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trade_lab.signals.base import Signal

ohlc_str = ['Open', 'High', 'Low', 'Close']

def price_normalization(data: pd.DataFrame) -> pd.DataFrame:
    cols = ['Open', 'Close', 'High', 'Low']
    return pd.DataFrame(np.log(data[cols].div(data['Close'].shift(1)).dropna()))

class OHLC(Signal):
    """OHLC signal - returns normalized input bars as a signal."""

    @property
    def name(self) -> str:
        return "OHLC"

    def run(self, bars: pd.DataFrame) -> pd.DataFrame:
        return price_normalization(bars[ohlc_str])

    def plot(self, signal_values: pd.Series) -> Any:

        fig = go.Figure(data=[go.Candlestick(
            x=signal_values.index,
            open=signal_values['Open'],
            high=signal_values['High'],
            low=signal_values['Low'],
            close=signal_values['Close']
        )])
        fig.update_layout(title="OHLC", xaxis_title="Date", yaxis_title="Price")
        return fig