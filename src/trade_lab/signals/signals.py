from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trade_lab.signals.base import BaseSignal


class OHLC(BaseSignal):
    """Log-normalised OHLCV signal.

    Normalisation formula per column:
        signal(t) = log( P(t) / Close(t-1) )

    All four columns are divided by the previous bar's Close,
    capturing overnight gaps in Open and intrabar range in High/Low.

    Parameters
    ----------
    source : BaseSignal | None
        Upstream signal to chain from.
    lag : int
        Bars to shift output backward. See ``BaseSignal`` for details.
    """

    def __init__(self, source: BaseSignal | None = None, lag: int = 0) -> None:
        super().__init__(source=source, lag=lag)

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        prev_close = df['Close'].shift(1)
        for col in ['Open', 'High', 'Low', 'Close']:
            df[f'signal__log_return_{col.lower()}'] = np.log(df[col] / prev_close)
        return df

    def plot(self, df: pd.DataFrame) -> None:
        fig = go.Figure()
        for col in self.output_columns:
            fig.add_trace(go.Scatter(
                x=df.index.values, y=df[col], mode='lines', name=col,
            ))
        fig.update_layout(
            title='OHLC Log Returns',
            xaxis_title='Date',
            yaxis_title='Log Return',
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [
            'signal__log_return_open',
            'signal__log_return_high',
            'signal__log_return_low',
            'signal__log_return_close',
        ]


class HeikinAshi(BaseSignal):
    """Heikin-Ashi bars, log-normalised at output.

    Computes HA bars from standard OHLCV, then applies the same
    log normalisation as OHLC:  log( HA_P(t) / HA_Close(t-1) ).

    Parameters
    ----------
    source : BaseSignal | None
        Upstream signal to chain from.
    lag : int
        Bars to shift output backward. See ``BaseSignal`` for details.
    """

    def __init__(self, source: BaseSignal | None = None, lag: int = 0) -> None:
        super().__init__(source=source, lag=lag)
        self._raw_df: pd.DataFrame | None = None

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)

        # --- HA bars from raw OHLCV ---
        ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4.0

        ha_open = np.empty(len(df))
        ha_open[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2.0
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0
        ha_open = pd.Series(ha_open, index=df.index)

        stacked = np.column_stack([
            df['High'].to_numpy(), ha_open.to_numpy(), ha_close.to_numpy(),
        ])
        ha_high = pd.Series(stacked.max(axis=1), index=df.index)

        stacked = np.column_stack([
            df['Low'].to_numpy(), ha_open.to_numpy(), ha_close.to_numpy(),
        ])
        ha_low = pd.Series(stacked.min(axis=1), index=df.index)

        # --- Save raw HA bars (price-level) for plotting ---
        self._raw_df = pd.DataFrame({
            'signal__ha_open':  ha_open,
            'signal__ha_high':  ha_high,
            'signal__ha_low':   ha_low,
            'signal__ha_close': ha_close,
        }, index=df.index)

        # --- Log normalisation against previous HA Close ---
        prev_ha_close = ha_close.shift(1)
        df['signal__ha_open']  = np.log(ha_open  / prev_ha_close)
        df['signal__ha_high']  = np.log(ha_high  / prev_ha_close)
        df['signal__ha_low']   = np.log(ha_low   / prev_ha_close)
        df['signal__ha_close'] = np.log(ha_close / prev_ha_close)

        return df

    def plot(self, df: pd.DataFrame) -> None:
        if self._raw_df is None:
            raise RuntimeError("Call compute() before plot().")
        fig = go.Figure(data=[
            go.Candlestick(
                x=self._raw_df.index.to_numpy(),
                open=self._raw_df['signal__ha_open'],
                high=self._raw_df['signal__ha_high'],
                low=self._raw_df['signal__ha_low'],
                close=self._raw_df['signal__ha_close'],
                name='Heikin-Ashi',
                showlegend=False,
            )
        ])
        fig.update_layout(
            title='Heikin-Ashi',
            xaxis_title='Date',
            yaxis_title='Price',
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        return [
            'signal__ha_open',
            'signal__ha_high',
            'signal__ha_low',
            'signal__ha_close',
        ]