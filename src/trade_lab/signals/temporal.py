from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trade_lab.signals.base import BaseSignal


class CyclicalTemporalSignal(BaseSignal):
    """Encodes a datetime component as cyclical sin/cos features.

    Reads from df.index (must be DatetimeIndex) — never accepts a source
    signal.

    Example
    -------
    >>> CyclicalTemporalSignal('month', 12)         # annual seasonality
    >>> CyclicalTemporalSignal('month', 6)          # bi-annual seasonality
    >>> CyclicalTemporalSignal('day_of_year', 90)   # quarterly seasonality

    Output columns (lag=0)::

        signal__<component>_p<period>_sin
        signal__<component>_p<period>_cos

    Parameters
    ----------
    component : str
        Calendar component to encode. One of ``'hour'``, ``'day_of_week'``,
        ``'day_of_month'``, ``'month'``, ``'day_of_year'``.
    period : float
        Natural cycle length for the component (e.g. 12 for months).
    lag : int
        Bars to shift output backward. See ``BaseSignal`` for details.
    """

    EXTRACTORS: dict = {
        "hour": lambda idx: idx.hour,
        "day_of_week": lambda idx: idx.dayofweek,
        "day_of_month": lambda idx: idx.day,
        "month": lambda idx: idx.month,
        "day_of_year": lambda idx: idx.dayofyear,
    }

    def __init__(self, component: str, period: float, lag: int = 0) -> None:
        super().__init__(source=None, lag=lag)  # index-based, never chained
        if component not in self.EXTRACTORS:
            raise ValueError(
                f"Unknown component '{component}'. "
                f"Valid: {list(self.EXTRACTORS.keys())}"
            )
        if period <= 0:
            raise ValueError("period must be positive")
        self.component = component
        self.period = period

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self.EXTRACTORS[self.component](df.index)
        angle = 2 * np.pi * raw / self.period
        sin_col, cos_col = self._raw_output_columns
        df[sin_col] = np.sin(angle)
        df[cos_col] = np.cos(angle)
        return df

    def plot(self, df: pd.DataFrame) -> None:
        sin_col, cos_col = self.output_columns
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[sin_col],
                mode="lines",
                name=sin_col,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[cos_col],
                mode="lines",
                name=cos_col,
            )
        )
        fig.update_layout(
            title=f"Cyclical {self.component} (period={self.period})",
            xaxis_title="Date",
            yaxis_title="Value",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        p = int(self.period) if self.period == int(self.period) else self.period
        base = f"signal__{self.component}_p{p}"
        return [f"{base}_sin", f"{base}_cos"]
