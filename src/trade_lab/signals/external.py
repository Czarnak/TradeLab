from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go

from trade_lab.indicators.base import BaseIndicator


class ExternalSignal(BaseIndicator):
    """Wraps an existing DataFrame column as a first-class ``BaseIndicator``.

    Designed for external data sources — VIX, FED funds rate, commodity
    prices, sentiment indices — that are already merged into the DataFrame
    before the TradeLab pipeline runs.  By subclassing ``BaseIndicator``,
    ``ExternalSignal`` integrates transparently with ``FeatureMatrix`` (auto-
    discovery via the ``indicator__`` prefix) and ``MLStrategy.indicators``
    without any subclassing or monkey-patching on the user side.

    Design rationale
    ----------------
    ``ExternalSignal`` intentionally does *not* accept upstream ``*signals``
    because it reads directly from a pre-existing DataFrame column rather than
    computing a derived value.  It carries no semantic directional
    interpretation, so ``to_signal_strength`` raises ``NotImplementedError``
    — it is **not** compatible with ``StandardStrategy``.

    Parameters
    ----------
    column : str
        Name of the source column that must already exist in the DataFrame
        passed to ``compute()`` (e.g. ``"VIX"``, ``"FED_Rate"``).
    name : str | None
        Friendly name used to build the output column.  Defaults to
        ``column.lower()`` with spaces, hyphens, and dots replaced by
        underscores.
    normalization : Callable[[pd.Series], pd.Series] | None
        Optional transformation applied to the source series before writing
        to the output column.  If ``None``, values are passed through
        unchanged.
    lag : int
        Bars to shift output backward.  Handled automatically by
        ``BaseIndicator.compute()`` — do not shift inside ``normalization``.

    Output columns
    --------------
    ``indicator__<name>__lag_0`` (no lag) or
    ``indicator__<name>__lag_0_lag{n}`` (with lag).

    The ``indicator__`` prefix ensures ``FeatureMatrix`` auto-discovers this
    column when scanning for features.

    Compatibility
    -------------
    - ``MLStrategy.indicators`` — fully compatible.
    - ``FeatureMatrix`` — fully compatible.
    - ``StandardStrategy`` — **not** compatible; ``to_signal_strength`` raises
      ``NotImplementedError``.

    Examples
    --------
    >>> import numpy as np
    >>> from trade_lab.signals.external import ExternalSignal
    >>> from trade_lab.indicators.oscillators import RSI
    >>> from trade_lab.strategies.ml_strategy import MLStrategy
    >>>
    >>> vix_signal = ExternalSignal(
    ...     column="VIX",
    ...     name="vix_normalized",
    ...     normalization=lambda s: (s - 50) / 50,
    ... )
    >>> fed_signal = ExternalSignal(
    ...     column="FED_Rate",
    ...     name="fed_rate_normalized",
    ...     normalization=lambda s: (s - 50) / 50,
    ... )
    >>>
    >>> # df must already contain "VIX" and "FED_Rate" columns
    >>> strategy = MLStrategy(
    ...     model=wrapped_model,
    ...     indicators=[RSI(period=14), vix_signal, fed_signal],
    ...     entry_threshold=0.1,
    ...     exit_threshold=0.05,
    ...     allow_long=True,
    ...     allow_short=True,
    ... )
    """

    def __init__(
        self,
        column: str,
        name: str | None = None,
        normalization: Callable[[pd.Series], pd.Series] | None = None,
        lag: int = 0,
    ) -> None:
        super().__init__(lag=lag)
        self.column = column
        self.name = (
            name
            if name is not None
            else column.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        )
        self.normalization = normalization

    # ------------------------------------------------------------------
    # BaseIndicator interface
    # ------------------------------------------------------------------

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write the (optionally normalised) source column to the output column.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``self.column``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with output column appended in-place.

        Raises
        ------
        KeyError
            If ``self.column`` is not present in ``df``.
        """
        if self.column not in df.columns:
            raise KeyError(
                f"ExternalSignal: column '{self.column}' not found in DataFrame. "
                "Ensure external data is merged into the DataFrame before running the pipeline."
            )
        series = df[self.column]
        if self.normalization is not None:
            series = self.normalization(series)
        df[self._raw_output_columns[0]] = series
        return df

    @property
    def _raw_output_columns(self) -> list[str]:
        """Output column name before lag suffix is applied."""
        return [f"indicator__{self.name}__lag_0"]

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """Not supported — ``ExternalSignal`` is for ``MLStrategy`` only.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "ExternalSignal does not support StandardStrategy. "
            "Use ExternalSignal with MLStrategy only."
        )

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        """Plot the output column as a line chart.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame after ``compute()`` has been called.
        ax : ignored
            Accepted for API compatibility; Plotly figures do not use axes.
        """
        col = self.output_columns[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.values,
                y=df[col],
                mode="lines",
                name=col,
            )
        )
        fig.update_layout(
            title=f"ExternalSignal: {self.name}",
            xaxis_title="Date",
            yaxis_title="Value",
        )
        fig.show()
