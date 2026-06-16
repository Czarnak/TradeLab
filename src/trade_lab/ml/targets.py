from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseTarget(ABC):
    """Generates training labels from OHLCV data.

    Labels are aligned with the DataFrame index — rows where
    the target cannot be computed (e.g. last N bars for future
    returns) will contain NaN and are dropped during preprocessing.
    """

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of target values aligned with ``df.index``."""
        ...


class FutureReturn(BaseTarget):
    """Target = tanh(log_return(t+n) * scale).

    Squashes future log-returns into [-1, 1] to match the
    tanh-activated model output.  The ``scale`` parameter controls
    sensitivity — higher values push small returns closer to ±1.

    With the default ``scale=10.0``:
        ±1%  move → ±0.10  target
        ±5%  move → ±0.46  target
        ±10% move → ±0.76  target
    """

    def __init__(self, periods: int = 5, column: str = "Close", scale: float = 10.0):
        self.periods = periods
        self.column = column
        self.scale = scale

    def generate(self, df: pd.DataFrame) -> pd.Series:
        future = df[self.column].shift(-self.periods)
        log_return = np.log(future / df[self.column])
        return pd.Series(
            np.tanh(log_return * self.scale), index=df.index, name="target"
        )


_DRIFT_METHODS = ("rolling", "ewma", "expanding")


class ExcessVolNormalizedReturn(BaseTarget):
    """Demeaned, vol-normalized future return — the model learns deviations
    from trend, not the trend itself.

    ``tanh( (fwd - mu_n) / (vol_n * scale) )`` where
        fwd   = log(Close[t+n] / Close[t])                     (future move)
        mu_n  = trailing drift of daily log-returns at t, * n  (subtracted)
        vol_n = stdev(daily log-returns, vol_window)[t] * sqrt(n)

    Both ``mu_n`` and ``vol_n`` use only data up to ``t`` (no lookahead in the
    normalization). Subtracting the drift forces the network to time
    above/below-trend moves instead of simply predicting the index drift.

    ``drift_method`` selects how the per-bar drift is estimated:

    * ``"rolling"`` — simple mean over the last ``drift_window`` bars. Default;
      reproduces the legacy target exactly.
    * ``"ewma"`` — exponentially-weighted mean (``span=drift_span`` or
      ``drift_window``). Tracks an accelerating regime more closely than the
      lagging trailing mean, so a persistent bull no longer leaves the label
      net-positive — more negative labels make short-side signals reachable.
    * ``"expanding"`` — mean over all bars up to ``t``.

    All estimators warm up over ``drift_window`` observations so the row count is
    comparable across methods.
    """

    def __init__(
        self,
        periods: int = 5,
        column: str = "Close",
        vol_window: int = 21,
        drift_window: int = 252,
        drift_method: str = "rolling",
        drift_span: int | None = None,
        scale: float = 2.0,
    ) -> None:
        if drift_method not in _DRIFT_METHODS:
            raise ValueError(
                f"drift_method must be one of {_DRIFT_METHODS}, got {drift_method!r}"
            )
        self.periods = periods
        self.column = column
        self.vol_window = vol_window
        self.drift_window = drift_window
        self.drift_method = drift_method
        self.drift_span = drift_span if drift_span is not None else drift_window
        self.scale = scale

    def _daily_drift(self, daily: pd.Series) -> pd.Series:
        if self.drift_method == "rolling":
            return daily.rolling(self.drift_window).mean()
        if self.drift_method == "ewma":
            return daily.ewm(span=self.drift_span, min_periods=self.drift_window).mean()
        return daily.expanding(min_periods=self.drift_window).mean()

    def generate(self, df: pd.DataFrame) -> pd.Series:
        close = df[self.column]
        daily = np.log(close / close.shift(1))
        mu_n = self._daily_drift(daily) * self.periods
        vol_n = daily.rolling(self.vol_window).std() * np.sqrt(self.periods)
        fwd = np.log(close.shift(-self.periods) / close)
        target = np.tanh((fwd - mu_n) / (vol_n * self.scale))
        return pd.Series(target, index=df.index, name="target")


class DirectionalTarget(BaseTarget):
    """Target = sign of future price change.

    Values in {-1, 0, +1}:
        +1  price went up
        -1  price went down
         0  unchanged (rare for real data)
    """

    def __init__(self, periods: int = 5, column: str = "Close"):
        self.periods = periods
        self.column = column

    def generate(self, df: pd.DataFrame) -> pd.Series:
        future = df[self.column].shift(-self.periods)
        return pd.Series(
            np.sign(future - df[self.column]), index=df.index, name="target"
        )
