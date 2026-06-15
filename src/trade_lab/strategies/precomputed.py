import pandas as pd

from .base import BaseStrategy


class PrecomputedSignalStrategy(BaseStrategy):
    """Replay a fixed prediction series as ``signal_strength``.

    Unlike :class:`MLStrategy`, no model is run: the caller supplies the
    predictions directly. Predictions are aligned to each bar by **index**
    (date), so a misordered or partial series cannot silently mis-map — bars
    with no prediction become ``NaN`` and are skipped by the engine.

    This makes it the substrate for engine-based threshold calibration: feed one
    out-of-fold prediction path and run the real :class:`BacktestEngine` across a
    grid of ``entry_threshold``/``exit_threshold`` values, with sizing, financing,
    leverage and execution timing all live — no need to re-predict per grid point.

    Parameters
    ----------
    signals : pd.Series
        Prediction values in ``[-1, 1]`` indexed by timestamp. Reindexed onto the
        backtest frame; missing timestamps yield ``NaN`` (no trade on that bar).
    **kwargs
        Forwarded to :class:`BaseStrategy` (``allow_long``, ``allow_short``,
        ``position_sizer``, ``entry_threshold``, ``exit_threshold``).
    """

    def __init__(self, signals: pd.Series, **kwargs):
        super().__init__(**kwargs)
        self.signals = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal_strength"] = self.signals.reindex(out.index)
        return out
