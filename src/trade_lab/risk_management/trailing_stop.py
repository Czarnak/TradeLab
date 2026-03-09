from __future__ import annotations

import pandas as pd

from trade_lab.risk_management.base import BaseTrailingStop
from trade_lab.risk_management.stop_loss import _parabolic_sar_seed


def _require_column(bar: pd.Series, column: str) -> float | None:
    if column not in bar.index:
        raise KeyError(
            f"MovingAverageTS requires column '{column}' to be present in the bar."
        )
    value = bar[column]
    if pd.isna(value):
        return None
    return float(value)


class FixedTS(BaseTrailingStop):
    def __init__(self, base_points: float, step_points: float = 10.0):
        super().__init__(step_points=step_points)
        self.base_points = base_points

    def compute_initial(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        return entry_price - direction * self.base_points

    def update(
        self,
        current_stop: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        new_stop = float(bar["Close"]) - direction * self.base_points
        return self._apply_step_guard(current_stop, new_stop, direction)


class SignalStrengthTS(BaseTrailingStop):
    def __init__(self, base_points: float, step_points: float = 10.0):
        super().__init__(step_points=step_points)
        self.base_points = base_points

    def compute_initial(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        return entry_price - direction * self.base_points * abs(signal_strength)

    def update(
        self,
        current_stop: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        new_stop = float(bar["Close"]) - direction * self.base_points * abs(
            signal_strength
        )
        return self._apply_step_guard(current_stop, new_stop, direction)


class MovingAverageTS(BaseTrailingStop):
    def __init__(self, column: str, step_points: float = 10.0):
        super().__init__(step_points=step_points)
        self.column = column

    def compute_initial(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float | None:
        return _require_column(bar, self.column)

    def update(
        self,
        current_stop: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        new_stop = _require_column(bar, self.column)
        if new_stop is None:
            return current_stop
        return self._apply_step_guard(current_stop, new_stop, direction)


class ParabolicSARTS(BaseTrailingStop):
    def __init__(
        self,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.2,
        step_points: float = 10.0,
    ):
        super().__init__(step_points=step_points)
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self._af: float | None = None
        self._ep: float | None = None
        self._sar: float | None = None

    def compute_initial(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        seed_sar, self._ep = _parabolic_sar_seed(entry_price, direction, bar)
        self._af = self.af_start
        self._sar = seed_sar + self._af * (self._ep - seed_sar)
        return self._sar

    def update(
        self,
        current_stop: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        if self._af is None or self._ep is None or self._sar is None:
            raise RuntimeError(
                "ParabolicSARTS must be initialised via compute_initial()."
            )

        high = float(bar["High"])
        low = float(bar["Low"])

        if direction == 1 and high > self._ep:
            self._ep = high
            self._af = min(self._af + self.af_step, self.af_max)
        elif direction == -1 and low < self._ep:
            self._ep = low
            self._af = min(self._af + self.af_step, self.af_max)

        self._sar = self._sar + self._af * (self._ep - self._sar)
        return self._apply_step_guard(current_stop, self._sar, direction)
