from __future__ import annotations

import pandas as pd

from trade_lab.risk_management.base import BaseStopLoss


def _parabolic_sar_seed(
    entry_price: float,
    direction: int,
    bar: pd.Series,
) -> tuple[float, float]:
    high = float(bar["High"])
    low = float(bar["Low"])
    if direction == 1:
        sar = min(entry_price, low)
        ep = max(entry_price, high)
    else:
        sar = max(entry_price, high)
        ep = min(entry_price, low)
    return sar, ep


def _parabolic_sar_value(
    entry_price: float,
    direction: int,
    bar: pd.Series,
    af_start: float,
) -> float:
    sar, ep = _parabolic_sar_seed(entry_price, direction, bar)
    return sar + af_start * (ep - sar)


class FixedSL(BaseStopLoss):
    def __init__(self, base_points: float):
        self.base_points = base_points

    def compute(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        return entry_price - direction * self.base_points


class SignalStrengthSL(BaseStopLoss):
    def __init__(self, base_points: float):
        self.base_points = base_points

    def compute(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        return entry_price - direction * self.base_points * abs(signal_strength)


class MovingAverageSL(BaseStopLoss):
    def __init__(self, column: str):
        self.column = column

    def compute(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float | None:
        if self.column not in bar.index:
            raise KeyError(
                f"MovingAverageSL requires column '{self.column}' to be present in the bar."
            )
        value = bar[self.column]
        if pd.isna(value):
            return None
        return float(value)


class ParabolicSARSL(BaseStopLoss):
    def __init__(
        self,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.2,
    ):
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max

    def compute(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float | None:
        if "sar" in bar.index and not pd.isna(bar["sar"]):
            return float(bar["sar"])
        return _parabolic_sar_value(entry_price, direction, bar, self.af_start)
