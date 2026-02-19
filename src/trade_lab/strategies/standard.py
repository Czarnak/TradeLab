import numpy as np
import pandas as pd
from .base import BaseStrategy
from trade_lab.indicators.base import BaseIndicator


class StandardStrategy(BaseStrategy):
    """
    Fuzzy weighted combination of indicator signal strengths.
    
    Weights are defined here at strategy level — indicators
    are unaware of their weights.

    Combination formula:
        raw     = Σ(weight_i x indicator_i.to_signal_strength())
        output  = tanh(raw)   # squashes to [-1, 1]
    """

    def __init__(
        self,
        indicators: list[tuple[BaseIndicator, float]],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.indicators = indicators  # (indicator, weight) pairs

    def _combine_strengths(self, df: pd.DataFrame) -> pd.Series:
        weighted_sum = sum(
            weight * indicator.to_signal_strength(df)
            for indicator, weight in self.indicators
        )
        return pd.Series(np.tanh(weighted_sum))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        for indicator, _ in self.indicators:
            df = indicator.compute(df)
        df['signal_strength'] = self._combine_strengths(df)
        return df