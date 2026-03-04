import pandas as pd
from .base import BaseStrategy
from trade_lab.indicators.base import BaseIndicator


class MLStrategy(BaseStrategy):
    """
    Keras model replaces weighted aggregation entirely.
    Indicators serve only as feature providers — no weights.

    Model is expected to output a single value in [-1.0, 1.0]
    (tanh output activation).

    Feature resolution:
        model.input_names must match DataFrame column names
        following the established naming convention.
    """

    def __init__(self, model, indicators: list[BaseIndicator], **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.indicators = indicators

    def _build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select columns matching model.input_names."""
        return df[self.model.input_names]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        for indicator in self.indicators:
            df = indicator.compute(df)
        features = self._build_feature_matrix(df)
        df["signal_strength"] = self.model.predict(features)
        return df
