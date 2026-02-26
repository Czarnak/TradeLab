import numpy as np
import pandas as pd

from trade_lab.indicators.base import BaseIndicator
from trade_lab.strategies.ml_strategy import MLStrategy
from trade_lab.strategies.standard import StandardStrategy


class DummyIndicator(BaseIndicator):
    def __init__(self, name: str, values):
        super().__init__()
        self.name = name
        self.values = pd.Series(values)
        self.compute_calls = 0
        self.signal_calls = 0

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self.compute_calls += 1
        out = self.get_data(df.copy())
        out[self._raw_output_columns[0]] = self.values.to_numpy()
        return out

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        self.signal_calls += 1
        return self.values

    def plot(self, df: pd.DataFrame, ax=None):
        return None

    @property
    def _raw_output_columns(self) -> list[str]:
        return [f"indicator__{self.name}"]


class DummyModel:
    def __init__(self, input_names, predictions):
        self.input_names = input_names
        self.predictions = np.array(predictions)
        self.last_features = None

    def predict(self, features: pd.DataFrame):
        self.last_features = features.copy()
        return self.predictions


def test_standard_strategy_computes_indicators_and_combines_signal_strength():
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    first = DummyIndicator("first", [0.2, 0.1, -0.4])
    second = DummyIndicator("second", [0.4, -0.2, 0.3])
    strategy = StandardStrategy(indicators=[(first, 0.5), (second, 1.5)])

    result = strategy.generate_signals(df.copy())

    expected_raw = 0.5 * first.values + 1.5 * second.values
    expected = np.tanh(expected_raw)
    np.testing.assert_allclose(result["signal_strength"].to_numpy(), expected.to_numpy())
    assert first.compute_calls == 1
    assert second.compute_calls == 1
    assert first.signal_calls == 1
    assert second.signal_calls == 1
    assert "indicator__first" in result.columns
    assert "indicator__second" in result.columns


def test_ml_strategy_builds_features_from_model_inputs_and_predicts():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    df = pd.DataFrame({"close": [10, 11, 12]}, index=index)
    feature_a = DummyIndicator("feature_a", [1.0, 2.0, 3.0])
    feature_b = DummyIndicator("feature_b", [0.5, 0.25, 0.75])
    model = DummyModel(
        input_names=["indicator__feature_b", "indicator__feature_a"],
        predictions=[-0.1, 0.0, 0.9],
    )
    strategy = MLStrategy(model=model, indicators=[feature_a, feature_b])

    result = strategy.generate_signals(df.copy())

    assert feature_a.compute_calls == 1
    assert feature_b.compute_calls == 1
    assert list(model.last_features.columns) == ["indicator__feature_b", "indicator__feature_a"]
    np.testing.assert_allclose(
        model.last_features.to_numpy(),
        result[["indicator__feature_b", "indicator__feature_a"]].to_numpy(),
    )
    np.testing.assert_allclose(result["signal_strength"].to_numpy(), np.array([-0.1, 0.0, 0.9]))
