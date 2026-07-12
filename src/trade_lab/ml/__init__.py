from trade_lab.ml.targets import (
    BaseTarget,
    FutureReturn,
    DirectionalTarget,
    ExcessVolNormalizedReturn,
)
from trade_lab.ml.preprocessing import prepare_features, FeatureScaler
from trade_lab.ml.validation import WalkForwardSplit, WalkForwardResult
from trade_lab.ml.models import (
    KerasModelWrapper,
    dense_model,
    ensemble_average_model,
    pinball_loss,
    quantile_model,
)

try:
    from trade_lab.ml.trainer import MLTrainer
except ModuleNotFoundError:
    # tensorflow/keras not installed — trainer unavailable
    MLTrainer = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseTarget",
    "FutureReturn",
    "DirectionalTarget",
    "ExcessVolNormalizedReturn",
    "prepare_features",
    "FeatureScaler",
    "KerasModelWrapper",
    "dense_model",
    "ensemble_average_model",
    "quantile_model",
    "pinball_loss",
    "WalkForwardSplit",
    "WalkForwardResult",
    "MLTrainer",
]
