"""Build Keras Sequential Dense models from a configuration dict.

Supports configurable:
- Number of hidden layers and units per layer
- Activation functions per layer
- Optimizer type and learning rate
- Loss function
- Task type (classification / regression)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from market_lab.utils.logging import get_logger

log = get_logger("ml.model_builder")

VALID_ACTIVATIONS = ("relu", "sigmoid", "tanh", "elu", "selu", "linear", "swish")
VALID_OPTIMIZERS = ("adam", "rmsprop", "sgd")
VALID_LOSSES = ("binary_crossentropy", "mse", "mae")


@dataclass
class LayerConfig:
    """Configuration for a single Dense hidden layer."""

    units: int = 64
    activation: str = "relu"


@dataclass
class ModelConfig:
    """Full model architecture and training configuration."""

    input_dim: int = 10
    layers: list[LayerConfig] = field(default_factory=lambda: [
        LayerConfig(64, "relu"),
        LayerConfig(32, "relu"),
    ])
    optimizer: str = "adam"
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    loss: str = "binary_crossentropy"
    task_type: Literal["classification", "regression"] = "classification"

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty = valid)."""
        errors = []
        if self.input_dim < 1:
            errors.append("input_dim must be >= 1")
        for i, layer in enumerate(self.layers):
            if layer.units < 1:
                errors.append(f"Layer {i}: units must be >= 1")
            if layer.activation not in VALID_ACTIVATIONS:
                errors.append(
                    f"Layer {i}: invalid activation '{layer.activation}'. "
                    f"Choose from {VALID_ACTIVATIONS}"
                )
        if self.optimizer not in VALID_OPTIMIZERS:
            errors.append(f"Invalid optimizer '{self.optimizer}'. Choose from {VALID_OPTIMIZERS}")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        if self.epochs < 1:
            errors.append("epochs must be >= 1")
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.loss not in VALID_LOSSES:
            errors.append(f"Invalid loss '{self.loss}'. Choose from {VALID_LOSSES}")
        return errors

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "input_dim": self.input_dim,
            "layers": [{"units": l.units, "activation": l.activation} for l in self.layers],
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "loss": self.loss,
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Deserialise from a plain dict."""
        layers = [LayerConfig(**l) for l in d.get("layers", [])]
        return cls(
            input_dim=d.get("input_dim", 10),
            layers=layers,
            optimizer=d.get("optimizer", "adam"),
            learning_rate=d.get("learning_rate", 0.001),
            epochs=d.get("epochs", 50),
            batch_size=d.get("batch_size", 32),
            loss=d.get("loss", "binary_crossentropy"),
            task_type=d.get("task_type", "classification"),
        )


def build_model(config: ModelConfig):
    """Build and compile a Keras Sequential model from config.

    Parameters
    ----------
    config : ModelConfig
        Architecture and training configuration.

    Returns
    -------
    keras.Sequential
        Compiled model ready for training.

    Raises
    ------
    ValueError
        If config validation fails.
    """
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid model config: {'; '.join(errors)}")

    import tensorflow as tf

    model = tf.keras.Sequential()

    # Input layer (implicit in first Dense)
    model.add(tf.keras.layers.Input(shape=(config.input_dim,)))

    # Hidden layers
    for layer_cfg in config.layers:
        model.add(tf.keras.layers.Dense(
            units=layer_cfg.units,
            activation=layer_cfg.activation,
        ))

    # Output layer
    if config.task_type == "classification":
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    else:  # regression
        model.add(tf.keras.layers.Dense(1, activation="linear"))

    # Optimizer
    if config.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    elif config.optimizer == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    # Metrics
    if config.task_type == "classification":
        metrics = ["accuracy"]
    else:
        metrics = ["mae"]

    model.compile(optimizer=optimizer, loss=config.loss, metrics=metrics)

    log.info(
        "Built model: %d layers, input_dim=%d, optimizer=%s, loss=%s",
        len(config.layers), config.input_dim, config.optimizer, config.loss,
    )
    return model


def default_classification_config(input_dim: int) -> ModelConfig:
    """Return a sensible default config for binary classification."""
    return ModelConfig(
        input_dim=input_dim,
        layers=[LayerConfig(64, "relu"), LayerConfig(32, "relu")],
        optimizer="adam",
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        loss="binary_crossentropy",
        task_type="classification",
    )


def default_regression_config(input_dim: int) -> ModelConfig:
    """Return a sensible default config for regression."""
    return ModelConfig(
        input_dim=input_dim,
        layers=[LayerConfig(64, "relu"), LayerConfig(32, "relu")],
        optimizer="adam",
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        loss="mse",
        task_type="regression",
    )
