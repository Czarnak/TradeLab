"""Optuna-based hyperparameter optimization for ML models.

Searches over: number of layers, units per layer, activation functions,
learning rate, epochs (bounded), batch size, and classification threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
import pandas as pd

from market_lab.ml.dataset_builder import BuiltDataset
from market_lab.ml.model_builder import (
    LayerConfig,
    ModelConfig,
    VALID_ACTIVATIONS,
    build_model,
)
from market_lab.ml.trainer import TrainingResult, train_model, compute_equity_curves
from market_lab.utils.config import OUTPUTS_DIR, ensure_dirs
from market_lab.utils.logging import get_logger

log = get_logger("ml.optimizer")

MLObjective = Literal["val_accuracy", "val_sharpe"]

# Activations suitable for hidden layers (exclude sigmoid/linear which are better as output)
_HIDDEN_ACTIVATIONS = ["relu", "elu", "selu", "tanh", "swish"]


def optimize_ml(
    dataset: BuiltDataset,
    bars_list: list[pd.DataFrame],
    objective: MLObjective = "val_accuracy",
    n_trials: int = 20,
    min_layers: int = 1,
    max_layers: int = 4,
    min_units: int = 16,
    max_units: int = 256,
    min_epochs: int = 10,
    max_epochs: int = 100,
    seed: int | None = 42,
    benchmark_dataset_idx: int = 0,
    progress_callback=None,
) -> dict:
    """Run hyperparameter optimization for an ML model.

    Parameters
    ----------
    dataset : BuiltDataset
        Pre-built train/val data.
    bars_list : list of pd.DataFrame
        Original OHLCV data (for equity curve computation).
    objective : str
        ``"val_accuracy"`` or ``"val_sharpe"``.
    n_trials : int
        Number of Optuna trials.
    min_layers, max_layers : int
        Range for number of hidden layers.
    min_units, max_units : int
        Range for units per layer.
    min_epochs, max_epochs : int
        Range for training epochs.
    seed : int or None
        Sampler seed.
    benchmark_dataset_idx : int
        Which dataset to use as benchmark for equity curves.
    progress_callback : callable or None
        Called with (trial_number, n_trials) after each trial.

    Returns
    -------
    dict with keys: best_config, best_value, trials_df, best_result.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="ml_opt")

    best_result: list[TrainingResult | None] = [None]

    def trial_objective(trial: optuna.Trial) -> float:
        # Suggest architecture
        n_layers = trial.suggest_int("n_layers", min_layers, max_layers)
        layers = []
        for i in range(n_layers):
            units = trial.suggest_int(f"layer_{i}_units", min_units, max_units, step=8)
            activation = trial.suggest_categorical(f"layer_{i}_activation", _HIDDEN_ACTIVATIONS)
            layers.append(LayerConfig(units=units, activation=activation))

        # Suggest training params
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", min_epochs, max_epochs)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])

        threshold = 0.5
        if dataset.config.task_type == "classification":
            threshold = trial.suggest_float("threshold", 0.3, 0.7, step=0.05)

        # Build config
        config = ModelConfig(
            input_dim=dataset.input_dim,
            layers=layers,
            optimizer=optimizer_name,
            learning_rate=lr,
            epochs=epochs,
            batch_size=batch_size,
            loss="binary_crossentropy" if dataset.config.task_type == "classification" else "mse",
            task_type=dataset.config.task_type,
        )

        # Build, train, evaluate
        try:
            model = build_model(config)
            result = train_model(model, dataset, config, verbose=0)
            compute_equity_curves(result, bars_list, threshold, benchmark_dataset_idx)

            if objective == "val_accuracy":
                value = result.final_val_metric
            elif objective == "val_sharpe":
                value = result.val_sharpe
            else:
                value = result.final_val_metric

            # Track best
            if best_result[0] is None or value > (
                best_result[0].val_accuracy if objective == "val_accuracy"
                else best_result[0].val_sharpe
            ):
                best_result[0] = result

        except Exception as exc:
            log.warning("Trial %d failed: %s", trial.number, exc)
            value = float("-inf")

        if progress_callback:
            progress_callback(trial.number + 1, n_trials)

        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(trial_objective, n_trials=n_trials)

    # Build trials DataFrame
    trials_data = []
    for t in study.trials:
        row = {**t.params, "value": t.value, "trial": t.number}
        trials_data.append(row)
    trials_df = pd.DataFrame(trials_data)

    # Reconstruct best config
    bp = study.best_params
    n_layers = bp.get("n_layers", 1)
    best_layers = []
    for i in range(n_layers):
        best_layers.append(LayerConfig(
            units=bp.get(f"layer_{i}_units", 64),
            activation=bp.get(f"layer_{i}_activation", "relu"),
        ))

    best_config = ModelConfig(
        input_dim=dataset.input_dim,
        layers=best_layers,
        optimizer=bp.get("optimizer", "adam"),
        learning_rate=bp.get("learning_rate", 0.001),
        epochs=bp.get("epochs", 50),
        batch_size=bp.get("batch_size", 32),
        loss="binary_crossentropy" if dataset.config.task_type == "classification" else "mse",
        task_type=dataset.config.task_type,
    )

    result_dict = {
        "best_config": best_config,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials_df": trials_df,
        "study": study,
        "best_result": best_result[0],
    }

    log.info(
        "ML optimization complete: best %s=%.4f over %d trials",
        objective, study.best_value, n_trials,
    )
    return result_dict


def save_ml_optimization_results(
    opt_result: dict,
    output_dir: Path | None = None,
) -> Path:
    """Save ML optimization results to disk."""
    ensure_dirs()
    out = output_dir or OUTPUTS_DIR / "ml_optimizations"
    out.mkdir(parents=True, exist_ok=True)

    # best config
    with open(out / "best_config.json", "w") as f:
        json.dump(opt_result["best_config"].to_dict(), f, indent=2)

    # best params
    best_params = {}
    for k, v in opt_result.get("best_params", {}).items():
        best_params[k] = v
    with open(out / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    # trials
    if "trials_df" in opt_result and not opt_result["trials_df"].empty:
        opt_result["trials_df"].to_csv(out / "trials.csv", index=False)

    log.info("ML optimization results saved to %s", out)
    return out
