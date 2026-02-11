"""Train a Keras model on a BuiltDataset with history tracking and equity curves.

Outputs:
- Training history (loss, accuracy/mae per epoch)
- Equity curves from model signals vs benchmark
- Saved model, config, history, and plots
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from market_lab.ml.dataset_builder import BuiltDataset
from market_lab.ml.model_builder import ModelConfig
from market_lab.utils.config import ML_OUTPUTS_DIR, ensure_dirs
from market_lab.utils.logging import get_logger

log = get_logger("ml.trainer")


@dataclass
class TrainingResult:
    """Container for all training outputs."""

    history: dict  # keras history.history dict
    model: object  # keras model
    config: ModelConfig
    dataset: BuiltDataset

    # Predictions
    train_probs: np.ndarray | None = None
    val_probs: np.ndarray | None = None

    # Equity curves (set after compute_equity_curves)
    train_equity: pd.Series | None = None
    val_equity: pd.Series | None = None
    train_benchmark: pd.Series | None = None
    val_benchmark: pd.Series | None = None

    # Final metrics
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    final_train_metric: float = 0.0
    final_val_metric: float = 0.0
    val_accuracy: float = 0.0
    val_sharpe: float = 0.0


def train_model(
    model,
    dataset: BuiltDataset,
    config: ModelConfig,
    verbose: int = 0,
    epoch_callback=None,
) -> TrainingResult:
    """Train a compiled Keras model on the dataset.

    Parameters
    ----------
    model : keras.Model
        Compiled model from ``build_model()``.
    dataset : BuiltDataset
        Train/val data.
    config : ModelConfig
        Training hyperparameters.
    verbose : int
        Keras verbosity (0=silent, 1=progress, 2=one line/epoch).
    epoch_callback : callable or None
        Called with (epoch, logs_dict) after each epoch for progress updates.

    Returns
    -------
    TrainingResult
    """
    import tensorflow as tf

    callbacks = []
    if epoch_callback is not None:
        class _ProgressCB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                epoch_callback(epoch + 1, logs or {})
        callbacks.append(_ProgressCB())

    history = model.fit(
        dataset.X_train,
        dataset.y_train,
        validation_data=(dataset.X_val, dataset.y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )

    hist = history.history

    # Predictions
    train_probs = model.predict(dataset.X_train, verbose=0).flatten()
    val_probs = model.predict(dataset.X_val, verbose=0).flatten()

    # Extract final metrics
    final_train_loss = hist["loss"][-1] if "loss" in hist else 0.0
    final_val_loss = hist["val_loss"][-1] if "val_loss" in hist else 0.0

    metric_key = "accuracy" if config.task_type == "classification" else "mae"
    val_metric_key = f"val_{metric_key}"
    final_train_metric = hist.get(metric_key, [0.0])[-1]
    final_val_metric = hist.get(val_metric_key, [0.0])[-1]

    result = TrainingResult(
        history=hist,
        model=model,
        config=config,
        dataset=dataset,
        train_probs=train_probs,
        val_probs=val_probs,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        final_train_metric=final_train_metric,
        final_val_metric=final_val_metric,
        val_accuracy=final_val_metric if config.task_type == "classification" else 0.0,
    )

    log.info(
        "Training complete: %d epochs, train_%s=%.4f, val_%s=%.4f",
        config.epochs, metric_key, final_train_metric, metric_key, final_val_metric,
    )
    return result


def compute_equity_curves(
    result: TrainingResult,
    bars_list: list[pd.DataFrame],
    threshold: float = 0.5,
    benchmark_dataset_idx: int = 0,
) -> None:
    """Compute equity curves from model signals vs a benchmark.

    For classification: go long if predicted prob > threshold, else flat.
    Benchmark: buy-and-hold on the selected dataset.

    Mutates ``result`` in place, setting equity and benchmark series.

    Parameters
    ----------
    result : TrainingResult
        Must have train_probs and val_probs populated.
    bars_list : list of pd.DataFrame
        Original OHLCV data (same order as used for dataset building).
    threshold : float
        Probability threshold for going long (classification only).
    benchmark_dataset_idx : int
        Which dataset from bars_list to use as benchmark.
    """
    ds = result.dataset

    # Build bar-aligned close series for train and val periods
    # We use the index from the dataset to look up closes
    all_bars = pd.concat(bars_list, axis=0)
    # Deduplicate if same bars appear (shouldn't normally)
    all_closes = all_bars["close"]

    def _equity_from_signals(probs, index, closes):
        """Simple equity: invest $1, go long when signal > threshold."""
        aligned_close = closes.reindex(index)
        if aligned_close.isna().all():
            return pd.Series(1.0, index=index)

        returns = aligned_close.pct_change().fillna(0.0)

        if result.config.task_type == "classification":
            signals = (probs > threshold).astype(float)
        else:
            # Regression: go long if predicted return > 0
            signals = (probs > 0).astype(float)

        strategy_returns = signals * returns
        equity = (1 + strategy_returns).cumprod()
        return equity

    def _benchmark_equity(index, closes):
        """Buy-and-hold benchmark."""
        aligned_close = closes.reindex(index)
        if aligned_close.isna().all():
            return pd.Series(1.0, index=index)
        returns = aligned_close.pct_change().fillna(0.0)
        return (1 + returns).cumprod()

    # Use benchmark dataset's closes for reference
    if benchmark_dataset_idx < len(bars_list):
        bench_closes = bars_list[benchmark_dataset_idx]["close"]
    else:
        bench_closes = all_closes

    # Train equity
    result.train_equity = _equity_from_signals(
        result.train_probs, ds.train_index, all_closes,
    )
    result.train_benchmark = _benchmark_equity(ds.train_index, bench_closes)

    # Val equity
    result.val_equity = _equity_from_signals(
        result.val_probs, ds.val_index, all_closes,
    )
    result.val_benchmark = _benchmark_equity(ds.val_index, bench_closes)

    # Compute validation Sharpe of strategy
    if result.val_equity is not None and len(result.val_equity) > 1:
        val_rets = result.val_equity.pct_change().dropna()
        if val_rets.std() > 0:
            result.val_sharpe = float(
                val_rets.mean() / val_rets.std() * np.sqrt(252)
            )

    log.info(
        "Equity curves computed: val_sharpe=%.4f, benchmark_idx=%d",
        result.val_sharpe, benchmark_dataset_idx,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_accuracy_curves(result: TrainingResult, path: Path) -> None:
    """Plot train and validation accuracy (or MAE) over epochs."""
    metric_key = "accuracy" if result.config.task_type == "classification" else "mae"
    val_key = f"val_{metric_key}"

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(result.history.get(metric_key, [])) + 1)

    if metric_key in result.history:
        ax.plot(epochs, result.history[metric_key], label=f"Train {metric_key}", linewidth=1.5)
    if val_key in result.history:
        ax.plot(epochs, result.history[val_key], label=f"Val {metric_key}", linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_key.capitalize())
    ax.set_title(f"Training & Validation {metric_key.capitalize()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curves(result: TrainingResult, path: Path) -> None:
    """Plot train and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(result.history.get("loss", [])) + 1)

    if "loss" in result.history:
        ax.plot(epochs, result.history["loss"], label="Train loss", linewidth=1.5)
    if "val_loss" in result.history:
        ax.plot(epochs, result.history["val_loss"], label="Val loss", linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity_curves(
    equity: pd.Series,
    benchmark: pd.Series,
    title: str,
    path: Path,
) -> None:
    """Plot model strategy equity vs benchmark."""
    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax, label="Model strategy", linewidth=1.5)
    benchmark.plot(ax=ax, label="Buy & hold", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (normalised)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save training outputs
# ---------------------------------------------------------------------------

def save_training_result(
    result: TrainingResult,
    run_id: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Save model, config, history, and plots to disk.

    Returns the output directory.
    """
    ensure_dirs()
    run_id = run_id or uuid.uuid4().hex[:8]
    out = (output_dir or ML_OUTPUTS_DIR) / run_id
    out.mkdir(parents=True, exist_ok=True)

    # Save model
    try:
        result.model.save(out / "model.keras")
        log.info("Model saved: %s", out / "model.keras")
    except Exception as exc:
        log.warning("Could not save model: %s", exc)

    # training_history.json
    serialisable_history = {}
    for k, v in result.history.items():
        serialisable_history[k] = [float(x) for x in v]
    with open(out / "training_history.json", "w") as f:
        json.dump(serialisable_history, f, indent=2)

    # config.json
    with open(out / "config.json", "w") as f:
        json.dump(result.config.to_dict(), f, indent=2)

    # Plots
    plot_accuracy_curves(result, out / "accuracy.png")
    plot_loss_curves(result, out / "loss.png")

    if result.train_equity is not None and result.train_benchmark is not None:
        plot_equity_curves(
            result.train_equity, result.train_benchmark,
            "Train: Model vs Benchmark", out / "train_equity.png",
        )
    if result.val_equity is not None and result.val_benchmark is not None:
        plot_equity_curves(
            result.val_equity, result.val_benchmark,
            "Validation: Model vs Benchmark", out / "val_equity.png",
        )

    # Summary metrics
    summary = {
        "final_train_loss": round(result.final_train_loss, 6),
        "final_val_loss": round(result.final_val_loss, 6),
        "final_train_metric": round(result.final_train_metric, 6),
        "final_val_metric": round(result.final_val_metric, 6),
        "val_sharpe": round(result.val_sharpe, 4),
        "epochs": result.config.epochs,
        "task_type": result.config.task_type,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Training result saved to %s", out)
    return out
