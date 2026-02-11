"""Compose selected features into X / y arrays with train/validation split.

Supports:
- Multiple feature builder selections with independent n_lags
- Binary classification (next-bar direction) or regression (next return)
- Time-series split (default, no shuffle) or optional shuffle
- Multiple datasets combined (for multi-symbol training)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from market_lab.ml.input_definitions import get_feature_builder
from market_lab.utils.logging import get_logger

log = get_logger("ml.dataset_builder")


@dataclass
class FeatureSelection:
    """A single feature builder selection with its lag parameter."""

    builder_name: str
    n_lags: int = 5


@dataclass
class DatasetConfig:
    """Configuration for building an ML dataset."""

    feature_selections: list[FeatureSelection] = field(default_factory=list)
    task_type: Literal["classification", "regression"] = "classification"
    split_ratio: float = 0.8
    shuffle: bool = False
    random_seed: int = 42


@dataclass
class BuiltDataset:
    """Container for the assembled dataset, ready for training."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: list[str]
    input_dim: int
    train_index: pd.DatetimeIndex
    val_index: pd.DatetimeIndex
    config: DatasetConfig


def compute_input_dim(feature_selections: Sequence[FeatureSelection]) -> int:
    """Calculate total input dimension from feature selections (no data needed)."""
    total = 0
    for fs in feature_selections:
        builder = get_feature_builder(fs.builder_name)
        total += builder.output_dim(fs.n_lags)
    return total


def build_features_df(
    bars: pd.DataFrame,
    feature_selections: Sequence[FeatureSelection],
) -> pd.DataFrame:
    """Build the combined feature DataFrame from bars and selections.

    Parameters
    ----------
    bars : pd.DataFrame
        Canonical OHLCV data.
    feature_selections : sequence of FeatureSelection
        Which features to build and with how many lags.

    Returns
    -------
    pd.DataFrame
        Combined features, rows with NaN from lagging included.
    """
    frames: list[pd.DataFrame] = []
    for fs in feature_selections:
        builder = get_feature_builder(fs.builder_name)
        feat_df = builder.build_features(bars, fs.n_lags)
        frames.append(feat_df)

    if not frames:
        raise ValueError("No feature selections provided.")

    combined = pd.concat(frames, axis=1)
    return combined


def build_target(
    bars: pd.DataFrame,
    task_type: Literal["classification", "regression"] = "classification",
) -> pd.Series:
    """Build target variable.

    Parameters
    ----------
    bars : pd.DataFrame
        Canonical OHLCV data.
    task_type : str
        ``"classification"`` → binary direction (1 if next close > current close),
        ``"regression"`` → log return of next bar.

    Returns
    -------
    pd.Series aligned to bars index (last row will be NaN).
    """
    close = bars["close"]
    if task_type == "classification":
        future_ret = close.shift(-1) / close - 1
        target = (future_ret > 0).astype(int)
        target.iloc[-1] = np.nan
    elif task_type == "regression":
        target = np.log(close.shift(-1) / close)
        target.iloc[-1] = np.nan
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    target.name = "target"
    return target


def build_dataset(
    bars_list: list[pd.DataFrame],
    config: DatasetConfig,
) -> BuiltDataset:
    """Build train/validation dataset from one or more OHLCV DataFrames.

    Parameters
    ----------
    bars_list : list of pd.DataFrame
        One or more OHLCV DataFrames (multi-symbol support).
    config : DatasetConfig
        Feature selections, task type, split ratio, etc.

    Returns
    -------
    BuiltDataset
    """
    if not bars_list:
        raise ValueError("bars_list must contain at least one DataFrame.")
    if not config.feature_selections:
        raise ValueError("At least one feature selection is required.")

    all_features: list[pd.DataFrame] = []
    all_targets: list[pd.Series] = []

    for bars in bars_list:
        feat_df = build_features_df(bars, config.feature_selections)
        target = build_target(bars, config.task_type)
        # Combine and drop NaN rows
        combined = feat_df.copy()
        combined["_target"] = target
        combined = combined.dropna()
        all_features.append(combined.drop(columns=["_target"]))
        all_targets.append(combined["_target"])

    # Concatenate across datasets
    X_df = pd.concat(all_features, axis=0)
    y_series = pd.concat(all_targets, axis=0)

    feature_names = list(X_df.columns)
    input_dim = len(feature_names)

    X = X_df.values.astype(np.float32)
    y = y_series.values.astype(np.float32)
    full_index = X_df.index

    n_total = len(X)
    n_train = int(n_total * config.split_ratio)

    if config.shuffle:
        rng = np.random.default_rng(config.random_seed)
        indices = rng.permutation(n_total)
        X = X[indices]
        y = y[indices]
        full_index = full_index[indices]

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    train_index = full_index[:n_train]
    val_index = full_index[n_train:]

    log.info(
        "Dataset built: %d features, %d train, %d val (task=%s)",
        input_dim, len(X_train), len(X_val), config.task_type,
    )

    return BuiltDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        input_dim=input_dim,
        train_index=train_index,
        val_index=val_index,
        config=config,
    )
