"""Result container for a completed ML optimisation run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import optuna
    from sklearn.preprocessing import StandardScaler

    from trade_lab.indicators.base import BaseIndicator
    from trade_lab.strategies.ml_strategy import MLStrategy


@dataclass
class MLOptimizationResult:
    """Complete output of an ``MLOptimizer`` run.

    Attributes
    ----------
    best_params : dict[str, Any]
        Optuna parameter values that achieved the best objective score.
    best_value : float
        Objective metric value for the best trial.
    metric : str
        Name of the metric that was optimised.
    direction : str
        ``'maximize'`` or ``'minimize'``.
    trials_df : pd.DataFrame
        One row per trial with parameters, value, state, and duration.
    study : optuna.Study
        The underlying Optuna study object.
    best_model : keras.Model
        Retrained Keras model with named inputs matching ``feature_names``.
    best_feature_spec : list[BaseIndicator]
        Indicators of the best trial, each with its ``lag`` baked in.
        Replaces the former ``list[LaggedIndicator]`` — lag is now a
        first-class attribute of every indicator.
    best_strategy : MLStrategy
        Ready-to-use strategy wrapping ``best_model`` and indicators.
    feature_names : list[str]
        Ordered feature column names in the model's input space.
    scaler : StandardScaler
        Fitted scaler from training data, kept for reference. It is **already
        folded into** ``best_model``'s first Dense layer, so ``best_strategy``
        and any export feed raw features directly — do NOT re-apply it before
        inference (that would double-scale).
    val_metrics : dict[str, float]
        Full metrics dict from backtesting on validation data.
    test_metrics : dict[str, float] | None
        Full metrics dict from backtesting on test data, or ``None``.
    train_df : pd.DataFrame
        Training DataFrame, retained for downstream operations (e.g.
        pruning with scaler refit).
    n_trials_completed : int
        Number of trials that finished without error.
    n_trials_failed : int
        Number of trials that failed or were pruned.
    """

    best_params: dict[str, Any]
    best_value: float
    metric: str
    direction: str
    trials_df: pd.DataFrame
    study: "optuna.Study"
    best_model: Any  # keras.Model
    best_feature_spec: list["BaseIndicator"]
    best_strategy: "MLStrategy"
    feature_names: list[str]
    scaler: "StandardScaler"
    val_metrics: dict[str, float]
    train_df: pd.DataFrame
    test_metrics: dict[str, float] | None = None
    n_trials_completed: int = 0
    n_trials_failed: int = 0

    def summary(self) -> str:
        """Return a human-readable summary of the ML optimisation result."""
        lines = [
            "MLOptimizationResult",
            f"  Metric:     {self.metric} ({self.direction})",
            f"  Best value: {self.best_value:.4f}",
            f"  Trials:     {self.n_trials_completed} completed, "
            f"{self.n_trials_failed} failed/pruned",
            f"  Features:   {len(self.feature_names)}",
            "",
            "  Best indicators:",
        ]
        for ind in self.best_feature_spec:
            lag_str = f", lag={ind.lag}" if ind.lag > 0 else ""
            lines.append(f"    {type(ind).__name__}(period={ind.period}{lag_str})")
        lines += [
            "",
            "  Validation metrics:",
        ]
        for k, v in self.val_metrics.items():
            lines.append(
                f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}"
            )
        if self.test_metrics:
            lines.append("")
            lines.append("  Test metrics:")
            for k, v in self.test_metrics.items():
                lines.append(
                    f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}"
                )
        return "\n".join(lines)
