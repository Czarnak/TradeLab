"""Result container for a completed ML optimisation run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import optuna
    from sklearn.preprocessing import StandardScaler

    from trade_lab.ml_optimization.feature_builder import LaggedIndicator
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
    best_feature_spec : list[LaggedIndicator]
        Lagged indicator configuration of the best trial.
    best_strategy : MLStrategy
        Ready-to-use strategy wrapping ``best_model`` and indicators.
    feature_names : list[str]
        Ordered feature column names in the model's input space.
    scaler : StandardScaler
        Fitted scaler from training data. Apply to new data before inference.
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
    study: 'optuna.Study'
    best_model: Any  # keras.Model
    best_feature_spec: list['LaggedIndicator']
    best_strategy: 'MLStrategy'
    feature_names: list[str]
    scaler: 'StandardScaler'
    val_metrics: dict[str, float]
    train_df: pd.DataFrame
    test_metrics: dict[str, float] | None = None
    n_trials_completed: int = 0
    n_trials_failed: int = 0

    def summary(self) -> str:
        """Return a human-readable summary of the ML optimisation result.

        Returns
        -------
        str
            Multi-line summary string ready for printing.
        """
        lines = [
            "=" * 54,
            "  ML Optimisation Result",
            "=" * 54,
            f"  Metric     : {self.metric} ({self.direction})",
            f"  Best value : {self.best_value:.4f}  (val)",
            f"  Trials     : {self.n_trials_completed} completed, "
            f"{self.n_trials_failed} failed/pruned",
            f"  Features   : {len(self.feature_names)}",
            "",
            "  Best parameters:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"    {k:<30} {v}")

        # Indicator and lag summary
        lines.append("")
        lines.append("  Selected indicators:")
        for li in self.best_feature_spec:
            ind = li.indicator
            cls_name = type(ind).__name__
            period = getattr(ind, 'period', '?')
            lags_str = ', '.join(str(lag) for lag in li.lags)
            lines.append(f"    {cls_name}(period={period})  lags=[{lags_str}]")

        # Validation metrics
        val_val = self.val_metrics.get(self.metric, float('nan'))
        lines += [
            "",
            f"  Validation {self.metric:<22} {val_val:.4f}",
        ]

        # Test metrics
        if self.test_metrics is not None:
            test_val = self.test_metrics.get(self.metric, float('nan'))
            lines.append(
                f"  Test       {self.metric:<22} {test_val:.4f}",
            )

        lines.append("=" * 54)
        return "\n".join(lines)
