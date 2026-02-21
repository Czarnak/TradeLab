"""Result container for a completed optimisation run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import optuna


@dataclass
class OptimizationResult:
    """Complete output of an ``OptunaOptimizer`` run.

    Attributes
    ----------
    best_params : dict[str, Any]
        Parameter values that achieved the best objective score on the
        training data.
    best_value : float
        Objective metric value for ``best_params`` on training data.
    metric : str
        Name of the metric that was optimised (e.g. ``'sharpe_ratio'``).
    direction : str
        ``'maximize'`` or ``'minimize'``.
    trials_df : pd.DataFrame
        One row per completed trial. Columns include all param names,
        ``value`` (objective score), ``state``, ``duration_seconds``,
        and ``trial_number``. Useful for plotting the optimisation
        landscape and diagnosing search behaviour.
    study : optuna.Study
        The underlying Optuna study object. Gives access to the full Optuna
        API: importance plots, pareto front (for future multi-objective use),
        re-running trials, etc.
    train_metrics : dict[str, float]
        Full metrics dict from running the best strategy on training data.
        Computed after optimisation completes so it covers all metrics, not
        just the optimisation objective.
    val_metrics : dict[str, float] | None
        Full metrics dict from running the best strategy on validation data.
        ``None`` if no validation DataFrame was provided.
    n_trials_completed : int
        Number of trials that finished without error.
    n_trials_failed : int
        Number of trials that raised an exception (caught and logged by Optuna).
    """

    best_params: dict[str, Any]
    best_value: float
    metric: str
    direction: str
    trials_df: pd.DataFrame
    study: 'optuna.Study'
    train_metrics: dict[str, float]
    val_metrics: dict[str, float] | None = None
    n_trials_completed: int = 0
    n_trials_failed: int = 0

    def summary(self) -> str:
        """Return a human-readable summary of the optimisation result.

        Returns
        -------
        str
            Multi-line summary string ready for printing.
        """
        lines = [
            "=" * 54,
            "  Optimisation Result",
            "=" * 54,
            f"  Metric     : {self.metric} ({self.direction})",
            f"  Best value : {self.best_value:.4f}  (train)",
            f"  Trials     : {self.n_trials_completed} completed, "
            f"{self.n_trials_failed} failed",
            "",
            "  Best parameters:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"    {k:<30} {v}")

        if self.val_metrics is not None:
            val_val = self.val_metrics.get(self.metric, float('nan'))
            lines += [
                "",
                f"  Validation {self.metric:<22} {val_val:.4f}",
            ]

        lines.append("=" * 54)
        return "\n".join(lines)