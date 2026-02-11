"""Optuna-based strategy parameter optimization.

Derives the search space from the strategy's ``parameters_schema()`` and
optimises a user-selected objective metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import optuna
import pandas as pd

from market_lab.backtest.engine import BacktestConfig, run_backtest
from market_lab.strategies.base import Strategy
from market_lab.utils.config import OUTPUTS_DIR, ensure_dirs
from market_lab.utils.logging import get_logger

log = get_logger("backtest.optimizer")

ObjectiveMetric = Literal[
    "sharpe",
    "sortino",
    "total_return_pct",
    "profit_factor",
    "max_drawdown_pct",
    "win_rate_pct",
]

# Metrics where higher is better
_MAXIMIZE = {"sharpe", "sortino", "total_return_pct", "profit_factor", "win_rate_pct"}
# Metrics where lower is better (drawdown is negative, so minimising = best)
_MINIMIZE = {"max_drawdown_pct"}


def _suggest_param(trial: optuna.Trial, name: str, schema: dict):
    """Use Optuna trial to suggest a value based on schema."""
    ptype = schema.get("type", "float")
    if ptype == "int":
        return trial.suggest_int(
            name,
            int(schema.get("min", 2)),
            int(schema.get("max", 200)),
            step=int(schema.get("step", 1)),
        )
    elif ptype == "float":
        return trial.suggest_float(
            name,
            float(schema.get("min", 0.0)),
            float(schema.get("max", 10.0)),
            step=schema.get("step"),
        )
    elif ptype == "bool":
        return trial.suggest_categorical(name, [True, False])
    elif ptype == "enum":
        return trial.suggest_categorical(name, schema.get("choices", []))
    else:
        return schema.get("default")


def optimize(
    strategy: Strategy,
    bars: pd.DataFrame,
    objective_metric: ObjectiveMetric = "sharpe",
    n_trials: int = 100,
    backtest_config: BacktestConfig | None = None,
    seed: int | None = 42,
    progress_callback=None,
) -> dict:
    """Run Optuna optimization.

    Parameters
    ----------
    strategy : Strategy
        Strategy instance to optimize.
    bars : pd.DataFrame
        OHLCV bars.
    objective_metric : str
        Metric to optimise.
    n_trials : int
        Number of Optuna trials.
    backtest_config : BacktestConfig or None
        Base backtest config (params will vary).
    seed : int or None
        Sampler seed.
    progress_callback : callable or None
        Called with (trial_number, n_trials) after each trial.

    Returns
    -------
    dict with keys: best_params, best_value, trials_df, study.
    """
    if backtest_config is None:
        backtest_config = BacktestConfig()

    direction = "maximize" if objective_metric in _MAXIMIZE else "minimize"
    schema = strategy.parameters_schema()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=f"{strategy.name}_opt",
    )

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for pname, pschema in schema.items():
            params[pname] = _suggest_param(trial, pname, pschema)

        signals = strategy.run(bars, params)
        result = run_backtest(bars, signals.signal, backtest_config)
        value = result.metrics.get(objective_metric, 0.0)

        if progress_callback:
            progress_callback(trial.number + 1, n_trials)

        return value

    # Suppress Optuna's default logging for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    # Build trials DataFrame
    trials_data = []
    for t in study.trials:
        row = {**t.params, "value": t.value, "trial": t.number}
        trials_data.append(row)
    trials_df = pd.DataFrame(trials_data)

    best = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials_df": trials_df,
        "study": study,
    }

    log.info(
        "Optimization complete: %s=%s, best %s=%.4f",
        strategy.name, study.best_params, objective_metric, study.best_value,
    )
    return best


def save_optimization_results(
    opt_result: dict,
    output_dir: Path | None = None,
) -> Path:
    """Save optimization results to disk."""
    ensure_dirs()
    out = output_dir or OUTPUTS_DIR / "optimizations"
    out.mkdir(parents=True, exist_ok=True)

    # best_params.json
    with open(out / "best_params.json", "w") as f:
        json.dump(opt_result["best_params"], f, indent=2, default=str)

    # trials.csv
    if "trials_df" in opt_result and not opt_result["trials_df"].empty:
        opt_result["trials_df"].to_csv(out / "trials.csv", index=False)

    log.info("Optimization results saved to %s", out)
    return out
