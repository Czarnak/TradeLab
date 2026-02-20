# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.3.1] - 2026-02-20

### Fixed

- `optimization/optimizer.py`: `_MINIMIZE_METRICS` incorrectly included
  `max_drawdown`, `avg_loss`, `long_avg_loss`, `short_avg_loss` — all of which
  are stored as negative numbers by `compute_metrics`. Optuna was minimising
  them, searching for the worst possible values. Removed from the set; these
  metrics are now correctly maximised (value closest to zero = best).
  `annual_volatility` and `total_commission` remain in `_MINIMIZE_METRICS`
  as they are genuinely positive.
- `monte_carlo/runner.py`: a single failing simulation (e.g. degenerate
  synthetic series) crashed the entire run and discarded all completed results.
  Added per-simulation `try/except` in `_run_single_simulation`; failures now
  record NaN for all metrics and print a warning to stderr.
- `monte_carlo/generators.py`: `CircularBlockBootstrap.generate()` silently
  produced incorrect results when `block_size >= n` (number of return
  observations). Added an explicit `ValueError` guard.
- `backtesting/engine.py`: `strategy` parameter is now optional (`None` by
  default). A clear `ValueError` is raised if `run()` or `run_on()` is called
  without a strategy set. Allows constructing an engine solely for
  `fetch_data()` without passing a placeholder `strategy=None`.
- `README.md`: updated optimization example to use a dedicated engine instance
  for `fetch_data()`, making the pattern explicit and semantically correct.

## [0.3.0] - 2026-02-20

### Added

- `optimization` module for Optuna-based strategy parameter search:
  - `param_space.py`:
    - `IntParam` — integer parameter with optional step size.
    - `FloatParam` — float parameter with optional log-scale sampling.
    - `CategoricalParam` — discrete choice parameter.
    - `ParamSpace` type alias for `list[IntParam | FloatParam | CategoricalParam]`.
  - `objective.py`:
    - `Objective` — picklable callable wrapping the strategy factory and
      backtest engine into an Optuna-compatible objective function.
    - `StrategyFactory` type alias for `Callable[[dict], BaseStrategy]`.
  - `optimizer.py`:
    - `OptunaOptimizer` — orchestrates Optuna study creation, trial execution,
      and result compilation. Supports:
      - TPE sampler (Bayesian, efficient for mixed discrete/continuous spaces).
      - Single-process (in-memory storage) and multi-process (SQLite storage).
      - Optional train/validation DataFrame split (validation is strictly
        post-hoc — never influences the search).
      - Auto-inferred optimisation direction from metric name.
      - `load_if_exists=True` study resumption from existing SQLite files.
  - `result.py`:
    - `OptimizationResult` — dataclass with `best_params`, `best_value`,
      `train_metrics`, `val_metrics`, `trials_df`, `study`, trial counts,
      and `summary()` human-readable output.
- `optuna>=3.5` added to core dependencies in `pyproject.toml`.
- `optuna[integration]>=3.5` added to `[ml]` optional extras (for future
  Keras pruning callback integration).

### Fixed

- `BacktestEngine._simulate()`: trade dicts were missing the `commission` key,
  causing `KeyError` in `compute_metrics` on any backtest with at least one trade.
  Round-trip commission (`entry_comm + exit_comm`) is now recorded per trade.

### Changed

- `BacktestEngine.__init__()`: `ticker`, `start`, `end` are now optional
  (`None` by default). A `ValueError` with a clear message is raised only when
  `run()` or `fetch_data()` is called without them. `run_on(df)` is unaffected.
  This enables clean optimizer usage where the engine is built per trial without
  needing ticker/date information.

## [0.2.0] - 2026-02-20

### Added

- `monte_carlo` module for strategy robustness validation:
  - `generators.py`:
    - `BaseGenerator` — abstract base class for all MC generators.
    - `ReturnShuffler` — naive baseline; randomly permutes log returns.
    - `BlockBootstrap` — Stationary Block Bootstrap (Politis & Romano, 1994);
      geometric-distributed block lengths preserve short-range autocorrelation.
    - `CircularBlockBootstrap` — circular variant that eliminates boundary bias.
    - `GBMSimulator` — Geometric Brownian Motion with Itô-corrected drift.
  - `runner.py`:
    - `MonteCarloRunner` — orchestrates N synthetic backtests; deterministic
      per-simulation seeding for full reproducibility.
    - `MonteCarloResult` — dataclass holding raw metric distributions.
  - `analysis.py`:
    - `MonteCarloAnalysis` — `summary()`, `distributions()`,
      `confidence_interval()`, `percentile_of()`, `available_metrics()`.
- `BacktestEngine.run_on(df)` — public method for running a backtest on a
  pre-built DataFrame (skips data download); used internally by the MC runner.
- `BacktestEngine.fetch_data()` — public wrapper around `_fetch_data()` to
  expose raw data retrieval for external use (e.g. passing to `MonteCarloRunner`).

## [0.1.0] - 2026-02-19

### Added

- Initial `trade_lab` package structure with modular domains:
  - `backtesting` (`engine.py`, `metrics.py`, `report.py`)
  - `strategies` (`base.py`, `standard.py`, `ml_strategy.py`)
  - `signals` (`base.py`, `signals.py`, `temporal.py`)
  - `indicators` (`base.py`, `moving_averages.py`, `oscillators.py`)
  - `sizing` (`base.py`, `fixed.py`, `risk_based.py`)
- Event-driven `BacktestEngine` with:
  - signal-threshold based entries/exits,
  - long and short execution paths,
  - commission/slippage modeling,
  - ATR helper for volatility-aware sizing,
  - trade log and equity curve outputs.
- Performance analytics via `compute_metrics`:
  - return, annualized return, volatility,
  - Sharpe and Sortino ratios,
  - max drawdown and trade-level stats.
- HTML report generation via `generate_report` with:
  - equity curve, drawdown, and price/trade overlays.
- Built-in indicators:
  - `SMA`, `EMA`, `RSI`.
- Built-in signals:
  - `OHLC`, `HeikinAshi`, `CyclicalTemporalSignal`.
- Position sizing implementations:
  - `FixedPositionSizer`,
  - `RiskBasedPositionSizer`.
- Example script:
  - `examples/simple_ema_strategy.py`.
- Test suite for strategy composition and temporal signals.
- GitHub Actions CI workflow for test and lint jobs.
