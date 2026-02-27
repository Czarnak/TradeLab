# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.6.0] - 2026-02-27

### Added

- Indicator library expansion:
  - `moving_averages.py`: `DEMA`, `TEMA`.
  - `oscillators.py`: `BollingerBands`, `CCI`, `Stochastic`, `ROC`, `TRIX`,
    `DPO`, `RVI`, `DeMarker`.
  - New modules:
    - `trend.py`: `ADX`, `ATR`, `MassIndex`.
    - `volume.py`: `OBV`, `ForceIndex`, `CHO`.
  - `trade_lab.indicators.__init__` now exports these classes in `__all__`.
- `mql5_export/indicator_registry.py` expanded with descriptors for the new
  moving-average, oscillator, trend/volatility, and volume indicators.
- `mql5_export` module extended with ML strategy export support.
  Export an `MLStrategy` with a trained `KerasModelWrapper` to a complete,
  ready-to-compile MetaTrader 5 Expert Advisor (`.mq5` file) with hardcoded
  Dense network weights. No Python runtime or ONNX dependency required at
  EA execution time.
  - `ml_validator.py`:
    - `validate_ml_strategy(strategy)` — pre-export checks for `MLStrategy`;
      returns `ValidationResult(is_valid, errors, warnings)`. Validates that
      `strategy.model` is a `KerasModelWrapper`, all Keras layers are Dense /
      Dropout / InputLayer / Concatenate, activations are relu / tanh / linear /
      sigmoid, the output layer has exactly 1 unit, and position sizer is
      supported. Warns for sigmoid output (convention expects tanh) and for
      models with > 10k parameters (large `.mq5` files).
  - `ml_introspector.py`:
    - `MLStrategyIntrospector.introspect(strategy)` — walks an `MLStrategy`
      object tree and returns an `MLStrategyConfig` dataclass.
    - `MLLayerConfig` — per-Dense-layer config: `index`, `units_in`,
      `units_out`, `activation`, `weights` (kernel as `list[list[float]]`),
      `biases` (list[float]). Dropout / InputLayer layers are excluded.
    - `MLStrategyConfig` — full config: `layers`, `feature_names`,
      `n_features`, `entry_threshold`, `exit_threshold`, `allow_long`,
      `allow_short`, `sizing`. Shares attribute names with `StrategyConfig`
      for fields consumed by `trade_logic.mq5.j2`, so the trade-logic
      sub-template is reused without modification.
  - `code_generator.py`:
    - `export_ml_to_mql5(strategy, output_path, ...)` — validates,
      introspects, renders via Jinja2, and writes a UTF-8 BOM `.mq5` file.
      Returns `MQL5ExportResult(filepath, code, validation,
      indicators_exported)` where `indicators_exported` contains layer
      architecture summaries.
  - `templates/ea_ml.mq5.j2` — master Jinja2 template for the ML EA,
    structured in seven sections:
    1. Property header + `#include <Trade\Trade.mqh>`.
    2. Input parameters: thresholds, magic number, spread limit, one
       `input double Feature_N_<name>` per model feature, sizing inputs.
    3. Hardcoded weight arrays: `const double LAYER{i}_W[in][out]` and
       `const double LAYER{i}_B[out]` for each Dense layer.
    4. Global variables (`CTrade`, optional ATR handle for risk sizing).
    5. Helper functions: `IsNewBar()`, `NormalizeLots()`.
    6. `ComputeMLSignal(const double &features[])` — pure-MQL5 forward pass;
       activation rendered conditionally (`MathMax(0.0,x)` for relu,
       `MathTanh(x)` for tanh, `1/(1+MathExp(-x))` for sigmoid, identity
       for linear).
    7. `OnInit` / `OnDeinit` / `OnTick` — `OnTick` assembles the feature
       array from inputs, calls `ComputeMLSignal`, then includes the
       existing `trade_logic.mq5.j2` sub-template unchanged.
  - `__init__.py` updated: `export_ml_to_mql5`, `MLStrategyIntrospector`,
    and `validate_ml_strategy` added to public API and `__all__`.

## [0.5.0] - 2026-02-26

### Added

- `mql5_export` module for translating a `StandardStrategy` into a complete,
  ready-to-compile MetaTrader 5 Expert Advisor (`.mq5` file). Install the
  optional dependency with `pip install 'TradeLab[mql5]'` (requires Jinja2).
  - `validators.py`:
    - `validate_strategy(strategy)` — pre-export checks; returns
      `ValidationResult(is_valid, errors, warnings)`. Validates indicator and
      signal types, position sizer type, and at least one indicator present.
      Emits warnings for MACD (unbounded signal line) and CMA (expanding window
      resets on EA restart).
  - `introspector.py`:
    - `StrategyIntrospector.introspect(strategy)` — walks a `StandardStrategy`
      object tree and returns a `StrategyConfig` dataclass with fully resolved
      `IndicatorConfig`, `SignalConfig`, and `SizingConfig` entries.
    - Var-name generation: one indicator of a type → plain name (e.g. `ema`);
      two of the same type → `_fast`/`_slow` ordered by period; three or more →
      `_1`/`_2`/`_3`. Input prefixes preserve abbreviations: `EMA_Fast`, `RSI`,
      `Larry_Williams` (not `Ema_Fast`).
  - `indicator_registry.py`:
    - `INDICATOR_REGISTRY` — maps all 8 indicator classes to
      `MQL5IndicatorDescriptor` entries describing handle type, buffer count,
      and MQL5 applied-price mapping. Supported: `SMA`, `EMA`, `WMA`, `CMA`,
      `RSI`, `MACD`, `Momentum`, `LarryWilliams`.
  - `signal_registry.py`:
    - `SIGNAL_REGISTRY` — maps `OHLC`, `HeikinAshi`, and
      `CyclicalTemporalSignal` to `MQL5SignalDescriptor` entries.
  - `sizing_registry.py`:
    - `SIZING_REGISTRY` — maps `None`, `FixedPositionSizer`, and
      `RiskBasedPositionSizer` to `MQL5SizingDescriptor` entries.
  - `code_generator.py`:
    - `export_to_mql5(strategy, symbol, timeframe, output_path, ...)` —
      validates, introspects, renders via Jinja2, and writes a UTF-8 BOM
      `.mq5` file (BOM required for MetaEditor compatibility). Returns
      `MQL5ExportResult(filepath, code, validation, indicators_exported)`.
  - `templates/ea_main.mq5.j2` — master Jinja2 template rendering all
    seven sections of the EA: property header, input parameters, global
    variables, static helpers (`IsNewBar`, `NormalizeLots`, `GetAppliedPrice`,
    `RollingStd`), signal-strength functions, `OnInit`/`OnDeinit`, `OnTick`.
  - `templates/helpers/signal_strength.mq5.j2` — one `GetXxxStrength()`
    function per indicator, each mirroring the Python `to_signal_strength()`
    formula exactly.
  - `templates/helpers/trade_logic.mq5.j2` — `CTrade`-based entry/exit logic
    mirroring `StandardStrategy`'s threshold interpretation, with conditional
    `allow_long` / `allow_short` blocks rendered at generation time.
  - `templates/indicators/` — standalone reference implementations for all 9
    indicator types (`sma`, `ema`, `wma`, `cma`, `rsi`, `macd`, `momentum`,
    `larry_williams`, `custom_base`).
  - `templates/signals/` — MQL5 reference implementations for `ohlc`
    (log-return), `heikin_ashi` (stateful HA bar computation), and `temporal`
    (cyclical sin/cos encoding of calendar features).
  - `templates/sizing/` — standalone lot-calculation snippets for `fixed`
    (balance × fraction / price) and `risk_based`
    (conviction-scaled, ATR-normalised sizing).
- `Jinja2>=3.1` added to both the `[mql5]` and `[dev]` optional extras in
  `pyproject.toml`.

## [0.4.0] - 2026-02-20

### Added

- `ml_optimization` module for Keras model feature selection and optimization:
  - `feature_builder.py`:
    - `LaggedIndicator` — composes any `BaseIndicator` with configurable lags.
      Lag 0 always included. Column convention: `indicator__ema_20__lag_3`.
    - `FeatureMatrix` — assembles lagged feature matrices from a list of
      `LaggedIndicator` instances. Computes log forward return as target `y`.
      Fits `StandardScaler` on training data; applies (never refits) on
      val/test. Exposes `feature_names` and `scaler` for deployment.
  - `search_space.py`:
    - `IndicatorSpec` — descriptor for one candidate indicator in the search:
      class, period range, lag range, max lags (default 5), optional flag.
  - `objective.py`:
    - `MLObjective` — picklable Optuna objective. Per trial: samples indicator
      config, builds feature matrices, trains Keras model with
      `KerasPruningCallback` (pruning on val_loss), evaluates via full
      backtest, returns chosen metric. Stores feature spec in trial user attrs
      for best-trial reconstruction.
  - `optimizer.py`:
    - `MLOptimizer` — orchestrates Optuna study with `MedianPruner`. After
      search, retrains best configuration from scratch for full `n_epochs`.
      Evaluates on val_df and optional test_df. Supports parallel workers
      via SQLite storage (same pattern as `OptunaOptimizer`).
  - `result.py`:
    - `MLOptimizationResult` — dataclass with `best_model`, `best_strategy`,
      `best_feature_spec`, `scaler`, `feature_names`, `val_metrics`,
      `test_metrics`, `train_df`, trial counts, and `summary()`.
  - `pruning.py`:
    - `ModelPruner` — post-training weight pruning. Two modes: global
      threshold (one cutoff across all layers) and per-layer percentile
      (bottom N% zeroed per layer). Dead feature detection on first Dense
      layer. `prune_model()` operates on raw Keras model; `prune_result()`
      operates on `MLOptimizationResult`, handles filtered feature spec,
      scaler refit, and fine-tuning of surviving weights.
- `[ml]` optional extras in `pyproject.toml` now include:
  - `tensorflow>=2.20.0`
  - `keras>=3.13`
  - `optuna-integration[keras]>=4.5`
  - `scikit-learn>=1.8`

### Fixed

- `backtesting/engine.py`: `_compute_atr()` now initialises `prev_close` as a
  float array, preventing `ValueError` on integer-typed OHLCV inputs when
  assigning `NaN` to the first element.
- `backtesting/metrics.py`: `_direction_stats()` now safely returns zeros when
  the trade log has no `direction` column (e.g. empty DataFrame from no-trade
  simulations), avoiding `KeyError` during metric computation.

### Changed

- `pyproject.toml`: core and ML dependency version floors were updated
  (including migration to `optuna-integration[keras]` in the `ml` extra).
- `requirements.txt` was removed in favor of `pyproject.toml` as the single
  source of dependency definitions.

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

