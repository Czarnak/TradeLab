# TradeLab

[![CI][ci-badge]][ci-link]
[![Python][python-badge]][python-link]
[![License: MIT][license-badge]][license-link]

TradeLab is a modular Python framework for strategy backtesting with a clear separation between:

- `signals` (feature generation),
- `indicators` (market state transforms + signal-strength mapping),
- `strategies` (how indicator strengths are combined),
- `risk_management` (take-profit, stop-loss, and trailing-stop policies),
- `position sizing`,
- `backtesting` (execution simulation, metrics, report generation),
- `monte_carlo` (synthetic data generation and robustness analysis),
- `optimization` (Optuna-based parameter and weight search),
- `mql5_export` (generate MetaTrader 5 Expert Advisors from `StandardStrategy` and `MLStrategy`).

## Features

- Event-driven backtesting engine with:
  - long/short support,
  - commission and slippage modeling,
  - trade logging, exit reasons, and equity curve output,
  - take-profit, stop-loss, and trailing-stop handling with TP/SL/TS priority over signal exits.
- Composable signal and indicator pipelines.
- Strategy abstractions:
  - `StandardStrategy` for weighted indicator combinations,
  - `MLStrategy` for model-driven predictions.
- Risk-management primitives:
  - take profit: `FixedTP`, `SignalStrengthTP`,
  - stop loss: `FixedSL`, `SignalStrengthSL`, `MovingAverageSL`, `ParabolicSARSL`,
  - trailing stop: `FixedTS`, `SignalStrengthTS`, `MovingAverageTS`, `ParabolicSARTS`.
- Position sizers:
  - fixed-fraction sizing,
  - risk-based sizing with volatility input.
- Built-in indicators/signals:
  - moving averages: `SMA`, `EMA`, `WMA`, `CMA`, `DEMA`, `TEMA`,
  - oscillators: `RSI`, `MACD`, `Momentum`, `LarryWilliams`, `BollingerBands`, `CCI`, `Stochastic`, `ROC`, `TRIX`, `DPO`, `RVI`, `DeMarker`,
  - volume/trend: `OBV`, `ForceIndex`, `CHO`, `ADX`, `ATR`, `MassIndex`,
  - statistical/kernel regression: `TriangularKernel`, `GaussianKernel`, `EpanechnikovKernel`, `LogisticKernel`, `LogLogisticKernel`, `CosineKernel`, `SincKernel`, `LaplaceKernel`, `QuarticKernel`, `ParabolicKernel`, `ExponentialKernel`, `SilvermanKernel`, `CauchyKernel`, `TentKernel`, `WaveKernel`, `PowerKernel`, `MortersKernel`, `SquareKernel`,
  - `OHLC`, `HeikinAshi`, `CyclicalTemporalSignal`.
- Performance metrics and HTML backtest report generation.
- Monte Carlo robustness testing:
  - Stationary Block Bootstrap (Politis & Romano),
  - Circular Block Bootstrap,
  - Geometric Brownian Motion simulation,
  - Return shuffling (naive baseline).
- Parameter optimisation (Optuna TPE):
  - indicator parameter search (periods, column choices),
  - indicator weight search,
  - optional train/validation split for out-of-sample evaluation,
  - parallel multi-process search (SQLite-backed).
- ML optimisation:
  - indicator inclusion/period/lag search,
  - Keras model training inside Optuna trials,
  - optional post-search model pruning.
- MQL5 Expert Advisor export:
  - `export_to_mql5` generates `.mq5` code from `StandardStrategy`,
  - `export_ml_to_mql5` generates `.mq5` code from `MLStrategy` with hardcoded Dense network weights,
  - `export_ml_to_mql5_onnx` generates `.mq5` code plus an `.onnx` model file for MT5 runtime inference,
  - pre-export validation for standard, ML hardcoded, and ML ONNX paths,
  - template-driven support for signals/sizing/indicator logic (standard path), including statistical/kernel-regression indicator templates and shared kernel helper functions,
  - Dense forward-pass logic (ML path), and ONNX runtime wiring (ML ONNX path).

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

For ML optimization workflows (TensorFlow/Keras + sklearn + Optuna integration):

```bash
pip install -e ".[ml]"
```

For MQL5 export workflows (Jinja2 templates for `.mq5` generation):

```bash
pip install -e ".[mql5]"
```

For ONNX-based ML export workflows (TensorFlow/Keras + Jinja2 + tf2onnx + onnx, Python 3.10-3.12):

```bash
pip install -e ".[onnx]"
```

Note: on Python 3.13, ONNX export dependencies are not installable yet because
`tf2onnx` currently requires `protobuf~=3.20` while TensorFlow 2.20+ requires
`protobuf>=5.28.0`.

## Quick Start

Run the provided example strategy:

```bash
python examples/simple_ema_strategy.py --ticker SPY --fast 20 --slow 50
```

This example:

- downloads historical data via Yahoo Finance,
- runs a two-EMA strategy through `BacktestEngine`,
- prints summary metrics,
- writes an HTML report to `outputs/simple_ema_report.html` by default.

## Programmatic Example

```python
from trade_lab.backtesting import BacktestEngine, generate_report
from trade_lab.indicators import EMA
from trade_lab.risk_management import FixedSL, FixedTP, SignalStrengthTS
from trade_lab.strategies import StandardStrategy
fast_ema = EMA(period=20)
slow_ema = EMA(period=50)

strategy = StandardStrategy(
    indicators=[(fast_ema, 1.0), (slow_ema, -1.0)],
    allow_long=True,
    allow_short=True,
    entry_threshold=0.2,
    exit_threshold=0.05,
)

strategy.take_profit = FixedTP(base_points=8.0)
strategy.stop_loss = FixedSL(base_points=5.0)
strategy.trailing_stop = SignalStrengthTS(base_points=6.0, step_points=1.0)

engine = BacktestEngine(
    strategy=strategy,
    ticker="SPY",
    start="2021-01-01",
    end="2026-01-01",
    initial_capital=100_000,
    commission=0.001,
    slippage=0.0005,
)

result = engine.run()
generate_report(result, output_path="outputs/backtest_report.html")
print(result.metrics)
print(result.trade_log.tail())
```

Risk objects are plain strategy attributes. They are not constructor parameters:

```python
from trade_lab.risk_management import FixedSL, FixedTP, SignalStrengthTS

strategy.take_profit = FixedTP(base_points=8.0)
strategy.stop_loss = FixedSL(base_points=5.0)
strategy.trailing_stop = SignalStrengthTS(base_points=6.0, step_points=1.0)
```

## Parameter Optimisation

Use the optimisation module to find the best indicator parameters and weights
for a strategy on a given dataset. The optimizer runs a configurable number of
backtest trials via Optuna's TPE sampler, which is Bayesian and significantly
more efficient than random or grid search.

```python
from trade_lab.optimization import (
    OptunaOptimizer, IntParam, FloatParam
)
from trade_lab.indicators import EMA, RSI
from trade_lab.strategies import StandardStrategy
from trade_lab.backtesting import BacktestEngine

# Fetch data once using a minimal engine (strategy not needed for fetch)
data_engine = BacktestEngine(ticker='SPY', start='2015-01-01', end='2024-01-01')
full_df = data_engine.fetch_data()


# Split data: train on 2015-2021, validate on 2022-2024
train_df = full_df[:'2021-12-31']
val_df   = full_df['2022-01-01':]

# The factory receives the sampled params dict and returns a strategy
def factory(params):
    return StandardStrategy(
        indicators=[
            (EMA(period=params['fast']), params['w_fast']),
            (EMA(period=params['slow']), params['w_slow']),
            (RSI(period=params['rsi']),  params['w_rsi']),
        ],
        entry_threshold=params['entry_thr'],
        exit_threshold=0.05,
    )

result = OptunaOptimizer(
    strategy_factory=factory,
    param_space=[
        IntParam('fast',      5,   50),
        IntParam('slow',      20,  200, step=5),
        IntParam('rsi',       7,   28),
        FloatParam('w_fast',  0.1, 3.0),
        FloatParam('w_slow',  0.1, 3.0),
        FloatParam('w_rsi',   0.1, 3.0),
        FloatParam('entry_thr', 0.1, 0.8),
    ],
    train_df=train_df,
    val_df=val_df,         # evaluated post-hoc, never influences search
    metric='sharpe_ratio', # direction inferred automatically
    n_trials=300,
    n_jobs=4,              # parallel workers; uses SQLite automatically
).optimize()

print(result.summary())
print(result.trials_df.sort_values('value', ascending=False).head(10))
```

### Key design points

The **strategy factory** pattern keeps the optimizer decoupled from strategy
internals. The optimizer samples numbers; the factory decides what they mean.
This means the same optimizer can tune indicator parameters, weights, thresholds,
position sizing fractions - anything your factory maps from the params dict.

The **validation DataFrame** is never shown to the optimizer during search. It
is evaluated once after the search completes using only `best_params`. This
gives a clean out-of-sample read without data leakage.

**Parallelism** is handled transparently: `n_jobs=1` uses fast in-memory storage;
`n_jobs > 1` creates a SQLite file under `./optuna_studies/` and spawns worker
processes. Ensure your factory and all objects it captures are picklable.

## Monte Carlo Robustness Testing

After running a backtest, use the Monte Carlo module to validate that the
strategy's performance is not the result of fitting to one specific historical
price path. The module generates synthetic OHLCV scenarios and runs the full
backtest pipeline on each one, collecting a distribution of performance metrics.

```python
from trade_lab.monte_carlo import BlockBootstrap, MonteCarloRunner, MonteCarloAnalysis

# Re-use the same engine configured above
original_df = engine.fetch_data()

runner = MonteCarloRunner(
    engine=engine,
    generator=BlockBootstrap(seed=42),
    n_simulations=500,
)

mc_result = runner.run(original_df)
analysis = MonteCarloAnalysis(mc_result)

# Distributional statistics across all 500 synthetic backtests
print(analysis.summary())

# Where does the real backtest Sharpe sit in the MC distribution?
real_sharpe = result.metrics['sharpe_ratio']
percentile = analysis.percentile_of('sharpe_ratio', real_sharpe)
print(f"Real Sharpe is at the {percentile:.1f}th percentile of MC distribution")

# 90% confidence interval for max drawdown across simulations
low, high = analysis.confidence_interval('max_drawdown', lower=5, upper=95)
print(f"Max drawdown 90% CI: [{low:.2%}, {high:.2%}]")
```

### Available Generators

| Generator | Preserves autocorrelation | Preserves tails | Use case |
|---|---|---|---|
| `BlockBootstrap` | Yes (short-range) | Yes | Primary robustness test |
| `CircularBlockBootstrap` | Yes (no boundary bias) | Yes | Complement to BlockBootstrap |
| `GBMSimulator` | No | No (log-normal) | Analytical baseline |
| `ReturnShuffler` | No | Yes | Naive baseline |

The `BlockBootstrap` and `CircularBlockBootstrap` generators are recommended
as the primary robustness tools. They preserve short-range autocorrelation and
volatility clustering - the temporal structure that trend-following and
mean-reversion strategies actually exploit. Shuffling destroys this structure;
GBM replaces it with an idealized model.

### Interpreting Results

The `percentile_of(metric, value)` method is the most useful single-number
check. For metrics where higher is better (e.g. Sharpe ratio):

- **Below 50th percentile** - the strategy underperforms the average synthetic path.
- **50-75th percentile** - moderate edge, may be partially path-dependent.
- **75-95th percentile** - strategy is robust across most synthetic paths.
- **Above 95th percentile** - investigate: strong edge or potential overfitting.

For metrics where lower is better (e.g. `max_drawdown`), invert the interpretation.

## ML Optimisation

`trade_lab.ml_optimization` combines indicator search, Keras training, and
backtest-driven objective evaluation.

```python
from trade_lab.indicators import EMA, RSI
from trade_lab.ml_optimization import IndicatorSpec, MLOptimizer

specs = [
    IndicatorSpec("ema", EMA, period_low=5, period_high=50, lag_low=0, lag_high=10, max_lags=3, optional=False),
    IndicatorSpec("rsi", RSI, period_low=7, period_high=28, lag_low=0, lag_high=5, max_lags=2, optional=True),
]

def build_model(n_features: int):
    import keras
    model = keras.Sequential([
        keras.layers.Dense(32, activation="relu", input_shape=(n_features,)),
        keras.layers.Dense(1, activation="tanh"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

ml_result = MLOptimizer(
    indicator_specs=specs,
    model_factory=build_model,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    metric="sharpe_ratio",
    n_trials=30,
    n_epochs=15,
).optimize()

print(ml_result.summary())
```

Optional pruning:

```python
from trade_lab.ml_optimization import ModelPruner

pruner = ModelPruner(percentile=20)
pruned_result, report = pruner.prune_result(ml_result)
print(report["zero_fraction"], report["dead_features"])
```

## MQL5 Export

`trade_lab.mql5_export` supports three export paths to MetaTrader 5 Expert
Advisors (`.mq5`).

### StandardStrategy Path

Supported components:

- Indicators: `SMA`, `EMA`, `WMA`, `CMA`, `RSI`, `MACD`, `Momentum`, `LarryWilliams`, and the statistical/kernel-regression family (`TriangularKernel`, `GaussianKernel`, `EpanechnikovKernel`, `LogisticKernel`, `LogLogisticKernel`, `CosineKernel`, `SincKernel`, `LaplaceKernel`, `QuarticKernel`, `ParabolicKernel`, `ExponentialKernel`, `SilvermanKernel`, `CauchyKernel`, `TentKernel`, `WaveKernel`, `PowerKernel`, `MortersKernel`, `SquareKernel`)
- Upstream signals: `OHLC`, `HeikinAshi`, `CyclicalTemporalSignal`
- Position sizing: `None`, `FixedPositionSizer`, `RiskBasedPositionSizer`
- Risk management:
  - directly exportable: `FixedTP`, `SignalStrengthTP`, `FixedSL`, `SignalStrengthSL`, `FixedTS`, `SignalStrengthTS`
  - warning-only/manual MT5 mapping: `MovingAverageSL`, `MovingAverageTS`, `ParabolicSARSL`, `ParabolicSARTS`

Indicator registry coverage (`indicator_registry.py`) additionally includes
descriptors for: `DEMA`, `TEMA`, `BollingerBands`, `CCI`, `Stochastic`, `ROC`,
`TRIX`, `DPO`, `RVI`, `DeMarker`, `OBV`, `ForceIndex`, `CHO`, `ADX`, `ATR`,
`MassIndex`, plus the shared statistical/kernel descriptor used by the
kernel-regression exporters.

The standard-path templates now include shared kernel helpers plus a dedicated
Jinja2 sub-template for statistical indicators, so exported `.mq5` files can
recreate the non-repainting kernel estimate, deviation scaling, and
`tanh((price - estimate) / scale)` signal-strength mapping used in Python.

When exportable risk objects are present, the generated EAs also emit risk
inputs, pass absolute TP/SL prices to `trade.Buy()` / `trade.Sell()`, and add
broker-side trailing-stop management logic. Non-exportable MA/SAR risk objects
produce warnings so they can be mapped manually in MT5.

```python
from trade_lab.indicators import EMA, RSI
from trade_lab.strategies import StandardStrategy
from trade_lab.mql5_export import export_to_mql5

strategy = StandardStrategy(
    indicators=[
        (EMA(period=20), 1.0),
        (EMA(period=50), -1.0),
        (RSI(period=14), 0.5),
    ],
    entry_threshold=0.3,
    exit_threshold=0.1,
    allow_long=True,
    allow_short=True,
)

result = export_to_mql5(
    strategy,
    symbol="EURUSD",
    timeframe="PERIOD_H1",
    output_path="outputs/TradeLab_EA.mq5",
    magic_number=123456,
)

print(result.filepath)
print(result.indicators_exported)
print(result.validation.warnings)
```

### MLStrategy Path

Use this path to export a trained Keras Dense network wrapped in
`KerasModelWrapper`.

Validation contract (`validate_ml_strategy`) before export:

- strategy must be `MLStrategy`,
- `strategy.model` must be `KerasModelWrapper(model, input_names)`,
- allowed layer types: `Dense`, `Dropout`, `InputLayer`, `Concatenate`,
- supported Dense activations: `relu`, `tanh`, `linear`, `sigmoid`,
- final Dense layer must have exactly `1` output unit,
- `input_names` must be non-empty,
- supported sizers: `None`, `FixedPositionSizer`, `RiskBasedPositionSizer`.
- risk config is introspected too; fixed and signal-strength TP/SL/TS variants
  are rendered directly, while MA/SAR variants emit warnings for manual MT5 mapping.

Warnings (non-fatal):

- sigmoid output layer (recommended convention is tanh),
- large models (`>10,000` parameters), which can inflate `.mq5` size and compile time.

`MLStrategyIntrospector` converts the wrapper/model into `MLStrategyConfig`
(`layers`, `feature_names`, thresholds, long/short flags, sizing) for template
rendering. Only Dense layers are exported; `Dropout`, `InputLayer`, and
`Concatenate` are skipped.

```python
import keras

from trade_lab.ml.models import KerasModelWrapper
from trade_lab.mql5_export import export_ml_to_mql5
from trade_lab.strategies import MLStrategy

model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(2,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="tanh"),
])
model.compile(optimizer="adam", loss="mse")

wrapped = KerasModelWrapper(
    model=model,
    input_names=["indicator__ema_20", "indicator__rsi_14"],
)

strategy = MLStrategy(
    model=wrapped,
    indicators=[],
    entry_threshold=0.3,
    exit_threshold=0.1,
    allow_long=True,
    allow_short=True,
)

result = export_ml_to_mql5(
    strategy,
    output_path="outputs/TradeLab_ML_EA.mq5",
    magic_number=654321,
)

print(result.filepath)
print(result.indicators_exported)  # feature list + Dense layer summaries
print(result.validation.warnings)
```

### MLStrategy ONNX Path

Use this path to export an `MLStrategy` as:

- a lightweight `.mq5` EA that calls MT5 `OnnxCreate`/`OnnxRun`,
- an external `.onnx` model file converted from the wrapped Keras model.

```python
from trade_lab.mql5_export import export_ml_to_mql5_onnx

result = export_ml_to_mql5_onnx(
    strategy,
    output_path="outputs/TradeLab_ML_ONNX_EA.mq5",
    onnx_output_path="outputs/TradeLab_ML_ONNX_EA.onnx",  # optional
    magic_number=654321,
)

print(result.filepath)       # .mq5 file
print(result.onnx_filepath)  # .onnx file
print(result.indicators_exported)
```

Deployment note:

- Copy the generated `.onnx` file to your MT5 data folder under `MQL5\Files\`.

## Core Concepts

### Signals

Signals subclass `BaseSignal` and append columns to a DataFrame.
They can be chained via `source`, except index-based temporal signals.

### Indicators

Indicators subclass `BaseIndicator` and provide:

- `compute(df)` for raw indicator columns,
- `to_signal_strength(df)` for normalized conviction in `[-1, 1]`.

### Strategies

Strategies subclass `BaseStrategy` and must output `signal_strength`.
The backtester interprets it with thresholds:

- `signal_strength > entry_threshold`: open/hold long,
- `signal_strength < -entry_threshold`: open/hold short,
- `abs(signal_strength) < exit_threshold`: close position.

### Position Sizing

Optional `BasePositionSizer` controls units per trade.
If no sizer is provided, the engine defaults to full-equity allocation.

## Package Layout

```text
src/trade_lab/
  backtesting/
    engine.py
    metrics.py
    report.py
  indicators/
    base.py
    moving_averages.py
    oscillators.py
    statistical.py
    trend.py
    volume.py
  ml/
    __init__.py
    models.py
    preprocessing.py
    targets.py
    trainer.py
    validation.py
  ml_optimization/
    __init__.py
    search_space.py
    feature_builder.py
    objective.py
    optimizer.py
    pruning.py
    result.py
  mql5_export/
    __init__.py
    code_generator.py
    introspector.py
    ml_introspector.py
    ml_validator.py
    onnx_exporter.py
    validators.py
    indicator_registry.py
    signal_registry.py
    sizing_registry.py
    templates/
  monte_carlo/
    __init__.py
    generators.py
    runner.py
    analysis.py
  optimization/
    __init__.py
    param_space.py
    objective.py
    optimizer.py
    result.py
  risk_management/
    __init__.py
    base.py
    take_profit.py
    stop_loss.py
    trailing_stop.py
  signals/
    base.py
    signals.py
    temporal.py
  sizing/
    base.py
    fixed.py
    risk_based.py
  strategies/
    base.py
    standard.py
    ml_strategy.py
```

## API Reference

Core modules:

- [`src/trade_lab/backtesting/engine.py`][api-engine]
- [`src/trade_lab/backtesting/metrics.py`][api-metrics]
- [`src/trade_lab/backtesting/report.py`][api-report]
- [`src/trade_lab/strategies/base.py`][api-strategy-base]
- [`src/trade_lab/strategies/standard.py`][api-strategy-standard]
- [`src/trade_lab/strategies/ml_strategy.py`][api-strategy-ml]
- [`src/trade_lab/indicators/base.py`][api-indicator-base]
- [`src/trade_lab/indicators/moving_averages.py`][api-indicator-ma]
- [`src/trade_lab/indicators/oscillators.py`][api-indicator-osc]
- [`src/trade_lab/indicators/statistical.py`][api-indicator-stat]
- [`src/trade_lab/indicators/trend.py`][api-indicator-trend]
- [`src/trade_lab/indicators/volume.py`][api-indicator-volume]
- [`src/trade_lab/risk_management/base.py`][api-risk-base]
- [`src/trade_lab/risk_management/take_profit.py`][api-risk-tp]
- [`src/trade_lab/risk_management/stop_loss.py`][api-risk-sl]
- [`src/trade_lab/risk_management/trailing_stop.py`][api-risk-ts]
- [`src/trade_lab/signals/base.py`][api-signal-base]
- [`src/trade_lab/signals/signals.py`][api-signal-signals]
- [`src/trade_lab/signals/temporal.py`][api-signal-temporal]
- [`src/trade_lab/sizing/base.py`][api-sizing-base]
- [`src/trade_lab/sizing/fixed.py`][api-sizing-fixed]
- [`src/trade_lab/sizing/risk_based.py`][api-sizing-risk]
- [`src/trade_lab/monte_carlo/generators.py`][api-mc-generators]
- [`src/trade_lab/monte_carlo/runner.py`][api-mc-runner]
- [`src/trade_lab/monte_carlo/analysis.py`][api-mc-analysis]
- [`src/trade_lab/optimization/param_space.py`][api-opt-params]
- [`src/trade_lab/optimization/objective.py`][api-opt-objective]
- [`src/trade_lab/optimization/optimizer.py`][api-opt-optimizer]
- [`src/trade_lab/optimization/result.py`][api-opt-result]
- [`src/trade_lab/ml_optimization/search_space.py`][api-mlopt-search]
- [`src/trade_lab/ml_optimization/feature_builder.py`][api-mlopt-features]
- [`src/trade_lab/ml_optimization/objective.py`][api-mlopt-objective]
- [`src/trade_lab/ml_optimization/optimizer.py`][api-mlopt-optimizer]
- [`src/trade_lab/ml_optimization/pruning.py`][api-mlopt-pruning]
- [`src/trade_lab/ml_optimization/result.py`][api-mlopt-result]
- [`src/trade_lab/mql5_export/__init__.py`][api-mql5-init]
- [`src/trade_lab/mql5_export/code_generator.py`][api-mql5-codegen]
- [`src/trade_lab/mql5_export/introspector.py`][api-mql5-introspector]
- [`src/trade_lab/mql5_export/ml_introspector.py`][api-mql5-ml-introspector]
- [`src/trade_lab/mql5_export/ml_validator.py`][api-mql5-ml-validator]
- [`src/trade_lab/mql5_export/onnx_exporter.py`][api-mql5-onnx-exporter]
- [`src/trade_lab/mql5_export/validators.py`][api-mql5-validators]
- [`src/trade_lab/mql5_export/indicator_registry.py`][api-mql5-indicator-reg]
- [`src/trade_lab/mql5_export/signal_registry.py`][api-mql5-signal-reg]
- [`src/trade_lab/mql5_export/sizing_registry.py`][api-mql5-sizing-reg]

Examples:

- [`examples/simple_ema_strategy.py`][example-ema]
- [`examples/standard_optimization_example.ipynb`][example-opt-standard]
- [`examples/ml_optimization_example.ipynb`][example-opt-ml]
- [`examples/monte_carlo_example.ipynb`][example-mc]
- [`examples/generating_ea.ipynb`][example-ea]

## Roadmap

- Stabilize and document public APIs for strategy, signal, and indicator extension.
- Add more built-in strategy templates (trend-following and mean-reversion variants).
- Extend ML optimisation with richer model families and time-series validation schemes.
- Improve reporting with richer trade analytics and export formats.
- Extend ML MQL5 export with automated live feature wiring options.
- Prepare packaging and release automation for PyPI distribution (maybe).

## Testing

Run the full suite:

```bash
pytest -q tests
```

Run focused risk-management + exporter tests:

```bash
pytest -q tests/test_risk_management.py tests/test_mql5_export.py tests/test_mql5_export_ml.py
```

Check focused coverage for ML and MQL5 code generation:

```bash
pytest -q --cov=trade_lab.ml --cov=trade_lab.mql5_export.code_generator --cov-report=term-missing tests/test_ml_module.py tests/test_mql5_export_ml.py
```

## Contributing

To contribute through GitHub:

1. Fork the repository and clone your fork locally.
2. Create a branch from `main` for your change.
3. Implement changes and add/update tests as needed.
4. Run `pytest -v` and ensure tests pass.
5. Commit and push your branch to your fork.
6. Open a Pull Request to `main` in `Czarnak/TradeLab`.
7. Describe what changed, why, and any follow-up work in the PR.

## License

MIT

---

*Created with Claude AI*

[ci-badge]: https://github.com/Czarnak/TradeLab/actions/workflows/ci.yml/badge.svg?branch=main
[ci-link]: https://github.com/Czarnak/TradeLab/actions/workflows/ci.yml
[python-badge]: https://img.shields.io/badge/python-3.10%2B-blue.svg
[python-link]: https://www.python.org/downloads/
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-link]: LICENSE
[api-engine]: src/trade_lab/backtesting/engine.py
[api-metrics]: src/trade_lab/backtesting/metrics.py
[api-report]: src/trade_lab/backtesting/report.py
[api-strategy-base]: src/trade_lab/strategies/base.py
[api-strategy-standard]: src/trade_lab/strategies/standard.py
[api-strategy-ml]: src/trade_lab/strategies/ml_strategy.py
[api-indicator-base]: src/trade_lab/indicators/base.py
[api-indicator-ma]: src/trade_lab/indicators/moving_averages.py
[api-indicator-osc]: src/trade_lab/indicators/oscillators.py
[api-indicator-stat]: src/trade_lab/indicators/statistical.py
[api-indicator-trend]: src/trade_lab/indicators/trend.py
[api-indicator-volume]: src/trade_lab/indicators/volume.py
[api-risk-base]: src/trade_lab/risk_management/base.py
[api-risk-tp]: src/trade_lab/risk_management/take_profit.py
[api-risk-sl]: src/trade_lab/risk_management/stop_loss.py
[api-risk-ts]: src/trade_lab/risk_management/trailing_stop.py
[api-signal-base]: src/trade_lab/signals/base.py
[api-signal-signals]: src/trade_lab/signals/signals.py
[api-signal-temporal]: src/trade_lab/signals/temporal.py
[api-sizing-base]: src/trade_lab/sizing/base.py
[api-sizing-fixed]: src/trade_lab/sizing/fixed.py
[api-sizing-risk]: src/trade_lab/sizing/risk_based.py
[api-mc-generators]: src/trade_lab/monte_carlo/generators.py
[api-mc-runner]: src/trade_lab/monte_carlo/runner.py
[api-mc-analysis]: src/trade_lab/monte_carlo/analysis.py
[api-opt-params]: src/trade_lab/optimization/param_space.py
[api-opt-objective]: src/trade_lab/optimization/objective.py
[api-opt-optimizer]: src/trade_lab/optimization/optimizer.py
[api-opt-result]: src/trade_lab/optimization/result.py
[api-mlopt-search]: src/trade_lab/ml_optimization/search_space.py
[api-mlopt-features]: src/trade_lab/ml_optimization/feature_builder.py
[api-mlopt-objective]: src/trade_lab/ml_optimization/objective.py
[api-mlopt-optimizer]: src/trade_lab/ml_optimization/optimizer.py
[api-mlopt-pruning]: src/trade_lab/ml_optimization/pruning.py
[api-mlopt-result]: src/trade_lab/ml_optimization/result.py
[api-mql5-init]: src/trade_lab/mql5_export/__init__.py
[api-mql5-codegen]: src/trade_lab/mql5_export/code_generator.py
[api-mql5-introspector]: src/trade_lab/mql5_export/introspector.py
[api-mql5-ml-introspector]: src/trade_lab/mql5_export/ml_introspector.py
[api-mql5-ml-validator]: src/trade_lab/mql5_export/ml_validator.py
[api-mql5-onnx-exporter]: src/trade_lab/mql5_export/onnx_exporter.py
[api-mql5-validators]: src/trade_lab/mql5_export/validators.py
[api-mql5-indicator-reg]: src/trade_lab/mql5_export/indicator_registry.py
[api-mql5-signal-reg]: src/trade_lab/mql5_export/signal_registry.py
[api-mql5-sizing-reg]: src/trade_lab/mql5_export/sizing_registry.py
[example-ema]: examples/simple_ema_strategy.py
[example-opt-standard]: examples/standard_optimization_example.ipynb
[example-opt-ml]: examples/ml_optimization_example.ipynb
[example-mc]: examples/monte_carlo_example.ipynb
[example-ea]: examples/generating_ea.ipynb
