# TradeLab

[![CI][ci-badge]][ci-link]
[![Python][python-badge]][python-link]
[![License: MIT][license-badge]][license-link]

TradeLab is a modular Python framework for strategy backtesting with a clear separation between:

- `signals` (feature generation),
- `indicators` (market state transforms + signal-strength mapping),
- `strategies` (how indicator strengths are combined),
- `position sizing`,
- `backtesting` (execution simulation, metrics, report generation).

## Features

- Event-driven backtesting engine with:
  - long/short support,
  - commission and slippage modeling,
  - trade logging and equity curve output.
- Composable signal and indicator pipelines.
- Strategy abstractions:
  - `StandardStrategy` for weighted indicator combinations,
  - `MLStrategy` for model-driven predictions.
- Position sizers:
  - fixed-fraction sizing,
  - risk-based sizing with volatility input.
- Built-in indicators/signals:
  - `SMA`, `EMA`, `RSI`,
  - `OHLC`, `HeikinAshi`, `CyclicalTemporalSignal`.
- Performance metrics and HTML backtest report generation.

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
- [`src/trade_lab/signals/base.py`][api-signal-base]
- [`src/trade_lab/signals/signals.py`][api-signal-signals]
- [`src/trade_lab/signals/temporal.py`][api-signal-temporal]
- [`src/trade_lab/sizing/base.py`][api-sizing-base]
- [`src/trade_lab/sizing/fixed.py`][api-sizing-fixed]
- [`src/trade_lab/sizing/risk_based.py`][api-sizing-risk]

Examples:

- [`examples/simple_ema_strategy.py`][example-ema]

## Roadmap

- Stabilize and document public APIs for strategy, signal, and indicator extension.
- Add more built-in strategy templates (trend-following and mean-reversion variants).
- Expand test coverage for backtest edge cases (missing data, sparse trades, no-trade windows).
- Introduce parameter sweep/optimization helpers for strategy research workflows.
- Improve reporting with richer trade analytics and export formats.
- Prepare packaging and release automation for PyPI distribution.

## Testing

Run tests with:

```bash
pytest -v
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
[api-signal-base]: src/trade_lab/signals/base.py
[api-signal-signals]: src/trade_lab/signals/signals.py
[api-signal-temporal]: src/trade_lab/signals/temporal.py
[api-sizing-base]: src/trade_lab/sizing/base.py
[api-sizing-fixed]: src/trade_lab/sizing/fixed.py
[api-sizing-risk]: src/trade_lab/sizing/risk_based.py
[example-ema]: examples/simple_ema_strategy.py
