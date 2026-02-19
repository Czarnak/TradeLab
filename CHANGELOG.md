# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

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
