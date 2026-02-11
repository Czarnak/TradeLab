# Market-Lab

Desktop application integrating **strategy backtesting + optimization** and **ML model building/training** (Keras).

Built with Python 3.11+, PySide6, and a clean modular architecture.

> **Phase 1 complete**: Core data layer, backtest engine, strategies, metrics, optimization, CLI, tests, and CI.
> **Phase 2A complete**: ML module — feature builders, dataset builder, Keras model builder, trainer with equity curves, Optuna hyperparameter optimization.
> **Phase 2B complete**: Full dark-themed GUI — Backtest tab, ML tab, dynamic settings dialogs, matplotlib embedding, pytest-qt tests.

---

## GUI

The desktop application uses a **dark Fusion theme** with three tabbed modules:

### Backtest Tab
- **Data loading**: Browse CSV, download from Yahoo Finance, or generate sample data
- **Strategy selection**: Dropdown with all registered strategies
- **Parameter editing**: Dynamic form dialog built from `parameters_schema()`
- **Engine config**: Initial capital, commission/slippage in bps, short toggle
- **Run backtest**: Background execution with progress bar, equity curve chart, metrics panel, sortable trades table
- **Optimize**: Optuna parameter search (100 trials) with best params auto-applied

### ML Tab
- **Multi-dataset support**: Load multiple OHLCV sources for combined training
- **Feature selection**: Checkboxes for each feature builder with adjustable lag counts, live input-dim display
- **Architecture config**: Layers, units, activation, optimizer, LR, epochs, batch size, signal threshold
- **Benchmark selector**: Dropdown to pick which dataset is the buy-and-hold comparison
- **Train**: Background training with 4-panel results (loss, accuracy/MAE, train equity, val equity)
- **Optimize**: Optuna hyperparameter search with best config auto-applied to UI

### Launch

```bash
market-lab          # GUI entry point
# or
python -m market_lab.main
```

---

## Setup

```bash
# Clone and create virtual environment
git clone <repo-url> market-lab
cd market-lab
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install
pip install -e ".[dev]"
```

## Quick Start

### Generate sample data

```bash
market-lab-cli generate-sample
```

Creates `data/examples/sample_ohlcv.csv` (500 OHLCV bars) and `data/examples/sample_ticks.csv` (2000 ticks) using geometric Brownian motion.

### Run a backtest

```bash
market-lab-cli backtest \
  --data data/examples/sample_ohlcv.csv \
  --strategy "MA Crossover" \
  --capital 100000
```

### Run parameter optimization

```bash
market-lab-cli optimize \
  --data data/examples/sample_ohlcv.csv \
  --strategy "MA Crossover" \
  --metric sharpe \
  --trials 100
```

### List available strategies

```bash
market-lab-cli list-strategies
```

### Launch GUI

```bash
market-lab
```

---

## Data Formats

### OHLCV bars CSV

```
Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,99.0,103.0,1000000
```

Accepted header aliases: `Datetime`/`time`/`ts` for timestamp; `o`/`h`/`l`/`c`/`vol` for OHLCV columns.

### Tick CSV

```
timestamp,bid,ask,volume
2023-01-01 09:30:00.100,100.01,100.05,50
```

Volume is optional. `offer` is accepted as an alias for `ask`. A `mid` column is computed automatically.

---

## Architecture

```
src/market_lab/
├── gui/           # PySide6 views, dialogs, models
├── data/          # Loaders, validators, Yahoo downloader, sample generator, Monte Carlo
├── backtest/      # Engine, metrics, reports, Optuna optimizer
├── strategies/    # Strategy plugin system + built-in strategies
├── ml/            # Feature builders, model builder, trainer, optimizer
├── gui/           # PySide6 desktop interface (Backtest, ML tabs)
└── utils/         # Logging, caching, threading helpers, config
```

No business logic lives inside GUI classes — only orchestration.

---

## Built-in Strategies

### MA Crossover
Dual moving-average crossover. Long when fast MA > slow MA, short when fast < slow.

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| fast_period | int | 10 | 2-200 |
| slow_period | int | 30 | 5-500 |
| ma_type | enum | SMA | SMA, EMA |
| allow_short | bool | True | - |

### Mean Reversion
Bollinger-band / z-score mean-reversion. Enter long below lower band, short above upper band.

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| lookback | int | 20 | 5-200 |
| entry_z | float | 2.0 | 0.5-4.0 |
| exit_z | float | 0.0 | -1.0-2.0 |
| allow_short | bool | True | - |

## How to Add a New Strategy

1. Create a file in `src/market_lab/strategies/`, e.g. `my_strategy.py`
2. Subclass `Strategy` and implement `name`, `parameters_schema()`, `run()`
3. Call `register_strategy(MyStrategy())` at module level
4. Import the module in `main.py` or `cli.py`

```python
from market_lab.strategies.base import Strategy, StrategySignals, register_strategy

class MyStrategy(Strategy):
    @property
    def name(self) -> str:
        return "My Strategy"

    def parameters_schema(self) -> dict:
        return {
            "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
        }

    def run(self, bars, params):
        signal = ...  # your logic
        return StrategySignals(signal=signal)

register_strategy(MyStrategy())
```

---

## Monte Carlo Data Variations

Generate synthetic variations of existing OHLCV data for robustness testing:

```python
from market_lab.data.monte_carlo import generate_variations, MCMethod
from market_lab.data.sample_generator import generate_ohlcv_bars

bars = generate_ohlcv_bars(n_bars=500)
results = generate_variations(
    bars,
    methods=[MCMethod.BOOTSTRAP, MCMethod.GBM_FITTED, MCMethod.NOISE_INJECTION],
    n_simulations=10,
)
# results["bootstrap"] -> list of 10 DataFrames
```

---

## ML Model Builder

Build, train, and optimize Keras Dense neural networks for market prediction.

### Feature Builders

| Name | Key | Description |
|------|-----|-------------|
| Close last N bars | `close_lags` | Raw close prices |
| Open last N bars | `open_lags` | Raw open prices |
| Log return last N bars | `return_lags` | ln(close/prev_close) |
| High-Low range last N bars | `hl_range_lags` | Volatility proxy |
| Volume last N bars | `volume_lags` | Trading volume |

### Programmatic Usage

```python
from market_lab.data.sample_generator import generate_ohlcv_bars
from market_lab.ml.dataset_builder import FeatureSelection, DatasetConfig, build_dataset
from market_lab.ml.model_builder import build_model, default_classification_config
from market_lab.ml.trainer import train_model, compute_equity_curves, save_training_result

# 1. Prepare data
bars = generate_ohlcv_bars(n_bars=500, seed=42)

# 2. Build dataset
ds_config = DatasetConfig(
    feature_selections=[
        FeatureSelection("return_lags", n_lags=10),
        FeatureSelection("volume_lags", n_lags=5),
    ],
    task_type="classification",
    split_ratio=0.8,
)
dataset = build_dataset([bars], ds_config)

# 3. Build and train model
model_config = default_classification_config(dataset.input_dim)
model = build_model(model_config)
result = train_model(model, dataset, model_config)

# 4. Compute equity curves (model vs buy-and-hold benchmark)
compute_equity_curves(result, [bars], threshold=0.5, benchmark_dataset_idx=0)

# 5. Save everything
save_training_result(result)
```

### ML Hyperparameter Optimization

```python
from market_lab.ml.optimizer import optimize_ml

opt_result = optimize_ml(
    dataset=dataset,
    bars_list=[bars],
    objective="val_accuracy",  # or "val_sharpe"
    n_trials=50,
)
print(f"Best accuracy: {opt_result['best_value']:.4f}")
print(f"Best config: {opt_result['best_config'].to_dict()}")
```

Optuna searches over: number of layers (1-4), units per layer, activation functions, optimizer, learning rate, epochs, batch size, and classification threshold.

### How to Add New ML Feature Definitions

1. Create a class in `src/market_lab/ml/input_definitions.py` subclassing `FeatureBuilder`
2. Implement `name`, `display_name`, and `build_features(bars, n_lags)`
3. Call `register_feature(MyFeature())` at module level

```python
from market_lab.ml.input_definitions import FeatureBuilder, register_feature

class RSIFeature(FeatureBuilder):
    @property
    def name(self) -> str:
        return "rsi"

    @property
    def display_name(self) -> str:
        return "RSI last N bars"

    def build_features(self, bars, n_lags):
        # Your RSI implementation here
        ...

register_feature(RSIFeature())
```

---

## Outputs Directory Structure

```
outputs/
├── backtests/<run_id>/
│   ├── metrics.json
│   ├── trades.csv
│   ├── equity.csv
│   ├── equity_curve.png
│   └── report.md
├── ml/<run_id>/
│   ├── model.keras
│   ├── config.json
│   ├── training_history.json
│   ├── summary.json
│   ├── accuracy.png
│   ├── loss.png
│   ├── train_equity.png
│   └── val_equity.png
├── ml_optimizations/
│   ├── best_config.json
│   ├── best_params.json
│   └── trials.csv
└── optimizations/
    ├── best_params.json
    └── trials.csv
```

---

## Testing

```bash
pytest -v                      # run all tests
pytest --cov=market_lab -v     # with coverage
```

153 tests covering: CSV schema parsing, sample generation, Monte Carlo variations, strategy execution, backtest engine correctness, metrics calculations, Optuna optimization, ML feature builders, dataset builder, model builder, trainer with equity curves, ML hyperparameter optimization, and GUI widget testing (matplotlib canvas, table models, dialogs, tabs, main window).

---

## CI/CD

GitHub Actions runs tests on Ubuntu and Windows with Python 3.11 and 3.12. See `.github/workflows/ci.yml`.

---

## License

MIT
