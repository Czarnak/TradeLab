# Market-Lab

Desktop application integrating **strategy backtesting + optimization**, **ML model building/training** (Keras), and **insider-trading scanning** (web + EDGAR verification).

Built with Python 3.11+, PySide6, and a clean modular architecture.

> **Phase 1 complete**: Core data layer, backtest engine, strategies, metrics, optimization, CLI, tests, and CI.
> ML module and Insider Scan module are planned for subsequent phases.

---

## Setup

```bash
# Clone and create virtual environment
git clone https://github.com/Czarnak/TradeLab.git

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
├── ml/            # Feature builders, model builder, trainer, optimizer (Phase 2)
├── insiders/      # Source parsers, EDGAR resolver, merger (Phase 3)
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

## Outputs Directory Structure

```
outputs/
├── backtests/<run_id>/
│   ├── metrics.json
│   ├── trades.csv
│   ├── equity.csv
│   ├── equity_curve.png
│   └── report.md
├── ml/<run_id>/           # (Phase 2)
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

70 tests covering: CSV schema parsing, sample generation, Monte Carlo variations, strategy execution, backtest engine correctness, metrics calculations, and Optuna optimization.

---

## CI/CD

GitHub Actions runs tests on Ubuntu and Windows with Python 3.11 and 3.12. See `.github/workflows/ci.yml`.

---

## Insider Scan (Phase 3 - planned)

Sources: secform4.com, openinsider.com, SEC EDGAR verification.
Rate limits: SEC requires max 10 req/s with proper User-Agent.
Senate data: Congress member list from official sources; family disclosure data requires paid APIs (documented limitation).

---

## License

MIT
