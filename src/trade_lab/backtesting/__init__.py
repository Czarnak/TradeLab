from trade_lab.backtesting.metrics import compute_metrics

try:
    from trade_lab.backtesting.engine import BacktestEngine, BacktestResult
except ModuleNotFoundError:
    # Keep lightweight utilities importable when optional runtime deps (e.g. yfinance)
    # needed by the engine are not installed.
    BacktestEngine = None
    BacktestResult = None

try:
    from trade_lab.backtesting.report import generate_report
except ModuleNotFoundError:
    generate_report = None

__all__ = ["BacktestEngine", "BacktestResult", "compute_metrics", "generate_report"]
