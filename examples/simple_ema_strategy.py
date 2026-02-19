from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

# Allow running this file directly from repository root without install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trade_lab.backtesting import BacktestEngine, generate_report
from trade_lab.indicators import EMA
from trade_lab.strategies import StandardStrategy


def default_dates() -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=365 * 5)
    return start.isoformat(), end.isoformat()


def build_parser() -> argparse.ArgumentParser:
    start_default, end_default = default_dates()
    parser = argparse.ArgumentParser(
        description="Run a simple EMA-based strategy backtest."
    )
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance ticker.")
    parser.add_argument("--start", default=start_default, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=end_default, help="End date YYYY-MM-DD.")
    parser.add_argument("--fast", type=int, default=20, help="Fast EMA period.")
    parser.add_argument("--slow", type=int, default=50, help="Slow EMA period.")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial capital.")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate.")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage rate.")
    parser.add_argument(
        "--report-path",
        default="outputs/simple_ema_report.html",
        help="Path for generated HTML report.",
    )
    return parser


def run_backtest(args: argparse.Namespace):
    if args.fast >= args.slow:
        raise ValueError("fast EMA period must be smaller than slow EMA period")

    fast_ema = EMA(period=args.fast)
    slow_ema = EMA(period=args.slow)

    strategy = StandardStrategy(
        indicators=[(fast_ema, 1.0), (slow_ema, -1.0)],
        allow_long=True,
        allow_short=True,
        entry_threshold=0.2,
        exit_threshold=0.05,
    )

    engine = BacktestEngine(
        strategy=strategy,
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
    )
    return engine.run()


def print_summary(result, report_path: str) -> None:
    metrics = result.metrics
    print("\nBacktest Summary")
    print(f"Total return:      {metrics['total_return']:.2%}")
    print(f"Annualized return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"Max drawdown:      {metrics['max_drawdown']:.2%}")
    print(f"Total trades:      {int(metrics['total_trades'])}")
    print(f"Win rate:          {metrics['win_rate']:.1%}")
    print(f"Report:            {report_path}")

    if not result.trade_log.empty:
        print("\nLast 5 trades:")
        print(result.trade_log.tail(5).to_string(index=False))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_backtest(args)

    report_file = Path(args.report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_path = generate_report(result, output_path=str(report_file))

    print_summary(result, report_path)


if __name__ == "__main__":
    main()
