"""CLI entry point for headless backtest and optimization."""

from __future__ import annotations

import argparse
import json
import sys

from market_lab.utils.config import ensure_dirs
from market_lab.utils.logging import setup_logging, get_logger

log = get_logger("cli")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a headless backtest."""
    import market_lab.strategies.ma_crossover  # noqa: F401
    import market_lab.strategies.mean_reversion  # noqa: F401

    from market_lab.data.loaders import load_csv, load_parquet
    from market_lab.strategies.base import get_strategy
    from market_lab.backtest.engine import BacktestConfig, run_backtest
    from market_lab.backtest.report import save_report

    path = args.data
    if path.endswith(".parquet"):
        bars = load_parquet(path)
    else:
        bars = load_csv(path, schema="bars")

    strategy = get_strategy(args.strategy)
    params = strategy.default_params()
    if args.params:
        params.update(json.loads(args.params))

    config = BacktestConfig(
        initial_capital=args.capital,
        allow_short=args.allow_short,
    )
    signals = strategy.run(bars, params)
    result = run_backtest(bars, signals.signal, config)

    out = save_report(result)
    print(f"\nBacktest complete. Report saved to: {out}")
    print(json.dumps(result.metrics, indent=2))


def cmd_generate_sample(args: argparse.Namespace) -> None:
    """Generate sample data files."""
    from market_lab.data.sample_generator import save_example_csvs

    saved = save_example_csvs()
    for name, path in saved.items():
        print(f"  {name}: {path}")


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run headless parameter optimization."""
    import market_lab.strategies.ma_crossover  # noqa: F401
    import market_lab.strategies.mean_reversion  # noqa: F401

    from market_lab.data.loaders import load_csv, load_parquet
    from market_lab.strategies.base import get_strategy
    from market_lab.backtest.engine import BacktestConfig
    from market_lab.backtest.optimizer import optimize, save_optimization_results

    path = args.data
    if path.endswith(".parquet"):
        bars = load_parquet(path)
    else:
        bars = load_csv(path, schema="bars")

    strategy = get_strategy(args.strategy)
    config = BacktestConfig(
        initial_capital=args.capital,
        allow_short=args.allow_short,
    )

    def progress(current, total):
        pct = current / total * 100
        print(f"\r  Trial {current}/{total} ({pct:.0f}%)", end="", flush=True)

    result = optimize(
        strategy=strategy,
        bars=bars,
        objective_metric=args.metric,
        n_trials=args.trials,
        backtest_config=config,
        seed=args.seed,
        progress_callback=progress,
    )

    print()
    out = save_optimization_results(result)
    print(f"\nOptimization complete. Results saved to: {out}")
    print(f"  Best {args.metric}: {result['best_value']:.4f}")
    print(f"  Best params: {json.dumps(result['best_params'], indent=2)}")


def cmd_list_strategies(args: argparse.Namespace) -> None:
    """List available strategies."""
    import market_lab.strategies.ma_crossover  # noqa: F401
    import market_lab.strategies.mean_reversion  # noqa: F401
    from market_lab.strategies.base import all_strategies

    for name, strat in all_strategies().items():
        schema = strat.parameters_schema()
        params_str = ", ".join(
            f"{k} ({v.get('type', '?')}, default={v.get('default')})"
            for k, v in schema.items()
        )
        print(f"  {name}")
        print(f"    Parameters: {params_str}")
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-lab-cli",
        description="Market-Lab command-line interface",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- backtest ---
    bt = sub.add_parser("backtest", help="Run a backtest")
    bt.add_argument("--data", required=True, help="Path to OHLCV CSV or Parquet")
    bt.add_argument("--strategy", required=True, help="Strategy name")
    bt.add_argument("--params", default=None, help='JSON string of params, e.g. \'{"fast_period": 5}\'')
    bt.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    bt.add_argument("--allow-short", action="store_true", default=True, help="Allow short positions")
    bt.add_argument("--no-short", dest="allow_short", action="store_false", help="Disable short positions")

    # --- optimize ---
    opt = sub.add_parser("optimize", help="Run parameter optimization")
    opt.add_argument("--data", required=True, help="Path to OHLCV CSV or Parquet")
    opt.add_argument("--strategy", required=True, help="Strategy name")
    opt.add_argument("--metric", default="sharpe", help="Objective metric")
    opt.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    opt.add_argument("--seed", type=int, default=42, help="Random seed")
    opt.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    opt.add_argument("--allow-short", action="store_true", default=True)
    opt.add_argument("--no-short", dest="allow_short", action="store_false")

    # --- generate-sample ---
    sub.add_parser("generate-sample", help="Generate sample CSV data files")

    # --- list-strategies ---
    sub.add_parser("list-strategies", help="List available strategies")

    return parser


def main() -> None:
    """CLI main entry point."""
    setup_logging()
    ensure_dirs()

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "backtest": cmd_backtest,
        "optimize": cmd_optimize,
        "generate-sample": cmd_generate_sample,
        "list-strategies": cmd_list_strategies,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except Exception as exc:
        log.error("Command failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
