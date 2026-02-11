"""Report generation for backtest results.

Exports to ``outputs/backtests/<run_id>/``:
- metrics.json
- trades.csv
- equity.csv
- report.md
- equity_curve.png
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from market_lab.backtest.engine import BacktestResult
from market_lab.utils.config import BACKTEST_OUTPUTS_DIR, ensure_dirs
from market_lab.utils.logging import get_logger

log = get_logger("backtest.report")


def _trades_to_df(result: BacktestResult) -> pd.DataFrame:
    """Convert trades list to a DataFrame."""
    if not result.trades:
        return pd.DataFrame()
    records = []
    for t in result.trades:
        records.append({
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_price": round(t.entry_price, 4),
            "exit_price": round(t.exit_price, 4),
            "quantity": round(t.quantity, 4),
            "pnl": round(t.pnl, 2),
            "commission": round(t.commission, 2),
            "bars_held": t.bars_held,
        })
    return pd.DataFrame(records)


def generate_equity_chart(result: BacktestResult, path: Path) -> None:
    """Save equity curve chart as PNG."""
    fig, ax = plt.subplots(figsize=(12, 5))
    result.equity_curve.plot(ax=ax, linewidth=1.2, color="#2196F3")
    ax.set_title("Equity Curve", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_markdown_report(result: BacktestResult) -> str:
    """Generate a Markdown report string."""
    m = result.metrics
    lines = [
        "# Backtest Report",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Return | {m.get('total_return_pct', 0):.2f}% |",
        f"| CAGR | {m.get('cagr_pct', 0):.2f}% |",
        f"| Max Drawdown | {m.get('max_drawdown_pct', 0):.2f}% |",
        f"| Sharpe Ratio | {m.get('sharpe', 0):.4f} |",
        f"| Sortino Ratio | {m.get('sortino', 0):.4f} |",
        f"| Profit Factor | {m.get('profit_factor', 0):.4f} |",
        f"| Win Rate | {m.get('win_rate_pct', 0):.2f}% |",
        f"| Avg Trade PnL | ${m.get('avg_trade_pnl', 0):.2f} |",
        f"| Total Trades | {m.get('total_trades', 0)} |",
        f"| Initial Capital | ${m.get('initial_capital', 0):,.2f} |",
        f"| Final Equity | ${m.get('final_equity', 0):,.2f} |",
        "",
        "## Configuration",
        "",
        f"- Commission: {result.config.commission_bps} bps",
        f"- Slippage: {result.config.slippage_bps} bps",
        f"- Position sizing: {result.config.position_sizing} ({result.config.size_value})",
        f"- Allow short: {result.config.allow_short}",
        "",
        "![Equity Curve](equity_curve.png)",
    ]
    return "\n".join(lines)


def save_report(
    result: BacktestResult,
    run_id: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Save full backtest report to disk.

    Returns the output directory.
    """
    ensure_dirs()
    run_id = run_id or uuid.uuid4().hex[:8]
    out = (output_dir or BACKTEST_OUTPUTS_DIR) / run_id
    out.mkdir(parents=True, exist_ok=True)

    # metrics.json
    with open(out / "metrics.json", "w") as f:
        json.dump(result.metrics, f, indent=2, default=str)

    # trades.csv
    trades_df = _trades_to_df(result)
    if not trades_df.empty:
        trades_df.to_csv(out / "trades.csv", index=False)

    # equity.csv
    result.equity_curve.to_csv(out / "equity.csv", header=True)

    # equity chart
    generate_equity_chart(result, out / "equity_curve.png")

    # report.md
    md = generate_markdown_report(result)
    (out / "report.md").write_text(md, encoding="utf-8")

    log.info("Report saved to %s", out)
    return out
