import pandas as pd

from trade_lab.backtesting.metrics import compute_metrics
from trade_lab.backtesting.report import _format_metrics_tables


def test_compute_metrics_adds_long_and_short_breakdown():
    equity_curve = pd.Series(
        [100_000.0, 101_000.0, 100_500.0, 102_000.0],
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )
    trade_log = pd.DataFrame(
        [
            {"direction": "long", "pnl": 100.0, "bars_held": 3, "commission": 1.0},
            {"direction": "long", "pnl": -50.0, "bars_held": 2, "commission": 1.0},
            {"direction": "short", "pnl": 200.0, "bars_held": 1, "commission": 1.0},
            {"direction": "short", "pnl": -300.0, "bars_held": 4, "commission": 1.0},
            {"direction": "short", "pnl": 0.0, "bars_held": 2, "commission": 1.0},
        ]
    )

    metrics = compute_metrics(equity_curve, trade_log)

    assert metrics["long_win_rate"] == 0.5
    assert metrics["long_avg_win"] == 100.0
    assert metrics["long_avg_loss"] == -50.0
    assert metrics["short_win_rate"] == 1 / 3
    assert metrics["short_avg_win"] == 200.0
    assert metrics["short_avg_loss"] == -150.0


def test_report_metrics_tables_include_responsive_grid_and_breakdown():
    metrics = {
        "total_return": 0.1,
        "annualized_return": 0.08,
        "sharpe_ratio": 1.2,
        "sortino_ratio": 1.5,
        "max_drawdown": -0.05,
        "annual_volatility": 0.12,
        "total_trades": 10,
        "win_rate": 0.6,
        "profit_factor": 1.4,
        "avg_win": 120.0,
        "avg_loss": -80.0,
        "avg_trade_bars": 4.0,
        "total_commission": 25.0,
        "long_win_rate": 0.7,
        "short_win_rate": 0.5,
        "long_avg_win": 140.0,
        "long_avg_loss": -90.0,
        "short_avg_win": 100.0,
        "short_avg_loss": -70.0,
    }

    html = _format_metrics_tables(metrics)

    assert 'class="metrics-grid"' in html
    assert "Overall Metrics" in html
    assert "Long / Short Breakdown" in html
    assert "Long Win Rate" in html
    assert "Short Avg Loss" in html
