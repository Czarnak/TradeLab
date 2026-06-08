import warnings

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


def test_compute_metrics_handles_empty_trade_log():
    equity_curve = pd.Series(
        [100_000.0, 99_000.0, 101_000.0],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )
    trade_log = pd.DataFrame(columns=["direction", "pnl", "bars_held", "commission"])

    metrics = compute_metrics(equity_curve, trade_log)

    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] == 0.0
    # No trades → profit factor is undefined (nan), distinct from a real 0.0.
    assert pd.isna(metrics["profit_factor"])
    assert metrics["avg_trade_bars"] == 0.0
    assert metrics["total_commission"] == 0.0
    assert metrics["long_win_rate"] == 0.0
    assert metrics["short_win_rate"] == 0.0


def test_sharpe_uses_zero_risk_free_rate_by_default():
    equity_curve = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0],
        index=pd.date_range("2026-01-01", periods=5, freq="D"),
    )
    trade_log = pd.DataFrame(columns=["direction", "pnl", "bars_held", "commission"])

    # Default risk-free rate is now 0.0 → a steadily rising curve has a positive
    # Sharpe (the old default of 10.0 = 1000% forced it deeply negative).
    metrics = compute_metrics(equity_curve, trade_log)
    assert metrics["sharpe_ratio"] > 0

    # A large annual risk-free fraction still drives Sharpe negative.
    high_rf = compute_metrics(equity_curve, trade_log, risk_free_rate=10.0)
    assert high_rf["sharpe_ratio"] < 0


def test_compute_metrics_ignores_infinite_returns_without_runtime_warnings():
    equity_curve = pd.Series(
        [100_000.0, 0.0, 10_000.0],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )
    trade_log = pd.DataFrame(columns=["direction", "pnl", "bars_held", "commission"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        metrics = compute_metrics(equity_curve, trade_log)

    runtime_messages = [
        str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)
    ]
    assert not any(
        "invalid value encountered in subtract" in msg for msg in runtime_messages
    )
    assert not any(
        "invalid value encountered in reduce" in msg for msg in runtime_messages
    )
    assert metrics["annual_volatility"] == 0.0
