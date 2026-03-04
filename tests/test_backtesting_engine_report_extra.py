from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import trade_lab.backtesting.engine as bt_engine
from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.backtesting.report import generate_report
from trade_lab.strategies.base import BaseStrategy


class _SignalStrategy(BaseStrategy):
    def __init__(self, signals: list[float], **kwargs):
        super().__init__(**kwargs)
        self._signals = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal_strength"] = pd.Series(self._signals, index=out.index)
        return out


def _sample_ohlcv(n: int = 6) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": [10, 11, 12, 13, 12, 11][:n],
            "High": [11, 12, 13, 14, 13, 12][:n],
            "Low": [9, 10, 11, 12, 11, 10][:n],
            "Close": [10, 11, 12, 11, 10, 9][:n],
            "Volume": [100, 110, 120, 130, 140, 150][:n],
        },
        index=idx,
    )


def test_engine_run_and_fetch_data_validate_required_inputs():
    strategy = _SignalStrategy([0.0, 0.0, 0.0])
    engine = BacktestEngine(strategy=strategy)

    with pytest.raises(ValueError, match="requires ticker, start, and end"):
        engine.run()
    with pytest.raises(ValueError, match="requires ticker, start, and end"):
        engine.fetch_data()


def test_engine_fetch_data_drops_multiindex_ticker_level(monkeypatch):
    strategy = _SignalStrategy([0.0] * 3)
    engine = BacktestEngine(
        strategy=strategy, ticker="SPY", start="2020-01-01", end="2020-01-05"
    )

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    data = pd.DataFrame(
        [[1, 2, 0, 1, 100]] * 3,
        index=idx,
        columns=pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["SPY"]],
            names=[None, "Ticker"],
        ),
    )

    monkeypatch.setattr(bt_engine.yf, "download", lambda *args, **kwargs: data)

    out = engine.fetch_data()
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_engine_run_fetches_data_and_runs_pipeline(monkeypatch):
    df = _sample_ohlcv(3)
    strategy = _SignalStrategy([0.0, 0.0, 0.0])
    engine = BacktestEngine(
        strategy=strategy, ticker="SPY", start="2020-01-01", end="2020-01-05"
    )

    monkeypatch.setattr(engine, "_fetch_data", lambda: df)

    result = engine.run()
    assert len(result.df) == 3


def test_engine_simulation_produces_long_and_short_trades():
    df = _sample_ohlcv(6)
    strategy = _SignalStrategy(
        [0.6, 0.0, -0.7, 0.0, 0.0, 0.0],
        allow_long=True,
        allow_short=True,
        entry_threshold=0.5,
        exit_threshold=0.1,
    )
    engine = BacktestEngine(
        strategy=strategy, initial_capital=10_000, commission=0.0, slippage=0.0
    )

    result = engine.run_on(df)

    assert len(result.equity_curve) == len(df)
    assert not result.trade_log.empty
    assert set(result.trade_log["direction"]) == {"long", "short"}
    assert "total_return" in result.metrics


def test_engine_simulation_handles_nan_signal_and_non_positive_equity():
    df = _sample_ohlcv(3)
    strategy = _SignalStrategy(
        [np.nan, 0.9, 0.0], entry_threshold=0.5, exit_threshold=0.1
    )
    engine = BacktestEngine(strategy=strategy, initial_capital=0.0)

    result = engine.run_on(df)

    assert (result.equity_curve.to_numpy() == 0.0).all()
    assert result.trade_log.empty


def test_engine_compute_atr_returns_expected_length_and_nans():
    atr = BacktestEngine._compute_atr(_sample_ohlcv(6), period=3)

    assert len(atr) == 6
    assert np.isnan(atr[0])
    assert np.isfinite(atr[-1])


def test_generate_report_writes_html_with_and_without_trades(tmp_path):
    base_df = _sample_ohlcv(4)
    equity = pd.Series([1000.0, 1010.0, 1005.0, 1020.0], index=base_df.index)
    metrics = {"total_return": 0.02, "max_drawdown": -0.01}

    no_trades = SimpleNamespace(
        df=base_df,
        equity_curve=equity,
        trade_log=pd.DataFrame(),
        metrics=metrics,
    )
    path_a = tmp_path / "report_a.html"
    out_a = generate_report(no_trades, output_path=str(path_a))
    assert out_a == str(path_a)
    assert "Backtest Report" in path_a.read_text(encoding="utf-8")

    trades = pd.DataFrame(
        [
            {
                "direction": "long",
                "entry_date": base_df.index[0],
                "entry_price": 10.0,
                "exit_date": base_df.index[1],
                "exit_price": 11.0,
            },
            {
                "direction": "short",
                "entry_date": base_df.index[2],
                "entry_price": 12.0,
                "exit_date": base_df.index[3],
                "exit_price": 11.0,
            },
        ]
    )
    with_trades = SimpleNamespace(
        df=base_df,
        equity_curve=equity,
        trade_log=trades,
        metrics=metrics,
    )
    path_b = tmp_path / "report_b.html"
    generate_report(with_trades, output_path=str(path_b))
    html = path_b.read_text(encoding="utf-8")
    assert "Overall Metrics" in html
    assert "Long / Short Breakdown" in html


def test_backtesting_init_fallback_import_paths(monkeypatch):
    module_name = "trade_lab_backtesting_init_fallback_test"
    module_path = Path("src/trade_lab/backtesting/__init__.py").resolve()
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {
            "trade_lab.backtesting.engine",
            "trade_lab.backtesting.report",
        }:
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    assert module.BacktestEngine is None
    assert module.BacktestResult is None
    assert module.generate_report is None
