"""Backtest tab: data loading, strategy selection, run, results display, and optimization."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from market_lab.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from market_lab.backtest.optimizer import optimize, save_optimization_results
from market_lab.backtest.report import save_report
from market_lab.data.loaders import load_csv, load_parquet
from market_lab.data.sample_generator import generate_ohlcv_bars
from market_lab.data.yahoo import VALID_INTERVALS, VALID_PERIODS
from market_lab.gui.settings_dialog import SettingsDialog
from market_lab.gui.widgets import ChartCanvas, PandasTableModel
from market_lab.strategies.base import Strategy, all_strategies, get_strategy
from market_lab.utils.logging import get_logger
from market_lab.utils.threading import run_in_background

log = get_logger("gui.backtest_tab")


class BacktestTab(QWidget):
    """Main backtest tab widget."""

    def __init__(self, status_callback=None, parent=None):
        super().__init__(parent)
        self._status = status_callback or (lambda msg: None)
        self._bars: pd.DataFrame | None = None
        self._strategy_params: dict = {}
        self._result: BacktestResult | None = None

        self._build_ui()
        self._refresh_strategies()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # --- Top: Data + Strategy controls ---
        top = QHBoxLayout()

        # Data group
        data_grp = QGroupBox("Data Source")
        data_lay = QVBoxLayout(data_grp)

        row1 = QHBoxLayout()
        self._btn_load_csv = QPushButton("Load CSV")
        self._btn_load_csv.clicked.connect(self._on_load_csv)
        row1.addWidget(self._btn_load_csv)

        self._btn_sample = QPushButton("Use Sample Data")
        self._btn_sample.clicked.connect(self._on_use_sample)
        row1.addWidget(self._btn_sample)
        data_lay.addLayout(row1)

        # Yahoo download
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Symbol:"))
        self._txt_symbol = QLineEdit("AAPL")
        self._txt_symbol.setMaximumWidth(80)
        row2.addWidget(self._txt_symbol)

        row2.addWidget(QLabel("Period:"))
        self._cmb_period = QComboBox()
        self._cmb_period.addItems(list(VALID_PERIODS))
        self._cmb_period.setCurrentText("1y")
        row2.addWidget(self._cmb_period)

        row2.addWidget(QLabel("Interval:"))
        self._cmb_interval = QComboBox()
        self._cmb_interval.addItems(list(VALID_INTERVALS))
        self._cmb_interval.setCurrentText("1d")
        row2.addWidget(self._cmb_interval)

        self._btn_yahoo = QPushButton("Download")
        self._btn_yahoo.clicked.connect(self._on_yahoo_download)
        row2.addWidget(self._btn_yahoo)
        data_lay.addLayout(row2)

        self._lbl_data_info = QLabel("No data loaded")
        data_lay.addWidget(self._lbl_data_info)
        top.addWidget(data_grp, stretch=3)

        # Strategy group
        strat_grp = QGroupBox("Strategy")
        strat_lay = QVBoxLayout(strat_grp)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Strategy:"))
        self._cmb_strategy = QComboBox()
        self._cmb_strategy.currentTextChanged.connect(self._on_strategy_changed)
        row3.addWidget(self._cmb_strategy)
        strat_lay.addLayout(row3)

        row4 = QHBoxLayout()
        self._btn_settings = QPushButton("Settings...")
        self._btn_settings.clicked.connect(self._on_settings)
        row4.addWidget(self._btn_settings)

        self._btn_run = QPushButton("Run Backtest")
        self._btn_run.setStyleSheet("font-weight: bold;")
        self._btn_run.clicked.connect(self._on_run)
        row4.addWidget(self._btn_run)

        self._btn_save = QPushButton("Save Report")
        self._btn_save.clicked.connect(self._on_save_report)
        self._btn_save.setEnabled(False)
        row4.addWidget(self._btn_save)
        strat_lay.addLayout(row4)

        # Config
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Capital:"))
        self._spn_capital = QDoubleSpinBox()
        self._spn_capital.setRange(1_000, 100_000_000)
        self._spn_capital.setValue(100_000)
        self._spn_capital.setPrefix("$")
        self._spn_capital.setDecimals(0)
        row5.addWidget(self._spn_capital)

        row5.addWidget(QLabel("Comm (bps):"))
        self._spn_comm = QDoubleSpinBox()
        self._spn_comm.setRange(0, 100)
        self._spn_comm.setValue(5)
        row5.addWidget(self._spn_comm)

        row5.addWidget(QLabel("Slip (bps):"))
        self._spn_slip = QDoubleSpinBox()
        self._spn_slip.setRange(0, 100)
        self._spn_slip.setValue(2)
        row5.addWidget(self._spn_slip)
        strat_lay.addLayout(row5)

        top.addWidget(strat_grp, stretch=2)
        main_layout.addLayout(top)

        # --- Optimization row ---
        opt_grp = QGroupBox("Optimization (Optuna)")
        opt_lay = QHBoxLayout(opt_grp)

        opt_lay.addWidget(QLabel("Metric:"))
        self._cmb_opt_metric = QComboBox()
        self._cmb_opt_metric.addItems([
            "sharpe", "sortino", "total_return_pct", "profit_factor",
            "max_drawdown_pct", "win_rate_pct",
        ])
        opt_lay.addWidget(self._cmb_opt_metric)

        opt_lay.addWidget(QLabel("Trials:"))
        self._spn_trials = QSpinBox()
        self._spn_trials.setRange(10, 5000)
        self._spn_trials.setValue(50)
        opt_lay.addWidget(self._spn_trials)

        self._btn_optimize = QPushButton("Optimize")
        self._btn_optimize.clicked.connect(self._on_optimize)
        opt_lay.addWidget(self._btn_optimize)

        self._lbl_opt_status = QLabel("")
        opt_lay.addWidget(self._lbl_opt_status, stretch=1)
        main_layout.addWidget(opt_grp)

        # --- Bottom: Results ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Metrics panel
        self._txt_metrics = QTextEdit()
        self._txt_metrics.setReadOnly(True)
        self._txt_metrics.setMaximumWidth(320)
        splitter.addWidget(self._txt_metrics)

        # Chart
        self._chart = ChartCanvas(self, width=8, height=4)
        splitter.addWidget(self._chart)

        # Trades table
        self._trades_model = PandasTableModel()
        self._trades_table = QTableView()
        self._trades_table.setModel(self._trades_model)
        self._trades_table.setSortingEnabled(True)
        self._trades_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        splitter.addWidget(self._trades_table)

        splitter.setSizes([250, 500, 350])
        main_layout.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_strategies(self) -> None:
        strats = all_strategies()
        self._cmb_strategy.clear()
        self._cmb_strategy.addItems(list(strats.keys()))
        if strats:
            first = list(strats.values())[0]
            self._strategy_params = first.default_params()

    def _current_strategy(self) -> Strategy | None:
        name = self._cmb_strategy.currentText()
        if not name:
            return None
        return get_strategy(name)

    def _set_data(self, df: pd.DataFrame, label: str) -> None:
        self._bars = df
        n = len(df)
        start = str(df.index[0])[:10] if n > 0 else "?"
        end = str(df.index[-1])[:10] if n > 0 else "?"
        self._lbl_data_info.setText(f"{label} — {n} bars ({start} → {end})")
        self._status(f"Data loaded: {label} ({n} bars)")

    def _show_result(self, result: BacktestResult) -> None:
        self._result = result
        self._btn_save.setEnabled(True)

        # Metrics
        m = result.metrics
        lines = []
        for k, v in m.items():
            lines.append(f"<b>{k}</b>: {v}")
        self._txt_metrics.setHtml("<br>".join(lines))

        # Equity curve chart
        self._chart.plot_series(
            result.equity_curve, title="Equity Curve", ylabel="Equity ($)"
        )

        # Trades table
        if result.trades:
            records = []
            for t in result.trades:
                records.append({
                    "Entry": str(t.entry_date)[:19],
                    "Exit": str(t.exit_date)[:19],
                    "Dir": "LONG" if t.direction == 1 else "SHORT",
                    "Entry$": round(t.entry_price, 2),
                    "Exit$": round(t.exit_price, 2),
                    "Qty": round(t.quantity, 2),
                    "PnL": round(t.pnl, 2),
                    "Bars": t.bars_held,
                })
            self._trades_model.set_dataframe(pd.DataFrame(records))
        else:
            self._trades_model.set_dataframe(pd.DataFrame())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load OHLCV Data", "", "CSV/Parquet (*.csv *.parquet);;All Files (*)"
        )
        if not path:
            return
        try:
            if path.endswith(".parquet"):
                df = load_parquet(path)
            else:
                df = load_csv(path, schema="bars")
            self._set_data(df, Path(path).name)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def _on_use_sample(self) -> None:
        df = generate_ohlcv_bars(n_bars=500, seed=42)
        self._set_data(df, "Sample GBM Data (500 bars)")

    def _on_yahoo_download(self) -> None:
        symbol = self._txt_symbol.text().strip()
        period = self._cmb_period.currentText()
        interval = self._cmb_interval.currentText()
        if not symbol:
            return

        self._status(f"Downloading {symbol}...")
        self._btn_yahoo.setEnabled(False)

        def do_download():
            from market_lab.data.yahoo import download
            return download(symbol, period, interval)

        def on_done(df):
            self._set_data(df, f"{symbol} ({interval}, {period})")
            self._btn_yahoo.setEnabled(True)

        def on_error(err):
            self._btn_yahoo.setEnabled(True)
            self._status(f"Download failed: {err[:100]}")
            QMessageBox.critical(self, "Download Error", str(err))

        run_in_background(do_download, on_result=on_done, on_error=on_error)

    def _on_strategy_changed(self, name: str) -> None:
        strat = self._current_strategy()
        if strat:
            self._strategy_params = strat.default_params()

    def _on_settings(self) -> None:
        strat = self._current_strategy()
        if not strat:
            return
        dlg = SettingsDialog(
            strat.parameters_schema(), self._strategy_params, title=f"{strat.name} Settings", parent=self
        )
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            self._strategy_params = dlg.get_values()
            self._status(f"Params updated: {self._strategy_params}")

    def _on_run(self) -> None:
        if self._bars is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        strat = self._current_strategy()
        if not strat:
            return

        self._status("Running backtest...")
        self._btn_run.setEnabled(False)

        bars = self._bars
        params = dict(self._strategy_params)
        config = BacktestConfig(
            initial_capital=self._spn_capital.value(),
            commission_bps=self._spn_comm.value(),
            slippage_bps=self._spn_slip.value(),
        )

        def do_backtest():
            signals = strat.run(bars, params)
            return run_backtest(bars, signals.signal, config)

        def on_done(result):
            self._show_result(result)
            self._btn_run.setEnabled(True)
            self._status(
                f"Backtest done: {result.metrics.get('total_return_pct', 0):.2f}% return, "
                f"{result.metrics.get('total_trades', 0)} trades"
            )

        def on_error(err):
            self._btn_run.setEnabled(True)
            self._status("Backtest failed")
            QMessageBox.critical(self, "Backtest Error", str(err))

        run_in_background(do_backtest, on_result=on_done, on_error=on_error)

    def _on_save_report(self) -> None:
        if self._result is None:
            return
        try:
            out = save_report(self._result)
            self._status(f"Report saved to {out}")
            QMessageBox.information(self, "Saved", f"Report saved to:\n{out}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_optimize(self) -> None:
        if self._bars is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        strat = self._current_strategy()
        if not strat:
            return

        metric = self._cmb_opt_metric.currentText()
        n_trials = self._spn_trials.value()
        config = BacktestConfig(
            initial_capital=self._spn_capital.value(),
            commission_bps=self._spn_comm.value(),
            slippage_bps=self._spn_slip.value(),
        )

        self._btn_optimize.setEnabled(False)
        self._lbl_opt_status.setText("Optimizing...")
        self._status(f"Optimizing {strat.name} ({n_trials} trials)...")

        bars = self._bars

        def do_opt():
            return optimize(
                strategy=strat,
                bars=bars,
                objective_metric=metric,
                n_trials=n_trials,
                backtest_config=config,
            )

        def on_done(result):
            self._btn_optimize.setEnabled(True)
            best_val = result["best_value"]
            best_params = result["best_params"]
            self._lbl_opt_status.setText(
                f"Best {metric}={best_val:.4f} | {json.dumps(best_params)}"
            )
            self._strategy_params = best_params
            self._status(f"Optimization done: best {metric}={best_val:.4f}")

            # Run backtest with best params
            signals = strat.run(bars, best_params)
            bt_result = run_backtest(bars, signals.signal, config)
            self._show_result(bt_result)

            try:
                save_optimization_results(result)
            except Exception:
                pass

        def on_error(err):
            self._btn_optimize.setEnabled(True)
            self._lbl_opt_status.setText("Optimization failed")
            self._status("Optimization failed")
            QMessageBox.critical(self, "Optimization Error", str(err))

        run_in_background(do_opt, on_result=on_done, on_error=on_error)
