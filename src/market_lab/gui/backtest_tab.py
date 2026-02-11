"""Backtest tab: data load, strategy select, parameter edit, run, report, optimize."""

from __future__ import annotations

import json

import pandas as pd
from PySide6.QtCore import Qt, QThreadPool, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from market_lab.gui.widgets import DataLoaderPanel, MplCanvas, SortableTableModel
from market_lab.utils.threading import Worker


class BacktestTab(QWidget):
    """Full backtest workflow: load data → pick strategy → configure → run → view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bars: pd.DataFrame | None = None
        self._result = None
        self._strategy_params: dict = {}
        self._bt_config: dict = {
            "initial_capital": 100_000,
            "commission_bps": 5,
            "slippage_bps": 2,
            "allow_short": True,
        }

        self._build_ui()
        self._load_strategies()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Top: data loader
        self.data_loader = DataLoaderPanel()
        self.data_loader.data_loaded.connect(self._on_data_loaded)
        root.addWidget(self.data_loader)

        # Data info label
        self.data_label = QLabel("No data loaded")
        self.data_label.setStyleSheet("color: #aaa; padding: 2px;")
        root.addWidget(self.data_label)

        # Middle: controls row
        ctrl_layout = QHBoxLayout()

        # Strategy selector
        strat_grp = QGroupBox("Strategy")
        strat_l = QHBoxLayout(strat_grp)
        self.strat_combo = QComboBox()
        self.strat_combo.currentTextChanged.connect(self._on_strategy_changed)
        strat_l.addWidget(self.strat_combo)
        btn_params = QPushButton("Parameters...")
        btn_params.clicked.connect(self._edit_params)
        strat_l.addWidget(btn_params)
        btn_bt_cfg = QPushButton("Engine Config...")
        btn_bt_cfg.clicked.connect(self._edit_bt_config)
        strat_l.addWidget(btn_bt_cfg)
        ctrl_layout.addWidget(strat_grp)

        # Action buttons
        action_grp = QGroupBox("Actions")
        action_l = QHBoxLayout(action_grp)
        self.btn_run = QPushButton("Run Backtest")
        self.btn_run.clicked.connect(self._run_backtest)
        self.btn_run.setEnabled(False)
        self.btn_optimize = QPushButton("Optimize")
        self.btn_optimize.clicked.connect(self._run_optimize)
        self.btn_optimize.setEnabled(False)
        action_l.addWidget(self.btn_run)
        action_l.addWidget(self.btn_optimize)
        ctrl_layout.addWidget(action_grp)

        root.addLayout(ctrl_layout)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        root.addWidget(self.progress)

        # Bottom: results area (splitter)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: chart
        self.chart = MplCanvas(self, width=7, height=4)
        splitter.addWidget(self.chart)

        # Right: tabs-like stacked area (metrics + trades)
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        # Metrics text
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(200)
        self.metrics_text.setPlaceholderText("Metrics will appear here after a backtest run.")
        right_l.addWidget(QLabel("Metrics"))
        right_l.addWidget(self.metrics_text)

        # Trades table
        right_l.addWidget(QLabel("Trades"))
        self.trades_model = SortableTableModel()
        self.trades_table = QTableView()
        self.trades_table.setModel(self.trades_model)
        self.trades_table.setSortingEnabled(True)
        self.trades_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        right_l.addWidget(self.trades_table)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Strategy loading
    # ------------------------------------------------------------------

    def _load_strategies(self):
        import market_lab.strategies.ma_crossover  # noqa: F401
        import market_lab.strategies.mean_reversion  # noqa: F401
        from market_lab.strategies.base import list_strategies

        self.strat_combo.blockSignals(True)
        self.strat_combo.clear()
        for name in list_strategies():
            self.strat_combo.addItem(name)
        self.strat_combo.blockSignals(False)
        if self.strat_combo.count() > 0:
            self._on_strategy_changed(self.strat_combo.currentText())

    def _on_strategy_changed(self, name: str):
        if not name:
            return
        from market_lab.strategies.base import get_strategy
        strat = get_strategy(name)
        self._strategy_params = strat.default_params()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @Slot(object, str)
    def _on_data_loaded(self, df: pd.DataFrame, label: str):
        self._bars = df
        n = len(df)
        start = df.index[0].strftime("%Y-%m-%d") if hasattr(df.index[0], "strftime") else str(df.index[0])
        end = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])
        self.data_label.setText(f"Loaded: {label}  |  {n} bars  |  {start} → {end}")
        self.btn_run.setEnabled(True)
        self.btn_optimize.setEnabled(True)

        # Plot price
        ax = self.chart.clear_and_get_ax()
        ax.plot(df.index, df["close"], color="#4fc3f7", linewidth=1)
        ax.set_title(f"{label} — Close Price")
        ax.set_ylabel("Price")
        self.chart.refresh()

    # ------------------------------------------------------------------
    # Parameter / config editing
    # ------------------------------------------------------------------

    def _edit_params(self):
        name = self.strat_combo.currentText()
        if not name:
            return
        from market_lab.strategies.base import get_strategy
        from market_lab.gui.settings_dialog import StrategySettingsDialog

        strat = get_strategy(name)
        dlg = StrategySettingsDialog(strat.parameters_schema(), self._strategy_params, self)
        if dlg.exec():
            self._strategy_params = dlg.get_params()

    def _edit_bt_config(self):
        from market_lab.gui.settings_dialog import BacktestConfigDialog

        dlg = BacktestConfigDialog(self._bt_config, self)
        if dlg.exec():
            self._bt_config = dlg.get_config()

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------

    def _run_backtest(self):
        if self._bars is None:
            return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setFormat("Running backtest...")

        bars = self._bars.copy()
        strat_name = self.strat_combo.currentText()
        params = dict(self._strategy_params)
        bt_cfg = dict(self._bt_config)

        def work():
            from market_lab.strategies.base import get_strategy
            from market_lab.backtest.engine import BacktestConfig, run_backtest

            strat = get_strategy(strat_name)
            signals = strat.run(bars, params)
            config = BacktestConfig(**bt_cfg)
            return run_backtest(bars, signals.signal, config)

        worker = Worker(work)
        worker.signals.result.connect(self._on_backtest_done)
        worker.signals.error.connect(self._on_backtest_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_backtest_done(self, result):
        self._result = result
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)

        # Metrics
        m = result.metrics
        lines = []
        for k, v in m.items():
            if isinstance(v, float):
                lines.append(f"{k:>22s}: {v:>12.4f}")
            else:
                lines.append(f"{k:>22s}: {v:>12}")
        self.metrics_text.setPlainText("\n".join(lines))

        # Trades table
        if result.trades:
            rows = []
            for t in result.trades:
                rows.append({
                    "entry_date": str(t.entry_date.date()) if hasattr(t.entry_date, "date") else str(t.entry_date),
                    "exit_date": str(t.exit_date.date()) if hasattr(t.exit_date, "date") else str(t.exit_date),
                    "direction": "LONG" if t.direction == 1 else "SHORT",
                    "entry_price": round(t.entry_price, 4),
                    "exit_price": round(t.exit_price, 4),
                    "quantity": t.quantity,
                    "pnl": round(t.pnl, 2),
                    "bars_held": t.bars_held,
                })
            self.trades_model.set_dataframe(pd.DataFrame(rows))
        else:
            self.trades_model.set_dataframe(pd.DataFrame())

        # Equity chart
        ax = self.chart.clear_and_get_ax()
        ax.plot(result.equity_curve.index, result.equity_curve.values,
                color="#4fc3f7", linewidth=1.2, label="Strategy")
        ax.axhline(y=m.get("initial_capital", 100_000), color="#888",
                    linestyle="--", alpha=0.5, label="Initial Capital")
        ax.set_title("Equity Curve")
        ax.set_ylabel("Equity ($)")
        ax.legend(facecolor="#353535", edgecolor="#555", labelcolor="#ccc")
        self.chart.refresh()

    @Slot(tuple)
    def _on_backtest_error(self, error_info):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        exc_type, exc_value, _ = error_info
        QMessageBox.critical(self, "Backtest Error", f"{exc_type.__name__}: {exc_value}")

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------

    def _run_optimize(self):
        if self._bars is None:
            return

        self.btn_optimize.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Optimizing... %p%")

        bars = self._bars.copy()
        strat_name = self.strat_combo.currentText()
        bt_cfg = dict(self._bt_config)
        n_trials = 100

        def work():
            from market_lab.strategies.base import get_strategy
            from market_lab.backtest.engine import BacktestConfig
            from market_lab.backtest.optimizer import optimize

            strat = get_strategy(strat_name)
            config = BacktestConfig(**bt_cfg)

            def progress_cb(current, total):
                # Will be called from worker thread; progress bar update
                # is coarse-grained but safe via signal
                pass

            return optimize(
                strategy=strat,
                bars=bars,
                objective_metric="sharpe",
                n_trials=n_trials,
                backtest_config=config,
                seed=42,
                progress_callback=progress_cb,
            )

        worker = Worker(work)
        worker.signals.result.connect(self._on_optimize_done)
        worker.signals.error.connect(self._on_optimize_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_optimize_done(self, result):
        self.progress.setVisible(False)
        self.btn_optimize.setEnabled(True)
        self.btn_run.setEnabled(True)

        best = result["best_params"]
        best_val = result["best_value"]

        # Apply best params
        self._strategy_params.update(best)

        msg = f"Best Sharpe: {best_val:.4f}\n\nBest Parameters:\n"
        for k, v in best.items():
            msg += f"  {k}: {v}\n"
        msg += "\nParameters have been applied. Click 'Run Backtest' to see results."
        QMessageBox.information(self, "Optimization Complete", msg)

    @Slot(tuple)
    def _on_optimize_error(self, error_info):
        self.progress.setVisible(False)
        self.btn_optimize.setEnabled(True)
        self.btn_run.setEnabled(True)
        exc_type, exc_value, _ = error_info
        QMessageBox.critical(self, "Optimization Error", f"{exc_type.__name__}: {exc_value}")
