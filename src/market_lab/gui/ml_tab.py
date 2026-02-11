"""ML tab: multi-dataset loading, feature selection, model config, train, equity, optimize."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, QThreadPool, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from market_lab.gui.widgets import DataLoaderPanel, MplCanvas, SortableTableModel
from market_lab.utils.threading import Worker


class MLTab(QWidget):
    """Full ML workflow: load → features → architecture → train → evaluate."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._datasets: list[tuple[pd.DataFrame, str]] = []  # (bars, label)
        self._training_result = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Scrollable top area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(420)
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # --- Data loading ---
        self.data_loader = DataLoaderPanel()
        self.data_loader.data_loaded.connect(self._on_data_loaded)
        top_layout.addWidget(self.data_loader)

        # Dataset list
        ds_grp = QGroupBox("Loaded Datasets")
        ds_l = QHBoxLayout(ds_grp)
        self.ds_list = QListWidget()
        self.ds_list.setMaximumHeight(80)
        ds_l.addWidget(self.ds_list)
        btn_rm = QPushButton("Remove Selected")
        btn_rm.clicked.connect(self._remove_dataset)
        ds_l.addWidget(btn_rm)
        top_layout.addWidget(ds_grp)

        # --- Feature selection + Model config side by side ---
        config_row = QHBoxLayout()

        # Feature selection
        feat_grp = QGroupBox("Feature Selection")
        feat_l = QVBoxLayout(feat_grp)
        self.feat_checks: dict[str, QCheckBox] = {}
        self.feat_lags: dict[str, QSpinBox] = {}

        from market_lab.ml.input_definitions import all_feature_builders
        for name, builder in all_feature_builders().items():
            row = QHBoxLayout()
            cb = QCheckBox(builder.display_name)
            cb.setChecked(name in ("return_lags", "volume_lags"))
            cb.stateChanged.connect(self._update_input_dim)
            self.feat_checks[name] = cb
            row.addWidget(cb)
            lag_spin = QSpinBox()
            lag_spin.setRange(1, 50)
            lag_spin.setValue(5)
            lag_spin.setPrefix("lags: ")
            lag_spin.setMaximumWidth(100)
            lag_spin.valueChanged.connect(self._update_input_dim)
            self.feat_lags[name] = lag_spin
            row.addWidget(lag_spin)
            feat_l.addLayout(row)

        self.dim_label = QLabel("Input dim: 0")
        self.dim_label.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        feat_l.addWidget(self.dim_label)
        config_row.addWidget(feat_grp)

        # Model architecture
        arch_grp = QGroupBox("Model Architecture")
        arch_form = QFormLayout(arch_grp)

        self.task_combo = QComboBox()
        self.task_combo.addItems(["classification", "regression"])
        arch_form.addRow("Task:", self.task_combo)

        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 6)
        self.layers_spin.setValue(2)
        arch_form.addRow("Hidden Layers:", self.layers_spin)

        self.units_spin = QSpinBox()
        self.units_spin.setRange(4, 512)
        self.units_spin.setSingleStep(8)
        self.units_spin.setValue(64)
        arch_form.addRow("Units / Layer:", self.units_spin)

        self.act_combo = QComboBox()
        self.act_combo.addItems(["relu", "elu", "selu", "tanh", "swish"])
        arch_form.addRow("Activation:", self.act_combo)

        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["adam", "rmsprop", "sgd"])
        arch_form.addRow("Optimizer:", self.opt_combo)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        arch_form.addRow("Learning Rate:", self.lr_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        arch_form.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setSingleStep(16)
        self.batch_spin.setValue(32)
        arch_form.addRow("Batch Size:", self.batch_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.50)
        arch_form.addRow("Signal Threshold:", self.threshold_spin)

        # Benchmark dropdown
        self.bench_combo = QComboBox()
        self.bench_combo.addItem("(first dataset)")
        arch_form.addRow("Benchmark:", self.bench_combo)

        config_row.addWidget(arch_grp)
        top_layout.addLayout(config_row)

        scroll.setWidget(top_widget)
        root.addWidget(scroll)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        self.btn_train = QPushButton("Train Model")
        self.btn_train.clicked.connect(self._run_train)
        self.btn_train.setEnabled(False)
        self.btn_optimize = QPushButton("Optimize Hyperparameters")
        self.btn_optimize.clicked.connect(self._run_optimize)
        self.btn_optimize.setEnabled(False)
        btn_row.addWidget(self.btn_train)
        btn_row.addWidget(self.btn_optimize)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        root.addWidget(self.progress)

        # --- Results area ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Charts (left)
        self.chart = MplCanvas(self, width=8, height=5)
        splitter.addWidget(self.chart)

        # Metrics (right)
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setPlaceholderText("Training metrics will appear here.")
        right_l.addWidget(QLabel("Training Summary"))
        right_l.addWidget(self.metrics_text)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        self._update_input_dim()

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    @Slot(object, str)
    def _on_data_loaded(self, df: pd.DataFrame, label: str):
        self._datasets.append((df, label))
        self.ds_list.addItem(f"{label}  ({len(df)} bars)")
        self.bench_combo.addItem(label)
        self._update_buttons()

    def _remove_dataset(self):
        row = self.ds_list.currentRow()
        if row < 0:
            return
        self._datasets.pop(row)
        self.ds_list.takeItem(row)
        # +1 because index 0 is "(first dataset)"
        if row + 1 < self.bench_combo.count():
            self.bench_combo.removeItem(row + 1)
        self._update_buttons()

    def _update_buttons(self):
        has_data = len(self._datasets) > 0
        self.btn_train.setEnabled(has_data)
        self.btn_optimize.setEnabled(has_data)

    # ------------------------------------------------------------------
    # Feature / dim tracking
    # ------------------------------------------------------------------

    def _update_input_dim(self):
        total = 0
        for name, cb in self.feat_checks.items():
            if cb.isChecked():
                total += self.feat_lags[name].value()
        self.dim_label.setText(f"Input dim: {total}")

    def _get_feature_selections(self):
        from market_lab.ml.dataset_builder import FeatureSelection
        sels = []
        for name, cb in self.feat_checks.items():
            if cb.isChecked():
                sels.append(FeatureSelection(name, self.feat_lags[name].value()))
        return sels

    # ------------------------------------------------------------------
    # Build config helpers
    # ------------------------------------------------------------------

    def _build_model_config(self, input_dim: int):
        from market_lab.ml.model_builder import ModelConfig, LayerConfig

        task = self.task_combo.currentText()
        n_layers = self.layers_spin.value()
        units = self.units_spin.value()
        act = self.act_combo.currentText()
        loss = "binary_crossentropy" if task == "classification" else "mse"

        layers = [LayerConfig(units, act) for _ in range(n_layers)]
        return ModelConfig(
            input_dim=input_dim,
            layers=layers,
            optimizer=self.opt_combo.currentText(),
            learning_rate=self.lr_spin.value(),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            loss=loss,
            task_type=task,
        )

    def _get_benchmark_idx(self) -> int:
        idx = self.bench_combo.currentIndex()
        return max(0, idx - 1)  # 0 is "(first dataset)"

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def _run_train(self):
        if not self._datasets:
            return

        sels = self._get_feature_selections()
        if not sels:
            QMessageBox.warning(self, "No Features", "Select at least one feature builder.")
            return

        self.btn_train.setEnabled(False)
        self.btn_optimize.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, self.epochs_spin.value())
        self.progress.setValue(0)
        self.progress.setFormat("Training... epoch %v / %m")

        bars_list = [d[0] for d in self._datasets]
        task = self.task_combo.currentText()
        threshold = self.threshold_spin.value()
        bench_idx = self._get_benchmark_idx()

        # Capture config params on main thread
        sels_copy = list(sels)

        def work():
            from market_lab.ml.dataset_builder import DatasetConfig, build_dataset
            from market_lab.ml.model_builder import build_model
            from market_lab.ml.trainer import train_model, compute_equity_curves

            ds_config = DatasetConfig(
                feature_selections=sels_copy,
                task_type=task,
                split_ratio=0.8,
            )
            dataset = build_dataset(bars_list, ds_config)
            model_config = self._build_model_config(dataset.input_dim)
            model = build_model(model_config)

            result = train_model(model, dataset, model_config, verbose=0)
            compute_equity_curves(result, bars_list, threshold, bench_idx)
            return result

        worker = Worker(work)
        worker.signals.result.connect(self._on_train_done)
        worker.signals.error.connect(self._on_train_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_train_done(self, result):
        self._training_result = result
        self.progress.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_optimize.setEnabled(True)

        # Metrics summary
        task = result.config.task_type
        metric_key = "accuracy" if task == "classification" else "mae"
        lines = [
            f"Task: {task}",
            f"Epochs: {result.config.epochs}",
            f"Train loss: {result.final_train_loss:.6f}",
            f"Val loss:   {result.final_val_loss:.6f}",
            f"Train {metric_key}: {result.final_train_metric:.4f}",
            f"Val {metric_key}:   {result.final_val_metric:.4f}",
            f"Val Sharpe:  {result.val_sharpe:.4f}",
        ]
        self.metrics_text.setPlainText("\n".join(lines))

        # 4-panel chart: loss, accuracy/mae, train equity, val equity
        axes = self.chart.clear_and_get_axes(nrows=2, ncols=2)
        hist = result.history

        # Loss
        ax = axes[0]
        if "loss" in hist:
            ax.plot(hist["loss"], color="#ef5350", linewidth=1, label="Train")
        if "val_loss" in hist:
            ax.plot(hist["val_loss"], color="#4fc3f7", linewidth=1, label="Val")
        ax.set_title("Loss")
        ax.legend(facecolor="#353535", edgecolor="#555", labelcolor="#ccc", fontsize=8)

        # Metric
        ax = axes[1]
        val_key = f"val_{metric_key}"
        if metric_key in hist:
            ax.plot(hist[metric_key], color="#ef5350", linewidth=1, label="Train")
        if val_key in hist:
            ax.plot(hist[val_key], color="#4fc3f7", linewidth=1, label="Val")
        ax.set_title(metric_key.capitalize())
        ax.legend(facecolor="#353535", edgecolor="#555", labelcolor="#ccc", fontsize=8)

        # Train equity
        ax = axes[2]
        if result.train_equity is not None:
            ax.plot(result.train_equity.values, color="#66bb6a", linewidth=1, label="Strategy")
        if result.train_benchmark is not None:
            ax.plot(result.train_benchmark.values, color="#888", linewidth=1,
                    linestyle="--", label="Benchmark")
        ax.set_title("Train Equity")
        ax.legend(facecolor="#353535", edgecolor="#555", labelcolor="#ccc", fontsize=8)

        # Val equity
        ax = axes[3]
        if result.val_equity is not None:
            ax.plot(result.val_equity.values, color="#66bb6a", linewidth=1, label="Strategy")
        if result.val_benchmark is not None:
            ax.plot(result.val_benchmark.values, color="#888", linewidth=1,
                    linestyle="--", label="Benchmark")
        ax.set_title("Val Equity")
        ax.legend(facecolor="#353535", edgecolor="#555", labelcolor="#ccc", fontsize=8)

        self.chart.refresh()

    @Slot(tuple)
    def _on_train_error(self, error_info):
        self.progress.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_optimize.setEnabled(True)
        exc_type, exc_value, _ = error_info
        QMessageBox.critical(self, "Training Error", f"{exc_type.__name__}: {exc_value}")

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------

    def _run_optimize(self):
        if not self._datasets:
            return

        sels = self._get_feature_selections()
        if not sels:
            QMessageBox.warning(self, "No Features", "Select at least one feature builder.")
            return

        self.btn_train.setEnabled(False)
        self.btn_optimize.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Optimizing hyperparameters...")

        bars_list = [d[0] for d in self._datasets]
        task = self.task_combo.currentText()
        bench_idx = self._get_benchmark_idx()
        sels_copy = list(sels)

        def work():
            from market_lab.ml.dataset_builder import DatasetConfig, build_dataset
            from market_lab.ml.optimizer import optimize_ml

            ds_config = DatasetConfig(
                feature_selections=sels_copy,
                task_type=task,
                split_ratio=0.8,
            )
            dataset = build_dataset(bars_list, ds_config)

            return optimize_ml(
                dataset=dataset,
                bars_list=bars_list,
                objective="val_sharpe" if task == "classification" else "val_accuracy",
                n_trials=20,
                min_epochs=5,
                max_epochs=60,
                seed=42,
                benchmark_dataset_idx=bench_idx,
            )

        worker = Worker(work)
        worker.signals.result.connect(self._on_optimize_done)
        worker.signals.error.connect(self._on_optimize_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_optimize_done(self, result):
        self.progress.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_optimize.setEnabled(True)

        cfg = result["best_config"]
        val = result["best_value"]

        # Apply best config to UI
        self.layers_spin.setValue(len(cfg.layers))
        if cfg.layers:
            self.units_spin.setValue(cfg.layers[0].units)
            idx = self.act_combo.findText(cfg.layers[0].activation)
            if idx >= 0:
                self.act_combo.setCurrentIndex(idx)
        idx = self.opt_combo.findText(cfg.optimizer)
        if idx >= 0:
            self.opt_combo.setCurrentIndex(idx)
        self.lr_spin.setValue(cfg.learning_rate)
        self.epochs_spin.setValue(cfg.epochs)
        self.batch_spin.setValue(cfg.batch_size)

        msg = f"Best value: {val:.4f}\n\nBest config applied to UI.\nClick 'Train Model' to train with optimal parameters."
        QMessageBox.information(self, "ML Optimization Complete", msg)

    @Slot(tuple)
    def _on_optimize_error(self, error_info):
        self.progress.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_optimize.setEnabled(True)
        exc_type, exc_value, _ = error_info
        QMessageBox.critical(self, "Optimization Error", f"{exc_type.__name__}: {exc_value}")
