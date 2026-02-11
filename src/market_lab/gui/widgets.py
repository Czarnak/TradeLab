"""Reusable GUI widgets: matplotlib canvas, sortable table, data loader panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Signal,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Matplotlib canvas
# ---------------------------------------------------------------------------

class MplCanvas(FigureCanvas):
    """Embeddable matplotlib canvas with dark styling."""

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("#2b2b2b")
        self.axes = self.fig.add_subplot(111)
        self._style_axes(self.axes)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _style_axes(ax):
        ax.set_facecolor("#353535")
        ax.tick_params(colors="#cccccc", which="both")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#eeeeee")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.grid(True, alpha=0.2, color="#888888")

    def clear_and_get_ax(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style_axes(ax)
        self.axes = ax
        return ax

    def clear_and_get_axes(self, nrows=1, ncols=2):
        self.fig.clear()
        axes = []
        for i in range(nrows * ncols):
            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            self._style_axes(ax)
            axes.append(ax)
        return axes

    def refresh(self):
        self.fig.tight_layout()
        self.draw()


# ---------------------------------------------------------------------------
# Pandas table model
# ---------------------------------------------------------------------------

class PandasTableModel(QAbstractTableModel):
    """Qt table model backed by a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iloc[index.row(), index.column()]
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)

    @property
    def dataframe(self):
        return self._df


class SortableTableModel(QSortFilterProxyModel):
    """Proxy adding sort/filter on top of PandasTableModel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = PandasTableModel(parent=self)
        self.setSourceModel(self._source)
        self.setDynamicSortFilter(True)

    def set_dataframe(self, df: pd.DataFrame):
        self._source.set_dataframe(df)

    @property
    def dataframe(self):
        return self._source.dataframe


# ---------------------------------------------------------------------------
# Data loader panel
# ---------------------------------------------------------------------------

class DataLoaderPanel(QWidget):
    """CSV file picker + Yahoo download + sample data generator."""

    data_loaded = Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # CSV
        csv_grp = QGroupBox("Load CSV")
        csv_l = QHBoxLayout(csv_grp)
        self.csv_path = QLineEdit()
        self.csv_path.setPlaceholderText("Path to OHLCV CSV...")
        self.csv_path.setReadOnly(True)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse)
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self._load_csv)
        csv_l.addWidget(self.csv_path, stretch=1)
        csv_l.addWidget(btn_browse)
        csv_l.addWidget(btn_load)
        layout.addWidget(csv_grp)

        # Yahoo
        yh_grp = QGroupBox("Yahoo Finance")
        yh_l = QHBoxLayout(yh_grp)
        self.sym_edit = QLineEdit()
        self.sym_edit.setPlaceholderText("AAPL")
        self.sym_edit.setMaximumWidth(100)
        self.period_cb = QComboBox()
        self.period_cb.addItems(["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
        self.period_cb.setCurrentIndex(3)
        self.interval_cb = QComboBox()
        self.interval_cb.addItems(["1m", "5m", "15m", "1h", "1d"])
        self.interval_cb.setCurrentIndex(4)
        btn_dl = QPushButton("Download")
        btn_dl.clicked.connect(self._download)
        for w, lbl in [(self.sym_edit, "Symbol:"), (self.period_cb, "Period:"),
                       (self.interval_cb, "Interval:")]:
            yh_l.addWidget(QLabel(lbl))
            yh_l.addWidget(w)
        yh_l.addWidget(btn_dl)
        layout.addWidget(yh_grp)

        # Sample
        sl = QHBoxLayout()
        btn_sample = QPushButton("Load Sample OHLCV (500 bars)")
        btn_sample.clicked.connect(self._gen_sample)
        sl.addWidget(btn_sample)
        sl.addStretch()
        layout.addLayout(sl)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "CSV File", "", "CSV (*.csv);;All (*)")
        if p:
            self.csv_path.setText(p)

    def _load_csv(self):
        p = self.csv_path.text().strip()
        if not p:
            return
        try:
            from market_lab.data.loaders import load_csv
            df = load_csv(p, schema="bars")
            self.data_loaded.emit(df, Path(p).stem)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", str(e))

    def _download(self):
        sym = self.sym_edit.text().strip().upper()
        if not sym:
            return
        try:
            from market_lab.data.yahoo import download
            df = download(sym, self.period_cb.currentText(), self.interval_cb.currentText())
            self.data_loaded.emit(df, f"{sym}_{self.interval_cb.currentText()}")
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", str(e))

    def _gen_sample(self):
        from market_lab.data.sample_generator import generate_ohlcv_bars
        self.data_loaded.emit(generate_ohlcv_bars(500, seed=42), "sample_500bars")
