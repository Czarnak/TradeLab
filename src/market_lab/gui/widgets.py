"""Reusable Qt widgets: matplotlib chart canvas and pandas table model."""

from __future__ import annotations

from typing import Any

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class ChartCanvas(FigureCanvas):
    """Embeddable matplotlib canvas widget."""

    def __init__(self, parent=None, width: int = 8, height: int = 4, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def clear(self) -> None:
        self.axes.clear()
        self.draw()

    def plot_series(
        self,
        series: pd.Series,
        title: str = "",
        ylabel: str = "",
        color: str = "#2196F3",
    ) -> None:
        self.axes.clear()
        self.axes.plot(series.index, series.values, linewidth=1.2, color=color)
        if title:
            self.axes.set_title(title, fontsize=12)
        if ylabel:
            self.axes.set_ylabel(ylabel)
        self.axes.grid(True, alpha=0.3)
        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.draw()


class PandasTableModel(QAbstractTableModel):
    """Qt table model backed by a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        self.beginResetModel()
        col_name = self._df.columns[column]
        ascending = order == Qt.SortOrder.AscendingOrder
        self._df = self._df.sort_values(col_name, ascending=ascending).reset_index(drop=True)
        self.endResetModel()
