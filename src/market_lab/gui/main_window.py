"""Main application window with tabbed layout."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QLabel,
)


class PlaceholderTab(QWidget):
    """Temporary placeholder tab for modules not yet integrated."""

    def __init__(self, message: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 16px; color: #888;")
        layout.addWidget(label)


class MainWindow(QMainWindow):
    """Market-Lab main window with Backtest, ML, and Insider Scan tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Market-Lab")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._init_backtest_tab()
        self._init_ml_tab()
        self._init_insider_tab()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _init_backtest_tab(self) -> None:
        try:
            from market_lab.gui.backtest_tab import BacktestTab
            self.backtest_tab = BacktestTab()
            self.tabs.addTab(self.backtest_tab, "Backtest")
        except ImportError:
            self.tabs.addTab(
                PlaceholderTab("Backtest module — coming in next phase"),
                "Backtest",
            )

    def _init_ml_tab(self) -> None:
        try:
            from market_lab.gui.ml_tab import MLTab
            self.ml_tab = MLTab()
            self.tabs.addTab(self.ml_tab, "ML")
        except ImportError:
            self.tabs.addTab(
                PlaceholderTab("ML module — coming in next phase"),
                "ML",
            )

    def _init_insider_tab(self) -> None:
        try:
            from market_lab.gui.insider_tab import InsiderTab
            self.insider_tab = InsiderTab()
            self.tabs.addTab(self.insider_tab, "Insider Scan")
        except ImportError:
            self.tabs.addTab(
                PlaceholderTab("Insider Scan module — coming in next phase"),
                "Insider Scan",
            )

    def log_status(self, message: str) -> None:
        """Update the status bar message."""
        self.status_bar.showMessage(message)
