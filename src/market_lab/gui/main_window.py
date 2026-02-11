"""Main window with dark Fusion theme and tabbed layout."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QLabel,
)


def apply_dark_theme(app: QApplication) -> None:
    """Apply a dark Fusion palette to the application."""
    app.setStyle("Fusion")

    palette = QPalette()

    # Base colours
    dark = QColor(45, 45, 45)
    medium = QColor(53, 53, 53)
    light_grey = QColor(180, 180, 180)
    white = QColor(220, 220, 220)
    accent = QColor(79, 195, 247)  # light blue
    disabled = QColor(127, 127, 127)
    dark_bg = QColor(35, 35, 35)

    palette.setColor(QPalette.ColorRole.Window, medium)
    palette.setColor(QPalette.ColorRole.WindowText, white)
    palette.setColor(QPalette.ColorRole.Base, dark)
    palette.setColor(QPalette.ColorRole.AlternateBase, medium)
    palette.setColor(QPalette.ColorRole.ToolTipBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipText, white)
    palette.setColor(QPalette.ColorRole.Text, white)
    palette.setColor(QPalette.ColorRole.Button, medium)
    palette.setColor(QPalette.ColorRole.ButtonText, white)
    palette.setColor(QPalette.ColorRole.BrightText, accent)
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.PlaceholderText, disabled)

    # Disabled state
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled)

    app.setPalette(palette)

    # Extra stylesheet polish
    app.setStyleSheet("""
        QToolTip {
            color: #dcdcdc;
            background-color: #2d2d2d;
            border: 1px solid #555;
            padding: 4px;
        }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 14px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QTabWidget::pane {
            border: 1px solid #555;
            border-radius: 2px;
        }
        QTabBar::tab {
            background: #353535;
            color: #aaa;
            padding: 8px 18px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #2d2d2d;
            color: #4fc3f7;
            border-bottom: 2px solid #4fc3f7;
        }
        QTabBar::tab:hover {
            color: #ddd;
        }
        QPushButton {
            background-color: #404040;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px 14px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
            border-color: #4fc3f7;
        }
        QPushButton:pressed {
            background-color: #353535;
        }
        QPushButton:disabled {
            background-color: #333;
            color: #666;
            border-color: #444;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 4px;
            text-align: center;
            background: #2d2d2d;
            height: 20px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #4fc3f7, stop:1 #29b6f6);
            border-radius: 3px;
        }
        QHeaderView::section {
            background: #353535;
            color: #ccc;
            border: 1px solid #444;
            padding: 4px;
        }
        QTableView {
            gridline-color: #444;
            alternate-background-color: #2f2f2f;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: #2d2d2d;
            width: 12px;
            height: 12px;
        }
        QScrollBar::handle {
            background: #555;
            border-radius: 4px;
            min-height: 20px;
        }
        QScrollBar::handle:hover {
            background: #666;
        }
        QTextEdit, QLineEdit {
            background: #2d2d2d;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px;
        }
        QComboBox {
            background: #2d2d2d;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px 8px;
        }
        QComboBox::drop-down {
            border-left: 1px solid #555;
        }
        QSpinBox, QDoubleSpinBox {
            background: #2d2d2d;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px;
        }
    """)


class PlaceholderTab(QWidget):
    """Temporary placeholder for unimplemented tabs."""

    def __init__(self, message: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 16px; color: #888;")
        layout.addWidget(label)


class MainWindow(QMainWindow):
    """Market-Lab main window with Backtest and ML tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Market-Lab")
        self.setMinimumSize(1100, 700)
        self.resize(1350, 850)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._init_backtest_tab()
        self._init_ml_tab()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _init_backtest_tab(self):
        try:
            from market_lab.gui.backtest_tab import BacktestTab
            self.backtest_tab = BacktestTab()
            self.tabs.addTab(self.backtest_tab, "Backtest")
        except Exception:
            self.tabs.addTab(PlaceholderTab("Backtest tab failed to load"), "Backtest")

    def _init_ml_tab(self):
        try:
            from market_lab.gui.ml_tab import MLTab
            self.ml_tab = MLTab()
            self.tabs.addTab(self.ml_tab, "ML")
        except Exception:
            self.tabs.addTab(PlaceholderTab("ML tab failed to load"), "ML")

    def log_status(self, message: str):
        self.status_bar.showMessage(message)
