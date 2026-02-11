"""Insider scan tab — placeholder for Phase 3 implementation."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class InsiderTab(QWidget):
    """Insider trading scanner tab."""

    def __init__(self, status_callback=None, parent=None):
        super().__init__(parent)
        self._status = status_callback or (lambda msg: None)
        layout = QVBoxLayout(self)
        label = QLabel(
            "<h2>Insider Scan</h2>"
            "<p>This module will be implemented in Phase 3.</p>"
            "<p>Features: secform4 / openinsider scraping, EDGAR verification, "
            "senate/congress scanning, and sortable results table.</p>"
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch()
