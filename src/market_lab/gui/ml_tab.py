"""ML model builder tab — placeholder for Phase 2 implementation."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class MLTab(QWidget):
    """ML model builder / trainer / optimizer tab."""

    def __init__(self, status_callback=None, parent=None):
        super().__init__(parent)
        self._status = status_callback or (lambda msg: None)
        layout = QVBoxLayout(self)
        label = QLabel(
            "<h2>ML Model Builder</h2>"
            "<p>This module will be implemented in Phase 2.</p>"
            "<p>Features: feature definitions, Keras model architecture, "
            "training, validation, and Optuna hyperparameter optimization.</p>"
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch()
