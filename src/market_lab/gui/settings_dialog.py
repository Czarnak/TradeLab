"""Dynamic settings dialog built from a strategy's parameters_schema()."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class StrategySettingsDialog(QDialog):
    """Modal dialog that dynamically builds input fields from a parameter schema.

    Usage::

        dlg = StrategySettingsDialog(schema, current_params, parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_params = dlg.get_params()
    """

    def __init__(self, schema: dict, current_params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Strategy Parameters")
        self.setMinimumWidth(380)

        self._widgets: dict[str, object] = {}
        self._schema = schema

        layout = QVBoxLayout(self)

        grp = QGroupBox("Parameters")
        form = QFormLayout(grp)

        for key, spec in schema.items():
            ptype = spec.get("type", "float")
            default = current_params.get(key, spec.get("default"))

            if ptype == "int":
                w = QSpinBox()
                w.setMinimum(spec.get("min", 0))
                w.setMaximum(spec.get("max", 999999))
                if "step" in spec:
                    w.setSingleStep(spec["step"])
                w.setValue(int(default) if default is not None else 0)
                self._widgets[key] = w

            elif ptype == "float":
                w = QDoubleSpinBox()
                w.setDecimals(4)
                w.setMinimum(spec.get("min", -1e9))
                w.setMaximum(spec.get("max", 1e9))
                if "step" in spec:
                    w.setSingleStep(spec["step"])
                w.setValue(float(default) if default is not None else 0.0)
                self._widgets[key] = w

            elif ptype == "bool":
                w = QCheckBox()
                w.setChecked(bool(default) if default is not None else True)
                self._widgets[key] = w

            elif ptype == "enum":
                w = QComboBox()
                choices = spec.get("choices", [])
                w.addItems([str(c) for c in choices])
                if default is not None and str(default) in [str(c) for c in choices]:
                    w.setCurrentText(str(default))
                self._widgets[key] = w

            else:
                w = QLabel(f"(unsupported type: {ptype})")
                self._widgets[key] = w

            label_text = key.replace("_", " ").title()
            form.addRow(label_text + ":", w)

        layout.addWidget(grp)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> dict:
        """Read current widget values and return as a dict."""
        params = {}
        for key, widget in self._widgets.items():
            spec = self._schema[key]
            ptype = spec.get("type", "float")

            if ptype == "int":
                params[key] = widget.value()
            elif ptype == "float":
                params[key] = widget.value()
            elif ptype == "bool":
                params[key] = widget.isChecked()
            elif ptype == "enum":
                params[key] = widget.currentText()
            else:
                params[key] = spec.get("default")

        return params


class BacktestConfigDialog(QDialog):
    """Dialog for backtest engine configuration (capital, commission, slippage)."""

    def __init__(self, current: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Backtest Configuration")
        self.setMinimumWidth(340)

        current = current or {}
        layout = QVBoxLayout(self)

        grp = QGroupBox("Engine Settings")
        form = QFormLayout(grp)

        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1_000, 100_000_000)
        self.capital_spin.setDecimals(0)
        self.capital_spin.setSingleStep(10_000)
        self.capital_spin.setValue(current.get("initial_capital", 100_000))
        form.addRow("Initial Capital:", self.capital_spin)

        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 100)
        self.commission_spin.setDecimals(1)
        self.commission_spin.setSuffix(" bps")
        self.commission_spin.setValue(current.get("commission_bps", 5))
        form.addRow("Commission:", self.commission_spin)

        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0, 100)
        self.slippage_spin.setDecimals(1)
        self.slippage_spin.setSuffix(" bps")
        self.slippage_spin.setValue(current.get("slippage_bps", 2))
        form.addRow("Slippage:", self.slippage_spin)

        self.short_check = QCheckBox("Allow short positions")
        self.short_check.setChecked(current.get("allow_short", True))
        form.addRow(self.short_check)

        layout.addWidget(grp)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> dict:
        return {
            "initial_capital": self.capital_spin.value(),
            "commission_bps": self.commission_spin.value(),
            "slippage_bps": self.slippage_spin.value(),
            "allow_short": self.short_check.isChecked(),
        }
