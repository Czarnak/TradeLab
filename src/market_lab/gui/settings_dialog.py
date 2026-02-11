"""Dynamic settings dialog built from a strategy's parameters_schema()."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    """Dialog that dynamically renders input controls from a parameter schema.

    Parameters
    ----------
    schema : dict
        Strategy ``parameters_schema()`` output.
    current_values : dict
        Current parameter values (used to pre-fill controls).
    title : str
        Dialog window title.
    """

    def __init__(
        self,
        schema: dict,
        current_values: dict | None = None,
        title: str = "Strategy Settings",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(380)

        self._schema = schema
        self._values = dict(current_values) if current_values else {}
        self._widgets: dict[str, object] = {}

        layout = QVBoxLayout(self)
        form = QFormLayout()

        for pname, pdef in schema.items():
            widget = self._create_widget(pname, pdef)
            label_text = pdef.get("description", pname)
            form.addRow(QLabel(label_text + ":"), widget)
            self._widgets[pname] = widget

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _create_widget(self, name: str, pdef: dict):
        ptype = pdef.get("type", "float")
        default = pdef.get("default")
        current = self._values.get(name, default)

        if ptype == "int":
            w = QSpinBox()
            w.setMinimum(int(pdef.get("min", 0)))
            w.setMaximum(int(pdef.get("max", 999_999)))
            w.setSingleStep(int(pdef.get("step", 1)))
            if current is not None:
                w.setValue(int(current))
            return w

        elif ptype == "float":
            w = QDoubleSpinBox()
            w.setMinimum(float(pdef.get("min", -999_999)))
            w.setMaximum(float(pdef.get("max", 999_999)))
            step = pdef.get("step")
            if step:
                w.setSingleStep(float(step))
                # Set decimals based on step precision
                dec = len(str(float(step)).rstrip("0").split(".")[-1])
                w.setDecimals(max(dec, 2))
            else:
                w.setDecimals(4)
                w.setSingleStep(0.01)
            if current is not None:
                w.setValue(float(current))
            return w

        elif ptype == "bool":
            w = QCheckBox()
            if current is not None:
                w.setChecked(bool(current))
            return w

        elif ptype == "enum":
            w = QComboBox()
            choices = pdef.get("choices", [])
            w.addItems([str(c) for c in choices])
            if current is not None and str(current) in [str(c) for c in choices]:
                w.setCurrentText(str(current))
            return w

        else:
            # Fallback: read-only label
            w = QLabel(str(current or ""))
            return w

    def get_values(self) -> dict:
        """Return the parameter values from the dialog widgets."""
        values = {}
        for pname, pdef in self._schema.items():
            widget = self._widgets[pname]
            ptype = pdef.get("type", "float")

            if ptype == "int":
                values[pname] = widget.value()
            elif ptype == "float":
                values[pname] = widget.value()
            elif ptype == "bool":
                values[pname] = widget.isChecked()
            elif ptype == "enum":
                text = widget.currentText()
                # Try to preserve original type
                choices = pdef.get("choices", [])
                for c in choices:
                    if str(c) == text:
                        values[pname] = c
                        break
                else:
                    values[pname] = text
            else:
                values[pname] = self._values.get(pname)

        return values
