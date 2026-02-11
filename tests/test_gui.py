"""GUI tests using pytest-qt for widget creation and basic interactions."""

from __future__ import annotations

import pandas as pd
import pytest

from PySide6.QtCore import Qt


@pytest.fixture
def sample_bars():
    from market_lab.data.sample_generator import generate_ohlcv_bars
    return generate_ohlcv_bars(n_bars=100, seed=42)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class TestMplCanvas:
    def test_create(self, qtbot):
        from market_lab.gui.widgets import MplCanvas
        canvas = MplCanvas()
        qtbot.addWidget(canvas)
        assert canvas.fig is not None
        assert canvas.axes is not None

    def test_clear_and_get_ax(self, qtbot):
        from market_lab.gui.widgets import MplCanvas
        canvas = MplCanvas()
        qtbot.addWidget(canvas)
        ax = canvas.clear_and_get_ax()
        assert ax is not None

    def test_clear_and_get_axes(self, qtbot):
        from market_lab.gui.widgets import MplCanvas
        canvas = MplCanvas()
        qtbot.addWidget(canvas)
        axes = canvas.clear_and_get_axes(1, 2)
        assert len(axes) == 2


class TestPandasTableModel:
    def test_set_dataframe(self, qtbot):
        from market_lab.gui.widgets import PandasTableModel
        model = PandasTableModel()
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        model.set_dataframe(df)
        assert model.rowCount() == 2
        assert model.columnCount() == 2

    def test_data_formatting(self, qtbot):
        from market_lab.gui.widgets import PandasTableModel
        from PySide6.QtCore import QModelIndex
        model = PandasTableModel()
        df = pd.DataFrame({"val": [3.14159]})
        model.set_dataframe(df)
        idx = model.index(0, 0)
        assert model.data(idx) == "3.1416"

    def test_empty(self, qtbot):
        from market_lab.gui.widgets import PandasTableModel
        model = PandasTableModel()
        assert model.rowCount() == 0
        assert model.columnCount() == 0


class TestSortableTableModel:
    def test_set_and_sort(self, qtbot):
        from market_lab.gui.widgets import SortableTableModel
        model = SortableTableModel()
        df = pd.DataFrame({"name": ["B", "A", "C"], "val": [2, 1, 3]})
        model.set_dataframe(df)
        assert model.rowCount() == 3
        model.sort(1, Qt.SortOrder.AscendingOrder)
        assert model.rowCount() == 3


class TestDataLoaderPanel:
    def test_create(self, qtbot):
        from market_lab.gui.widgets import DataLoaderPanel
        panel = DataLoaderPanel()
        qtbot.addWidget(panel)
        assert panel is not None

    def test_sample_data_signal(self, qtbot):
        from market_lab.gui.widgets import DataLoaderPanel
        panel = DataLoaderPanel()
        qtbot.addWidget(panel)
        received = []
        panel.data_loaded.connect(lambda df, label: received.append((df, label)))
        panel._gen_sample()
        assert len(received) == 1
        df, label = received[0]
        assert len(df) == 500
        assert label == "sample_500bars"


# ---------------------------------------------------------------------------
# Settings dialogs
# ---------------------------------------------------------------------------

class TestStrategySettingsDialog:
    def test_create_and_get(self, qtbot):
        from market_lab.gui.settings_dialog import StrategySettingsDialog
        schema = {
            "fast": {"type": "int", "default": 10, "min": 2, "max": 200},
            "slow": {"type": "int", "default": 30, "min": 5, "max": 500},
            "allow_short": {"type": "bool", "default": True},
        }
        dlg = StrategySettingsDialog(schema, {"fast": 10, "slow": 30, "allow_short": True})
        qtbot.addWidget(dlg)
        params = dlg.get_params()
        assert params["fast"] == 10
        assert params["slow"] == 30
        assert params["allow_short"] is True

    def test_enum_parameter(self, qtbot):
        from market_lab.gui.settings_dialog import StrategySettingsDialog
        schema = {
            "ma_type": {"type": "enum", "default": "SMA", "choices": ["SMA", "EMA"]},
        }
        dlg = StrategySettingsDialog(schema, {"ma_type": "SMA"})
        qtbot.addWidget(dlg)
        params = dlg.get_params()
        assert params["ma_type"] == "SMA"


class TestBacktestConfigDialog:
    def test_defaults(self, qtbot):
        from market_lab.gui.settings_dialog import BacktestConfigDialog
        dlg = BacktestConfigDialog()
        qtbot.addWidget(dlg)
        cfg = dlg.get_config()
        assert cfg["initial_capital"] == 100_000
        assert cfg["allow_short"] is True


# ---------------------------------------------------------------------------
# Tab creation (no full interaction, just instantiation)
# ---------------------------------------------------------------------------

class TestBacktestTab:
    def test_create(self, qtbot):
        from market_lab.gui.backtest_tab import BacktestTab
        tab = BacktestTab()
        qtbot.addWidget(tab)
        assert tab.btn_run is not None
        assert tab.btn_optimize is not None
        assert not tab.btn_run.isEnabled()

    def test_data_loaded_enables_buttons(self, qtbot, sample_bars):
        from market_lab.gui.backtest_tab import BacktestTab
        tab = BacktestTab()
        qtbot.addWidget(tab)
        tab._on_data_loaded(sample_bars, "test")
        assert tab.btn_run.isEnabled()
        assert tab.btn_optimize.isEnabled()


class TestMLTab:
    def test_create(self, qtbot):
        from market_lab.gui.ml_tab import MLTab
        tab = MLTab()
        qtbot.addWidget(tab)
        assert tab.btn_train is not None
        assert not tab.btn_train.isEnabled()

    def test_data_loaded_enables_train(self, qtbot, sample_bars):
        from market_lab.gui.ml_tab import MLTab
        tab = MLTab()
        qtbot.addWidget(tab)
        tab._on_data_loaded(sample_bars, "test")
        assert tab.btn_train.isEnabled()

    def test_input_dim_tracking(self, qtbot):
        from market_lab.gui.ml_tab import MLTab
        tab = MLTab()
        qtbot.addWidget(tab)
        # With default checked features, dim should be > 0
        tab._update_input_dim()
        text = tab.dim_label.text()
        assert "Input dim:" in text


class TestMainWindow:
    def test_create(self, qtbot):
        from market_lab.gui.main_window import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        assert win.tabs.count() == 2

    def test_tab_names(self, qtbot):
        from market_lab.gui.main_window import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        assert "Backtest" in names
        assert "ML" in names

    def test_status_bar(self, qtbot):
        from market_lab.gui.main_window import MainWindow
        win = MainWindow()
        qtbot.addWidget(win)
        win.log_status("Testing")
        assert win.status_bar.currentMessage() == "Testing"
