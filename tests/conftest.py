import sys
import types
from pathlib import Path
from enum import Enum

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import plotly.graph_objects  # noqa: F401
except ModuleNotFoundError:
    class _DummyFigure:
        def add_trace(self, *args, **kwargs):
            return None

        def add_hline(self, *args, **kwargs):
            return None

        def update_layout(self, *args, **kwargs):
            return None

        def show(self, *args, **kwargs):
            return None

        def to_html(self, *args, **kwargs):
            return "<div></div>"

    class _DummyScatter:
        def __init__(self, *args, **kwargs):
            pass

    plotly = types.ModuleType("plotly")
    graph_objects = types.ModuleType("graph_objects")
    subplots = types.ModuleType("subplots")
    graph_objects.Figure = _DummyFigure
    graph_objects.Scatter = _DummyScatter
    subplots.make_subplots = lambda *args, **kwargs: _DummyFigure()
    plotly.graph_objects = graph_objects
    plotly.subplots = subplots

    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = graph_objects
    sys.modules["plotly.subplots"] = subplots

try:
    import yfinance  # noqa: F401
except ModuleNotFoundError:
    yfinance = types.ModuleType("yfinance")

    def _dummy_download(*args, **kwargs):
        raise RuntimeError("yfinance is not available in the test environment")

    yfinance.download = _dummy_download
    sys.modules["yfinance"] = yfinance

try:
    import optuna  # noqa: F401
except ModuleNotFoundError:
    optuna = types.ModuleType("optuna")

    class _TrialPruned(Exception):
        pass

    class _TrialState(Enum):
        COMPLETE = "COMPLETE"
        FAIL = "FAIL"
        PRUNED = "PRUNED"

    class _DummyStudy:
        def __init__(self, *args, **kwargs):
            self.best_params = {}
            self.best_value = 0.0
            self.trials = []

        def optimize(self, *args, **kwargs):
            return None

    class _DummyTPESampler:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyMedianPruner:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyKerasPruningCallback:
        def __init__(self, *args, **kwargs):
            pass

    def _dummy_create_study(*args, **kwargs):
        return _DummyStudy()

    class _Logging:
        WARNING = 30
        INFO = 20

        @staticmethod
        def set_verbosity(level):
            return None

    optuna.Study = _DummyStudy
    optuna.Trial = object
    optuna.TrialPruned = _TrialPruned
    optuna.create_study = _dummy_create_study
    optuna.logging = _Logging
    optuna.exceptions = types.SimpleNamespace(TrialPruned=_TrialPruned)
    optuna.samplers = types.SimpleNamespace(TPESampler=_DummyTPESampler)
    optuna.pruners = types.SimpleNamespace(MedianPruner=_DummyMedianPruner)
    optuna.trial = types.SimpleNamespace(TrialState=_TrialState)

    integration = types.ModuleType("integration")
    integration.KerasPruningCallback = _DummyKerasPruningCallback
    optuna.integration = integration

    sys.modules["optuna"] = optuna
    sys.modules["optuna.integration"] = integration
