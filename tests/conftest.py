import sys
import types
from pathlib import Path

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

        def update_layout(self, *args, **kwargs):
            return None

        def show(self, *args, **kwargs):
            return None

    class _DummyScatter:
        def __init__(self, *args, **kwargs):
            pass

    plotly = types.ModuleType("plotly")
    graph_objects = types.ModuleType("graph_objects")
    graph_objects.Figure = _DummyFigure
    graph_objects.Scatter = _DummyScatter
    plotly.graph_objects = graph_objects

    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = graph_objects
