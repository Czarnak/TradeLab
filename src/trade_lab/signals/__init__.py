from trade_lab.signals.base import BaseSignal

__all__ = [
    "BaseSignal",
    "ExternalSignal",
    "OHLC",
    "HeikinAshi",
    "CyclicalTemporalSignal",
]


def __getattr__(name: str):
    if name == "ExternalSignal":
        from trade_lab.signals.external import ExternalSignal

        return ExternalSignal
    if name == "OHLC":
        from trade_lab.signals.signals import OHLC

        return OHLC
    if name == "HeikinAshi":
        from trade_lab.signals.signals import HeikinAshi

        return HeikinAshi
    if name == "CyclicalTemporalSignal":
        from trade_lab.signals.temporal import CyclicalTemporalSignal

        return CyclicalTemporalSignal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
