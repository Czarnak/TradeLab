"""Registry mapping TradeLab signal classes to MQL5 rendering metadata.

Describes how each upstream signal is computed in MQL5 and whether it
requires persistent global state across ticks.
"""

from __future__ import annotations

from dataclasses import dataclass

from trade_lab.signals.signals import OHLC, HeikinAshi
from trade_lab.signals.temporal import CyclicalTemporalSignal


@dataclass
class MQL5SignalDescriptor:
    """MQL5 rendering metadata for a TradeLab signal.

    Parameters
    ----------
    signal_type : str
        Snake-case identifier: ``'ohlc'``, ``'heikin_ashi'``,
        ``'cyclical_temporal'``.
    mql5_approach : str
        Human-readable description of the MQL5 computation strategy.
    requires_state : bool
        True when the signal needs persistent global variables across ticks
        (e.g. Heikin-Ashi requires the previous HA open/close).
    template : str
        Relative path to the Jinja2 sub-template that renders this signal,
        e.g. ``'signals/heikin_ashi.mq5.j2'``.
    """

    signal_type: str
    mql5_approach: str
    requires_state: bool
    template: str


SIGNAL_REGISTRY: dict[type, MQL5SignalDescriptor] = {
    OHLC: MQL5SignalDescriptor(
        signal_type="ohlc",
        mql5_approach=(
            "Computed per-bar using iOpen/iHigh/iLow/iClose with shift. "
            "log_return_x = MathLog(iX(0) / iClose(1)). No global state needed."
        ),
        requires_state=False,
        template="signals/ohlc.mq5.j2",
    ),
    HeikinAshi: MQL5SignalDescriptor(
        signal_type="heikin_ashi",
        mql5_approach=(
            "Iterative HA bar computation. Global variables g_ha_open_prev and "
            "g_ha_close_prev carry state across ticks. Initialised on first bar "
            "using (Open[0]+Close[0])/2 and HA_Close[0]."
        ),
        requires_state=True,
        template="signals/heikin_ashi.mq5.j2",
    ),
    CyclicalTemporalSignal: MQL5SignalDescriptor(
        signal_type="cyclical_temporal",
        mql5_approach=(
            "Extract datetime component via iTime() + TimeToStruct(). "
            "Compute angle = 2*M_PI*component/period, then MathSin/MathCos."
        ),
        requires_state=False,
        template="signals/temporal.mq5.j2",
    ),
}

# MQL5 MqlDateTime field names for each TradeLab temporal component.
# Used by the CyclicalTemporalSignal template to extract the right field.
TEMPORAL_COMPONENT_MQL5: dict[str, str] = {
    "hour": "dt.hour",
    "day_of_week": "dt.day_of_week",
    "day_of_month": "dt.day",
    "month": "dt.mon",
    "day_of_year": "dt.day_of_year",
}
