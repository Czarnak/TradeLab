"""Registry mapping TradeLab position sizer classes to MQL5 rendering metadata.

Describes the lot-calculation approach and any auxiliary handles (e.g. ATR)
required for each sizer type.
"""
from __future__ import annotations

from dataclasses import dataclass

from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.sizing.risk_based import RiskBasedPositionSizer


@dataclass
class MQL5SizingDescriptor:
    """MQL5 rendering metadata for a TradeLab position sizer.

    Parameters
    ----------
    sizer_type : str
        Snake-case identifier: ``'none'``, ``'fixed'``, ``'risk_based'``.
    mql5_approach : str
        Human-readable description of the MQL5 lot-calculation strategy.
    requires_atr : bool
        True when an ``iATR`` handle must be created in ``OnInit`` and
        released in ``OnDeinit`` (risk-based sizer only).
    template : str
        Relative path to the Jinja2 sub-template for input declarations and
        lot-calculation code.
    """

    sizer_type: str
    mql5_approach: str
    requires_atr: bool
    template: str


SIZING_REGISTRY: dict[type | None, MQL5SizingDescriptor] = {
    type(None): MQL5SizingDescriptor(
        sizer_type="none",
        mql5_approach=(
            "Fixed lot size exposed as 'input double Lots = 0.1'. "
            "Lot calculation: use Lots directly, normalised via NormalizeLots()."
        ),
        requires_atr=False,
        template="sizing/fixed.mq5.j2",
    ),
    FixedPositionSizer: MQL5SizingDescriptor(
        sizer_type="fixed",
        mql5_approach=(
            "lots = AccountInfoDouble(ACCOUNT_BALANCE) * SizingFraction / price. "
            "Normalised with NormalizeLots(). Matches Python: equity * fraction / price."
        ),
        requires_atr=False,
        template="sizing/fixed.mq5.j2",
    ),
    RiskBasedPositionSizer: MQL5SizingDescriptor(
        sizer_type="risk_based",
        mql5_approach=(
            "risk_capital = balance * MaxFraction * MathAbs(signal_strength). "
            "lots = risk_capital / (atr_value * RiskMultiplier). "
            "ATR via iATR() handle. Normalised with NormalizeLots(). "
            "Matches Python: risk_capital / (volatility * risk_multiplier)."
        ),
        requires_atr=True,
        template="sizing/risk_based.mq5.j2",
    ),
}
