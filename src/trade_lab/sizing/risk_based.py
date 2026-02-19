from trade_lab.sizing.base import BasePositionSizer


class RiskBasedPositionSizer(BasePositionSizer):
    """Position size scales with conviction and inverse volatility.

    Formula:
        risk_capital = equity × max_fraction × |signal_strength|
        units        = risk_capital / (volatility × risk_multiplier)

    Higher conviction (|signal_strength| → 1) and lower volatility
    produce larger positions.  The risk_multiplier controls how many
    ATR-widths of risk one unit represents (default 2).

    Parameters
    ----------
    max_fraction : float
        Maximum fraction of equity at full conviction (e.g. 0.02 = 2%).
    risk_multiplier : float
        Number of volatility units (e.g. ATR) per share of risk.
        Higher values produce smaller positions.
    """

    def __init__(self, max_fraction: float = 0.02, risk_multiplier: float = 2.0):
        if not 0 < max_fraction <= 1:
            raise ValueError("max_fraction must be in (0, 1]")
        if risk_multiplier <= 0:
            raise ValueError("risk_multiplier must be positive")
        self.max_fraction = max_fraction
        self.risk_multiplier = risk_multiplier

    def compute_size(
        self,
        signal_strength: float,
        equity: float,
        price: float,
        volatility: float | None = None,
    ) -> float:
        if volatility is None or volatility <= 0:
            return 0.0
        risk_capital = equity * self.max_fraction * abs(signal_strength)
        return risk_capital / (volatility * self.risk_multiplier)
