from abc import ABC, abstractmethod


class BasePositionSizer(ABC):
    """
    Computes position size given current market state and signal strength.
    
    signal_strength is passed through so that size can optionally
    scale with conviction (e.g. RiskBasedPositionSizer).
    """

    @abstractmethod
    def compute_size(
        self,
        signal_strength: float,
        equity: float,
        price: float,
        volatility: float | None = None
    ) -> float:
        """Returns position size in number of units/shares."""
        ...