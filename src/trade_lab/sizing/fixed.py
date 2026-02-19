from trade_lab.sizing.base import BasePositionSizer


class FixedPositionSizer(BasePositionSizer):
    """Allocates a fixed fraction of equity per trade.

    Parameters
    ----------
    fraction : float
        Fraction of equity to allocate (e.g. 0.02 = 2%).
        Must be in (0, 1].
    """

    def __init__(self, fraction: float = 0.02):
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction

    def compute_size(
        self,
        signal_strength: float,
        equity: float,
        price: float,
        volatility: float | None = None,
    ) -> float:
        capital = equity * self.fraction
        return capital / price
