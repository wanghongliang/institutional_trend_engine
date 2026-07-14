"""
Position sizing.

Computes the number of shares/contracts to trade based on:
  - Account equity
  - Risk per trade (% of equity)
  - Stop-loss distance (in price)

Formula:  shares = (equity × risk_per_trade) / stop_distance

Also enforces a maximum position size cap.
"""

from dataclasses import dataclass


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    shares: int
    risk_amount: float       # dollar risk
    stop_distance: float     # price distance to stop
    risk_per_share: float


class PositionSizer:
    """
    Risk-based position sizer.

    Parameters
    ----------
    risk_per_trade : float
        Fraction of equity to risk per trade (e.g., 0.01 = 1%).
    max_position_size : int
        Maximum number of shares regardless of calculation.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_position_size: int = 100,
    ):
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size

    def calculate(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
    ) -> PositionSize:
        """
        Calculate position size.

        Parameters
        ----------
        equity : float
            Current account equity.
        entry_price : float
            Planned entry price.
        stop_price : float
            Planned stop-loss price.

        Returns
        -------
        PositionSize
        """
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return PositionSize(
                shares=0, risk_amount=0,
                stop_distance=0, risk_per_share=0,
            )

        risk_amount = equity * self.risk_per_trade
        shares = int(risk_amount / stop_distance)
        shares = min(shares, self.max_position_size)
        shares = max(shares, 0)

        return PositionSize(
            shares=shares,
            risk_amount=risk_amount,
            stop_distance=stop_distance,
            risk_per_share=stop_distance,
        )

    def calculate_atr_based(
        self,
        equity: float,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 1.5,
    ) -> PositionSize:
        """
        Calculate position size using ATR-based stop.

        stop = entry − (atr_multiplier × ATR)  [for longs]
        """
        stop_price = entry_price - atr_multiplier * atr
        return self.calculate(equity, entry_price, stop_price)
