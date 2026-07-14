"""
Take-profit management.

Implements:
  1. Fixed R-multiple target (default 2R)
  2. Dynamic scaling: take partial profits at 1R, 2R
  3. Structure-based exit: exit at next swing high/low
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TakeProfitType(str, Enum):
    FIXED_R = "fixed_r"
    SCALED = "scaled"
    STRUCTURE = "structure"
    NONE = "none"


@dataclass
class TakeProfitLevel:
    """Take-profit target state."""
    target_price: float
    tp_type: TakeProfitType
    r_multiple_target: float
    partial_filled: bool       # has partial profit been taken?
    entry_price: float
    direction: int


class TakeProfitManager:
    """
    Take-profit manager with scaling support.

    Parameters
    ----------
    take_profit_r : float
        Target R multiple (default 2.0).
    partial_r : float
        R multiple for partial exit (default 1.0).
    partial_pct : float
        Fraction of position to close at partial target (default 0.5).
    """

    def __init__(
        self,
        take_profit_r: float = 2.0,
        partial_r: float = 1.0,
        partial_pct: float = 0.5,
    ):
        self.take_profit_r = take_profit_r
        self.partial_r = partial_r
        self.partial_pct = partial_pct

    def initialize(
        self,
        entry_price: float,
        stop_price: float,
        direction: int,
    ) -> TakeProfitLevel:
        """Create the initial take-profit level."""
        risk = abs(entry_price - stop_price)
        if direction > 0:
            target = entry_price + self.take_profit_r * risk
        else:
            target = entry_price - self.take_profit_r * risk

        return TakeProfitLevel(
            target_price=target,
            tp_type=TakeProfitType.SCALED,
            r_multiple_target=self.take_profit_r,
            partial_filled=False,
            entry_price=entry_price,
            direction=direction,
        )

    def should_take_partial(
        self,
        level: TakeProfitLevel,
        current_price: float,
    ) -> bool:
        """Check if partial profit should be taken."""
        if level.partial_filled:
            return False
        risk = abs(level.entry_price - level.target_price) / max(self.take_profit_r, 0.01)
        if risk <= 0:
            return False
        if level.direction > 0:
            partial_target = level.entry_price + self.partial_r * risk
            return current_price >= partial_target
        else:
            partial_target = level.entry_price - self.partial_r * risk
            return current_price <= partial_target

    def should_take_full(
        self,
        level: TakeProfitLevel,
        current_price: float,
    ) -> bool:
        """Check if full profit should be taken."""
        if level.direction > 0:
            return current_price >= level.target_price
        else:
            return current_price <= level.target_price

    def mark_partial_filled(self, level: TakeProfitLevel) -> None:
        """Mark that partial profit has been taken."""
        level.partial_filled = True

    def update_structure_target(
        self,
        level: TakeProfitLevel,
        swing_price: float,
    ) -> None:
        """Update target to a structure-based level (e.g., next swing high)."""
        if level.direction > 0 and swing_price > level.entry_price:
            level.target_price = min(swing_price, level.target_price) if level.target_price > level.entry_price else swing_price
        elif level.direction < 0 and swing_price < level.entry_price:
            level.target_price = max(swing_price, level.target_price) if level.target_price < level.entry_price else swing_price
