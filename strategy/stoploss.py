"""
Stop-loss management.

Implements a multi-stage stop-loss system:
  1. Initial stop: entry_price − (initial_stop_atr × ATR)
  2. Break-even: move stop to entry price when price reaches 1R
  3. ATR trailing: trail stop at price − (trailing_stop_atr × ATR)

The stop is always the *tightest* of the three.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class StopType(str, Enum):
    INITIAL = "initial"
    BREAK_EVEN = "break_even"
    TRAILING = "trailing"
    NONE = "none"


@dataclass
class StopLevel:
    """Current stop-loss state."""
    stop_price: float
    stop_type: StopType
    r_multiple: float          # current R multiple of the trade
    entry_price: float
    highest_price: float       # for long; lowest for short
    direction: int             # +1 long, -1 short


class StopLossManager:
    """
    Multi-stage stop-loss manager.

    Parameters
    ----------
    initial_stop_atr : float
        ATR multiplier for initial stop (e.g., 1.5).
    trailing_stop_atr : float
        ATR multiplier for trailing stop (e.g., 2.0).
    break_even_r : float
        R multiple at which to move stop to breakeven (e.g., 1.0).
    """

    def __init__(
        self,
        initial_stop_atr: float = 1.5,
        trailing_stop_atr: float = 2.0,
        break_even_r: float = 1.0,
    ):
        self.initial_stop_atr = initial_stop_atr
        self.trailing_stop_atr = trailing_stop_atr
        self.break_even_r = break_even_r

    def initialize(
        self,
        entry_price: float,
        atr: float,
        direction: int,
    ) -> StopLevel:
        """Create the initial stop level for a new trade."""
        if direction > 0:  # long
            stop = entry_price - self.initial_stop_atr * atr
        else:  # short
            stop = entry_price + self.initial_stop_atr * atr

        return StopLevel(
            stop_price=stop,
            stop_type=StopType.INITIAL,
            r_multiple=0.0,
            entry_price=entry_price,
            highest_price=entry_price,
            direction=direction,
        )

    def update(
        self,
        level: StopLevel,
        current_price: float,
        current_atr: float,
    ) -> StopLevel:
        """
        Update the stop level based on the latest price and ATR.

        Returns the updated StopLevel (mutates the input in place
        and returns it for convenience).
        """
        if level.direction > 0:  # long
            level.highest_price = max(level.highest_price, current_price)
            risk = abs(level.entry_price - level.stop_price)
            if risk <= 0:
                return level
            level.r_multiple = (current_price - level.entry_price) / risk

            # Compute candidate stops
            # 1. Initial stop (keep as floor)
            initial_stop = level.entry_price - self.initial_stop_atr * current_atr

            # 2. Break-even stop
            be_stop = level.entry_price
            use_be = level.r_multiple >= self.break_even_r

            # 3. Trailing stop
            trailing_stop = level.highest_price - self.trailing_stop_atr * current_atr

            # Pick the tightest (highest) stop
            candidates = [level.stop_price]  # never move stop backward
            if use_be:
                candidates.append(be_stop)
            candidates.append(trailing_stop)

            new_stop = max(candidates)
            level.stop_price = new_stop

            if level.r_multiple >= self.break_even_r and new_stop >= be_stop:
                level.stop_type = StopType.BREAK_EVEN
            elif new_stop >= trailing_stop:
                level.stop_type = StopType.TRAILING
            else:
                level.stop_type = StopType.INITIAL

        else:  # short
            level.highest_price = min(level.highest_price, current_price)
            risk = abs(level.stop_price - level.entry_price)
            if risk <= 0:
                return level
            level.r_multiple = (level.entry_price - current_price) / risk

            initial_stop = level.entry_price + self.initial_stop_atr * current_atr
            be_stop = level.entry_price
            use_be = level.r_multiple >= self.break_even_r
            trailing_stop = level.highest_price + self.trailing_stop_atr * current_atr

            candidates = [level.stop_price]
            if use_be:
                candidates.append(be_stop)
            candidates.append(trailing_stop)

            new_stop = min(candidates)
            level.stop_price = new_stop

            if level.r_multiple >= self.break_even_r and new_stop <= be_stop:
                level.stop_type = StopType.BREAK_EVEN
            elif new_stop <= trailing_stop:
                level.stop_type = StopType.TRAILING
            else:
                level.stop_type = StopType.INITIAL

        return level

    def is_stopped(self, level: StopLevel, current_price: float) -> bool:
        """Check if the current price hits the stop."""
        if level.direction > 0:
            return current_price <= level.stop_price
        else:
            return current_price >= level.stop_price
