"""
Money management / risk management.

Implements circuit breakers and daily risk limits:
  - Max daily loss (percentage of equity)
  - Max consecutive losses
  - Daily trade count limit

When any limit is hit, the system stops trading for the day.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DailyRiskState:
    """Risk state for a single trading day."""
    date: date
    starting_equity: float
    current_equity: float
    trades_today: int = 0
    losses_today: int = 0
    wins_today: int = 0
    consecutive_losses: int = 0
    pnl_today: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""


class MoneyManager:
    """
    Daily risk manager with circuit breakers.

    Parameters
    ----------
    max_daily_loss_pct : float
        Max daily loss as fraction of starting equity (e.g., 0.03 = 3%).
    max_consecutive_losses : int
        Halt after this many consecutive losing trades.
    max_trades_per_day : int
        Max trades allowed per day.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        max_consecutive_losses: int = 3,
        max_trades_per_day: int = 10,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_trades_per_day = max_trades_per_day
        self._daily_states: Dict[date, DailyRiskState] = {}

    def get_daily_state(self, d: date, equity: float) -> DailyRiskState:
        """Get or create the risk state for date *d*."""
        if d not in self._daily_states:
            self._daily_states[d] = DailyRiskState(
                date=d,
                starting_equity=equity,
                current_equity=equity,
            )
        return self._daily_states[d]

    def can_trade(self, d: date, equity: float) -> tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns
        -------
        (bool, str)
            (allowed, reason_if_not)
        """
        state = self.get_daily_state(d, equity)

        if state.is_halted:
            return False, state.halt_reason

        if state.trades_today >= self.max_trades_per_day:
            state.is_halted = True
            state.halt_reason = f"Max trades per day ({self.max_trades_per_day}) reached"
            return False, state.halt_reason

        daily_loss = (state.starting_equity - equity) / state.starting_equity
        if daily_loss >= self.max_daily_loss_pct:
            state.is_halted = True
            state.halt_reason = (
                f"Max daily loss ({self.max_daily_loss_pct:.1%}) reached: "
                f"loss={daily_loss:.2%}"
            )
            logger.warning("Circuit breaker: %s", state.halt_reason)
            return False, state.halt_reason

        if state.consecutive_losses >= self.max_consecutive_losses:
            state.is_halted = True
            state.halt_reason = (
                f"Max consecutive losses ({self.max_consecutive_losses}) reached"
            )
            logger.warning("Circuit breaker: %s", state.halt_reason)
            return False, state.halt_reason

        return True, ""

    def record_trade(
        self,
        d: date,
        equity: float,
        pnl: float,
        is_win: bool,
    ) -> None:
        """Record a completed trade."""
        state = self.get_daily_state(d, equity)
        state.trades_today += 1
        state.current_equity = equity
        state.pnl_today += pnl

        if is_win:
            state.wins_today += 1
            state.consecutive_losses = 0
        else:
            state.losses_today += 1
            state.consecutive_losses += 1

    def daily_summary(self, d: date) -> dict:
        """Get a summary of the day's risk state."""
        state = self._daily_states.get(d)
        if not state:
            return {}
        return {
            "date": str(state.date),
            "starting_equity": state.starting_equity,
            "ending_equity": state.current_equity,
            "trades": state.trades_today,
            "wins": state.wins_today,
            "losses": state.losses_today,
            "pnl": state.pnl_today,
            "halted": state.is_halted,
            "halt_reason": state.halt_reason,
        }
