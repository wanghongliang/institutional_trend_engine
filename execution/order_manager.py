"""
Order manager.

Tracks the lifecycle of orders and open positions:
  PENDING → SUBMITTED → PARTIAL → FILLED / CANCELLED / REJECTED

In live mode, this queries the Schwab API for order status.
In backtest mode, orders are filled immediately (with optional slippage).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from execution.broker import OrderRequest, OrderResult, SchwabBroker
from utils.logger import setup_logger

logger = setup_logger(__name__)


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class ManagedOrder:
    """A tracked order with full lifecycle state."""
    order_id: str
    symbol: str
    quantity: int
    filled_quantity: int
    instruction: str        # BUY / SELL
    order_type: str         # MARKET / LIMIT
    price: Optional[float]
    avg_fill_price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    commission: float = 0.0


@dataclass
class Position:
    """An open position."""
    symbol: str
    quantity: int
    avg_price: float
    direction: int          # +1 long, -1 short
    entry_time: datetime
    stop_price: float
    target_price: float


class OrderManager:
    """
    Order and position lifecycle manager.

    Parameters
    ----------
    broker : SchwabBroker | None
        Live broker adapter. When None, runs in simulation mode.
    commission_per_trade : float
        Commission per trade (0.0 for Schwab equities).
    slippage_ticks : float
        Simulated slippage in price units.
    """

    def __init__(
        self,
        broker: SchwabBroker | None = None,
        commission_per_trade: float = 0.0,
        slippage_ticks: float = 0.01,
    ):
        self.broker = broker
        self.commission_per_trade = commission_per_trade
        self.slippage_ticks = slippage_ticks
        self.orders: List[ManagedOrder] = []
        self.position: Optional[Position] = None
        self._order_counter = 0

    def submit(
        self,
        request: OrderRequest,
        sim_price: Optional[float] = None,
    ) -> ManagedOrder:
        """
        Submit an order.

        In live mode, calls broker.place_order().
        In sim mode, fills immediately at sim_price ± slippage.

        Parameters
        ----------
        request : OrderRequest
            The order to submit.
        sim_price : float | None
            Fill price for simulation mode.

        Returns
        -------
        ManagedOrder
        """
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"

        managed = ManagedOrder(
            order_id=order_id,
            symbol=request.symbol,
            quantity=request.quantity,
            filled_quantity=0,
            instruction=request.instruction,
            order_type=request.order_type,
            price=request.price,
            avg_fill_price=None,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
            commission=self.commission_per_trade,
        )

        if self.broker and self.broker.connected:
            # Live mode
            result = self.broker.place_order(request)
            if result.success:
                managed.status = OrderStatus.SUBMITTED
                managed.order_id = result.order_id or order_id
            else:
                managed.status = OrderStatus.REJECTED
                logger.error("Order rejected: %s", result.message)
        else:
            # Simulation mode — fill immediately
            fill_price = sim_price or request.price or 0.0
            if request.instruction == "BUY":
                fill_price += self.slippage_ticks
            else:
                fill_price -= self.slippage_ticks

            managed.filled_quantity = request.quantity
            managed.avg_fill_price = fill_price
            managed.status = OrderStatus.FILLED
            self._update_position(request, fill_price)

        self.orders.append(managed)
        return managed

    def _update_position(self, request: OrderRequest, fill_price: float) -> None:
        """Update the internal position tracker after a fill."""
        direction = 1 if request.instruction == "BUY" else -1

        if self.position is None:
            self.position = Position(
                symbol=request.symbol,
                quantity=request.quantity,
                avg_price=fill_price,
                direction=direction,
                entry_time=datetime.now(),
                stop_price=0,
                target_price=0,
            )
        else:
            # Adding or reducing
            if direction == self.position.direction:
                # Adding to position — update average
                total_qty = self.position.quantity + request.quantity
                self.position.avg_price = (
                    (self.position.avg_price * self.position.quantity +
                     fill_price * request.quantity) / total_qty
                )
                self.position.quantity = total_qty
            else:
                # Reducing / closing
                self.position.quantity -= request.quantity
                if self.position.quantity <= 0:
                    self.position = None

    def close_position(self, sim_price: Optional[float] = None) -> Optional[ManagedOrder]:
        """Close the entire current position."""
        if self.position is None:
            return None

        instruction = "SELL" if self.position.direction > 0 else "BUY"
        request = OrderRequest(
            symbol=self.position.symbol,
            quantity=self.position.quantity,
            order_type="MARKET",
            instruction=instruction,
            asset_type="EQUITY",
        )
        return self.submit(request, sim_price=sim_price)

    def update_stop_target(self, stop: float, target: float) -> None:
        """Update stop and target on the current position."""
        if self.position:
            self.position.stop_price = stop
            self.position.target_price = target

    def is_flat(self) -> bool:
        """True when no position is open."""
        return self.position is None

    def position_pnl(self, current_price: float) -> float:
        """Unrealized P&L of current position."""
        if self.position is None:
            return 0.0
        if self.position.direction > 0:
            return (current_price - self.position.avg_price) * self.position.quantity
        else:
            return (self.position.avg_price - current_price) * self.position.quantity

    def trade_log(self) -> List[dict]:
        """Return a list of all filled orders as dicts."""
        return [
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "instruction": o.instruction,
                "quantity": o.quantity,
                "fill_price": o.avg_fill_price,
                "status": o.status.value,
                "timestamp": o.timestamp,
                "commission": o.commission,
            }
            for o in self.orders
            if o.status == OrderStatus.FILLED
        ]
