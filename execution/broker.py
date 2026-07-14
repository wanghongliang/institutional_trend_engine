"""
Schwab broker adapter.

Wraps the schwabdev.Client to provide a unified broker interface
that the execution engine can call.  This isolates all Schwab-specific
logic in one place.

Methods mirror a generic broker interface so that additional brokers
(IBKR, Alpaca, etc.) can be added later by implementing the same API.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class OrderRequest:
    """Normalized order request."""
    symbol: str
    quantity: int
    order_type: str       # "MARKET" | "LIMIT"
    instruction: str      # "BUY" | "SELL"
    price: Optional[float] = None
    asset_type: str = "EQUITY"
    duration: str = "DAY"
    session: str = "NORMAL"


@dataclass
class OrderResult:
    """Result of placing an order."""
    success: bool
    order_id: Optional[str]
    message: str
    raw_response: Optional[dict] = None


class SchwabBroker:
    """
    Schwab API broker adapter.

    Wraps schwabdev.Client for account access and order placement.

    Usage
    -----
    >>> broker = SchwabBroker(client)
    >>> broker.connect()
    >>> result = broker.place_order(OrderRequest(
    ...     symbol="SPY", quantity=10, order_type="MARKET", instruction="BUY"
    ... ))
    """

    def __init__(self, client=None):
        self.client = client
        self.account_hash: Optional[str] = None
        self.connected = False

    def connect(self) -> bool:
        """Authenticate and retrieve account info."""
        if self.client is None:
            logger.warning("No Schwab client provided — running in simulation mode")
            return False

        try:
            resp = self.client.linked_accounts()
            if resp.status_code == 200:
                accounts = resp.json()
                if accounts:
                    self.account_hash = accounts[0].get("hashValue")
                    self.connected = True
                    logger.info("Connected to Schwab. Account hash: %s", self.account_hash)
                    return True
            logger.error("Failed to connect: %s", resp.text[:200])
            return False
        except Exception as e:
            logger.error("Connection error: %s", e)
            return False

    def get_account_equity(self) -> float:
        """Get current account equity (liquidation value)."""
        if not self.connected or not self.account_hash:
            return 0.0
        try:
            resp = self.client.account_details(self.account_hash, fields="positions")
            if resp.status_code == 200:
                data = resp.json()
                # Navigate Schwab's nested JSON structure
                ag = data.get("securitiesAccount", {}).get("currentBalances", {})
                return float(ag.get("liquidationValue", 0))
        except Exception as e:
            logger.error("Error getting equity: %s", e)
        return 0.0

    def get_positions(self) -> list:
        """Get current open positions."""
        if not self.connected or not self.account_hash:
            return []
        try:
            resp = self.client.account_details(self.account_hash, fields="positions")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("securitiesAccount", {}).get("positions", [])
        except Exception as e:
            logger.error("Error getting positions: %s", e)
        return []

    def place_order(self, order: OrderRequest) -> OrderResult:
        """
        Place an order via Schwab API.

        Parameters
        ----------
        order : OrderRequest
            Normalized order request.

        Returns
        -------
        OrderResult
        """
        if not self.connected or not self.account_hash:
            logger.warning("Not connected — order not placed: %s %d %s",
                          order.instruction, order.quantity, order.symbol)
            return OrderResult(
                success=False, order_id=None,
                message="Not connected to Schwab",
            )

        # Build Schwab order payload
        schwab_order = {
            "orderType": order.order_type,
            "session": order.session,
            "duration": order.duration,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": order.instruction,
                    "quantity": order.quantity,
                    "instrument": {
                        "symbol": order.symbol,
                        "assetType": order.asset_type,
                    },
                }
            ],
        }

        if order.order_type == "LIMIT" and order.price is not None:
            schwab_order["price"] = str(order.price)

        try:
            resp = self.client.place_order(self.account_hash, schwab_order)
            if resp.status_code in (200, 201):
                order_id = resp.headers.get("location", "/").split("/")[-1]
                logger.info("Order placed: %s %d %s (id=%s)",
                           order.instruction, order.quantity, order.symbol, order_id)
                return OrderResult(
                    success=True, order_id=order_id,
                    message="Order placed successfully",
                )
            else:
                return OrderResult(
                    success=False, order_id=None,
                    message=f"API error: {resp.status_code} {resp.text[:200]}",
                )
        except Exception as e:
            logger.error("Order placement error: %s", e)
            return OrderResult(
                success=False, order_id=None,
                message=f"Exception: {e}",
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self.connected or not self.account_hash:
            return False
        try:
            resp = self.client.cancel_order(self.account_hash, order_id)
            return resp.status_code == 200
        except Exception as e:
            logger.error("Cancel error: %s", e)
            return False

    def get_quote(self, symbol: str) -> dict:
        """Get a real-time quote."""
        if not self.connected:
            return {}
        try:
            resp = self.client.quote(symbol)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error("Quote error: %s", e)
        return {}
