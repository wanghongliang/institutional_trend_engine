"""
Real-time market data via Schwab streaming API (schwabdev).

Wraps the schwabdev.Stream client to produce a clean callback-driven
data feed that the engine can subscribe to.

Two modes:
  1. **Live** — connects to Schwab Streamer and emits bar/tick events.
  2. **Replay** — replays a historical DataFrame bar-by-bar (for backtest).
"""

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Level-1 equity field indices (Schwab spec) ──────────────────
# https://developer.schwab.com/products/trader-api--individual/details/specifications/Market%20Data%20Production
L1_FIELDS = {
    "0": "symbol",
    "1": "bid_price",
    "2": "ask_price",
    "3": "last_price",
    "4": "bid_size",
    "5": "ask_size",
    "6": "ask_id",
    "7": "bid_id",
    "8": "total_volume",
    # ... extend as needed
}


@dataclass
class Tick:
    """A single Level-1 quote/trade update."""
    timestamp: datetime
    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    total_volume: int


@dataclass
class Bar:
    """An OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataFeed:
    """
    Real-time data feed built on schwabdev.Stream.

    Aggregates Level-1 ticks into 1-minute bars and calls
    registered callbacks on each bar close.

    Usage
    -----
    >>> feed = MarketDataFeed(client, symbols=["SPY"])
    >>> feed.on_bar(my_strategy.update)
    >>> feed.start()          # blocks; run in a thread if needed
    """

    def __init__(self, client=None, symbols: List[str] | None = None):
        self.client = client
        self.symbols = symbols or []
        self._bar_callbacks: List[Callable[[Bar], None]] = []
        self._tick_callbacks: List[Callable[[Tick], None]] = []
        self._streamer = None
        self._running = False

        # current building bar per symbol
        self._current_bars: Dict[str, Bar] = {}
        self._last_bar_minute: Dict[str, int] = {}

        # cumulative volume tracking for delta
        self._prev_volume: Dict[str, int] = {}

    # ── callback registration ───────────────────────────────────

    def on_bar(self, callback: Callable[[Bar], None]) -> None:
        """Register a callback fired on each bar close."""
        self._bar_callbacks.append(callback)

    def on_tick(self, callback: Callable[[Tick], None]) -> None:
        """Register a callback fired on each tick."""
        self._tick_callbacks.append(callback)

    # ── tick → bar aggregation ──────────────────────────────────

    def _process_tick(self, tick: Tick) -> None:
        """Aggregate tick into current 1-min bar; emit on minute rollover."""
        for cb in self._tick_callbacks:
            cb(tick)

        sym = tick.symbol
        minute_key = tick.timestamp.replace(second=0, microsecond=0)
        minute_int = int(minute_key.timestamp())

        # minute rollover → close previous bar
        if sym in self._current_bars and self._last_bar_minute.get(sym) != minute_int:
            closed = self._current_bars[sym]
            for cb in self._bar_callbacks:
                cb(closed)
            self._current_bars[sym] = None

        # build / update current bar
        if sym not in self._current_bars or self._current_bars[sym] is None:
            self._current_bars[sym] = Bar(
                timestamp=minute_key,
                open=tick.last_price,
                high=tick.last_price,
                low=tick.last_price,
                close=tick.last_price,
                volume=0,
            )
            self._last_bar_minute[sym] = minute_int

        bar = self._current_bars[sym]
        bar.high = max(bar.high, tick.last_price)
        bar.low = min(bar.low, tick.last_price)
        bar.close = tick.last_price

        # volume delta
        prev_vol = self._prev_volume.get(sym, 0)
        delta_vol = max(0, tick.total_volume - prev_vol)
        bar.volume += delta_vol
        self._prev_volume[sym] = tick.total_volume

    def _flush_current_bars(self) -> None:
        """Emit any unfinished bars (call on stop)."""
        for sym, bar in self._current_bars.items():
            if bar is not None:
                for cb in self._bar_callbacks:
                    cb(bar)
        self._current_bars.clear()

    # ── streaming ───────────────────────────────────────────────

    def start(self) -> None:
        """Start streaming.  Blocks the calling thread."""
        if self.client is None:
            raise RuntimeError("No Schwab client — cannot stream live data.")

        try:
            import schwabdev
        except ImportError:
            raise RuntimeError("schwabdev not installed. Run: pip install schwabdev")

        self._streamer = schwabdev.Stream(self.client)
        self._running = True

        def handler(message: str) -> None:
            self._handle_stream_message(message)

        self._streamer.start(handler)

        # subscribe to Level-1 equities
        symbols_str = ",".join(self.symbols)
        self._streamer.send(
            self._streamer.level_one_equities(symbols_str, "0,1,2,3,4,5,8")
        )
        logger.info("Streaming started for %s", self.symbols)

    def stop(self) -> None:
        """Stop streaming and flush pending bars."""
        self._running = False
        if self._streamer:
            self._streamer.stop()
        self._flush_current_bars()
        logger.info("Streaming stopped.")

    def _handle_stream_message(self, message: str) -> None:
        """Parse a Schwab stream message and route to _process_tick."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Schwab stream messages have a "data" list of records
        records = data.get("data", [])
        service = data.get("service", "")

        for rec in records:
            content = rec.get("content", {})
            sym = content.get("key", rec.get("key", ""))
            if not sym:
                continue

            ts_ms = content.get("47", int(time.time() * 1000))
            tick = Tick(
                timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                symbol=sym,
                last_price=float(content.get("3", 0)),
                bid_price=float(content.get("1", 0)),
                ask_price=float(content.get("2", 0)),
                bid_size=int(content.get("4", 0)),
                ask_size=int(content.get("5", 0)),
                total_volume=int(content.get("8", 0)),
            )
            if tick.last_price > 0:
                self._process_tick(tick)

    # ── replay mode (for backtest) ──────────────────────────────

    def replay(
        self,
        df: pd.DataFrame,
        speed: float = 1.0,
    ) -> None:
        """
        Replay a historical DataFrame bar-by-bar, calling bar callbacks.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.
        speed : float
            Replay speed multiplier (1.0 = real time, 100 = fast).
        """
        logger.info("Replaying %d bars at %.1fx speed", len(df), speed)
        delay = 0.0 if speed <= 0 else 60.0 / speed  # 1-min bars

        for ts, row in df.iterrows():
            bar = Bar(
                timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
            for cb in self._bar_callbacks:
                cb(bar)
            if delay > 0:
                time.sleep(delay)

        logger.info("Replay complete.")
