"""
Market Structure Engine — the core of the Institutional Trend Engine.

Implements ICT/SMC-style market structure recognition in a
fully quantitative, backtestable way.

Components
----------
1. SwingDetector         → raw swing highs/lows
2. StructureBuilder      → HH / HL / LH / LL classification
3. BOSDetector           → Break of Structure (trend continuation)
4. CHoCHDetector         → Change of Character (trend reversal)
5. LiquiditySweepDetector → stop-hunt / liquidity grab detection
6. FalseBreakDetector    → weak breakout that fails
7. TrendClassifier       → bullish / bearish / range

The engine processes bars incrementally and maintains a live
snapshot of the current market structure.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from indicators.atr import atr
from indicators.swing import SwingDetector, SwingPoint
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Enums ───────────────────────────────────────────────────────

class StructureType(str, Enum):
    """Swing structure classification."""
    HH = "higher_high"   # bullish continuation
    HL = "higher_low"    # bullish continuation
    LH = "lower_high"    # bearish continuation
    LL = "lower_low"     # bearish continuation
    EQ = "equal"         # no change


class TrendDirection(str, Enum):
    """Overall trend direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGE = "range"
    UNKNOWN = "unknown"


class StructureEvent(str, Enum):
    """Market structure events."""
    BOS_BULL = "bos_bull"           # bullish break of structure
    BOS_BEAR = "bos_bear"           # bearish break of structure
    CHOCH_BULL = "choch_bull"       # bullish change of character
    CHOCH_BEAR = "choch_bear"       # bearish change of character
    LIQUIDITY_SWEEP_HIGH = "liq_sweep_high"
    LIQUIDITY_SWEEP_LOW = "liq_sweep_low"
    FALSE_BREAK_HIGH = "false_break_high"
    FALSE_BREAK_LOW = "false_break_low"
    NONE = "none"


# ── Data containers ─────────────────────────────────────────────

@dataclass
class StructurePoint:
    """A classified swing point with structure info."""
    swing: SwingPoint
    structure_type: StructureType
    trend: TrendDirection


@dataclass
class MarketStructureSnapshot:
    """Live snapshot of market structure at a given bar."""
    timestamp: pd.Timestamp
    trend: TrendDirection
    last_swing_high: Optional[SwingPoint]
    last_swing_low: Optional[SwingPoint]
    last_structure_type: StructureType
    last_event: StructureEvent
    distance_to_swing_high_atr: float   # how far price is from last swing high (in ATR)
    distance_to_swing_low_atr: float    # how far price is from last swing low (in ATR)
    structure_points: List[StructurePoint] = field(default_factory=list)


# ── Engine ──────────────────────────────────────────────────────

class MarketStructureEngine:
    """
    The Market Structure Engine.

    Processes OHLCV data and produces a stream of StructureEvents
    and maintains a live MarketStructureSnapshot.

    Parameters
    ----------
    swing_window : int
        Bars on each side for swing confirmation.
    min_atr : float
        Minimum swing excursion in ATR units.
    false_break_lookback : int
        Bars to wait before declaring a breakout "false".
    liquidity_sweep_threshold : float
        Fraction of ATR that price must exceed the swing and
        reverse to count as a sweep.
    """

    def __init__(
        self,
        swing_window: int = 3,
        min_atr: float = 0.5,
        false_break_lookback: int = 5,
        liquidity_sweep_threshold: float = 0.2,
    ):
        self.swing_detector = SwingDetector(swing_window, min_atr)
        self.false_break_lookback = false_break_lookback
        self.liquidity_sweep_threshold = liquidity_sweep_threshold

        self.structure_points: List[StructurePoint] = []
        self.events: List[tuple] = []  # (timestamp, event)
        self.snapshots: List[MarketStructureSnapshot] = []

        self._trend = TrendDirection.UNKNOWN
        self._prev_high: Optional[SwingPoint] = None
        self._prev_low: Optional[SwingPoint] = None

        # Track which swing levels have already been broken (prevent re-firing)
        self._bos_fired_high_price: Optional[float] = None
        self._bos_fired_low_price: Optional[float] = None

    # ── public API ──────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> List[MarketStructureSnapshot]:
        """
        Full analysis of a DataFrame.

        Returns a list of snapshots, one per bar.
        """
        atr_series = atr(df["high"], df["low"], df["close"], period=14)
        swings = self.swing_detector.detect(df, atr_series)

        # Build structure points
        self._build_structure_points(swings)

        # Process bar-by-bar for events and snapshots
        snapshots = []
        swing_idx = 0
        events_per_bar = {}

        for i, (ts, row) in enumerate(df.iterrows()):
            current_atr = atr_series.iloc[i] if i < len(atr_series) else np.nan

            # Check for structure events at this bar
            event = StructureEvent.NONE

            # BOS / CHoCH detection
            event = self._check_bos_choch(ts, row, current_atr)

            # Liquidity sweep
            if event == StructureEvent.NONE:
                event = self._check_liquidity_sweep(ts, row, current_atr)

            # False break
            if event == StructureEvent.NONE:
                event = self._check_false_break(ts, row, current_atr)

            if event != StructureEvent.NONE:
                self.events.append((ts, event))

            # Build snapshot
            last_sh = self.swing_detector.last_swing_high()
            last_sl = self.swing_detector.last_swing_low()

            dist_high = 0.0
            dist_low = 0.0
            if last_sh and current_atr and current_atr > 0:
                dist_high = (last_sh.price - row["close"]) / current_atr
            if last_sl and current_atr and current_atr > 0:
                dist_low = (row["close"] - last_sl.price) / current_atr

            last_struct = (
                self.structure_points[-1].structure_type
                if self.structure_points
                else StructureType.EQ
            )

            snap = MarketStructureSnapshot(
                timestamp=ts,
                trend=self._trend,
                last_swing_high=last_sh,
                last_swing_low=last_sl,
                last_structure_type=last_struct,
                last_event=event,
                distance_to_swing_high_atr=dist_high,
                distance_to_swing_low_atr=dist_low,
                structure_points=list(self.structure_points[-10:]),
            )
            snapshots.append(snap)

        self.snapshots = snapshots
        return snapshots

    def current_snapshot(self) -> Optional[MarketStructureSnapshot]:
        """Return the most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    # ── structure building ──────────────────────────────────────

    def _build_structure_points(self, swings: List[SwingPoint]) -> None:
        """Classify each swing as HH/HL/LH/LL and update trend."""
        self.structure_points = []
        prev_high: Optional[SwingPoint] = None
        prev_low: Optional[SwingPoint] = None

        for sw in swings:
            if sw.type == "high":
                if prev_high is None:
                    st = StructureType.EQ
                elif sw.price > prev_high.price:
                    st = StructureType.HH
                    self._trend = TrendDirection.BULLISH
                elif sw.price < prev_high.price:
                    st = StructureType.LH
                else:
                    st = StructureType.EQ
                prev_high = sw
            else:  # low
                if prev_low is None:
                    st = StructureType.EQ
                elif sw.price > prev_low.price:
                    st = StructureType.HL
                    self._trend = TrendDirection.BULLISH
                elif sw.price < prev_low.price:
                    st = StructureType.LL
                    self._trend = TrendDirection.BEARISH
                else:
                    st = StructureType.EQ
                prev_low = sw

            self.structure_points.append(StructurePoint(
                swing=sw,
                structure_type=st,
                trend=self._trend,
            ))

        self._prev_high = prev_high
        self._prev_low = prev_low

    # ── BOS / CHoCH ─────────────────────────────────────────────

    def _check_bos_choch(
        self,
        ts: pd.Timestamp,
        row: pd.Series,
        current_atr: float,
    ) -> StructureEvent:
        """
        Break of Structure (BOS): price breaks the last swing in
        the direction of the prevailing trend (continuation).

        Change of Character (CHoCH): price breaks the last swing
        *against* the prevailing trend (reversal).

        Each swing level can only fire one BOS/CHoCH event — once
        broken, it won't fire again until a new swing replaces it.
        """
        last_high = self.swing_detector.last_swing_high()
        last_low = self.swing_detector.last_swing_low()

        # Bullish BOS/CHoCH: close breaks above last swing high
        if last_high and row["close"] > last_high.price:
            # Skip if this swing level already fired
            if self._bos_fired_high_price != last_high.price:
                self._bos_fired_high_price = last_high.price
                if self._trend == TrendDirection.BULLISH:
                    return StructureEvent.BOS_BULL
                elif self._trend == TrendDirection.BEARISH:
                    self._trend = TrendDirection.BULLISH
                    return StructureEvent.CHOCH_BULL

        # Bearish BOS/CHoCH: close breaks below last swing low
        if last_low and row["close"] < last_low.price:
            if self._bos_fired_low_price != last_low.price:
                self._bos_fired_low_price = last_low.price
                if self._trend == TrendDirection.BEARISH:
                    return StructureEvent.BOS_BEAR
                elif self._trend == TrendDirection.BULLISH:
                    self._trend = TrendDirection.BEARISH
                    return StructureEvent.CHOCH_BEAR

        return StructureEvent.NONE

    # ── Liquidity Sweep ─────────────────────────────────────────

    def _check_liquidity_sweep(
        self,
        ts: pd.Timestamp,
        row: pd.Series,
        current_atr: float,
    ) -> StructureEvent:
        """
        Liquidity Sweep: price briefly exceeds a swing high/low
        (grabbing stop orders) then immediately reverses.

        Detection: bar high exceeds swing high but close is back
        below it (bearish sweep) — or mirror for lows (bullish sweep).
        """
        if not current_atr or current_atr <= 0:
            return StructureEvent.NONE

        threshold = self.liquidity_sweep_threshold * current_atr

        last_high = self.swing_detector.last_swing_high()
        if last_high:
            # Price poked above swing high but closed back below
            if (row["high"] > last_high.price and
                    row["close"] < last_high.price - threshold):
                return StructureEvent.LIQUIDITY_SWEEP_HIGH

        last_low = self.swing_detector.last_swing_low()
        if last_low:
            if (row["low"] < last_low.price and
                    row["close"] > last_low.price + threshold):
                return StructureEvent.LIQUIDITY_SWEEP_LOW

        return StructureEvent.NONE

    # ── False Break ─────────────────────────────────────────────

    def _check_false_break(
        self,
        ts: pd.Timestamp,
        row: pd.Series,
        current_atr: float,
    ) -> StructureEvent:
        """
        False Break: price breaks a swing level but with weak
        momentum (low volume, no follow-through) and reverses.

        Simplified check: broke swing high but close is back below
        and bar is bearish (close < open).
        """
        last_high = self.swing_detector.last_swing_high()
        if last_high:
            # Check recent bars for a breakout that failed
            if (row["high"] > last_high.price and
                    row["close"] < row["open"] and
                    row["close"] < last_high.price):
                return StructureEvent.FALSE_BREAK_HIGH

        last_low = self.swing_detector.last_swing_low()
        if last_low:
            if (row["low"] < last_low.price and
                    row["close"] > row["open"] and
                    row["close"] > last_low.price):
                return StructureEvent.FALSE_BREAK_LOW

        return StructureEvent.NONE

    # ── helpers ─────────────────────────────────────────────────

    def near_swing_high(self, price: float, atr_val: float, ratio: float = 0.2) -> bool:
        """Is *price* within *ratio* ATR below the last swing high (approaching from below)?"""
        sh = self.swing_detector.last_swing_high()
        if not sh or not atr_val or atr_val <= 0:
            return False
        # Only count when price is at or below the swing high (hasn't broken yet)
        distance = sh.price - price
        return 0 <= distance <= ratio * atr_val

    def near_swing_low(self, price: float, atr_val: float, ratio: float = 0.2) -> bool:
        """Is *price* within *ratio* ATR above the last swing low (approaching from above)?"""
        sl = self.swing_detector.last_swing_low()
        if not sl or not atr_val or atr_val <= 0:
            return False
        # Only count when price is at or above the swing low (hasn't broken yet)
        distance = price - sl.price
        return 0 <= distance <= ratio * atr_val

    def events_dataframe(self) -> pd.DataFrame:
        """Export detected events to a DataFrame."""
        if not self.events:
            return pd.DataFrame(columns=["timestamp", "event"])
        return pd.DataFrame(self.events, columns=["timestamp", "event"])
