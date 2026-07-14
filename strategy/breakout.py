"""
Breakout & Early Entry strategy.

This implements the document's core trading logic:

1. **Early Entry**: When price is within 0.2 ATR of a swing high
   AND momentum factors are synchronously increasing (velocity up,
   acceleration up, OFI up, VWAP slope up) → enter before the breakout.

2. **Breakout Entry**: When price actually breaks above the swing high
   with strong volume → add to position.

3. **Exit**: When CHoCH or BOS against position occurs, or trailing
   stop is hit.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import SignalType, TradeState
from indicators.market_structure import (
    MarketStructureEngine,
    StructureEvent,
    TrendDirection,
)
from indicators.trend_score import TrendScoreEngine
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Signal:
    """A trading signal with full context."""
    timestamp: pd.Timestamp
    signal_type: SignalType
    price: float
    atr: float
    score: float             # turning score at signal time
    regime: str              # market regime
    reason: str              # human-readable explanation


class BreakoutStrategy:
    """
    Early-entry + breakout strategy.

    The strategy combines:
      - MarketStructureEngine for swing/BOS/CHoCH
      - TrendScoreEngine for turning-point detection
      - StateMachine for regime filtering

    Only generates signals when the regime supports trend trading.
    """

    def __init__(
        self,
        structure_engine: MarketStructureEngine,
        score_engine: TrendScoreEngine,
        early_entry_atr_ratio: float = 0.2,
        min_score_for_early_entry: float = 10.0,
        allowed_regimes: set | None = None,
    ):
        self.structure = structure_engine
        self.scorer = score_engine
        self.early_entry_atr_ratio = early_entry_atr_ratio
        self.min_score_for_early_entry = min_score_for_early_entry
        self.allowed_regimes = allowed_regimes or {
            "trend_day", "open_drive", "trend_exhaustion", "unknown",
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        regime_df: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        """
        Generate trading signals for the entire DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data.
        regime_df : pd.DataFrame | None
            Market regime per bar (from StateMachine.to_dataframe).
            If None, all bars are treated as tradeable.

        Returns
        -------
        list[Signal]
            Trading signals.
        """
        # Run structure analysis
        snapshots = self.structure.analyze(df)

        # Run score engine
        scored = self.scorer.compute(df)

        # Pre-compute ATR once (not inside the loop)
        from indicators.atr import atr as calc_atr
        atr_series = calc_atr(df["high"], df["low"], df["close"], period=14)

        signals = []
        position_state = TradeState.FLAT

        for i in range(len(df)):
            ts = df.index[i]
            row = df.iloc[i]
            close = row["close"]
            atr_val = atr_series.iloc[i] if i < len(atr_series) else 0
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            snap = snapshots[i]
            score_row = scored.iloc[i]
            bull_score = score_row["bull_score"]
            bear_score = score_row["bear_score"]
            is_turning = score_row["is_turning"]
            direction = score_row["turning_direction"]

            # Regime check
            regime_str = "unknown"
            if regime_df is not None and ts in regime_df.index:
                regime_str = regime_df.loc[ts, "regime"]
            regime_tradeable = regime_str in self.allowed_regimes

            event = snap.last_event
            trend = snap.trend

            # Helper: check if price is near a swing level using the
            # snapshot's swing (which is correct for bar i, not global)
            def _near_high():
                sh = snap.last_swing_high
                if not sh or atr_val <= 0:
                    return False
                dist = sh.price - close
                return 0 <= dist <= self.early_entry_atr_ratio * atr_val

            def _near_low():
                sl = snap.last_swing_low
                if not sl or atr_val <= 0:
                    return False
                dist = close - sl.price
                return 0 <= dist <= self.early_entry_atr_ratio * atr_val

            # ── Exit logic ───────────────────────────────────────
            if position_state == TradeState.IN_POSITION or position_state == TradeState.EARLY_ENTRY:
                # CHoCH against position → exit
                if event == StructureEvent.CHOCH_BEAR:
                    signals.append(Signal(
                        timestamp=ts, signal_type=SignalType.EXIT_LONG,
                        price=close, atr=atr_val, score=max(bull_score, bear_score),
                        regime=regime_str,
                        reason="CHoCH bearish — trend reversal detected",
                    ))
                    position_state = TradeState.FLAT
                    continue

                if event == StructureEvent.CHOCH_BULL:
                    signals.append(Signal(
                        timestamp=ts, signal_type=SignalType.EXIT_SHORT,
                        price=close, atr=atr_val, score=max(bull_score, bear_score),
                        regime=regime_str,
                        reason="CHoCH bullish — trend reversal detected",
                    ))
                    position_state = TradeState.FLAT
                    continue

                # Liquidity sweep against position → exit
                if event == StructureEvent.LIQUIDITY_SWEEP_HIGH:
                    signals.append(Signal(
                        timestamp=ts, signal_type=SignalType.EXIT_LONG,
                        price=close, atr=atr_val, score=max(bull_score, bear_score),
                        regime=regime_str,
                        reason="Liquidity sweep high — stop hunt detected",
                    ))
                    position_state = TradeState.FLAT
                    continue

                if event == StructureEvent.LIQUIDITY_SWEEP_LOW:
                    signals.append(Signal(
                        timestamp=ts, signal_type=SignalType.EXIT_SHORT,
                        price=close, atr=atr_val, score=max(bull_score, bear_score),
                        regime=regime_str,
                        reason="Liquidity sweep low — stop hunt detected",
                    ))
                    position_state = TradeState.FLAT
                    continue

            # ── Entry logic ──────────────────────────────────────
            if position_state == TradeState.FLAT and regime_tradeable:
                # Check for bullish early entry
                if (_near_high()
                        and bull_score >= self.min_score_for_early_entry
                        and direction > 0):
                    signals.append(Signal(
                        timestamp=ts, signal_type=SignalType.EARLY_BUY,
                        price=close, atr=atr_val, score=bull_score,
                        regime=regime_str,
                        reason=f"Early entry: {self.early_entry_atr_ratio} ATR from swing high, "
                               f"bull score={bull_score:.0f}",
                    ))
                    position_state = TradeState.EARLY_ENTRY
                    continue

                # Check for bearish early entry
                if (_near_low()
                        and bear_score >= self.min_score_for_early_entry
                        and direction < 0):
                    signals.append(Signal(
                        timestamp=ts, signal_type=SignalType.EARLY_SELL,
                        price=close, atr=atr_val, score=bear_score,
                        regime=regime_str,
                        reason=f"Early entry: {self.early_entry_atr_ratio} ATR from swing low, "
                               f"bear score={bear_score:.0f}",
                    ))
                    position_state = TradeState.EARLY_ENTRY
                    continue

                # Breakout entry: BOS in trend direction
                if event == StructureEvent.BOS_BULL and trend == TrendDirection.BULLISH:
                    signals.append(Signal(
                        timestamp=ts, signal_type=SignalType.BREAKOUT_BUY,
                        price=close, atr=atr_val, score=bull_score,
                        regime=regime_str,
                        reason="Bullish BOS — breakout above swing high",
                    ))
                    position_state = TradeState.IN_POSITION
                    continue

                if event == StructureEvent.BOS_BEAR and trend == TrendDirection.BEARISH:
                    signals.append(Signal(
                        timestamp=ts, signal_type=SignalType.BREAKOUT_SELL,
                        price=close, atr=atr_val, score=bear_score,
                        regime=regime_str,
                        reason="Bearish BOS — breakdown below swing low",
                    ))
                    position_state = TradeState.IN_POSITION
                    continue

            # ── Add position on confirmed breakout after early entry ──
            if position_state == TradeState.EARLY_ENTRY:
                if event == StructureEvent.BOS_BULL:
                    signals.append(Signal(
                        timestamp=ts, signal_type=SignalType.BREAKOUT_BUY,
                        price=close, atr=atr_val, score=bull_score,
                        regime=regime_str,
                        reason="Add position: breakout confirmed after early entry",
                    ))
                    position_state = TradeState.ADD_POSITION
                    continue

                if event == StructureEvent.BOS_BEAR:
                    signals.append(Signal(
                        timestamp=ts, signal_type=SignalType.BREAKOUT_SELL,
                        price=close, atr=atr_val, score=bear_score,
                        regime=regime_str,
                        reason="Add position: breakdown confirmed after early entry",
                    ))
                    position_state = TradeState.ADD_POSITION
                    continue

        logger.info("Generated %d signals", len(signals))
        return signals

    def signals_to_dataframe(self, signals: list[Signal]) -> pd.DataFrame:
        """Convert signals to a DataFrame."""
        if not signals:
            return pd.DataFrame(columns=[
                "timestamp", "signal_type", "price", "atr", "score", "regime", "reason"
            ])
        return pd.DataFrame([
            {
                "timestamp": s.timestamp,
                "signal_type": s.signal_type.value,
                "price": s.price,
                "atr": s.atr,
                "score": s.score,
                "regime": s.regime,
                "reason": s.reason,
            }
            for s in signals
        ]).set_index("timestamp")
