"""
Swing Low Elevated Strategy — 底部抬高买入策略.

A strict trend-following reversal strategy:

1. Detect swing lows using fractal-based SwingDetector.
2. For each confirmed swing low, check whether its price is HIGHER
   than the previous TWO swing lows (底部抬高 / bottom-rising pattern).
3. Only if the bottom is rising, generate a BUY signal at the close
   of the confirmation bar.
4. Non-elevated swing lows are tracked (for visualization / stats)
   but do NOT trigger entry.
5. Stop loss  = entry_price - 1 * ATR.
6. Take profit = entry_price + 2 * ATR.
7. Risk : Reward = 1 : 2.

Only one position at a time. No pyramiding.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from indicators.atr import atr as calc_atr
from indicators.swing import SwingDetector, SwingPoint
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ElevatedSwingSignal:
    """A trading signal from the elevated swing low strategy."""
    timestamp: pd.Timestamp
    signal_type: str            # "BUY" or "REJECT"
    price: float
    atr: float
    stop_loss: float
    take_profit: float
    swing_low_price: float
    swing_low_bar: int
    prev_swing_lows: list       # prices of the previous two swing lows
    is_elevated: bool           # True = bottom rising, False = rejected
    reason: str


class SwingLowElevatedStrategy:
    """
    Swing Low Elevated Strategy (底部抬高).

    Buys at each confirmed swing low ONLY when the swing low price is
    higher than the previous two swing lows, indicating a rising bottom
    pattern (底部抬高).

    Parameters
    ----------
    swing_window : int
        Bars on each side to confirm a swing (default 3).
    atr_period : int
        ATR calculation period (default 14).
    stop_atr_multiplier : float
        Stop loss = entry - stop_atr_multiplier * ATR (default 1.0).
    tp_atr_multiplier : float
        Take profit = entry + tp_atr_multiplier * ATR (default 2.0).
    require_n_prev_lows : int
        Number of previous swing lows the current must exceed (default 2).
    """

    def __init__(
        self,
        swing_window: int = 3,
        atr_period: int = 14,
        stop_atr_multiplier: float = 1.0,
        tp_atr_multiplier: float = 2.0,
        require_n_prev_lows: int = 2,
        swing_min_atr: float = 0.0,
    ):
        self.swing_window = swing_window
        self.atr_period = atr_period
        self.stop_atr_multiplier = stop_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.require_n_prev_lows = require_n_prev_lows
        self.swing_min_atr = swing_min_atr

        self.swing_detector = SwingDetector(
            window=swing_window,
            min_atr=swing_min_atr,
        )
        self.swings: List[SwingPoint] = []
        self.all_signals: List[ElevatedSwingSignal] = []   # includes rejected
        self.buy_signals: List[ElevatedSwingSignal] = []   # only BUY

    # ── Public API ──────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> List[ElevatedSwingSignal]:
        """
        Generate signals by scanning every confirmed swing low.

        For each swing low at bar *i* (confirmed at bar *i + window*):
          - Collect the previous N swing lows (N = require_n_prev_lows).
          - If fewer than N prior lows exist → REJECT (not enough history).
          - If current.price > ALL previous N lows → BUY (底部抬高).
          - Otherwise → REJECT (bottom not rising).

        Returns only BUY signals.  See `all_signals` for rejected ones.
        """
        # Compute ATR
        atr_series = calc_atr(
            df["high"], df["low"], df["close"], period=self.atr_period
        )

        # Detect all swing points
        self.swings = self.swing_detector.detect(df, atr_series)
        self.swing_detector.swings = self.swings

        self.all_signals = []
        self.buy_signals = []

        w = self.swing_window

        # Collect swing lows in chronological order
        swing_lows = [sw for sw in self.swings if sw.type == "low"]

        for idx, sw in enumerate(swing_lows):
            # Confirmation bar = swing bar + window
            confirm_bar = sw.bar_index + w
            if confirm_bar >= len(df):
                continue

            ts = df.index[confirm_bar]
            close = df["close"].iloc[confirm_bar]
            atr_val = atr_series.iloc[confirm_bar]

            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # ── Gather previous N swing lows ──
            prev_lows = swing_lows[:idx]  # all swing lows before this one
            n_prev = min(len(prev_lows), self.require_n_prev_lows)
            prev_prices = [p.price for p in prev_lows[-n_prev:]] \
                if n_prev > 0 else []

            # ── Check bottom-rising condition ──
            if n_prev < self.require_n_prev_lows:
                # Not enough history
                is_elevated = False
                reason = (
                    f"REJECT: only {n_prev} previous swing low(s), "
                    f"need {self.require_n_prev_lows}"
                )
            else:
                prev_n = prev_lows[-self.require_n_prev_lows:]
                prev_n_prices = [p.price for p in prev_n]
                is_elevated = sw.price > max(prev_n_prices)

                if is_elevated:
                    reason = (
                        f"BUY: swing low {sw.price:.2f} > "
                        f"prev {self.require_n_prev_lows} lows "
                        f"{prev_n_prices} (底部抬高)"
                    )
                else:
                    reason = (
                        f"REJECT: swing low {sw.price:.2f} <= "
                        f"prev {self.require_n_prev_lows} lows "
                        f"{prev_n_prices} (底部未抬高)"
                    )

            stop_loss = close - self.stop_atr_multiplier * atr_val
            take_profit = close + self.tp_atr_multiplier * atr_val

            sig = ElevatedSwingSignal(
                timestamp=ts,
                signal_type="BUY" if is_elevated else "REJECT",
                price=close,
                atr=atr_val,
                stop_loss=stop_loss,
                take_profit=take_profit,
                swing_low_price=sw.price,
                swing_low_bar=sw.bar_index,
                prev_swing_lows=prev_prices,
                is_elevated=is_elevated,
                reason=reason,
            )

            self.all_signals.append(sig)
            if is_elevated:
                self.buy_signals.append(sig)

        n_total = len(self.all_signals)
        n_buy = len(self.buy_signals)
        n_reject = n_total - n_buy
        logger.info(
            "SwingLowElevated: %d swing lows, %d BUY (底部抬高), %d rejected",
            n_total, n_buy, n_reject,
        )
        return self.buy_signals

    # ── DataFrame helpers ───────────────────────────────────────

    def all_signals_to_dataframe(self) -> pd.DataFrame:
        """All signals (BUY + REJECT) as a DataFrame."""
        if not self.all_signals:
            return pd.DataFrame(columns=[
                "timestamp", "signal_type", "price", "atr",
                "stop_loss", "take_profit", "swing_low_price",
                "swing_low_bar", "prev_swing_lows", "is_elevated",
                "reason",
            ])
        return pd.DataFrame([
            {
                "timestamp": s.timestamp,
                "signal_type": s.signal_type,
                "price": s.price,
                "atr": s.atr,
                "stop_loss": s.stop_loss,
                "take_profit": s.take_profit,
                "swing_low_price": s.swing_low_price,
                "swing_low_bar": s.swing_low_bar,
                "prev_swing_lows": str(s.prev_swing_lows),
                "is_elevated": s.is_elevated,
                "reason": s.reason,
            }
            for s in self.all_signals
        ]).set_index("timestamp")

    def buy_signals_to_dataframe(self) -> pd.DataFrame:
        """Only BUY signals as a DataFrame."""
        if not self.buy_signals:
            return pd.DataFrame(columns=[
                "timestamp", "signal_type", "price", "atr",
                "stop_loss", "take_profit", "swing_low_price",
                "swing_low_bar", "prev_swing_lows", "reason",
            ])
        return pd.DataFrame([
            {
                "timestamp": s.timestamp,
                "signal_type": s.signal_type,
                "price": s.price,
                "atr": s.atr,
                "stop_loss": s.stop_loss,
                "take_profit": s.take_profit,
                "swing_low_price": s.swing_low_price,
                "swing_low_bar": s.swing_low_bar,
                "prev_swing_lows": str(s.prev_swing_lows),
                "reason": s.reason,
            }
            for s in self.buy_signals
        ]).set_index("timestamp")
