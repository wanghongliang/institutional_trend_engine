"""
Swing Low Reversal Strategy.

A simple, clean mean-reversion strategy:

1. Detect swing lows using fractal-based SwingDetector.
2. When a swing low is confirmed (window bars after the low),
   buy at the close of the confirmation bar.
3. Stop loss = entry_price - 1 * ATR.
4. Take profit = entry_price + 2 * ATR.

Risk : Reward = 1 : 2

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
class SwingSignal:
    """A trading signal from the swing low strategy."""
    timestamp: pd.Timestamp
    signal_type: str          # "BUY" or "SELL"
    price: float
    atr: float
    stop_loss: float
    take_profit: float
    swing_low_price: float    # the swing low that triggered the entry
    swing_low_bar: int        # bar index of the swing low
    reason: str


class SwingLowStrategy:
    """
    Swing Low Reversal Strategy.

    Buys at each confirmed swing low with:
      - Stop loss: 1 ATR below entry
      - Take profit: 2 ATR above entry

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
    """

    def __init__(
        self,
        swing_window: int = 3,
        atr_period: int = 14,
        stop_atr_multiplier: float = 1.0,
        tp_atr_multiplier: float = 2.0,
        swing_min_atr: float = 0.0,
    ):
        self.swing_window = swing_window
        self.atr_period = atr_period
        self.stop_atr_multiplier = stop_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.swing_min_atr = swing_min_atr

        self.swing_detector = SwingDetector(
            window=swing_window,
            min_atr=swing_min_atr,
        )
        self.swings: List[SwingPoint] = []
        self.signals: List[SwingSignal] = []

    def generate_signals(self, df: pd.DataFrame) -> List[SwingSignal]:
        """
        Generate BUY signals at each confirmed swing low.

        A swing low at bar i is confirmed at bar i + window
        (when window bars to the right are all higher).
        The BUY signal fires at the close of bar i + window.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex.

        Returns
        -------
        List[SwingSignal]
            Trading signals (BUY only; exits are handled by the
            backtest engine via stop-loss / take-profit).
        """
        # Compute ATR
        atr_series = calc_atr(
            df["high"], df["low"], df["close"], period=self.atr_period
        )

        # Detect all swing points
        self.swings = self.swing_detector.detect(df, atr_series)
        self.swing_detector.swings = self.swings

        # Generate signals at each swing low
        self.signals = []
        w = self.swing_window

        for sw in self.swings:
            if sw.type != "low":
                continue

            # The confirmation bar is sw.bar_index + w
            confirm_bar = sw.bar_index + w
            if confirm_bar >= len(df):
                continue

            ts = df.index[confirm_bar]
            close = df["close"].iloc[confirm_bar]
            atr_val = atr_series.iloc[confirm_bar]

            if np.isnan(atr_val) or atr_val <= 0:
                continue

            stop_loss = close - self.stop_atr_multiplier * atr_val
            take_profit = close + self.tp_atr_multiplier * atr_val

            self.signals.append(SwingSignal(
                timestamp=ts,
                signal_type="BUY",
                price=close,
                atr=atr_val,
                stop_loss=stop_loss,
                take_profit=take_profit,
                swing_low_price=sw.price,
                swing_low_bar=sw.bar_index,
                reason=f"Swing low confirmed at {sw.price:.2f} "
                       f"(bar {sw.bar_index}), entry at {close:.2f}, "
                       f"SL={stop_loss:.2f}, TP={take_profit:.2f}",
            ))

        logger.info(
            "SwingLowStrategy: %d swing lows detected, %d BUY signals generated",
            sum(1 for s in self.swings if s.type == "low"),
            len(self.signals),
        )
        return self.signals

    def signals_to_dataframe(self) -> pd.DataFrame:
        """Convert signals to a DataFrame."""
        if not self.signals:
            return pd.DataFrame(columns=[
                "timestamp", "signal_type", "price", "atr",
                "stop_loss", "take_profit", "swing_low_price",
                "swing_low_bar", "reason",
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
                "reason": s.reason,
            }
            for s in self.signals
        ]).set_index("timestamp")
