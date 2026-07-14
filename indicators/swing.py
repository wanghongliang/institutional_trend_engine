"""
Swing High / Swing Low detector.

Implements dynamic swing-point detection using a fractal-based
approach with ATR filtering.  Output feeds the Market Structure
Engine for BOS / CHoCH / liquidity-sweep detection.

Algorithm
---------
A Swing High is confirmed when *window* bars on each side are
all lower than the candidate bar's high.  Swing Lows are the mirror.

Swings are then filtered: only keep swings where the price
difference exceeds *min_atr * ATR* to avoid noise.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    """A detected swing high or low."""
    timestamp: pd.Timestamp
    price: float
    type: str          # "high" or "low"
    bar_index: int     # position in the original series


class SwingDetector:
    """
    Detect swing highs and lows from OHLC data.

    Parameters
    ----------
    window : int
        Number of bars on each side that must be lower (for highs)
        or higher (for lows) to confirm a swing.
    min_atr : float
        Minimum swing excursion in ATR units.  Swings smaller than
        this are discarded as noise.
    """

    def __init__(self, window: int = 3, min_atr: float = 0.5):
        self.window = window
        self.min_atr = min_atr
        self.swings: List[SwingPoint] = []

    def detect(self, df: pd.DataFrame, atr_series: pd.Series) -> List[SwingPoint]:
        """
        Detect all swing points in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        atr_series : pd.Series
            ATR aligned to df.

        Returns
        -------
        List[SwingPoint]
            Sorted by timestamp.
        """
        self.swings = []
        highs = df["high"].values
        lows = df["low"].values
        timestamps = df.index
        atr_vals = atr_series.reindex(df.index).values
        n = len(highs)

        w = self.window

        for i in range(w, n - w):
            # ── swing high test ──
            is_high = True
            for j in range(1, w + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_high = False
                    break

            if is_high:
                atr_val = atr_vals[i] if not np.isnan(atr_vals[i]) else 0
                if atr_val > 0:
                    # check minimum excursion vs previous swing
                    prev_swing_price = self._last_swing_price("high")
                    if prev_swing_price is not None:
                        excursion = abs(highs[i] - prev_swing_price) / atr_val
                        if excursion < self.min_atr:
                            continue  # too small, skip

                self.swings.append(SwingPoint(
                    timestamp=timestamps[i],
                    price=float(highs[i]),
                    type="high",
                    bar_index=i,
                ))

            # ── swing low test ──
            is_low = True
            for j in range(1, w + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break

            if is_low:
                atr_val = atr_vals[i] if not np.isnan(atr_vals[i]) else 0
                if atr_val > 0:
                    prev_swing_price = self._last_swing_price("low")
                    if prev_swing_price is not None:
                        excursion = abs(lows[i] - prev_swing_price) / atr_val
                        if excursion < self.min_atr:
                            continue

                self.swings.append(SwingPoint(
                    timestamp=timestamps[i],
                    price=float(lows[i]),
                    type="low",
                    bar_index=i,
                ))

        self.swings.sort(key=lambda s: s.bar_index)
        return self.swings

    def _last_swing_price(self, swing_type: str) -> float | None:
        """Return the price of the most recent swing of *swing_type*."""
        for s in reversed(self.swings):
            if s.type == swing_type:
                return s.price
        return None

    def last_swing_high(self) -> SwingPoint | None:
        """Most recent swing high."""
        for s in reversed(self.swings):
            if s.type == "high":
                return s
        return None

    def last_swing_low(self) -> SwingPoint | None:
        """Most recent swing low."""
        for s in reversed(self.swings):
            if s.type == "low":
                return s
        return None

    def recent_swings(self, n: int = 10) -> List[SwingPoint]:
        """Return the last *n* swing points."""
        return self.swings[-n:]

    def to_dataframe(self) -> pd.DataFrame:
        """Export swings to a DataFrame."""
        if not self.swings:
            return pd.DataFrame(columns=["timestamp", "price", "type", "bar_index"])
        return pd.DataFrame([
            {"timestamp": s.timestamp, "price": s.price, "type": s.type, "bar_index": s.bar_index}
            for s in self.swings
        ])
