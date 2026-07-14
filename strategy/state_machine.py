"""
Market Regime / State Machine.

Classifies each bar into one of five market states:
  - TREND_DAY          : strong directional day, trend-following
  - RANGE_DAY          : choppy, mean-reversion only
  - OPEN_DRIVE         : strong move in first 30 min (AM session)
  - TREND_EXHAUSTION   : trend losing momentum, prepare for reversal
  - FALSE_BREAKOUT     : breakout failed, expect reversal

The regime classification gates the strategy: only trade breakouts
on TREND_DAY or OPEN_DRIVE; avoid trading on RANGE_DAY.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config import MarketRegime


@dataclass
class RegimeSnapshot:
    """Market regime at a given bar."""
    timestamp: pd.Timestamp
    regime: MarketRegime
    trend_strength: float     # 0-1, how strong the trend is
    range_score: float        # 0-1, how range-bound the market is
    bar_index: int


class StateMachine:
    """
    Market regime classifier.

    Uses a combination of:
      - Opening range (first 30 min high/low)
      - ADX-like trend strength (slope / ATR ratio)
      - Realized volatility expansion
      - Price position relative to VWAP
    """

    def __init__(
        self,
        opening_range_minutes: int = 30,
        trend_strength_threshold: float = 0.15,
        range_threshold: float = 0.3,
    ):
        self.opening_range_minutes = opening_range_minutes
        self.trend_strength_threshold = trend_strength_threshold
        self.range_threshold = range_threshold

    def classify(self, df: pd.DataFrame) -> list[RegimeSnapshot]:
        """
        Classify each bar's market regime.

        Returns
        -------
        list[RegimeSnapshot]
            One snapshot per bar.
        """
        snapshots = []
        n = len(df)

        if n == 0:
            return snapshots

        # Compute opening range (first N minutes of each day)
        dates = df.index.date
        or_high = {}
        or_low = {}

        current_date = None
        bar_count = 0
        for i, ts in enumerate(df.index):
            d = ts.date()
            if d != current_date:
                current_date = d
                bar_count = 0
                or_high[d] = df["high"].iloc[i]
                or_low[d] = df["low"].iloc[i]
            else:
                bar_count += 1
                if bar_count < self.opening_range_minutes:
                    or_high[d] = max(or_high[d], df["high"].iloc[i])
                    or_low[d] = min(or_low[d], df["low"].iloc[i])

        # Compute trend strength via rolling slope / ATR
        from indicators.momentum import slope
        from indicators.atr import atr

        slope_series = slope(df["close"], window=20)
        atr_series = atr(df["high"], df["low"], df["close"], period=14)

        # Normalize trend strength: |slope| / ATR
        trend_strength = (slope_series.abs() / atr_series.replace(0, np.nan)).clip(0, 1)

        # Range score: 1 - trend_strength
        range_score = 1.0 - trend_strength

        # VWAP
        from indicators.vwap import vwap
        vwap_series = vwap(df["high"], df["low"], df["close"], df["volume"])

        for i, ts in enumerate(df.index):
            d = ts.date()
            ts_strength = trend_strength.iloc[i] if i < len(trend_strength) else 0
            r_score = range_score.iloc[i] if i < len(range_score) else 1

            # Determine regime
            regime = MarketRegime.UNKNOWN

            # Check if within opening range period
            bars_today = 0
            for j in range(i, -1, -1):
                if df.index[j].date() == d:
                    bars_today += 1
                else:
                    break

            is_opening = bars_today <= self.opening_range_minutes
            close = df["close"].iloc[i]
            or_h = or_high.get(d, close)
            or_l = or_low.get(d, close)
            vw = vwap_series.iloc[i] if i < len(vwap_series) else close

            # Open Drive: price breaks opening range within first 30 min
            if is_opening and bars_today > 5:
                if close > or_h * 1.001 and ts_strength > self.trend_strength_threshold:
                    regime = MarketRegime.OPEN_DRIVE
                elif close < or_l * 0.999 and ts_strength > self.trend_strength_threshold:
                    regime = MarketRegime.OPEN_DRIVE
                else:
                    regime = MarketRegime.UNKNOWN

            if regime == MarketRegime.UNKNOWN:
                if ts_strength > self.trend_strength_threshold:
                    # Check for exhaustion: trend strong but decelerating
                    if i > 5:
                        accel = df["close"].diff().diff().iloc[i]
                        if accel is not None and accel < 0:
                            regime = MarketRegime.TREND_EXHAUSTION
                        else:
                            regime = MarketRegime.TREND_DAY
                    else:
                        regime = MarketRegime.TREND_DAY
                elif r_score > (1 - self.range_threshold):
                    regime = MarketRegime.RANGE_DAY
                else:
                    # Check false breakout: broke OR but came back
                    if (close > or_h or close < or_l) and i > self.opening_range_minutes:
                        # Did it reverse?
                        if i > 1:
                            prev_close = df["close"].iloc[i - 1]
                            if abs(close - prev_close) / max(close, 0.01) < 0.001:
                                regime = MarketRegime.FALSE_BREAKOUT
                            else:
                                regime = MarketRegime.TREND_DAY
                        else:
                            regime = MarketRegime.UNKNOWN
                    else:
                        regime = MarketRegime.UNKNOWN

            snapshots.append(RegimeSnapshot(
                timestamp=ts,
                regime=regime,
                trend_strength=float(ts_strength) if not np.isnan(ts_strength) else 0,
                range_score=float(r_score) if not np.isnan(r_score) else 1,
                bar_index=i,
            ))

        return snapshots

    def to_dataframe(self, snapshots: list[RegimeSnapshot]) -> pd.DataFrame:
        """Convert snapshots to a DataFrame."""
        return pd.DataFrame([
            {
                "timestamp": s.timestamp,
                "regime": s.regime.value,
                "trend_strength": s.trend_strength,
                "range_score": s.range_score,
            }
            for s in snapshots
        ]).set_index("timestamp")
