"""
Momentum Acceleration Strategy — 动量突破 + 加速度交易.

Core philosophy:
    "Only catch the fastest, strongest segment of the main trend."

Trading flow:
    1. Patiently wait in cash — no pre-judging, no early entry.
    2. Detect consolidation (蓄力) phases using rolling range analysis
       and scipy.signal.argrelextrema for swing boundaries.
    3. Identify momentum breakout — price breaks out of the
       consolidation range with strong velocity.
    4. Confirm acceleration — velocity is accelerating (2nd derivative
       positive for longs, negative for shorts) AND the acceleration
       magnitude is increasing over the last few bars.
    5. Enter IMMEDIATELY at the close of the signal bar — no delay.
    6. Very tight stop loss (default 0.5 ATR).
    7. Fixed 2:1 risk-reward — take profit at 2× stop distance.
    8. Holding period: 2–20 bars (ultra-short intraday).

Supports both LONG and SHORT directions.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from indicators.atr import atr as calc_atr
from indicators.momentum import velocity, acceleration
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════

@dataclass
class MomentumSignal:
    """A trading signal from the momentum acceleration strategy."""
    timestamp: pd.Timestamp
    direction: str               # "LONG" or "SHORT"
    price: float                 # entry price (close of signal bar)
    atr: float
    velocity: float              # price velocity at signal bar
    acceleration: float          # price acceleration at signal bar
    stop_loss: float
    take_profit: float
    consolidation_high: float    # upper boundary of consolidation
    consolidation_low: float     # lower boundary of consolidation
    consolidation_bars: int      # how many bars was the consolidation
    reason: str


@dataclass
class ConsolidationZone:
    """A detected consolidation (蓄力) zone."""
    start_bar: int
    end_bar: int
    high: float
    low: float
    range_pct: float             # (high - low) / mean_close
    n_bars: int


# ══════════════════════════════════════════════════════════════════
#  Strategy
# ══════════════════════════════════════════════════════════════════

class MomentumAccelerationStrategy:
    """
    Momentum Breakout + Acceleration Trading Strategy.

    Detects consolidation phases, then enters when price breaks out
    with strong momentum AND confirmed acceleration.

    Parameters
    ----------
    extrema_order : int
        Order parameter for scipy.signal.argrelextrema (default 5).
        Determines how many bars on each side must be lower/higher
        to confirm a local extremum.
    consolidation_window : int
        Rolling window to detect range-bound price action (default 20).
    consolidation_range_max : float
        Relative threshold for consolidation: a zone forms when the
        rolling range is below this fraction of its own rolling median
        (default 0.7 = 70% of median range = compressed volatility).
    velocity_min_atr : float
        Minimum |velocity| as fraction of ATR to qualify as momentum
        (default 0.3).
    accel_confirm_bars : int
        Number of bars to confirm acceleration trend (default 3).
    accel_min_atr : float
        Minimum |acceleration| as fraction of ATR (default 0.1).
    stop_atr_multiplier : float
        Stop loss distance = stop_atr_multiplier × ATR (default 0.5).
    tp_atr_multiplier : float
        Take profit distance = tp_atr_multiplier × ATR (default 1.0).
        This gives a fixed 2:1 risk-reward ratio.
    max_holding_bars : int
        Maximum holding period in bars (default 20).
    min_holding_bars : int
        Minimum holding period before TP/SL can trigger (default 2).
        Stop loss still triggers immediately to protect capital.
    atr_period : int
        ATR calculation period (default 14).
    """

    def __init__(
        self,
        extrema_order: int = 5,
        consolidation_window: int = 20,
        consolidation_range_max: float = 0.7,
        velocity_min_atr: float = 0.3,
        accel_confirm_bars: int = 3,
        accel_min_atr: float = 0.1,
        stop_atr_multiplier: float = 0.5,
        tp_atr_multiplier: float = 1.0,
        max_holding_bars: int = 20,
        min_holding_bars: int = 2,
        atr_period: int = 14,
    ):
        self.extrema_order = extrema_order
        self.consolidation_window = consolidation_window
        self.consolidation_range_max = consolidation_range_max
        self.velocity_min_atr = velocity_min_atr
        self.accel_confirm_bars = accel_confirm_bars
        self.accel_min_atr = accel_min_atr
        self.stop_atr_multiplier = stop_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.max_holding_bars = max_holding_bars
        self.min_holding_bars = min_holding_bars
        self.atr_period = atr_period

        self.signals: List[MomentumSignal] = []
        self.consolidation_zones: List[ConsolidationZone] = []
        self.all_bars_analysis: List[dict] = []  # per-bar analysis for debugging

    # ── Public API ──────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> List[MomentumSignal]:
        """
        Scan every bar and generate entry signals when:
          1. Price was recently in a consolidation zone.
          2. Price breaks out of the consolidation range.
          3. Velocity exceeds the momentum threshold.
          4. Acceleration is in the breakout direction and increasing.

        Only one signal per direction per consolidation breakout.
        Returns the list of confirmed entry signals.
        """
        n = len(df)
        if n < self.consolidation_window + self.extrema_order + 5:
            logger.warning(
                "Not enough bars (%d) for momentum analysis "
                "(need >= %d)",
                n, self.consolidation_window + self.extrema_order + 5,
            )
            return []

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ── Compute indicators ──────────────────────────────────
        atr_series = calc_atr(high, low, close, period=self.atr_period)
        vel = velocity(close, window=1)
        accel = acceleration(close, window=1)

        # ── Detect local extrema using scipy ────────────────────
        high_idx = argrelextrema(high.values, np.greater, order=self.extrema_order)[0]
        low_idx = argrelextrema(low.values, np.less, order=self.extrema_order)[0]

        logger.info(
            "argrelextrema (order=%d): %d swing highs, %d swing lows",
            self.extrema_order, len(high_idx), len(low_idx),
        )

        # ── Detect consolidation zones ──────────────────────────
        self.consolidation_zones = self._detect_consolidation_zones(df, atr_series)

        logger.info(
            "Detected %d consolidation zones",
            len(self.consolidation_zones),
        )

        # ── Build a lookup: bar_index -> active consolidation zone ──
        # A zone is "active" for breakout detection from its end_bar
        # until a new zone forms or `consolidation_window` bars pass.
        zone_by_bar = {}
        for zone in self.consolidation_zones:
            # Zone is active from end_bar to end_bar + consolidation_window
            for b in range(zone.end_bar, min(zone.end_bar + self.consolidation_window, n)):
                if b not in zone_by_bar:
                    zone_by_bar[b] = zone

        # ── Scan for breakout + acceleration signals ────────────
        self.signals = []
        self.all_bars_analysis = []

        # Track which zones have already generated a signal
        used_zones = set()

        for i in range(self.consolidation_window + self.extrema_order, n):
            ts = df.index[i]
            c = close.iloc[i]
            h = high.iloc[i]
            l = low.iloc[i]
            atr_val = atr_series.iloc[i]
            vel_val = vel.iloc[i]
            accel_val = accel.iloc[i]

            if np.isnan(atr_val) or atr_val <= 0:
                self.all_bars_analysis.append({
                    "bar": i, "timestamp": ts,
                    "state": "no_atr",
                })
                continue

            # Check if there's an active consolidation zone
            zone = zone_by_bar.get(i)
            if zone is None or id(zone) in used_zones:
                self.all_bars_analysis.append({
                    "bar": i, "timestamp": ts,
                    "state": "waiting",
                    "close": c, "atr": atr_val,
                })
                continue

            # ── Check breakout conditions ───────────────────────
            # LONG: close breaks above consolidation high
            long_breakout = c > zone.high
            # SHORT: close breaks below consolidation low
            short_breakout = c < zone.low

            if not long_breakout and not short_breakout:
                self.all_bars_analysis.append({
                    "bar": i, "timestamp": ts,
                    "state": "in_consolidation",
                    "close": c, "zone_high": zone.high,
                    "zone_low": zone.low,
                })
                continue

            # ── Check momentum (velocity threshold) ─────────────
            vel_threshold = self.velocity_min_atr * atr_val
            if long_breakout and vel_val <= vel_threshold:
                self.all_bars_analysis.append({
                    "bar": i, "timestamp": ts,
                    "state": "breakout_no_momentum",
                    "close": c, "velocity": vel_val,
                    "vel_threshold": vel_threshold,
                })
                continue
            if short_breakout and vel_val >= -vel_threshold:
                self.all_bars_analysis.append({
                    "bar": i, "timestamp": ts,
                    "state": "breakout_no_momentum",
                    "close": c, "velocity": vel_val,
                    "vel_threshold": vel_threshold,
                })
                continue

            # ── Check acceleration confirmation ─────────────────
            # For LONG: acceleration must be positive AND increasing
            # For SHORT: acceleration must be negative AND decreasing (more negative)
            accel_threshold = self.accel_min_atr * atr_val

            if long_breakout:
                # Need positive acceleration
                if accel_val <= accel_threshold:
                    self.all_bars_analysis.append({
                        "bar": i, "timestamp": ts,
                        "state": "no_acceleration",
                        "close": c, "acceleration": accel_val,
                        "accel_threshold": accel_threshold,
                    })
                    continue

                # Check acceleration is increasing over last N bars
                recent_accel = accel.iloc[max(0, i - self.accel_confirm_bars):i + 1]
                accel_increasing = recent_accel.iloc[-1] >= recent_accel.iloc[0]

                if not accel_increasing:
                    self.all_bars_analysis.append({
                        "bar": i, "timestamp": ts,
                        "state": "accel_not_increasing",
                        "close": c, "acceleration": accel_val,
                    })
                    continue

                # ── ALL CONDITIONS MET — LONG SIGNAL ──
                stop_loss = c - self.stop_atr_multiplier * atr_val
                take_profit = c + self.tp_atr_multiplier * atr_val

                sig = MomentumSignal(
                    timestamp=ts,
                    direction="LONG",
                    price=c,
                    atr=atr_val,
                    velocity=vel_val,
                    acceleration=accel_val,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    consolidation_high=zone.high,
                    consolidation_low=zone.low,
                    consolidation_bars=zone.n_bars,
                    reason=(
                        f"LONG: breakout above consolidation high "
                        f"{zone.high:.2f}, vel={vel_val:.4f} > "
                        f"{vel_threshold:.4f}, accel={accel_val:.4f} > "
                        f"{accel_threshold:.4f} and increasing"
                    ),
                )
                self.signals.append(sig)
                used_zones.add(id(zone))

                self.all_bars_analysis.append({
                    "bar": i, "timestamp": ts,
                    "state": "LONG_SIGNAL",
                    "close": c, "signal": sig,
                })

            elif short_breakout:
                # Need negative acceleration
                if accel_val >= -accel_threshold:
                    self.all_bars_analysis.append({
                        "bar": i, "timestamp": ts,
                        "state": "no_acceleration",
                        "close": c, "acceleration": accel_val,
                        "accel_threshold": accel_threshold,
                    })
                    continue

                # Check acceleration is decreasing (more negative) over last N bars
                recent_accel = accel.iloc[max(0, i - self.accel_confirm_bars):i + 1]
                accel_decreasing = recent_accel.iloc[-1] <= recent_accel.iloc[0]

                if not accel_decreasing:
                    self.all_bars_analysis.append({
                        "bar": i, "timestamp": ts,
                        "state": "accel_not_increasing",
                        "close": c, "acceleration": accel_val,
                    })
                    continue

                # ── ALL CONDITIONS MET — SHORT SIGNAL ──
                stop_loss = c + self.stop_atr_multiplier * atr_val
                take_profit = c - self.tp_atr_multiplier * atr_val

                sig = MomentumSignal(
                    timestamp=ts,
                    direction="SHORT",
                    price=c,
                    atr=atr_val,
                    velocity=vel_val,
                    acceleration=accel_val,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    consolidation_high=zone.high,
                    consolidation_low=zone.low,
                    consolidation_bars=zone.n_bars,
                    reason=(
                        f"SHORT: breakout below consolidation low "
                        f"{zone.low:.2f}, vel={vel_val:.4f} < "
                        f"{-vel_threshold:.4f}, accel={accel_val:.4f} < "
                        f"{-accel_threshold:.4f} and decreasing"
                    ),
                )
                self.signals.append(sig)
                used_zones.add(id(zone))

                self.all_bars_analysis.append({
                    "bar": i, "timestamp": ts,
                    "state": "SHORT_SIGNAL",
                    "close": c, "signal": sig,
                })

        n_long = sum(1 for s in self.signals if s.direction == "LONG")
        n_short = sum(1 for s in self.signals if s.direction == "SHORT")
        logger.info(
            "MomentumAcceleration: %d signals (%d LONG, %d SHORT) "
            "from %d consolidation zones",
            len(self.signals), n_long, n_short,
            len(self.consolidation_zones),
        )
        return self.signals

    # ── Consolidation detection ────────────────────────────────

    def _detect_consolidation_zones(
        self,
        df: pd.DataFrame,
        atr_series: pd.Series,
    ) -> List[ConsolidationZone]:
        """
        Detect consolidation (蓄力) zones using adaptive relative analysis.

        Instead of a fixed absolute threshold, uses a **percentile-based**
        approach that adapts to any data volatility:

          1. Compute rolling range over `consolidation_window` bars.
          2. Compute the rolling median of this range over a longer window.
          3. Consolidation = current range < 70% of its own rolling median.

        This identifies periods where volatility is compressed relative
        to the recent norm — true "蓄力" (accumulation) zones.

        Returns a list of ConsolidationZone objects.
        """
        n = len(df)
        w = self.consolidation_window
        zones: List[ConsolidationZone] = []

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Rolling high/low/mean over the consolidation window
        rolling_high = high.rolling(w).max()
        rolling_low = low.rolling(w).min()
        rolling_mean = close.rolling(w).mean()

        # Range as fraction of mean price
        range_frac = (rolling_high - rolling_low) / rolling_mean

        # Adaptive threshold: rolling median of range_frac
        # Consolidation = range below (consolidation_range_max × median)
        # consolidation_range_max is repurposed as the relative ratio (default 0.7)
        long_window = w * 3  # 3x the consolidation window for context
        range_median = range_frac.rolling(long_window).median()
        relative_threshold = range_median * self.consolidation_range_max

        # Scan for consolidation zones
        i = w + long_window
        while i < n:
            if np.isnan(range_frac.iloc[i]) or np.isnan(relative_threshold.iloc[i]):
                i += 1
                continue

            is_consolidating = range_frac.iloc[i] < relative_threshold.iloc[i]

            if is_consolidating:
                start = i
                # Extend while range stays below 1.3x threshold
                while i < n and (
                    not np.isnan(range_frac.iloc[i])
                    and not np.isnan(relative_threshold.iloc[i])
                    and range_frac.iloc[i] < relative_threshold.iloc[i] * 1.3
                ):
                    i += 1

                end = i - 1
                if end > start + self.extrema_order:
                    # Use actual high/low during the consolidation period
                    zone_high = high.iloc[start:end + 1].max()
                    zone_low = low.iloc[start:end + 1].min()
                    zone_range_frac = (zone_high - zone_low) / close.iloc[end]

                    zone = ConsolidationZone(
                        start_bar=start,
                        end_bar=end,
                        high=zone_high,
                        low=zone_low,
                        range_pct=zone_range_frac,
                        n_bars=end - start + 1,
                    )
                    zones.append(zone)

                # Skip ahead to avoid overlapping zones
                i += self.extrema_order
            else:
                i += 1

        return zones

    # ── DataFrame helpers ───────────────────────────────────────

    def signals_to_dataframe(self) -> pd.DataFrame:
        """All signals as a DataFrame."""
        if not self.signals:
            return pd.DataFrame(columns=[
                "timestamp", "direction", "price", "atr",
                "velocity", "acceleration", "stop_loss", "take_profit",
                "consolidation_high", "consolidation_low",
                "consolidation_bars", "reason",
            ])
        return pd.DataFrame([
            {
                "timestamp": s.timestamp,
                "direction": s.direction,
                "price": s.price,
                "atr": s.atr,
                "velocity": s.velocity,
                "acceleration": s.acceleration,
                "stop_loss": s.stop_loss,
                "take_profit": s.take_profit,
                "consolidation_high": s.consolidation_high,
                "consolidation_low": s.consolidation_low,
                "consolidation_bars": s.consolidation_bars,
                "reason": s.reason,
            }
            for s in self.signals
        ]).set_index("timestamp")

    def zones_to_dataframe(self) -> pd.DataFrame:
        """Consolidation zones as a DataFrame."""
        if not self.consolidation_zones:
            return pd.DataFrame(columns=[
                "start_bar", "end_bar", "high", "low",
                "range_pct", "n_bars",
            ])
        return pd.DataFrame([
            {
                "start_bar": z.start_bar,
                "end_bar": z.end_bar,
                "high": z.high,
                "low": z.low,
                "range_pct": z.range_pct,
                "n_bars": z.n_bars,
            }
            for z in self.consolidation_zones
        ])
