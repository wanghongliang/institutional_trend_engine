"""
Volume Delta indicator.

Volume Delta measures the net buying/selling pressure by estimating
how much volume was aggressive-buy vs aggressive-sell.

When intraday Level-2 data is available, delta is computed exactly.
Otherwise, we estimate it from bar close position within the range:

    delta ≈ volume × (2 × (close - low) / (high - low) - 1)

This gives:
  +volume → close near high (buyers in control)
  -volume → close near low  (sellers in control)
  0       → close at mid-range
"""

import numpy as np
import pandas as pd


def volume_delta(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Estimate per-bar volume delta.

    Returns
    -------
    pd.Series
        Signed volume delta (positive = net buy, negative = net sell).
    """
    rng = (high - low).replace(0, np.nan)
    close_position = (close - low) / rng  # 0 = at low, 1 = at high
    # map [0,1] → [-1, +1]
    buy_ratio = 2 * close_position - 1
    delta = volume * buy_ratio
    return delta.fillna(0)


def cumulative_delta(delta: pd.Series, reset_daily: bool = True) -> pd.Series:
    """
    Cumulative volume delta (CVD).

    Rising CVD → sustained buying pressure.
    Falling CVD → sustained selling pressure.
    Divergence between price and CVD → potential reversal.
    """
    if reset_daily and hasattr(delta.index, "date"):
        dates = delta.index.date
        return delta.groupby(dates).cumsum()
    return delta.cumsum()


def delta_acceleration(delta: pd.Series, window: int = 1) -> pd.Series:
    """
    Change in volume delta — captures shifts in order-flow pressure.

    When delta acceleration turns positive while price is still
    falling, it signals absorption / hidden buying.
    """
    return delta.diff(window)


def delta_reversal_bull(delta: pd.Series, lookback: int = 5) -> pd.Series:
    """
    Detect bullish delta reversal: was negative (selling), now positive (buying).

    Returns
    -------
    pd.Series
        Boolean: True on bullish reversal bar.
    """
    was_negative = delta.shift(lookback) < 0
    now_positive = delta > 0
    return was_negative & now_positive


def delta_reversal_bear(delta: pd.Series, lookback: int = 5) -> pd.Series:
    """
    Detect bearish delta reversal: was positive (buying), now negative (selling).

    Returns
    -------
    pd.Series
        Boolean: True on bearish reversal bar.
    """
    was_positive = delta.shift(lookback) > 0
    now_negative = delta < 0
    return was_positive & now_negative


def delta_reversal(delta: pd.Series, lookback: int = 5) -> pd.Series:
    """
    Detect any delta reversal (bullish or bearish).

    Returns
    -------
    pd.Series
        Boolean: True on any reversal bar.
    """
    return delta_reversal_bull(delta, lookback) | delta_reversal_bear(delta, lookback)
