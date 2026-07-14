"""
VWAP (Volume-Weighted Average Price).

VWAP is the institutional benchmark.  The strategy uses:
  - price crossing above VWAP → bullish
  - price reclaiming VWAP after losing it → strong bullish
  - VWAP slope turning up → trend confirmation
"""

import numpy as np
import pandas as pd


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    reset_daily: bool = True,
) -> pd.Series:
    """
    Compute VWAP, optionally resetting at each session open.

    Parameters
    ----------
    high, low, close, volume : pd.Series
        Bar OHLCV components (DatetimeIndex expected).
    reset_daily : bool
        If True, cumulative VWAP resets each trading day.

    Returns
    -------
    pd.Series
        VWAP values.
    """
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume

    if reset_daily and hasattr(typical_price.index, "date"):
        # Group by calendar date
        dates = typical_price.index.date
        cum_pv = pv.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
    else:
        cum_pv = pv.cumsum()
        cum_vol = volume.cumsum()

    vwap_val = cum_pv / cum_vol.replace(0, np.nan)
    return vwap_val


def vwap_slope(vwap_series: pd.Series, window: int = 5) -> pd.Series:
    """
    Slope of VWAP over *window* bars.
    Positive slope → VWAP rising (bullish bias).
    """
    return vwap_series.diff(window) / window


def vwap_reclaim_signal(close: pd.Series, vwap_series: pd.Series) -> pd.Series:
    """
    Detect VWAP reclaim: price was below VWAP, now back above.

    This is a strong institutional signal — it means buyers
    stepped back in after a pullback.

    Returns
    -------
    pd.Series
        Boolean: True on reclaim bar.
    """
    above = close > vwap_series
    was_below = close.shift(1) <= vwap_series.shift(1)
    return above & was_below
