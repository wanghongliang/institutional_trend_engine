"""
ATR (Average True Range) indicator.

ATR measures market volatility by decomposing the entire range
of a bar.  Used throughout the engine for:
  - position sizing
  - stop-loss placement
  - swing-size filtering
  - early-entry proximity checks
"""

import pandas as pd

from utils.math_utils import true_range


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Classic Wilder's ATR.

    Parameters
    ----------
    high, low, close : pd.Series
        Bar OHLC components.
    period : int
        Smoothing period (default 14).

    Returns
    -------
    pd.Series
        ATR values (NaN for warm-up).
    """
    tr = true_range(high, low, close)

    # Wilder's smoothing = EMA with alpha = 1/period
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def atr_percent(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR as a percentage of close price."""
    return atr(high, low, close, period) / close * 100.0
