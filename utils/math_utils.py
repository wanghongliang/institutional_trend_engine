"""
Math utilities used across the engine.

Pure functions — no state, no I/O.  Easy to unit-test.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def linear_regression_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling linear-regression slope for *series*.

    Uses the closed-form OLS estimator:
        slope = (n * Σxy − Σx * Σy) / (n * Σx² − (Σx)²)

    Parameters
    ----------
    series : pd.Series
        Values to fit (typically close prices).
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Slope for each bar (NaN for warm-up).
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    n = window
    x = np.arange(n, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x ** 2).sum()
    denom = n * sum_x2 - sum_x ** 2

    if denom == 0:
        return pd.Series(np.nan, index=series.index)

    def _slope(y: np.ndarray) -> float:
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        return (n * sum_xy - sum_x * sum_y) / denom

    return series.rolling(window).apply(_slope, raw=True)


def curvature(series: pd.Series, window: int) -> pd.Series:
    """
    Estimate curvature (second derivative) via second difference,
    smoothed over *window* bars.

    Returns
    -------
    pd.Series
        Curvature estimate (positive = concave up / accelerating).
    """
    first_diff = series.diff()
    second_diff = first_diff.diff()
    return second_diff.rolling(window).mean()


def normalize_0_1(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Boolean series: True where *fast* crosses above *slow*."""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Boolean series: True where *fast* crosses below *slow*."""
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division returning 0 where denominator is 0."""
    return numerator / denominator.replace(0, np.nan)


def rolling_high(series: pd.Series, window: int) -> pd.Series:
    """Rolling maximum."""
    return series.rolling(window).max()


def rolling_low(series: pd.Series, window: int) -> pd.Series:
    """Rolling minimum."""
    return series.rolling(window).min()


def true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    True Range: max(H-L, |H-prev_C|, |L-prev_C|).

    The first bar returns H - L.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
