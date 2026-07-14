"""
Momentum indicators: Velocity, Acceleration, Curvature, and Slope.

These are the core of the Turning Point Detection framework.

Velocity     = d(Price)/dt              → first difference
Acceleration = d(Velocity)/dt            → second difference (zero-crossing = turning signal)
Curvature    = smoothed second derivative → captures acceleration of acceleration
Slope        = rolling linear regression slope → trend direction strength
"""

import pandas as pd

from utils.math_utils import linear_regression_slope, curvature


def velocity(close: pd.Series, window: int = 1) -> pd.Series:
    """
    Price velocity = rate of change (first difference).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    window : int
        Difference period (default 1 = bar-to-bar).

    Returns
    -------
    pd.Series
        Velocity values.
    """
    return close.diff(window)


def acceleration(close: pd.Series, window: int = 1) -> pd.Series:
    """
    Price acceleration = change in velocity (second difference).

    **Zero-crossing** of acceleration is the earliest turning signal:
    when acceleration goes from negative to positive, the downtrend
    is decelerating even though price may still be falling.

    Returns
    -------
    pd.Series
        Acceleration values.
    """
    return velocity(close, window).diff()


def acceleration_zero_crossing(accel: pd.Series) -> pd.Series:
    """
    Detect acceleration zero-crossing (neg→pos = bullish, pos→neg = bearish).

    Returns
    -------
    pd.Series
        +1 on bullish zero-cross, -1 on bearish zero-cross, 0 otherwise.
    """
    bullish = (accel > 0) & (accel.shift(1) <= 0)
    bearish = (accel < 0) & (accel.shift(1) >= 0)
    signal = pd.Series(0, index=accel.index)
    signal[bullish] = 1
    signal[bearish] = -1
    return signal


def slope(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling linear-regression slope of close prices.

    Positive slope → uptrend.
    Negative slope → downtrend.
    Slope crossing zero → trend change.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    window : int
        Regression window (default 20 bars).

    Returns
    -------
    pd.Series
        Slope values.
    """
    return linear_regression_slope(close, window)


def slope_sign_change(slope_series: pd.Series) -> pd.Series:
    """
    Detect slope sign change (neg→pos or pos→neg).

    Returns
    -------
    pd.Series
        +1 on neg→pos, -1 on pos→neg, 0 otherwise.
    """
    bullish = (slope_series > 0) & (slope_series.shift(1) <= 0)
    bearish = (slope_series < 0) & (slope_series.shift(1) >= 0)
    signal = pd.Series(0, index=slope_series.index)
    signal[bullish] = 1
    signal[bearish] = -1
    return signal


def curvature_indicator(close: pd.Series, window: int = 10) -> pd.Series:
    """
    Smoothed second derivative — captures how fast acceleration
    is changing.  High positive curvature = price curve is
    bending upward sharply.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    window : int
        Smoothing window.

    Returns
    -------
    pd.Series
        Curvature values.
    """
    return curvature(close, window)


def curvature_threshold_break(
    close: pd.Series,
    window: int = 10,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Boolean: curvature exceeds *threshold* (curvature increasing).

    This is layer 3 of the turning-point framework: even when slope
    is still small, high curvature signals an impending move.
    """
    curv = curvature_indicator(close, window)
    return curv > threshold
