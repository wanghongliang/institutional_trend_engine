"""
Realized Volatility indicator.

RV measures the actual volatility experienced over a recent window.
The strategy looks for **volatility compression** (RV declining)
followed by **volatility expansion** (RV spiking) — this pattern
often precedes trend initiation.
"""

import numpy as np
import pandas as pd


def realized_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Annualized realized volatility from log returns.

    RV = std(log_returns, window) × sqrt(annualization_factor)

    For 1-minute bars and US equity market (390 min/day, 252 days):
        annual_factor = sqrt(390 × 252)

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int
        Rolling window for std calculation.

    Returns
    -------
    pd.Series
        Realized volatility (annualized).
    """
    log_ret = np.log(close / close.shift(1))
    # Annualization for 1-min bars: sqrt(390 * 252) ≈ 313.5
    annual_factor = np.sqrt(390 * 252)
    return log_ret.rolling(period).std() * annual_factor


def rv_compression(rv: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Detect volatility compression: current RV < threshold × average RV.

    Parameters
    ----------
    rv : pd.Series
        Realized volatility series.
    lookback : int
        Window for computing the average RV.

    Returns
    -------
    pd.Series
        Boolean: True when RV is compressed.
    """
    avg_rv = rv.rolling(lookback).mean()
    return rv < (avg_rv * 0.5)  # below 50% of average


def rv_expansion(rv: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Detect volatility expansion: RV spikes above recent average
    after a period of compression.

    This is the "volatility release" that often marks trend starts.
    """
    avg_rv = rv.rolling(lookback).mean()
    # Use a more sensitive compression threshold: below 70% of average
    was_compressed = (rv.shift(3) < avg_rv.shift(3) * 0.7)
    now_expanded = rv > avg_rv
    return was_compressed & now_expanded


def atr_based_rv(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
) -> pd.Series:
    """
    Simple volatility proxy using ATR as fraction of price.

    Useful when log-return based RV is too noisy for short windows.
    """
    from indicators.atr import atr
    a = atr(high, low, close, atr_period)
    return a / close
