"""
EMA (Exponential Moving Average) indicator.

EMA gives more weight to recent prices, making it more responsive
than SMA.  Used for trend confirmation (fast/slow crossover).
"""

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential moving average.

    Parameters
    ----------
    series : pd.Series
        Input values (typically close prices).
    period : int
        Lookback period.

    Returns
    -------
    pd.Series
        EMA values.
    """
    return series.ewm(span=period, adjust=False).mean()


def ema_crossover_signal(
    close: pd.Series,
    fast: int = 9,
    slow: int = 21,
) -> pd.Series:
    """
    Return +1 on bullish crossover, -1 on bearish, 0 otherwise.

    This is a *confirmation* signal only — the engine does not
    trade solely on EMA crossovers.
    """
    ema_f = ema(close, fast)
    ema_s = ema(close, slow)
    bullish = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
    bearish = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))
    signal = pd.Series(0, index=close.index)
    signal[bullish] = 1
    signal[bearish] = -1
    return signal
