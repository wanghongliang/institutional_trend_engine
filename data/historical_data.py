"""
Historical market data retrieval via the Schwab API (schwabdev).

Provides a clean interface to fetch OHLCV bars and optionally
cache them as Parquet for fast subsequent loads.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Schwab price-history frequency codes ────────────────────────
FREQ_MAP = {
    "1": "1",      # 1 minute
    "5": "5",      # 5 minutes
    "10": "10",    # 10 minutes
    "15": "15",    # 15 minutes
    "30": "30",    # 30 minutes
    "60": "60",    # 1 hour
    "D": "D",      # daily
    "W": "W",      # weekly
}

# Schwab period type → period value mapping
# period_type: 1=day, 2=month, 3=year, 4=ytd
PERIOD_MAP = {
    "day": ("1", "1"),       # 1 day of 1-min bars
    "2day": ("1", "2"),
    "1week": ("1", "5"),
    "month": ("2", "1"),     # 1 month
    "3month": ("2", "3"),
    "6month": ("2", "6"),
    "year": ("3", "1"),      # 1 year
    "ytd": ("4", "1"),
}


@dataclass
class PriceHistoryRequest:
    """Parameters for a Schwab price-history request."""
    symbol: str
    period: str = "day"          # key in PERIOD_MAP
    frequency: str = "1"         # key in FREQ_MAP
    need_extended_hours: bool = True


class HistoricalData:
    """
    Fetch historical OHLCV bars from Schwab.

    Usage
    -----
    >>> hd = HistoricalData(client)
    >>> df = hd.get_bars("SPY", period="day", frequency="1")
    """

    def __init__(self, client=None):
        """
        Parameters
        ----------
        client : schwabdev.Client | None
            Authenticated Schwab client.  When *None*, the instance
            can still parse pre-fetched JSON (useful for backtest).
        """
        self.client = client

    def get_bars(
        self,
        symbol: str,
        period: str = "day",
        frequency: str = "1",
        need_extended_hours: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical candles and return a tidy DataFrame.

        Returns
        -------
        pd.DataFrame  indexed by DatetimeIndex (UTC), columns:
            open, high, low, close, volume
        """
        if self.client is None:
            raise RuntimeError(
                "No Schwab client — cannot fetch live data. "
                "Pass a client or use the backtest data loader."
            )

        period_type, period_value = PERIOD_MAP.get(
            period, ("1", "1")
        )
        freq = FREQ_MAP.get(frequency, "1")

        logger.info(
            "Fetching %s bars: period=%s freq=%s",
            symbol, period, frequency,
        )

        resp = self.client.price_history(
            symbol,
            periodType=period_type,
            period=period_value,
            frequencyType="1",   # 1 = minute, 0 = daily, etc.
            frequency=freq,
            needExtendedHoursData=need_extended_hours,
        )

        if resp.status_code != 200:
            logger.error("Schwab API error %s: %s",
                         resp.status_code, resp.text[:300])
            return pd.DataFrame()

        data = resp.json()
        candles = data.get("candles", [])
        if not candles:
            logger.warning("No candles returned for %s", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        # Schwab returns epoch-millis in UTC
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("datetime")
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.sort_index()

        logger.info("Got %d bars for %s", len(df), symbol)
        return df

    def get_bars_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1",
    ) -> pd.DataFrame:
        """
        Fetch bars between *start* and *end* (UTC datetimes).

        Schwab's API accepts startDate / endDate as epoch-millis.
        """
        if self.client is None:
            raise RuntimeError("No Schwab client connected.")

        freq = FREQ_MAP.get(frequency, "1")
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        resp = self.client.price_history(
            symbol,
            periodType="1",
            frequencyType="1",
            frequency=freq,
            startDate=start_ms,
            endDate=end_ms,
            needExtendedHoursData=True,
        )

        if resp.status_code != 200:
            logger.error("API error %s", resp.status_code)
            return pd.DataFrame()

        candles = resp.json().get("candles", [])
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
        return df.sort_index()

    @staticmethod
    def save_parquet(df: pd.DataFrame, path: str) -> None:
        """Cache a DataFrame to Parquet."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_parquet(path)
        logger.info("Saved %d rows → %s", len(df), path)

    @staticmethod
    def load_parquet(path: str) -> pd.DataFrame:
        """Load a cached Parquet file."""
        df = pd.read_parquet(path)
        logger.info("Loaded %d rows ← %s", len(df), path)
        return df
