"""
Global configuration for the Institutional Trend Engine.

All tunable parameters live here so strategies, risk, and backtest
modules share a single source of truth.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class MarketRegime(str, Enum):
    """Market state classification."""
    TREND_DAY = "trend_day"
    RANGE_DAY = "range_day"
    OPEN_DRIVE = "open_drive"
    TREND_EXHAUSTION = "trend_exhaustion"
    FALSE_BREAKOUT = "false_breakout"
    UNKNOWN = "unknown"


class TradeState(str, Enum):
    """Order/trade lifecycle states."""
    FLAT = "flat"
    EARLY_ENTRY = "early_entry"
    IN_POSITION = "in_position"
    ADD_POSITION = "add_position"
    EXITING = "exiting"


class SignalType(str, Enum):
    """Signal types from the signal engine."""
    EARLY_BUY = "early_buy"
    BREAKOUT_BUY = "breakout_buy"
    EARLY_SELL = "early_sell"
    BREAKOUT_SELL = "breakout_sell"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    NO_SIGNAL = "no_signal"


@dataclass
class Config:
    """Master configuration object."""

    # ── Schwab API ──────────────────────────────────────────────
    schwab_app_key: str = ""
    schwab_app_secret: str = ""
    schwab_callback_url: str = "https://127.0.0.1:8443/callback"

    # ── Trading symbol ──────────────────────────────────────────
    symbol: str = "SPY"
    asset_type: str = "EQUITY"

    # ── Candle / data ───────────────────────────────────────────
    bar_frequency: str = "1"          # 1-minute bars
    history_period: str = "day"       # for initial load
    lookback_bars: int = 500          # bars to keep in memory

    # ── Swing detection ─────────────────────────────────────────
    swing_window: int = 3             # bars on each side for fractal
    swing_min_atr: float = 0.5        # minimum swing size in ATR units

    # ── ATR ─────────────────────────────────────────────────────
    atr_period: int = 14

    # ── EMA ─────────────────────────────────────────────────────
    ema_fast: int = 9
    ema_slow: int = 21

    # ── VWAP ────────────────────────────────────────────────────
    vwap_reset_daily: bool = True

    # ── Momentum / Acceleration ─────────────────────────────────
    velocity_window: int = 1          # diff period for velocity
    slope_window: int = 20            # linear regression window
    curvature_window: int = 10        # second derivative window

    # ── Realized volatility ─────────────────────────────────────
    rv_period: int = 20
    rv_compression_threshold: float = 0.5  # fraction of recent avg

    # ── Turning Score weights (must sum to 100) ─────────────────
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "acceleration": 25,
        "ofi": 20,
        "curvature": 15,
        "slope": 15,
        "vwap": 10,
        "volume_delta": 5,
        "rv": 5,
        "gex": 5,
    })
    score_threshold: float = 60.0     # turning score threshold (80 when GEX available)

    # ── Early entry ─────────────────────────────────────────────
    early_entry_atr_ratio: float = 0.2  # within 0.2 ATR of swing high/low

    # ── Risk management ─────────────────────────────────────────
    risk_per_trade: float = 0.01      # 1% account risk per trade
    max_position_size: int = 100      # max shares/contracts
    initial_stop_atr: float = 1.5     # initial stop = 1.5 ATR
    trailing_stop_atr: float = 2.0    # trailing stop = 2.0 ATR
    break_even_r: float = 1.0         # move to breakeven at 1R
    take_profit_r: float = 2.0        # take profit at 2R
    max_daily_loss_pct: float = 0.03  # 3% daily loss circuit breaker
    max_consecutive_losses: int = 3   # stop after 3 consecutive losses

    # ── Backtest ────────────────────────────────────────────────
    initial_capital: float = 100000.0
    commission_per_trade: float = 0.0  # Schwab zero commission for equities
    slippage_ticks: float = 0.01       # simulated slippage

    # ── Visualization ───────────────────────────────────────────
    chart_height: int = 800
    chart_width: int = 1400
    show_volume: bool = True
    show_orderflow: bool = True

    # ── Logging ─────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "logs/engine.log"

    def validate(self) -> List[str]:
        """Return a list of configuration errors (empty = valid)."""
        errors = []
        total_weight = sum(self.score_weights.values())
        if abs(total_weight - 100.0) > 0.01:
            errors.append(
                f"Score weights must sum to 100, got {total_weight}"
            )
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.05:
            errors.append("risk_per_trade should be between 0 and 0.05")
        if self.atr_period < 2:
            errors.append("atr_period must be >= 2")
        return errors


# ── Singleton config ────────────────────────────────────────────
_config: Config | None = None


def get_config() -> Config:
    """Return the global Config singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(cfg: Config) -> None:
    """Override the global config (e.g., from CLI or .env)."""
    global _config
    _config = cfg
