"""
Momentum Acceleration Backtest Engine.

Dedicated backtest engine for the MomentumAccelerationStrategy.

Flow:
  1. Strategy generates LONG/SHORT signals at momentum + acceleration
     breakouts.
  2. For each bar:
     a. If flat and a signal fires -> enter position immediately.
     b. If in position -> check stop loss, take profit, and time-based
        exit (max_holding_bars).
  3. Track equity curve, trade log, and signals.

Supports both LONG and SHORT positions.
Only one position at a time. No pyramiding.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from indicators.atr import atr as calc_atr
from indicators.momentum import velocity, acceleration
from risk.position import PositionSizer
from strategy.momentum_acceleration import (
    MomentumAccelerationStrategy,
    MomentumSignal,
    ConsolidationZone,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MomentumTrade:
    """A completed round-trip trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str            # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    pnl: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str          # "Stop Loss", "Take Profit", "Time Exit"
    holding_bars: int
    atr_at_entry: float
    velocity_at_entry: float
    acceleration_at_entry: float


@dataclass
class MomentumBacktestResult:
    """Full backtest results."""
    trades: List[MomentumTrade]
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    metrics: dict
    signals_df: pd.DataFrame
    zones_df: pd.DataFrame
    velocity_series: pd.Series
    acceleration_series: pd.Series
    strategy: MomentumAccelerationStrategy


class MomentumBacktestEngine:
    """
    Event-driven backtest engine for the Momentum Acceleration strategy.

    Parameters
    ----------
    initial_capital : float
        Starting account equity.
    risk_per_trade : float
        Fraction of equity risked per trade (default 1%).
    max_position_size : int
        Maximum shares per trade.
    commission : float
        Per-trade commission.
    slippage : float
        Simulated slippage per share.
    stop_atr : float
        Stop loss = stop_atr × ATR (default 0.5 — very tight).
    tp_atr : float
        Take profit = tp_atr × ATR (default 1.0 — 2:1 R:R).
    max_holding_bars : int
        Force exit after this many bars (default 20).
    min_holding_bars : int
        Minimum holding bars before TP can trigger (default 2).
        Stop loss always triggers immediately.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_per_trade: float = 0.01,
        max_position_size: int = 10000,
        commission: float = 0.0,
        slippage: float = 0.01,
        stop_atr: float = 0.5,
        tp_atr: float = 1.0,
        max_holding_bars: int = 20,
        min_holding_bars: int = 2,
        # Strategy parameters
        extrema_order: int = 5,
        consolidation_window: int = 20,
        consolidation_range_max: float = 0.7,
        velocity_min_atr: float = 0.3,
        accel_confirm_bars: int = 3,
        accel_min_atr: float = 0.1,
        atr_period: int = 14,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.commission = commission
        self.slippage = slippage
        self.stop_atr = stop_atr
        self.tp_atr = tp_atr
        self.max_holding_bars = max_holding_bars
        self.min_holding_bars = min_holding_bars

        # Strategy parameters
        self.extrema_order = extrema_order
        self.consolidation_window = consolidation_window
        self.consolidation_range_max = consolidation_range_max
        self.velocity_min_atr = velocity_min_atr
        self.accel_confirm_bars = accel_confirm_bars
        self.accel_min_atr = accel_min_atr
        self.atr_period = atr_period

    def run(self, df: pd.DataFrame) -> MomentumBacktestResult:
        """Run the backtest on OHLCV data."""
        logger.info("Starting Momentum Acceleration backtest on %d bars...", len(df))

        # ── Generate signals ────────────────────────────────────
        strategy = MomentumAccelerationStrategy(
            extrema_order=self.extrema_order,
            consolidation_window=self.consolidation_window,
            consolidation_range_max=self.consolidation_range_max,
            velocity_min_atr=self.velocity_min_atr,
            accel_confirm_bars=self.accel_confirm_bars,
            accel_min_atr=self.accel_min_atr,
            stop_atr_multiplier=self.stop_atr,
            tp_atr_multiplier=self.tp_atr,
            max_holding_bars=self.max_holding_bars,
            min_holding_bars=self.min_holding_bars,
            atr_period=self.atr_period,
        )
        signals = strategy.generate_signals(df)
        signals_df = strategy.signals_to_dataframe()
        zones_df = strategy.zones_to_dataframe()

        # Build signal lookup: bar index -> signal
        signal_by_bar = {}
        for sig in signals:
            if sig.timestamp in df.index:
                bar_idx = df.index.get_loc(sig.timestamp)
                signal_by_bar[bar_idx] = sig

        # Compute indicators for the full series
        atr_series = calc_atr(
            df["high"], df["low"], df["close"], period=self.atr_period
        )
        vel_series = velocity(df["close"], window=1)
        accel_series = acceleration(df["close"], window=1)

        # Position sizer
        sizer = PositionSizer(
            risk_per_trade=self.risk_per_trade,
            max_position_size=self.max_position_size,
        )

        # ── Simulate bar by bar ─────────────────────────────────
        equity = self.initial_capital
        equity_curve = []
        trades: List[MomentumTrade] = []
        position = None  # dict with entry info

        for i in range(len(df)):
            ts = df.index[i]
            row = df.iloc[i]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = atr_series.iloc[i] if i < len(atr_series) else np.nan

            if np.isnan(atr_val) or atr_val <= 0:
                if position is not None:
                    unrealized = self._unrealized_pnl(position, close)
                    equity_curve.append(equity + unrealized)
                else:
                    equity_curve.append(equity)
                continue

            # ── Check exits first (stop / target / time) ────────
            if position is not None:
                exit_price = None
                exit_reason = ""
                holding_bars = i - position["entry_bar"]

                # Stop loss always triggers (even within min_holding_bars)
                if position["direction"] == "LONG":
                    if low <= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "Stop Loss"
                    elif (holding_bars >= self.min_holding_bars
                          and high >= position["take_profit"]):
                        exit_price = position["take_profit"]
                        exit_reason = "Take Profit"
                else:  # SHORT
                    if high >= position["stop_loss"]:
                        exit_price = position["stop_loss"]
                        exit_reason = "Stop Loss"
                    elif (holding_bars >= self.min_holding_bars
                          and low <= position["take_profit"]):
                        exit_price = position["take_profit"]
                        exit_reason = "Take Profit"

                # Time-based exit (max holding period)
                if exit_price is None and holding_bars >= self.max_holding_bars:
                    exit_price = close
                    exit_reason = "Time Exit"

                if exit_price is not None:
                    pnl = self._calculate_pnl(
                        position, exit_price
                    )
                    pnl -= self.commission * 2

                    risk_amount = abs(
                        position["entry_price"] - position["stop_loss"]
                    ) * position["quantity"]
                    r_mult = pnl / risk_amount if risk_amount > 0 else 0

                    trades.append(MomentumTrade(
                        entry_time=position["entry_time"],
                        exit_time=ts,
                        direction=position["direction"],
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        stop_loss=position["stop_loss"],
                        take_profit=position["take_profit"],
                        quantity=position["quantity"],
                        pnl=pnl,
                        pnl_pct=pnl / (
                            position["entry_price"] * position["quantity"]
                        ) if position["entry_price"] > 0 else 0,
                        r_multiple=r_mult,
                        exit_reason=exit_reason,
                        holding_bars=holding_bars,
                        atr_at_entry=position["atr"],
                        velocity_at_entry=position["velocity"],
                        acceleration_at_entry=position["acceleration"],
                    ))

                    equity += pnl
                    position = None

            # ── Check entries ────────────────────────────────────
            if position is None and i in signal_by_bar:
                sig = signal_by_bar[i]
                entry_price = close

                # Apply slippage
                if sig.direction == "LONG":
                    fill_price = entry_price + self.slippage
                    stop_loss = fill_price - self.stop_atr * atr_val
                    take_profit = fill_price + self.tp_atr * atr_val
                else:  # SHORT
                    fill_price = entry_price - self.slippage
                    stop_loss = fill_price + self.stop_atr * atr_val
                    take_profit = fill_price - self.tp_atr * atr_val

                # Size position based on risk
                ps = sizer.calculate(equity, fill_price, stop_loss)
                if ps.shares > 0:
                    position = {
                        "entry_time": ts,
                        "entry_bar": i,
                        "direction": sig.direction,
                        "entry_price": fill_price,
                        "quantity": ps.shares,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "atr": atr_val,
                        "velocity": sig.velocity,
                        "acceleration": sig.acceleration,
                    }

                    logger.debug(
                        "%s @ %.2f  SL=%.2f  TP=%.2f  qty=%d  "
                        "vel=%.4f  accel=%.4f",
                        sig.direction, fill_price, stop_loss,
                        take_profit, ps.shares,
                        sig.velocity, sig.acceleration,
                    )

            # ── Update equity (mark-to-market) ──────────────────
            if position is not None:
                unrealized = self._unrealized_pnl(position, close)
                equity_curve.append(equity + unrealized)
            else:
                equity_curve.append(equity)

        # ── Build results ───────────────────────────────────────
        equity_series = pd.Series(equity_curve, index=df.index, name="equity")

        trade_log = pd.DataFrame([
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "r_multiple": t.r_multiple,
                "exit_reason": t.exit_reason,
                "holding_bars": t.holding_bars,
                "atr_at_entry": t.atr_at_entry,
                "velocity_at_entry": t.velocity_at_entry,
                "acceleration_at_entry": t.acceleration_at_entry,
            }
            for t in trades
        ])

        metrics = self._compute_metrics(trades, equity_series)

        logger.info(
            "Momentum backtest complete: %d trades, "
            "final equity=%.2f, return=%.2f%%",
            len(trades), equity_series.iloc[-1],
            (equity_series.iloc[-1] / self.initial_capital - 1) * 100,
        )

        return MomentumBacktestResult(
            trades=trades,
            equity_curve=equity_series,
            trade_log=trade_log,
            metrics=metrics,
            signals_df=signals_df,
            zones_df=zones_df,
            velocity_series=vel_series,
            acceleration_series=accel_series,
            strategy=strategy,
        )

    # ── Helpers ─────────────────────────────────────────────────

    def _unrealized_pnl(self, position: dict, current_price: float) -> float:
        """Calculate unrealized P&L for an open position."""
        if position["direction"] == "LONG":
            return (current_price - position["entry_price"]) * position["quantity"]
        else:  # SHORT
            return (position["entry_price"] - current_price) * position["quantity"]

    def _calculate_pnl(self, position: dict, exit_price: float) -> float:
        """Calculate realized P&L for a closed position."""
        if position["direction"] == "LONG":
            return (exit_price - position["entry_price"]) * position["quantity"]
        else:  # SHORT
            return (position["entry_price"] - exit_price) * position["quantity"]

    def _compute_metrics(
        self, trades: List[MomentumTrade], equity: pd.Series
    ) -> dict:
        """Compute performance metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_return_pct": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
                "avg_r_multiple": 0,
                "max_r_multiple": 0,
                "min_r_multiple": 0,
                "profit_factor": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "final_equity": equity.iloc[-1] if len(equity) > 0 else self.initial_capital,
                "initial_capital": self.initial_capital,
                "stop_loss_exits": 0,
                "take_profit_exits": 0,
                "time_exits": 0,
                "long_trades": 0,
                "short_trades": 0,
                "avg_holding_bars": 0,
            }

        pnls = np.array([t.pnl for t in trades])
        r_multiples = np.array([t.r_multiple for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        holding_bars = np.array([t.holding_bars for t in trades])

        total_return = (equity.iloc[-1] / self.initial_capital - 1) * 100

        # Max drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min() * 100

        # Sharpe ratio (per-trade, annualized)
        if len(pnls) > 1 and pnls.std() > 0:
            sharpe = pnls.mean() / pnls.std() * np.sqrt(252)
        else:
            sharpe = 0

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Exit reason counts
        sl_exits = sum(1 for t in trades if "Stop" in t.exit_reason)
        tp_exits = sum(1 for t in trades if "Take" in t.exit_reason)
        time_exits = sum(1 for t in trades if "Time" in t.exit_reason)

        # Direction counts
        long_trades = sum(1 for t in trades if t.direction == "LONG")
        short_trades = sum(1 for t in trades if t.direction == "SHORT")

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "total_return_pct": total_return,
            "total_pnl": pnls.sum(),
            "avg_pnl": pnls.mean(),
            "avg_r_multiple": r_multiples.mean(),
            "max_r_multiple": r_multiples.max(),
            "min_r_multiple": r_multiples.min(),
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "avg_win": wins.mean() if len(wins) > 0 else 0,
            "avg_loss": losses.mean() if len(losses) > 0 else 0,
            "final_equity": equity.iloc[-1],
            "initial_capital": self.initial_capital,
            "stop_loss_exits": sl_exits,
            "take_profit_exits": tp_exits,
            "time_exits": time_exits,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "avg_holding_bars": holding_bars.mean(),
        }
