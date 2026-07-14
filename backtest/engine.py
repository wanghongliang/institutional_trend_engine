"""
Backtest engine.

Simulates the full trading pipeline on historical data:
  1. Run SignalEngine to generate signals
  2. For each bar, process signals and manage positions
  3. Apply stop-loss and take-profit rules
  4. Track equity curve and trade log

The engine is designed to be deterministic: given the same data
and config, it always produces the same result.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from config import Config, SignalType, get_config
from execution.order_manager import OrderManager
from risk.money_management import MoneyManager
from risk.position import PositionSizer
from strategy.signal_engine import SignalEngine
from strategy.stoploss import StopLossManager, StopType
from strategy.take_profit import TakeProfitManager
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Trade:
    """A completed round-trip trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    direction: int           # +1 long, -1 short
    pnl: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str
    signal_score: float


@dataclass
class BacktestResult:
    """Full backtest results."""
    trades: List[Trade]
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    metrics: dict
    signals_df: pd.DataFrame
    scored_df: pd.DataFrame


class BacktestEngine:
    """
    Event-driven backtest engine.

    Usage
    -----
    >>> engine = BacktestEngine(initial_capital=100000)
    >>> result = engine.run(df)
    >>> print(result.metrics)
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.initial_capital = self.config.initial_capital
        self.commission = self.config.commission_per_trade
        self.slippage = self.config.slippage_ticks

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest on historical OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex.

        Returns
        -------
        BacktestResult
        """
        logger.info("Starting backtest on %d bars...", len(df))

        # ── Initialize components ───────────────────────────────
        signal_engine = SignalEngine(self.config)
        position_sizer = PositionSizer(
            risk_per_trade=self.config.risk_per_trade,
            max_position_size=self.config.max_position_size,
        )
        money_manager = MoneyManager(
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            max_consecutive_losses=self.config.max_consecutive_losses,
        )
        stop_manager = StopLossManager(
            initial_stop_atr=self.config.initial_stop_atr,
            trailing_stop_atr=self.config.trailing_stop_atr,
            break_even_r=self.config.break_even_r,
        )
        tp_manager = TakeProfitManager(
            take_profit_r=self.config.take_profit_r,
        )

        # ── Generate signals ────────────────────────────────────
        signals = signal_engine.run(df)
        signals_df = signal_engine.signals_df
        scored_df = signal_engine.scored_df

        # Build a signal lookup: timestamp → signal
        signal_map = {s.timestamp: s for s in signals}

        # ── Simulate bar by bar ─────────────────────────────────
        from indicators.atr import atr as calc_atr
        atr_series = calc_atr(df["high"], df["low"], df["close"], period=14)

        equity = self.initial_capital
        equity_curve = []
        trades: List[Trade] = []

        # Position state
        position = None  # dict with entry info
        stop_level = None
        tp_level = None

        for i, (ts, row) in enumerate(df.iterrows()):
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = atr_series.iloc[i] if i < len(atr_series) else np.nan
            if np.isnan(atr_val) or atr_val <= 0:
                equity_curve.append(equity)
                continue

            current_date = ts.date() if hasattr(ts, "date") else date.today()

            # ── Check exits first (stop / target) ───────────────
            if position is not None:
                # Update stop
                stop_level = stop_manager.update(stop_level, close, atr_val)

                exit_price = None
                exit_reason = ""

                # Stop loss hit?
                if stop_manager.is_stopped(stop_level, low if position["direction"] > 0 else high):
                    exit_price = stop_level.stop_price
                    exit_reason = f"Stop ({stop_level.stop_type.value})"

                # Take profit hit?
                elif tp_manager.should_take_full(tp_level, high if position["direction"] > 0 else low):
                    exit_price = tp_level.target_price
                    exit_reason = "Take profit"

                # Exit signal?
                elif ts in signal_map:
                    sig = signal_map[ts]
                    if position["direction"] > 0 and sig.signal_type in (SignalType.EXIT_LONG, SignalType.BREAKOUT_SELL):
                        exit_price = close
                        exit_reason = f"Signal: {sig.signal_type.value}"
                    elif position["direction"] < 0 and sig.signal_type in (SignalType.EXIT_SHORT, SignalType.BREAKOUT_BUY):
                        exit_price = close
                        exit_reason = f"Signal: {sig.signal_type.value}"

                if exit_price is not None:
                    # Close position
                    if position["direction"] > 0:
                        pnl = (exit_price - position["entry_price"]) * position["quantity"]
                    else:
                        pnl = (position["entry_price"] - exit_price) * position["quantity"]
                    pnl -= self.commission * 2  # entry + exit commission

                    risk_amount = abs(position["entry_price"] - position["initial_stop"]) * position["quantity"]
                    r_mult = pnl / risk_amount if risk_amount > 0 else 0

                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=ts,
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        quantity=position["quantity"],
                        direction=position["direction"],
                        pnl=pnl,
                        pnl_pct=pnl / (position["entry_price"] * position["quantity"]) if position["entry_price"] > 0 else 0,
                        r_multiple=r_mult,
                        exit_reason=exit_reason,
                        signal_score=position.get("score", 0),
                    ))

                    equity += pnl
                    money_manager.record_trade(current_date, equity, pnl, pnl > 0)
                    position = None
                    stop_level = None
                    tp_level = None

            # ── Check entries ────────────────────────────────────
            if position is None and ts in signal_map:
                sig = signal_map[ts]

                can_trade, reason = money_manager.can_trade(current_date, equity)
                if not can_trade:
                    logger.debug("Trade blocked: %s", reason)
                else:
                    direction = 0
                    if sig.signal_type in (SignalType.EARLY_BUY, SignalType.BREAKOUT_BUY):
                        direction = 1
                    elif sig.signal_type in (SignalType.EARLY_SELL, SignalType.BREAKOUT_SELL):
                        direction = -1

                    if direction != 0:
                        # Calculate stop
                        entry_price = close
                        if direction > 0:
                            stop_price = entry_price - self.config.initial_stop_atr * atr_val
                        else:
                            stop_price = entry_price + self.config.initial_stop_atr * atr_val

                        # Size position
                        ps = position_sizer.calculate(equity, entry_price, stop_price)
                        if ps.shares > 0:
                            # Apply slippage
                            fill_price = entry_price + (self.slippage if direction > 0 else -self.slippage)

                            position = {
                                "entry_time": ts,
                                "entry_price": fill_price,
                                "quantity": ps.shares,
                                "direction": direction,
                                "initial_stop": stop_price,
                                "score": sig.score,
                            }

                            stop_level = stop_manager.initialize(
                                fill_price, atr_val, direction
                            )
                            # Override with our stop
                            stop_level.stop_price = stop_price

                            tp_level = tp_manager.initialize(
                                fill_price, stop_price, direction
                            )

                            logger.debug(
                                "Entry %s @ %.2f, stop=%.2f, qty=%d, score=%.0f",
                                "LONG" if direction > 0 else "SHORT",
                                fill_price, stop_price, ps.shares, sig.score
                            )

            # ── Update equity (mark-to-market) ──────────────────
            if position is not None:
                unrealized = 0
                if position["direction"] > 0:
                    unrealized = (close - position["entry_price"]) * position["quantity"]
                else:
                    unrealized = (position["entry_price"] - close) * position["quantity"]
                equity_curve.append(equity + unrealized)
            else:
                equity_curve.append(equity)

        # ── Build results ───────────────────────────────────────
        equity_series = pd.Series(equity_curve, index=df.index, name="equity")

        trade_log = pd.DataFrame([
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "direction": t.direction,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "r_multiple": t.r_multiple,
                "exit_reason": t.exit_reason,
                "signal_score": t.signal_score,
            }
            for t in trades
        ])

        metrics = self._compute_metrics(trades, equity_series)

        logger.info(
            "Backtest complete: %d trades, final equity=%.2f, return=%.2f%%",
            len(trades), equity_series.iloc[-1],
            (equity_series.iloc[-1] / self.initial_capital - 1) * 100
        )

        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            trade_log=trade_log,
            metrics=metrics,
            signals_df=signals_df,
            scored_df=scored_df,
        )

    def _compute_metrics(self, trades: List[Trade], equity: pd.Series) -> dict:
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
            }

        pnls = np.array([t.pnl for t in trades])
        r_multiples = np.array([t.r_multiple for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        total_return = (equity.iloc[-1] / self.initial_capital - 1) * 100

        # Max drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min() * 100

        # Sharpe ratio (simplified: per-trade, not annualized)
        if len(pnls) > 1 and pnls.std() > 0:
            sharpe = pnls.mean() / pnls.std() * np.sqrt(252)
        else:
            sharpe = 0

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

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
        }
