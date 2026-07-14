"""
Swing Low Elevated Backtest Engine — 底部抬高策略回测引擎.

Dedicated backtest engine for the SwingLowElevatedStrategy.

Flow:
  1. SwingLowElevatedStrategy generates BUY signals only at swing lows
     where the bottom is rising (当前低点 > 前两个拐点低点).
  2. For each bar:
     a. If flat and a BUY signal fires -> enter long.
     b. If in position -> check stop loss (1 ATR) and take profit (2 ATR).
  3. Track equity curve, trade log, and all signals (BUY + REJECT).

Only one position at a time. No pyramiding.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from indicators.atr import atr as calc_atr
from risk.position import PositionSizer
from strategy.swing_low_elevated import (
    SwingLowElevatedStrategy,
    ElevatedSwingSignal,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ElevatedTrade:
    """A completed round-trip trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    pnl: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str
    swing_low_price: float
    atr_at_entry: float
    prev_swing_lows: list


@dataclass
class ElevatedBacktestResult:
    """Full backtest results."""
    trades: List[ElevatedTrade]
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    metrics: dict
    buy_signals_df: pd.DataFrame
    all_signals_df: pd.DataFrame
    swing_lows_df: pd.DataFrame
    strategy: SwingLowElevatedStrategy


class SwingElevatedEngine:
    """
    Event-driven backtest engine for the Swing Low Elevated strategy.

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
    swing_window : int
        Bars on each side to confirm a swing low.
    atr_period : int
        ATR period.
    stop_atr : float
        Stop loss = entry - stop_atr * ATR.
    tp_atr : float
        Take profit = entry + tp_atr * ATR.
    require_n_prev_lows : int
        Number of previous swing lows the current must exceed (default 2).
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_per_trade: float = 0.01,
        max_position_size: int = 10000,
        commission: float = 0.0,
        slippage: float = 0.01,
        swing_window: int = 3,
        atr_period: int = 14,
        stop_atr: float = 1.0,
        tp_atr: float = 2.0,
        require_n_prev_lows: int = 2,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.commission = commission
        self.slippage = slippage
        self.swing_window = swing_window
        self.atr_period = atr_period
        self.stop_atr = stop_atr
        self.tp_atr = tp_atr
        self.require_n_prev_lows = require_n_prev_lows

    def run(self, df: pd.DataFrame) -> ElevatedBacktestResult:
        """
        Run the backtest on OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex.

        Returns
        -------
        ElevatedBacktestResult
        """
        logger.info("Starting Swing Low Elevated backtest on %d bars...", len(df))

        # ── Generate signals ────────────────────────────────────
        strategy = SwingLowElevatedStrategy(
            swing_window=self.swing_window,
            atr_period=self.atr_period,
            stop_atr_multiplier=self.stop_atr,
            tp_atr_multiplier=self.tp_atr,
            require_n_prev_lows=self.require_n_prev_lows,
        )
        buy_signals = strategy.generate_signals(df)
        buy_signals_df = strategy.buy_signals_to_dataframe()
        all_signals_df = strategy.all_signals_to_dataframe()
        swings_df = strategy.swing_detector.to_dataframe()

        # Build signal lookup: bar index -> signal
        signal_by_bar = {}
        for sig in buy_signals:
            if sig.timestamp in df.index:
                bar_idx = df.index.get_loc(sig.timestamp)
                signal_by_bar[bar_idx] = sig

        # Compute ATR for the full series
        atr_series = calc_atr(
            df["high"], df["low"], df["close"], period=self.atr_period
        )

        # Position sizer
        sizer = PositionSizer(
            risk_per_trade=self.risk_per_trade,
            max_position_size=self.max_position_size,
        )

        # ── Simulate bar by bar ─────────────────────────────────
        equity = self.initial_capital
        equity_curve = []
        trades: List[ElevatedTrade] = []
        position = None

        for i in range(len(df)):
            ts = df.index[i]
            row = df.iloc[i]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = atr_series.iloc[i] if i < len(atr_series) else np.nan

            if np.isnan(atr_val) or atr_val <= 0:
                equity_curve.append(
                    equity if position is None
                    else equity + (close - position["entry_price"]) * position["quantity"]
                )
                continue

            # ── Check exits first (stop / target) ───────────────
            if position is not None:
                exit_price = None
                exit_reason = ""

                # Stop loss hit?
                if low <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_reason = "Stop Loss (1 ATR)"

                # Take profit hit?
                elif high >= position["take_profit"]:
                    exit_price = position["take_profit"]
                    exit_reason = "Take Profit (2 ATR)"

                if exit_price is not None:
                    pnl = (exit_price - position["entry_price"]) * position["quantity"]
                    pnl -= self.commission * 2

                    risk_amount = abs(
                        position["entry_price"] - position["stop_loss"]
                    ) * position["quantity"]
                    r_mult = pnl / risk_amount if risk_amount > 0 else 0

                    trades.append(ElevatedTrade(
                        entry_time=position["entry_time"],
                        exit_time=ts,
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        stop_loss=position["stop_loss"],
                        take_profit=position["take_profit"],
                        quantity=position["quantity"],
                        pnl=pnl,
                        pnl_pct=pnl / (position["entry_price"] * position["quantity"])
                            if position["entry_price"] > 0 else 0,
                        r_multiple=r_mult,
                        exit_reason=exit_reason,
                        swing_low_price=position["swing_low_price"],
                        atr_at_entry=position["atr"],
                        prev_swing_lows=position.get("prev_swing_lows", []),
                    ))

                    equity += pnl
                    position = None

            # ── Check entries ────────────────────────────────────
            if position is None and i in signal_by_bar:
                sig = signal_by_bar[i]
                entry_price = close

                # Apply slippage
                fill_price = entry_price + self.slippage

                # Calculate stop and target
                stop_loss = fill_price - self.stop_atr * atr_val
                take_profit = fill_price + self.tp_atr * atr_val

                # Size position based on risk
                ps = sizer.calculate(equity, fill_price, stop_loss)
                if ps.shares > 0:
                    position = {
                        "entry_time": ts,
                        "entry_price": fill_price,
                        "quantity": ps.shares,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "swing_low_price": sig.swing_low_price,
                        "atr": atr_val,
                        "prev_swing_lows": sig.prev_swing_lows,
                    }

                    logger.debug(
                        "BUY @ %.2f  SL=%.2f  TP=%.2f  qty=%d  "
                        "swing_low=%.2f  prev_lows=%s",
                        fill_price, stop_loss, take_profit, ps.shares,
                        sig.swing_low_price, sig.prev_swing_lows,
                    )

            # ── Update equity (mark-to-market) ──────────────────
            if position is not None:
                unrealized = (close - position["entry_price"]) * position["quantity"]
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
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "r_multiple": t.r_multiple,
                "exit_reason": t.exit_reason,
                "swing_low_price": t.swing_low_price,
                "atr_at_entry": t.atr_at_entry,
                "prev_swing_lows": str(t.prev_swing_lows),
            }
            for t in trades
        ])

        metrics = self._compute_metrics(trades, equity_series)

        logger.info(
            "Swing Elevated backtest complete: %d trades, "
            "final equity=%.2f, return=%.2f%%",
            len(trades), equity_series.iloc[-1],
            (equity_series.iloc[-1] / self.initial_capital - 1) * 100,
        )

        return ElevatedBacktestResult(
            trades=trades,
            equity_curve=equity_series,
            trade_log=trade_log,
            metrics=metrics,
            buy_signals_df=buy_signals_df,
            all_signals_df=all_signals_df,
            swing_lows_df=swings_df,
            strategy=strategy,
        )

    def _compute_metrics(
        self, trades: List[ElevatedTrade], equity: pd.Series
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
        }
