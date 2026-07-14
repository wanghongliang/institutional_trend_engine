"""
Swing Low Strategy Visualization.

Produces a multi-panel chart showing:
  1. Price with Swing Low markers and BUY signals
  2. Stop Loss / Take Profit levels for each trade
  3. Equity Curve
  4. R-Multiple distribution

Uses matplotlib for static rendering.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Color palette (light theme, Chinese convention: red=up, green=down) ──
COLOR_BG = "#FAFAFA"
COLOR_GRID = "#E0E0E0"
COLOR_PRICE = "#333333"
COLOR_UP = "#E53935"
COLOR_DOWN = "#43A047"
COLOR_BUY = "#1565C0"
COLOR_STOP = "#FF6F00"
COLOR_TP = "#00838F"
COLOR_SWING_LOW = "#7B1FA2"
COLOR_EQUITY = "#1565C0"


def plot_swing_chart(
    df: pd.DataFrame,
    swings_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    trade_log: pd.DataFrame,
    equity_curve: pd.Series,
    initial_capital: float = 100000.0,
    save_path: str = "output/swing_chart.png",
    title: str = "Swing Low Reversal Strategy",
    last_n_bars: int = 500,
) -> str:
    """
    Generate a multi-panel chart for the swing low strategy.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    swings_df : pd.DataFrame
        Swing points from SwingDetector.to_dataframe().
    signals_df : pd.DataFrame
        Signals from SwingLowStrategy.signals_to_dataframe().
    trade_log : pd.DataFrame
        Completed trades from the backtest.
    equity_curve : pd.Series
        Equity curve aligned to df index.
    initial_capital : float
        Starting capital for reference line.
    save_path : str
        Output file path.
    title : str
        Chart title.
    last_n_bars : int
        Show only the last N bars for readability.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    # Slice to last N bars
    df = df.iloc[-last_n_bars:].copy()

    # Filter swings to visible range
    if not swings_df.empty:
        start_ts = df.index[0]
        swings_visible = swings_df[swings_df["timestamp"] >= start_ts].copy()
    else:
        swings_visible = swings_df

    # Filter signals to visible range
    if not signals_df.empty:
        start_ts = df.index[0]
        signals_visible = signals_df[signals_df.index >= start_ts].copy()
    else:
        signals_visible = signals_df

    # Filter trades to visible range
    if not trade_log.empty:
        trades_visible = trade_log[trade_log["entry_time"] >= start_ts].copy()
    else:
        trades_visible = trade_log

    # ── Create figure with 3 panels ───────────────────────────
    fig, axes = plt.subplots(
        3, 1,
        figsize=(16, 12),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        facecolor=COLOR_BG,
    )

    # ── Panel 1: Price + Swing Lows + BUY signals + SL/TP ─────
    ax_price = axes[0]
    ax_price.set_facecolor(COLOR_BG)

    # Plot close price
    ax_price.plot(df.index, df["close"], color=COLOR_PRICE,
                  linewidth=1, label="Close Price")

    # Plot swing lows
    if not swings_visible.empty:
        swing_lows = swings_visible[swings_visible["type"] == "low"]
        for _, sw in swing_lows.iterrows():
            ax_price.scatter(sw["timestamp"], sw["price"],
                             color=COLOR_SWING_LOW, marker="v",
                             s=50, zorder=4, alpha=0.7)

    # Plot BUY signals and SL/TP lines
    if not signals_visible.empty:
        for ts, row in signals_visible.iterrows():
            # BUY marker
            ax_price.scatter(ts, row["price"], color=COLOR_BUY,
                             marker="^", s=120, zorder=6,
                             edgecolors="black", linewidths=1)

    # Plot SL/TP lines for each trade
    if not trades_visible.empty:
        for _, trade in trades_visible.iterrows():
            entry_ts = trade["entry_time"]
            exit_ts = trade["exit_time"]

            # Stop loss line (orange)
            ax_price.hlines(trade["stop_loss"], entry_ts, exit_ts,
                            colors=COLOR_STOP, linestyles="--",
                            linewidth=1, alpha=0.6)

            # Take profit line (teal)
            ax_price.hlines(trade["take_profit"], entry_ts, exit_ts,
                            colors=COLOR_TP, linestyles="--",
                            linewidth=1, alpha=0.6)

            # Entry marker
            ax_price.scatter(entry_ts, trade["entry_price"],
                             color=COLOR_BUY, marker="o",
                             s=60, zorder=5, edgecolors="black")

            # Exit marker
            exit_color = COLOR_UP if trade["pnl"] > 0 else COLOR_DOWN
            exit_marker = "s" if trade["pnl"] > 0 else "X"
            ax_price.scatter(exit_ts, trade["exit_price"],
                             color=exit_color, marker=exit_marker,
                             s=60, zorder=5, edgecolors="black")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=COLOR_PRICE, linewidth=1, label="Close Price"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor=COLOR_SWING_LOW,
                   markersize=8, label="Swing Low"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_BUY,
                   markersize=10, markeredgecolor="black", label="BUY Signal"),
        plt.Line2D([0], [0], color=COLOR_STOP, linestyle="--", linewidth=1, label="Stop Loss (1 ATR)"),
        plt.Line2D([0], [0], color=COLOR_TP, linestyle="--", linewidth=1, label="Take Profit (2 ATR)"),
    ]
    ax_price.legend(handles=legend_elements, loc="upper left", fontsize=8)

    ax_price.set_title(title, fontsize=14, fontweight="bold")
    ax_price.grid(True, color=COLOR_GRID, alpha=0.5)

    # ── Panel 2: Equity Curve ─────────────────────────────────
    ax_eq = axes[1]
    ax_eq.set_facecolor(COLOR_BG)

    eq_slice = equity_curve.iloc[-last_n_bars:]
    ax_eq.plot(eq_slice.index, eq_slice.values, color=COLOR_EQUITY, linewidth=1.5)
    ax_eq.fill_between(eq_slice.index, eq_slice.values, initial_capital,
                       where=eq_slice.values >= initial_capital,
                       color=COLOR_UP, alpha=0.1, label="Profit")
    ax_eq.fill_between(eq_slice.index, eq_slice.values, initial_capital,
                       where=eq_slice.values < initial_capital,
                       color=COLOR_DOWN, alpha=0.1, label="Loss")
    ax_eq.axhline(initial_capital, color="#666", linestyle="-", linewidth=0.5)
    ax_eq.set_title("Equity Curve", fontsize=12)
    ax_eq.legend(loc="upper left", fontsize=8)
    ax_eq.grid(True, color=COLOR_GRID, alpha=0.5)

    # ── Panel 3: R-Multiple per trade ─────────────────────────
    ax_r = axes[2]
    ax_r.set_facecolor(COLOR_BG)

    if not trade_log.empty:
        r_values = trade_log["r_multiple"].values
        trade_nums = range(1, len(r_values) + 1)
        colors = [COLOR_UP if r > 0 else COLOR_DOWN for r in r_values]
        ax_r.bar(trade_nums, r_values, color=colors, alpha=0.7, width=0.6)
        ax_r.axhline(0, color="#333", linewidth=0.5)
        ax_r.axhline(2, color=COLOR_TP, linestyle="--", linewidth=0.8, label="TP = 2R")
        ax_r.axhline(-1, color=COLOR_STOP, linestyle="--", linewidth=0.8, label="SL = -1R")
        ax_r.legend(loc="upper right", fontsize=8)

    ax_r.set_title("R-Multiple per Trade", fontsize=12)
    ax_r.set_xlabel("Trade #")
    ax_r.grid(True, color=COLOR_GRID, alpha=0.5)

    # Format x-axis
    for ax in axes[:2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    logger.info("Swing chart saved to %s", save_path)
    return save_path


def generate_swing_report(
    result,
    symbol: str = "",
    strategy_params: dict = None,
) -> str:
    """
    Generate a human-readable performance report for the swing low strategy.

    Parameters
    ----------
    result : SwingBacktestResult
        Backtest results.
    symbol : str
        Trading symbol.
    strategy_params : dict
        Strategy parameters for the report header.

    Returns
    -------
    str
        Multi-line report text.
    """
    m = result.metrics
    lines = []
    lines.append("=" * 60)
    lines.append("  SWING LOW REVERSAL STRATEGY — BACKTEST REPORT")
    lines.append("=" * 60)
    if symbol:
        lines.append(f"  Symbol: {symbol}")
    if strategy_params:
        lines.append(f"  Swing Window:     {strategy_params.get('swing_window', 3)}")
        lines.append(f"  ATR Period:       {strategy_params.get('atr_period', 14)}")
        lines.append(f"  Stop Loss:        {strategy_params.get('stop_atr', 1.0)} ATR")
        lines.append(f"  Take Profit:      {strategy_params.get('tp_atr', 2.0)} ATR")
        lines.append(f"  Risk per Trade:   {strategy_params.get('risk_per_trade', 0.01)*100:.1f}%")
    lines.append("")

    # ── Overview ──
    lines.append("-- OVERVIEW --")
    lines.append(f"  Initial Capital:     ${m.get('initial_capital', 0):>12,.2f}")
    lines.append(f"  Final Equity:        ${m.get('final_equity', 0):>12,.2f}")
    lines.append(f"  Total Return:         {m.get('total_return_pct', 0):>11.2f}%")
    lines.append(f"  Max Drawdown:         {m.get('max_drawdown_pct', 0):>11.2f}%")
    lines.append(f"  Sharpe Ratio:         {m.get('sharpe_ratio', 0):>11.2f}")
    lines.append("")

    # ── Trade statistics ──
    lines.append("-- TRADE STATISTICS --")
    lines.append(f"  Total Trades:         {m.get('total_trades', 0):>11d}")
    lines.append(f"  Winning Trades:       {m.get('winning_trades', 0):>11d}")
    lines.append(f"  Losing Trades:        {m.get('losing_trades', 0):>11d}")
    lines.append(f"  Win Rate:             {m.get('win_rate', 0):>11.1f}%")
    lines.append(f"  Profit Factor:        {m.get('profit_factor', 0):>11.2f}")
    lines.append(f"  Stop Loss Exits:      {m.get('stop_loss_exits', 0):>11d}")
    lines.append(f"  Take Profit Exits:    {m.get('take_profit_exits', 0):>11d}")
    lines.append("")

    # ── P&L breakdown ──
    lines.append("-- P&L BREAKDOWN --")
    lines.append(f"  Total P&L:           ${m.get('total_pnl', 0):>12,.2f}")
    lines.append(f"  Average P&L:         ${m.get('avg_pnl', 0):>12,.2f}")
    lines.append(f"  Average Win:         ${m.get('avg_win', 0):>12,.2f}")
    lines.append(f"  Average Loss:        ${m.get('avg_loss', 0):>12,.2f}")
    lines.append("")

    # ── R-multiple ──
    lines.append("-- R-MULTIPLE --")
    lines.append(f"  Average R:            {m.get('avg_r_multiple', 0):>11.2f}")
    lines.append(f"  Max R:                {m.get('max_r_multiple', 0):>11.2f}")
    lines.append(f"  Min R:                {m.get('min_r_multiple', 0):>11.2f}")
    lines.append("")

    # ── Trade details ──
    if not result.trade_log.empty:
        lines.append("-- TRADE LOG (last 15) --")
        recent = result.trade_log.tail(15)
        for _, t in recent.iterrows():
            lines.append(
                f"  {t['entry_time']}  BUY  "
                f"entry={t['entry_price']:.2f}  SL={t['stop_loss']:.2f}  "
                f"TP={t['take_profit']:.2f}  exit={t['exit_price']:.2f}  "
                f"P&L=${t['pnl']:.2f}  R={t['r_multiple']:.2f}  "
                f"[{t['exit_reason']}]"
            )
        lines.append("")

    # ── Signals summary ──
    if not result.signals_df.empty:
        lines.append("-- SIGNALS SUMMARY --")
        lines.append(f"  Total BUY signals:   {len(result.signals_df):>11d}")
        swing_low_count = len(result.swing_lows_df[result.swing_lows_df["type"] == "low"]) \
            if not result.swing_lows_df.empty else 0
        lines.append(f"  Swing lows detected: {swing_low_count:>11d}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
