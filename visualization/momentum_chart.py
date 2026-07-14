"""
Momentum Acceleration Strategy Visualization.

Produces a 4-panel chart:
  1. Price with consolidation zones (shaded), entry/exit markers,
     SL/TP lines for each trade.
  2. Velocity (first difference of close).
  3. Acceleration (second difference of close).
  4. Equity Curve.

Uses matplotlib for static rendering.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Color palette (light theme) ──
COLOR_BG = "#FAFAFA"
COLOR_GRID = "#E0E0E0"
COLOR_PRICE = "#333333"
COLOR_UP = "#E53935"         # red = up (Chinese convention)
COLOR_DOWN = "#43A047"       # green = down
COLOR_LONG = "#1565C0"       # blue — long entry
COLOR_SHORT = "#E65100"      # deep orange — short entry
COLOR_STOP = "#FF6F00"
COLOR_TP = "#00838F"
COLOR_CONSOLIDATION = "#FFF9C4"  # light yellow — consolidation zone
COLOR_CONSOLIDATION_EDGE = "#FBC02D"
COLOR_VELOCITY = "#7B1FA2"
COLOR_ACCELERATION = "#00695C"
COLOR_EQUITY = "#1565C0"


def plot_momentum_chart(
    df: pd.DataFrame,
    zones_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    trade_log: pd.DataFrame,
    equity_curve: pd.Series,
    velocity_series: pd.Series,
    acceleration_series: pd.Series,
    initial_capital: float = 100000.0,
    save_path: str = "output/momentum_chart.png",
    title: str = "Momentum Acceleration Strategy",
    last_n_bars: int = 500,
) -> str:
    """
    Generate a 4-panel chart for the momentum acceleration strategy.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    zones_df : pd.DataFrame
        Consolidation zones from MomentumAccelerationStrategy.
    signals_df : pd.DataFrame
        Entry signals (LONG / SHORT).
    trade_log : pd.DataFrame
        Completed trades from the backtest.
    equity_curve : pd.Series
        Equity curve aligned to df index.
    velocity_series : pd.Series
        Price velocity (first difference).
    acceleration_series : pd.Series
        Price acceleration (second difference).
    initial_capital : float
        Starting capital.
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
    start_ts = df.index[0]
    vel_slice = velocity_series.iloc[-last_n_bars:]
    accel_slice = acceleration_series.iloc[-last_n_bars:]
    eq_slice = equity_curve.iloc[-last_n_bars:]

    # ── Filter data to visible range ──
    zones_visible = zones_df.copy()
    if not zones_visible.empty and "end_bar" in zones_visible.columns:
        # Convert bar indices to timestamps
        zones_visible = zones_visible[
            zones_visible["end_bar"] < len(velocity_series)
        ].copy()

    if not signals_df.empty:
        signals_visible = signals_df[signals_df.index >= start_ts].copy()
    else:
        signals_visible = signals_df

    if not trade_log.empty:
        trades_visible = trade_log[trade_log["entry_time"] >= start_ts].copy()
    else:
        trades_visible = trade_log

    # ── Create figure with 4 panels ───────────────────────────
    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 16),
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
        facecolor=COLOR_BG,
    )

    # ════════════════════════════════════════════════════════════
    # Panel 1: Price + Consolidation zones + Entry/Exit + SL/TP
    # ════════════════════════════════════════════════════════════
    ax_price = axes[0]
    ax_price.set_facecolor(COLOR_BG)

    # Plot close price
    ax_price.plot(df.index, df["close"], color=COLOR_PRICE,
                  linewidth=1, label="Close Price")

    # ── Shade consolidation zones ──
    if not zones_visible.empty:
        full_df_len = len(velocity_series)
        offset = full_df_len - last_n_bars
        for _, zone in zones_visible.iterrows():
            start_bar = int(zone["start_bar"])
            end_bar = int(zone["end_bar"])
            # Adjust for visible slice
            vis_start = start_bar - offset
            vis_end = end_bar - offset
            if vis_end < 0 or vis_start >= last_n_bars:
                continue
            vis_start = max(0, vis_start)
            vis_end = min(last_n_bars - 1, vis_end)

            if vis_end > vis_start:
                x_start = df.index[vis_start]
                x_end = df.index[vis_end]
                zone_high = zone["high"]
                zone_low = zone["low"]

                # Draw shaded rectangle
                rect = Rectangle(
                    (mdates.date2num(x_start), zone_low),
                    mdates.date2num(x_end) - mdates.date2num(x_start),
                    zone_high - zone_low,
                    facecolor=COLOR_CONSOLIDATION,
                    edgecolor=COLOR_CONSOLIDATION_EDGE,
                    linewidth=1,
                    alpha=0.5,
                    zorder=2,
                )
                ax_price.add_patch(rect)

                # Draw horizontal lines for zone boundaries
                ax_price.hlines(
                    zone_high, x_start, x_end,
                    colors=COLOR_CONSOLIDATION_EDGE,
                    linestyles=":", linewidth=0.8, alpha=0.6,
                )
                ax_price.hlines(
                    zone_low, x_start, x_end,
                    colors=COLOR_CONSOLIDATION_EDGE,
                    linestyles=":", linewidth=0.8, alpha=0.6,
                )

    # ── Plot entry/exit markers and SL/TP for each trade ──
    if not trades_visible.empty:
        for _, trade in trades_visible.iterrows():
            entry_ts = trade["entry_time"]
            exit_ts = trade["exit_time"]
            direction = trade["direction"]

            # Entry marker
            entry_color = COLOR_LONG if direction == "LONG" else COLOR_SHORT
            entry_marker = "^" if direction == "LONG" else "v"
            ax_price.scatter(entry_ts, trade["entry_price"],
                             color=entry_color, marker=entry_marker,
                             s=120, zorder=6, edgecolors="black",
                             linewidths=1)

            # Stop loss line
            ax_price.hlines(trade["stop_loss"], entry_ts, exit_ts,
                            colors=COLOR_STOP, linestyles="--",
                            linewidth=1, alpha=0.6)

            # Take profit line
            ax_price.hlines(trade["take_profit"], entry_ts, exit_ts,
                            colors=COLOR_TP, linestyles="--",
                            linewidth=1, alpha=0.6)

            # Exit marker
            exit_color = COLOR_UP if trade["pnl"] > 0 else COLOR_DOWN
            exit_marker = "s" if trade["pnl"] > 0 else "X"
            ax_price.scatter(exit_ts, trade["exit_price"],
                             color=exit_color, marker=exit_marker,
                             s=60, zorder=5, edgecolors="black")

    # ── Plot signal markers (even if no trade completed) ──
    if not signals_visible.empty:
        for ts, row in signals_visible.iterrows():
            if ts in trades_visible["entry_time"].values:
                continue  # already plotted as trade entry
            sig_color = COLOR_LONG if row["direction"] == "LONG" else COLOR_SHORT
            sig_marker = "^" if row["direction"] == "LONG" else "v"
            ax_price.scatter(ts, row["price"],
                             color=sig_color, marker=sig_marker,
                             s=80, zorder=5, edgecolors="black",
                             linewidths=0.5, alpha=0.6)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=COLOR_PRICE, linewidth=1, label="Close Price"),
        plt.Line2D([0], [0], color=COLOR_CONSOLIDATION_EDGE, linewidth=3,
                   alpha=0.5, label="Consolidation Zone"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_LONG,
                   markersize=10, markeredgecolor="black", label="LONG Entry"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor=COLOR_SHORT,
                   markersize=10, markeredgecolor="black", label="SHORT Entry"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_UP,
                   markersize=8, markeredgecolor="black", label="Win Exit"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor=COLOR_DOWN,
                   markersize=8, markeredgecolor="black", label="Loss Exit"),
        plt.Line2D([0], [0], color=COLOR_STOP, linestyle="--", linewidth=1,
                   label="Stop Loss (0.5 ATR)"),
        plt.Line2D([0], [0], color=COLOR_TP, linestyle="--", linewidth=1,
                   label="Take Profit (1.0 ATR)"),
    ]
    ax_price.legend(handles=legend_elements, loc="upper left", fontsize=7.5)

    ax_price.set_title(title, fontsize=14, fontweight="bold")
    ax_price.grid(True, color=COLOR_GRID, alpha=0.5)

    # ════════════════════════════════════════════════════════════
    # Panel 2: Velocity
    # ════════════════════════════════════════════════════════════
    ax_vel = axes[1]
    ax_vel.set_facecolor(COLOR_BG)

    ax_vel.plot(vel_slice.index, vel_slice.values,
                color=COLOR_VELOCITY, linewidth=1)
    ax_vel.fill_between(vel_slice.index, 0, vel_slice.values,
                        where=vel_slice.values >= 0,
                        color=COLOR_UP, alpha=0.15)
    ax_vel.fill_between(vel_slice.index, 0, vel_slice.values,
                        where=vel_slice.values < 0,
                        color=COLOR_DOWN, alpha=0.15)
    ax_vel.axhline(0, color="#666", linewidth=0.5)
    ax_vel.set_title("Price Velocity (dP/dt)", fontsize=12)
    ax_vel.grid(True, color=COLOR_GRID, alpha=0.5)

    # ════════════════════════════════════════════════════════════
    # Panel 3: Acceleration
    # ════════════════════════════════════════════════════════════
    ax_accel = axes[2]
    ax_accel.set_facecolor(COLOR_BG)

    ax_accel.plot(accel_slice.index, accel_slice.values,
                  color=COLOR_ACCELERATION, linewidth=1)
    ax_accel.fill_between(accel_slice.index, 0, accel_slice.values,
                          where=accel_slice.values >= 0,
                          color=COLOR_UP, alpha=0.15)
    ax_accel.fill_between(accel_slice.index, 0, accel_slice.values,
                          where=accel_slice.values < 0,
                          color=COLOR_DOWN, alpha=0.15)
    ax_accel.axhline(0, color="#666", linewidth=0.5)
    ax_accel.set_title("Price Acceleration (d2P/dt2)", fontsize=12)
    ax_accel.grid(True, color=COLOR_GRID, alpha=0.5)

    # ════════════════════════════════════════════════════════════
    # Panel 4: Equity Curve
    # ════════════════════════════════════════════════════════════
    ax_eq = axes[3]
    ax_eq.set_facecolor(COLOR_BG)

    ax_eq.plot(eq_slice.index, eq_slice.values,
               color=COLOR_EQUITY, linewidth=1.5)
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

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    logger.info("Momentum chart saved to %s", save_path)
    return save_path


def generate_momentum_report(
    result,
    symbol: str = "",
    strategy_params: dict = None,
) -> str:
    """
    Generate a human-readable performance report for the momentum
    acceleration strategy.
    """
    m = result.metrics
    lines = []
    lines.append("=" * 64)
    lines.append("  MOMENTUM ACCELERATION STRATEGY")
    lines.append("  (Momentum Breakout + Acceleration Trading)")
    lines.append("  Only catch the fastest, strongest main trend")
    lines.append("=" * 64)
    if symbol:
        lines.append(f"  Symbol: {symbol}")
    if strategy_params:
        lines.append(f"  Extrema Order:         {strategy_params.get('extrema_order', 5)}")
        lines.append(f"  Consolidation Window:  {strategy_params.get('consolidation_window', 20)}")
        lines.append(f"  Range Max:             {strategy_params.get('consolidation_range_max', 0.004)*100:.2f}%")
        lines.append(f"  Velocity Min (ATR):    {strategy_params.get('velocity_min_atr', 0.3)}")
        lines.append(f"  Accel Confirm Bars:    {strategy_params.get('accel_confirm_bars', 3)}")
        lines.append(f"  Accel Min (ATR):       {strategy_params.get('accel_min_atr', 0.1)}")
        lines.append(f"  Stop Loss:             {strategy_params.get('stop_atr', 0.5)} ATR")
        lines.append(f"  Take Profit:           {strategy_params.get('tp_atr', 1.0)} ATR")
        lines.append(f"  Risk:Reward:           1:2")
        lines.append(f"  Max Holding Bars:      {strategy_params.get('max_holding_bars', 20)}")
        lines.append(f"  Min Holding Bars:      {strategy_params.get('min_holding_bars', 2)}")
        lines.append(f"  Risk per Trade:        {strategy_params.get('risk_per_trade', 0.01)*100:.1f}%")
    lines.append("")

    # ── Signal stats ──
    lines.append("-- SIGNAL DETECTION --")
    n_zones = len(result.zones_df) if not result.zones_df.empty else 0
    n_signals = len(result.signals_df) if not result.signals_df.empty else 0
    n_long = len(result.signals_df[result.signals_df["direction"] == "LONG"]) \
        if not result.signals_df.empty else 0
    n_short = len(result.signals_df[result.signals_df["direction"] == "SHORT"]) \
        if not result.signals_df.empty else 0
    lines.append(f"  Consolidation Zones:  {n_zones:>10d}")
    lines.append(f"  Total Signals:        {n_signals:>10d}")
    lines.append(f"  LONG Signals:         {n_long:>10d}")
    lines.append(f"  SHORT Signals:        {n_short:>10d}")
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
    lines.append(f"  LONG Trades:          {m.get('long_trades', 0):>11d}")
    lines.append(f"  SHORT Trades:         {m.get('short_trades', 0):>11d}")
    lines.append(f"  Avg Holding Bars:     {m.get('avg_holding_bars', 0):>11.1f}")
    lines.append("")

    # ── Exit analysis ──
    lines.append("-- EXIT ANALYSIS --")
    lines.append(f"  Stop Loss Exits:      {m.get('stop_loss_exits', 0):>11d}")
    lines.append(f"  Take Profit Exits:    {m.get('take_profit_exits', 0):>11d}")
    lines.append(f"  Time Exits:           {m.get('time_exits', 0):>11d}")
    lines.append("")

    # ── P&L breakdown ──
    lines.append("-- P&L BREAKDOWN --")
    lines.append(f"  Total P&L:           ${m.get('total_pnl', 0):>12,.2f}")
    lines.append(f"  Average P&L:         ${m.get('avg_pnl', 0):>12,.2f}")
    lines.append(f"  Average Win:         ${m.get('avg_win', 0):>12,.2f}")
    lines.append(f"  Average Loss:        ${m.get('avg_loss', 0):>12,.2f}")
    lines.append("")

    # ── R-multiple ──
    lines.append("-- R-MULTIPLE (Risk:Reward = 1:2) --")
    lines.append(f"  Average R:            {m.get('avg_r_multiple', 0):>11.2f}")
    lines.append(f"  Max R:                {m.get('max_r_multiple', 0):>11.2f}")
    lines.append(f"  Min R:                {m.get('min_r_multiple', 0):>11.2f}")
    lines.append("")

    # ── Trade details ──
    if not result.trade_log.empty:
        lines.append("-- TRADE LOG (last 20) --")
        recent = result.trade_log.tail(20)
        for _, t in recent.iterrows():
            lines.append(
                f"  {t['entry_time']}  {t['direction']:5s}  "
                f"entry={t['entry_price']:.2f}  SL={t['stop_loss']:.2f}  "
                f"TP={t['take_profit']:.2f}  exit={t['exit_price']:.2f}  "
                f"P&L=${t['pnl']:.2f}  R={t['r_multiple']:.2f}  "
                f"hold={t['holding_bars']}b  [{t['exit_reason']}]"
            )
        lines.append("")

    # ── Signal details ──
    if not result.signals_df.empty:
        lines.append("-- SIGNALS (last 15) --")
        recent_sigs = result.signals_df.tail(15)
        for ts, row in recent_sigs.iterrows():
            lines.append(
                f"  {ts}  {row['direction']:5s}  "
                f"price={row['price']:.2f}  "
                f"vel={row['velocity']:.4f}  "
                f"accel={row['acceleration']:.4f}  "
                f"zone=[{row['consolidation_low']:.2f}, "
                f"{row['consolidation_high']:.2f}]  "
                f"({int(row['consolidation_bars'])}b)"
            )
        lines.append("")

    # ── Consolidation zones ──
    if not result.zones_df.empty:
        lines.append("-- CONSOLIDATION ZONES (last 10) --")
        recent_zones = result.zones_df.tail(10)
        for _, z in recent_zones.iterrows():
            lines.append(
                f"  bars [{int(z['start_bar'])}-{int(z['end_bar'])}]  "
                f"high={z['high']:.2f}  low={z['low']:.2f}  "
                f"range={z['range_pct']*100:.3f}%  "
                f"({int(z['n_bars'])} bars)"
            )
        lines.append("")

    lines.append("=" * 64)
    return "\n".join(lines)
