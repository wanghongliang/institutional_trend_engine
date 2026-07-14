"""
Visualization module.

Produces TradingView-style charts showing:
  - Price candles with Swing High/Low markers
  - BOS / CHoCH / Liquidity Sweep annotations
  - BUY / SELL / ADD signals
  - Volume Delta subplot
  - Turning Score heatmap
  - Equity curve

Uses matplotlib for static rendering.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from config import SignalType
from indicators.market_structure import MarketStructureEngine, StructureEvent
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Color palette (light theme) ─────────────────────────────────
COLOR_BG = "#FAFAFA"
COLOR_GRID = "#E0E0E0"
COLOR_UP = "#E53935"         # red = up (Chinese convention)
COLOR_DOWN = "#43A047"       # green = down
COLOR_BUY = "#1565C0"        # blue
COLOR_SELL = "#FF6F00"       # orange
COLOR_SWING_HIGH = "#7B1FA2" # purple
COLOR_SWING_LOW = "#00838F"  # teal
COLOR_BOS = "#D81B60"        # pink
COLOR_CHOCH = "#F57F17"      # amber
COLOR_VWAP = "#6A1B9A"       # dark purple


def plot_full_chart(
    df: pd.DataFrame,
    structure_engine: MarketStructureEngine,
    signals_df: pd.DataFrame,
    equity_curve: Optional[pd.Series] = None,
    scored_df: Optional[pd.DataFrame] = None,
    save_path: str = "output/chart.png",
    title: str = "Institutional Trend Engine",
    last_n_bars: int = 300,
) -> str:
    """
    Generate a full multi-panel chart.

    Panels (top to bottom):
      1. Price + Swings + Signals + VWAP
      2. Volume Delta
      3. Turning Score
      4. Equity Curve (if provided)

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    # Slice to last N bars for readability
    df = df.iloc[-last_n_bars:]
    swings = structure_engine.swing_detector.swings
    events = structure_engine.events

    # Filter swings/events to visible range
    if len(df) > 0:
        start_ts = df.index[0]
        swings = [s for s in swings if s.timestamp >= start_ts]
        events = [(ts, ev) for ts, ev in events if ts >= start_ts]

    # Determine subplots
    n_panels = 3
    if equity_curve is not None:
        n_panels += 1

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(16, 4 * n_panels),
        gridspec_kw={"height_ratios": [3, 1, 1] + ([1] if equity_curve is not None else [])},
        facecolor=COLOR_BG,
    )
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: Price + Swings + Signals ───────────────────────
    ax_price = axes[0]
    ax_price.set_facecolor(COLOR_BG)

    # Plot close price
    ax_price.plot(df.index, df["close"], color="#333333", linewidth=1, label="Close")

    # Plot VWAP if available
    if scored_df is not None and "vwap" in scored_df.columns:
        vwap_slice = scored_df["vwap"].iloc[-last_n_bars:]
        ax_price.plot(vwap_slice.index, vwap_slice.values,
                      color=COLOR_VWAP, linewidth=1, linestyle="--", alpha=0.7, label="VWAP")

    # Plot swing highs and lows
    for sw in swings:
        if sw.type == "high":
            ax_price.scatter(sw.timestamp, sw.price, color=COLOR_SWING_HIGH,
                             marker="v", s=60, zorder=5)
            ax_price.annotate(f"SH\n{sw.price:.2f}",
                              (sw.timestamp, sw.price),
                              textcoords="offset points", xytext=(0, 12),
                              fontsize=7, color=COLOR_SWING_HIGH, ha="center")
        else:
            ax_price.scatter(sw.timestamp, sw.price, color=COLOR_SWING_LOW,
                             marker="^", s=60, zorder=5)
            ax_price.annotate(f"SL\n{sw.price:.2f}",
                              (sw.timestamp, sw.price),
                              textcoords="offset points", xytext=(0, -15),
                              fontsize=7, color=COLOR_SWING_LOW, ha="center")

    # Plot structure events
    for ts, event in events:
        if ts not in df.index:
            # find nearest
            idx = df.index.get_indexer([ts], method="nearest")[0]
            if idx < 0 or idx >= len(df):
                continue
            ts = df.index[idx]
        price = df.loc[ts, "close"] if ts in df.index else df["close"].iloc[-1]

        if event == StructureEvent.BOS_BULL:
            ax_price.axvline(ts, color=COLOR_BOS, alpha=0.3, linestyle="-")
            ax_price.annotate("BOS↑", (ts, price), fontsize=7,
                              color=COLOR_BOS, fontweight="bold")
        elif event == StructureEvent.BOS_BEAR:
            ax_price.axvline(ts, color=COLOR_BOS, alpha=0.3, linestyle="-")
            ax_price.annotate("BOS↓", (ts, price), fontsize=7,
                              color=COLOR_BOS, fontweight="bold")
        elif event == StructureEvent.CHOCH_BULL:
            ax_price.axvline(ts, color=COLOR_CHOCH, alpha=0.4, linestyle="--")
            ax_price.annotate("CHoCH↑", (ts, price), fontsize=7,
                              color=COLOR_CHOCH, fontweight="bold")
        elif event == StructureEvent.CHOCH_BEAR:
            ax_price.axvline(ts, color=COLOR_CHOCH, alpha=0.4, linestyle="--")
            ax_price.annotate("CHoCH↓", (ts, price), fontsize=7,
                              color=COLOR_CHOCH, fontweight="bold")

    # Plot signals
    if not signals_df.empty:
        visible_signals = signals_df[signals_df.index >= df.index[0]]
        for ts, row in visible_signals.iterrows():
            sig_type = row["signal_type"]
            price = row["price"]

            if sig_type == SignalType.EARLY_BUY.value:
                ax_price.scatter(ts, price, color=COLOR_BUY, marker="^",
                                 s=120, zorder=6, edgecolors="black")
                ax_price.annotate("BUY1", (ts, price),
                                  textcoords="offset points", xytext=(0, -20),
                                  fontsize=8, color=COLOR_BUY, fontweight="bold", ha="center")
            elif sig_type == SignalType.BREAKOUT_BUY.value:
                ax_price.scatter(ts, price, color=COLOR_BUY, marker="^",
                                 s=150, zorder=6, edgecolors="black", linewidths=1.5)
                ax_price.annotate("BUY2", (ts, price),
                                  textcoords="offset points", xytext=(0, -20),
                                  fontsize=8, color=COLOR_BUY, fontweight="bold", ha="center")
            elif sig_type == SignalType.EARLY_SELL.value:
                ax_price.scatter(ts, price, color=COLOR_SELL, marker="v",
                                 s=120, zorder=6, edgecolors="black")
                ax_price.annotate("SELL1", (ts, price),
                                  textcoords="offset points", xytext=(0, 12),
                                  fontsize=8, color=COLOR_SELL, fontweight="bold", ha="center")
            elif sig_type == SignalType.BREAKOUT_SELL.value:
                ax_price.scatter(ts, price, color=COLOR_SELL, marker="v",
                                 s=150, zorder=6, edgecolors="black", linewidths=1.5)
                ax_price.annotate("SELL2", (ts, price),
                                  textcoords="offset points", xytext=(0, 12),
                                  fontsize=8, color=COLOR_SELL, fontweight="bold", ha="center")
            elif sig_type in (SignalType.EXIT_LONG.value, SignalType.EXIT_SHORT.value):
                ax_price.scatter(ts, price, color="#666666", marker="x",
                                 s=80, zorder=6)

    ax_price.set_title(title, fontsize=14, fontweight="bold")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, color=COLOR_GRID, alpha=0.5)

    # ── Panel 2: Volume Delta ───────────────────────────────────
    ax_delta = axes[1]
    ax_delta.set_facecolor(COLOR_BG)
    if scored_df is not None and "vol_delta" in scored_df.columns:
        delta_slice = scored_df["vol_delta"].iloc[-last_n_bars:]
        colors = [COLOR_UP if d > 0 else COLOR_DOWN for d in delta_slice]
        ax_delta.bar(delta_slice.index, delta_slice.values, color=colors, width=0.001, alpha=0.7)
    ax_delta.set_title("Volume Delta", fontsize=10)
    ax_delta.axhline(0, color="#333", linewidth=0.5)
    ax_delta.grid(True, color=COLOR_GRID, alpha=0.5)

    # ── Panel 3: Turning Score ──────────────────────────────────
    ax_score = axes[2]
    ax_score.set_facecolor(COLOR_BG)
    if scored_df is not None and "turning_score" in scored_df.columns:
        score_slice = scored_df["turning_score"].iloc[-last_n_bars:]
        dir_slice = scored_df["turning_direction"].iloc[-last_n_bars:]
        colors = [COLOR_UP if d > 0 else COLOR_DOWN if d < 0 else "#999" for d in dir_slice]
        ax_score.bar(score_slice.index, score_slice.values, color=colors, width=0.001, alpha=0.7)
        ax_score.axhline(80, color=COLOR_CHOCH, linestyle="--", linewidth=0.8, label="Threshold")
    ax_score.set_title("Turning Score", fontsize=10)
    ax_score.set_ylim(0, 105)
    ax_score.grid(True, color=COLOR_GRID, alpha=0.5)

    # ── Panel 4: Equity Curve ───────────────────────────────────
    if equity_curve is not None:
        ax_eq = axes[3]
        ax_eq.set_facecolor(COLOR_BG)
        eq_slice = equity_curve.iloc[-last_n_bars:]
        ax_eq.plot(eq_slice.index, eq_slice.values, color=COLOR_BUY, linewidth=1.5)
        ax_eq.fill_between(eq_slice.index, eq_slice.values,
                           self_initial_capital(eq_slice),
                           color=COLOR_BUY, alpha=0.1)
        ax_eq.set_title("Equity Curve", fontsize=10)
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
    logger.info("Chart saved to %s", save_path)
    return save_path


def self_initial_capital(series):
    """Helper to get initial capital level for fill."""
    return series.iloc[0] if len(series) > 0 else 0
