"""
Backtest performance report.

Generates a formatted text report and optionally saves it to a file.
"""

from typing import Optional

import pandas as pd

from backtest.engine import BacktestResult


def generate_report(result: BacktestResult, symbol: str = "") -> str:
    """
    Generate a human-readable performance report.

    Returns
    -------
    str
        Multi-line report text.
    """
    m = result.metrics
    lines = []
    lines.append("=" * 60)
    lines.append("  INSTITUTIONAL TREND ENGINE — BACKTEST REPORT")
    lines.append("=" * 60)
    if symbol:
        lines.append(f"  Symbol: {symbol}")
    lines.append("")

    # ── Overview ──
    lines.append("── OVERVIEW ──")
    lines.append(f"  Initial Capital:     ${m.get('initial_capital', 0):>12,.2f}")
    lines.append(f"  Final Equity:        ${m.get('final_equity', 0):>12,.2f}")
    lines.append(f"  Total Return:         {m.get('total_return_pct', 0):>11.2f}%")
    lines.append(f"  Max Drawdown:         {m.get('max_drawdown_pct', 0):>11.2f}%")
    lines.append(f"  Sharpe Ratio:         {m.get('sharpe_ratio', 0):>11.2f}")
    lines.append("")

    # ── Trade statistics ──
    lines.append("── TRADE STATISTICS ──")
    lines.append(f"  Total Trades:         {m.get('total_trades', 0):>11d}")
    lines.append(f"  Winning Trades:       {m.get('winning_trades', 0):>11d}")
    lines.append(f"  Losing Trades:        {m.get('losing_trades', 0):>11d}")
    lines.append(f"  Win Rate:             {m.get('win_rate', 0):>11.1f}%")
    lines.append(f"  Profit Factor:        {m.get('profit_factor', 0):>11.2f}")
    lines.append("")

    # ── P&L breakdown ──
    lines.append("── P&L BREAKDOWN ──")
    lines.append(f"  Total P&L:           ${m.get('total_pnl', 0):>12,.2f}")
    lines.append(f"  Average P&L:         ${m.get('avg_pnl', 0):>12,.2f}")
    lines.append(f"  Average Win:         ${m.get('avg_win', 0):>12,.2f}")
    lines.append(f"  Average Loss:        ${m.get('avg_loss', 0):>12,.2f}")
    lines.append("")

    # ── R-multiple ──
    lines.append("── R-MULTIPLE ──")
    lines.append(f"  Average R:            {m.get('avg_r_multiple', 0):>11.2f}")
    lines.append(f"  Max R:                {m.get('max_r_multiple', 0):>11.2f}")
    lines.append(f"  Min R:                {m.get('min_r_multiple', 0):>11.2f}")
    lines.append("")

    # ── Trade details ──
    if not result.trade_log.empty:
        lines.append("── TRADE LOG (last 10) ──")
        recent = result.trade_log.tail(10)
        for _, t in recent.iterrows():
            direction = "LONG " if t["direction"] > 0 else "SHORT"
            lines.append(
                f"  {t['entry_time']}  {direction}  "
                f"entry={t['entry_price']:.2f}  exit={t['exit_price']:.2f}  "
                f"P&L=${t['pnl']:.2f}  R={t['r_multiple']:.2f}  "
                f"[{t['exit_reason']}]"
            )
        lines.append("")

    # ── Signals summary ──
    if not result.signals_df.empty:
        lines.append("── SIGNALS SUMMARY ──")
        counts = result.signals_df["signal_type"].value_counts()
        for sig_type, count in counts.items():
            lines.append(f"  {sig_type:<20s}  {count:>5d}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def save_report(report_text: str, path: str) -> None:
    """Save the report text to a file."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
