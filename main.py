"""
Institutional Trend Engine — Main Entry Point

Modes:
  1. backtest  — run strategy on historical data, generate report + chart
  2. live      — connect to Schwab API, stream real-time data
  3. demo      — generate synthetic data and run a full demonstration
  4. swing     — Swing Low reversal strategy (buy at every swing low, 1 ATR SL, 2 ATR TP)
  5. swing2    — Swing Low Elevated strategy (底部抬高: buy only when current swing low
                 > previous 2 swing lows, 1 ATR SL, 2 ATR TP)
  6. momentum  — Momentum Acceleration strategy (动量突破+加速度交易:
                 catch the fastest trend segment, 0.5 ATR SL, 1.0 ATR TP,
                 2-20 bar holding period)

Usage:
  python main.py backtest                    # backtest with synthetic data
  python main.py backtest --data spy.csv     # backtest with CSV data
  python main.py live                        # live trading (requires .env)
  python main.py demo                        # full demo with charts
  python main.py swing                       # swing low strategy demo
  python main.py swing --data spy.csv        # swing strategy on CSV data
  python main.py swing --bars 1000           # swing strategy with 1000 synthetic bars
  python main.py swing2                      # elevated swing low strategy (底部抬高)
  python main.py swing2 --bars 1000          # with 1000 synthetic bars
  python main.py swing2 --data spy.csv       # on CSV data
  python main.py swing2 --prev-lows 2        # require 2 previous lows (default)
  python main.py momentum                    # momentum acceleration strategy
  python main.py momentum --bars 1000        # with 1000 synthetic bars
  python main.py momentum --data spy.csv     # on CSV data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config, set_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════
#  Synthetic data generator (for demo / testing without Schwab API)
# ══════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    n_bars: int = 500,
    start_price: float = 6000.0,
    trend: float = 0.02,
    volatility: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data with trend + mean-reversion cycles.

    Produces data with swing highs/lows, volatility compression/
    expansion, and volume patterns that exercise all engine modules.

    The data alternates between trending phases and consolidation
    phases to create realistic market structure.
    """
    rng = np.random.RandomState(seed)
    rng = np.random.RandomState(np.random.randint(3,30))
    # Generate price path with multiple regimes:
    # - Trending phases (strong directional moves)
    # - Consolidation phases (range-bound)
    # - Reversal phases (trend change)
    close = np.zeros(n_bars)
    close[0] = start_price

    phase_length = n_bars // 5  # ~5 phases
    phase_trends = [0.05, -0.03, 0.04, -0.02, 0.06]  # alternating trends

    for i in range(1, n_bars):
        phase_idx = min(i // phase_length, len(phase_trends) - 1)
        base_trend = phase_trends[phase_idx]

        # Add cyclic mean-reversion within each phase
        cycle = 3 * np.sin(2 * np.pi * i / 30)

        # Add noise
        noise = rng.normal(0, volatility)

        close[i] = close[i-1] + base_trend + cycle * 0.1 + noise

    # Generate OHLC from close
    intrabar_range = rng.uniform(0.5, 2.0, n_bars) * volatility
    high = close + rng.uniform(0, 1, n_bars) * intrabar_range
    low = close - rng.uniform(0, 1, n_bars) * intrabar_range
    opn = np.roll(close, 1)
    opn[0] = close[0]
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))

    # Volume: higher on breakouts and trend changes
    base_vol = 100000
    volume = base_vol + rng.uniform(0.3, 1.5, n_bars) * base_vol * 0.1
    # Spike volume on large moves
    big_moves = np.abs(np.diff(close, prepend=close[0])) > volatility * 1.5
    volume[big_moves] *= 2.5
    # Higher volume at phase transitions
    for p in range(1, len(phase_trends)):
        transition_bar = p * phase_length
        if transition_bar < n_bars:
            volume[transition_bar:transition_bar+5] *= 2.0

    # Build DatetimeIndex (1-min bars, market hours)
    dates = pd.date_range("2025-01-13 09:30", periods=n_bars, freq="1min", tz="UTC")

    df = pd.DataFrame({
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume.astype(int),
    }, index=dates)

    return df


# ══════════════════════════════════════════════════════════════════
#  Mode: Backtest
# ══════════════════════════════════════════════════════════════════

def run_backtest(
    df: pd.DataFrame,
    config: Config | None = None,
    symbol: str = "SYNTH",
    output_dir: str = "output",
) -> None:
    """Run a full backtest and save results."""
    from backtest.engine import BacktestEngine
    from backtest.report import generate_report, save_report
    from visualization.chart import plot_full_chart
    from strategy.signal_engine import SignalEngine

    config = config or get_config()
    os.makedirs(output_dir, exist_ok=True)

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(df)

    # Generate report
    report = generate_report(result, symbol=symbol)
    #print(report)

    report_path = os.path.join(output_dir, "backtest_report.txt")
    save_report(report, report_path)

    # Save trade log
    if not result.trade_log.empty:
        trade_path = os.path.join(output_dir, "trades.csv")
        result.trade_log.to_csv(trade_path, index=False)
        logger.info("Trade log saved to %s", trade_path)

    # Generate chart
    signal_engine = SignalEngine(config)
    signal_engine.run(df)

    chart_path = os.path.join(output_dir, "chart.png")
    plot_full_chart(
        df=df,
        structure_engine=signal_engine.structure_engine,
        signals_df=result.signals_df,
        equity_curve=result.equity_curve,
        scored_df=result.scored_df,
        save_path=chart_path,
        title=f"Institutional Trend Engine — {symbol}",
    )

    # Save equity curve
    eq_path = os.path.join(output_dir, "equity_curve.csv")
    result.equity_curve.to_csv(eq_path, header=True)
    logger.info("Equity curve saved to %s", eq_path)

    logger.info("All outputs saved to %s/", output_dir)


# ══════════════════════════════════════════════════════════════════
#  Mode: Live
# ══════════════════════════════════════════════════════════════════

def run_live(config: Config | None = None) -> None:
    """Connect to Schwab API and run live."""
    from dotenv import load_dotenv
    load_dotenv()

    config = config or get_config()
    config.schwab_app_key = os.getenv("app_key", "")
    config.schwab_app_secret = os.getenv("app_secret", "")
    config.schwab_callback_url = os.getenv("callback_url", config.schwab_callback_url)

    if not config.schwab_app_key or not config.schwab_app_secret:
        logger.error("Missing Schwab API credentials. Check .env file.")
        logger.info("Copy .env.example to .env and fill in your credentials.")
        return

    try:
        import schwabdev
    except ImportError:
        logger.error("schwabdev not installed. Run: pip install schwabdev")
        return

    client = schwabdev.Client(
        config.schwab_app_key,
        config.schwab_app_secret,
        config.schwab_callback_url,
    )

    from data.historical_data import HistoricalData
    from data.market_data import MarketDataFeed
    from execution.broker import SchwabBroker
    from execution.order_manager import OrderManager
    from strategy.signal_engine import SignalEngine

    # Connect broker
    broker = SchwabBroker(client)
    broker.connect()

    # Load historical data
    hd = HistoricalData(client)
    df = hd.get_bars(config.symbol, period="day", frequency="1")

    if df.empty:
        logger.error("No historical data received")
        return

    # Initialize signal engine with historical data
    signal_engine = SignalEngine(config)
    signals = signal_engine.run(df)
    logger.info("Initial signals: %d", len(signals))

    # Start streaming
    feed = MarketDataFeed(client, [config.symbol])

    def on_bar(bar):
        logger.info("Bar: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%d",
                     bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume)
        # In production: append bar to df, recompute signals, execute trades

    feed.on_bar(on_bar)
    feed.start()


# ══════════════════════════════════════════════════════════════════
#  Mode: Demo
# ══════════════════════════════════════════════════════════════════

def run_demo(output_dir: str = "output", n_bars: int = 500) -> None:
    """Run a full demonstration with synthetic data."""
    logger.info("=" * 60)
    logger.info("  INSTITUTIONAL TREND ENGINE — DEMO MODE")
    logger.info("=" * 60)

    # Generate synthetic data
    df = generate_synthetic_data(n_bars=n_bars, seed=42)
    logger.info("Generated %d bars of synthetic data", len(df))

    # Run backtest
    run_backtest(df, symbol="SYNTH-DEMO", output_dir=output_dir)


# ══════════════════════════════════════════════════════════════════
#  Mode: Swing Low Strategy
# ══════════════════════════════════════════════════════════════════

def run_swing_backtest(
    df: pd.DataFrame,
    symbol: str = "SYNTH",
    output_dir: str = "output",
    swing_window: int = 3,
    atr_period: int = 14,
    stop_atr: float = 1.0,
    tp_atr: float = 2.0,
    risk_per_trade: float = 0.01,
    initial_capital: float = 100000.0,
) -> None:
    """Run the Swing Low reversal strategy backtest."""
    from backtest.swing_engine import SwingBacktestEngine
    from visualization.swing_chart import plot_swing_chart, generate_swing_report

    os.makedirs(output_dir, exist_ok=True)

    strategy_params = {
        "swing_window": swing_window,
        "atr_period": atr_period,
        "stop_atr": stop_atr,
        "tp_atr": tp_atr,
        "risk_per_trade": risk_per_trade,
    }

    # Run backtest
    engine = SwingBacktestEngine(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        swing_window=swing_window,
        atr_period=atr_period,
        stop_atr=stop_atr,
        tp_atr=tp_atr,
    )
    result = engine.run(df)

    # Generate and print report
    report = generate_swing_report(result, symbol=symbol, strategy_params=strategy_params)
    print(report)

    report_path = os.path.join(output_dir, "swing_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)

    # Save trade log
    if not result.trade_log.empty:
        trade_path = os.path.join(output_dir, "swing_trades.csv")
        result.trade_log.to_csv(trade_path, index=False)
        logger.info("Trade log saved to %s", trade_path)

    # Save signals
    if not result.signals_df.empty:
        sig_path = os.path.join(output_dir, "swing_signals.csv")
        result.signals_df.to_csv(sig_path)
        logger.info("Signals saved to %s", sig_path)

    # Save equity curve
    eq_path = os.path.join(output_dir, "swing_equity.csv")
    result.equity_curve.to_csv(eq_path, header=True)
    logger.info("Equity curve saved to %s", eq_path)

    # Generate chart
    chart_path = os.path.join(output_dir, "swing_chart.png")
    plot_swing_chart(
        df=df,
        swings_df=result.swing_lows_df,
        signals_df=result.signals_df,
        trade_log=result.trade_log,
        equity_curve=result.equity_curve,
        initial_capital=initial_capital,
        save_path=chart_path,
        title=f"Swing Low Reversal Strategy — {symbol} (SL={stop_atr}ATR, TP={tp_atr}ATR)",
    )

    logger.info("All outputs saved to %s/", output_dir)


def run_swing_demo(
    output_dir: str = "output",
    n_bars: int = 500,
) -> None:
    """Run a full demonstration of the Swing Low strategy with synthetic data."""
    logger.info("=" * 60)
    logger.info("  SWING LOW REVERSAL STRATEGY — DEMO MODE")
    logger.info("  Buy at each swing low, 1 ATR stop, 2 ATR target")
    logger.info("=" * 60)

    df = generate_synthetic_data(n_bars=n_bars, seed=42)
    logger.info("Generated %d bars of synthetic data", len(df))

    run_swing_backtest(df, symbol="SYNTH-DEMO", output_dir=output_dir)


# ══════════════════════════════════════════════════════════════════
#  Mode: Swing Low Elevated Strategy (底部抬高)
# ══════════════════════════════════════════════════════════════════

def run_swing2_backtest(
    df: pd.DataFrame,
    symbol: str = "SYNTH",
    output_dir: str = "output",
    swing_window: int = 3,
    atr_period: int = 14,
    stop_atr: float = 1.0,
    tp_atr: float = 2.0,
    require_prev_lows: int = 2,
    risk_per_trade: float = 0.01,
    initial_capital: float = 100000.0,
) -> None:
    """Run the Swing Low Elevated (底部抬高) strategy backtest."""
    from backtest.swing_elevated_engine import SwingElevatedEngine
    from visualization.swing_elevated_chart import (
        plot_elevated_chart,
        generate_elevated_report,
    )

    os.makedirs(output_dir, exist_ok=True)

    strategy_params = {
        "swing_window": swing_window,
        "atr_period": atr_period,
        "stop_atr": stop_atr,
        "tp_atr": tp_atr,
        "require_n_prev_lows": require_prev_lows,
        "risk_per_trade": risk_per_trade,
    }

    # Run backtest
    engine = SwingElevatedEngine(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        swing_window=swing_window,
        atr_period=atr_period,
        stop_atr=stop_atr,
        tp_atr=tp_atr,
        require_n_prev_lows=require_prev_lows,
    )
    result = engine.run(df)

    # Generate and print report
    report = generate_elevated_report(
        result, symbol=symbol, strategy_params=strategy_params
    )
    print(report)

    report_path = os.path.join(output_dir, "swing2_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)

    # Save trade log
    if not result.trade_log.empty:
        trade_path = os.path.join(output_dir, "swing2_trades.csv")
        result.trade_log.to_csv(trade_path, index=False)
        logger.info("Trade log saved to %s", trade_path)

    # Save all signals (BUY + REJECT)
    if not result.all_signals_df.empty:
        sig_path = os.path.join(output_dir, "swing2_all_signals.csv")
        result.all_signals_df.to_csv(sig_path)
        logger.info("All signals saved to %s", sig_path)

    # Save BUY signals only
    if not result.buy_signals_df.empty:
        buy_path = os.path.join(output_dir, "swing2_buy_signals.csv")
        result.buy_signals_df.to_csv(buy_path)
        logger.info("BUY signals saved to %s", buy_path)

    # Save equity curve
    eq_path = os.path.join(output_dir, "swing2_equity.csv")
    result.equity_curve.to_csv(eq_path, header=True)
    logger.info("Equity curve saved to %s", eq_path)

    # Generate chart
    chart_path = os.path.join(output_dir, "swing2_chart.png")
    plot_elevated_chart(
        df=df,
        swings_df=result.swing_lows_df,
        all_signals_df=result.all_signals_df,
        trade_log=result.trade_log,
        equity_curve=result.equity_curve,
        initial_capital=initial_capital,
        save_path=chart_path,
        title=f"Swing Low Elevated (Rising Bottom) — {symbol} "
              f"(SL={stop_atr}ATR, TP={tp_atr}ATR, prev={require_prev_lows})",
    )

    logger.info("All outputs saved to %s/", output_dir)


def run_swing2_demo(
    output_dir: str = "output",
    n_bars: int = 500,
) -> None:
    """Run a full demonstration of the Swing Low Elevated strategy."""
    logger.info("=" * 64)
    logger.info("  SWING LOW ELEVATED STRATEGY — 底部抬高买入策略")
    logger.info("  Buy only when swing low > previous 2 swing lows")
    logger.info("  1 ATR stop loss, 2 ATR take profit (R:R = 1:2)")
    logger.info("=" * 64)

    df = generate_synthetic_data(n_bars=n_bars, seed=42)
    logger.info("Generated %d bars of synthetic data", len(df))

    run_swing2_backtest(df, symbol="SYNTH-DEMO", output_dir=output_dir)


# ══════════════════════════════════════════════════════════════════
#  Mode: Momentum Acceleration Strategy (动量突破+加速度交易)
# ══════════════════════════════════════════════════════════════════

def run_momentum_backtest(
    df: pd.DataFrame,
    symbol: str = "SYNTH",
    output_dir: str = "output",
    extrema_order: int = 5,
    consolidation_window: int = 20,
    consolidation_range_max: float = 0.004,
    velocity_min_atr: float = 0.3,
    accel_confirm_bars: int = 3,
    accel_min_atr: float = 0.1,
    stop_atr: float = 0.5,
    tp_atr: float = 1.0,
    max_holding_bars: int = 20,
    min_holding_bars: int = 2,
    risk_per_trade: float = 0.01,
    initial_capital: float = 100000.0,
) -> None:
    """Run the Momentum Acceleration strategy backtest."""
    from backtest.momentum_engine import MomentumBacktestEngine
    from visualization.momentum_chart import (
        plot_momentum_chart,
        generate_momentum_report,
    )

    os.makedirs(output_dir, exist_ok=True)

    strategy_params = {
        "extrema_order": extrema_order,
        "consolidation_window": consolidation_window,
        "consolidation_range_max": consolidation_range_max,
        "velocity_min_atr": velocity_min_atr,
        "accel_confirm_bars": accel_confirm_bars,
        "accel_min_atr": accel_min_atr,
        "stop_atr": stop_atr,
        "tp_atr": tp_atr,
        "max_holding_bars": max_holding_bars,
        "min_holding_bars": min_holding_bars,
        "risk_per_trade": risk_per_trade,
    }

    # Run backtest
    engine = MomentumBacktestEngine(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        stop_atr=stop_atr,
        tp_atr=tp_atr,
        max_holding_bars=max_holding_bars,
        min_holding_bars=min_holding_bars,
        extrema_order=extrema_order,
        consolidation_window=consolidation_window,
        consolidation_range_max=consolidation_range_max,
        velocity_min_atr=velocity_min_atr,
        accel_confirm_bars=accel_confirm_bars,
        accel_min_atr=accel_min_atr,
    )
    result = engine.run(df)

    # Generate and print report
    report = generate_momentum_report(
        result, symbol=symbol, strategy_params=strategy_params
    )
    print(report)

    report_path = os.path.join(output_dir, "momentum_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)

    # Save trade log
    if not result.trade_log.empty:
        trade_path = os.path.join(output_dir, "momentum_trades.csv")
        result.trade_log.to_csv(trade_path, index=False)
        logger.info("Trade log saved to %s", trade_path)

    # Save signals
    if not result.signals_df.empty:
        sig_path = os.path.join(output_dir, "momentum_signals.csv")
        result.signals_df.to_csv(sig_path)
        logger.info("Signals saved to %s", sig_path)

    # Save consolidation zones
    if not result.zones_df.empty:
        zone_path = os.path.join(output_dir, "momentum_zones.csv")
        result.zones_df.to_csv(zone_path, index=False)
        logger.info("Consolidation zones saved to %s", zone_path)

    # Save equity curve
    eq_path = os.path.join(output_dir, "momentum_equity.csv")
    result.equity_curve.to_csv(eq_path, header=True)
    logger.info("Equity curve saved to %s", eq_path)

    # Generate chart
    chart_path = os.path.join(output_dir, "momentum_chart.png")
    plot_momentum_chart(
        df=df,
        zones_df=result.zones_df,
        signals_df=result.signals_df,
        trade_log=result.trade_log,
        equity_curve=result.equity_curve,
        velocity_series=result.velocity_series,
        acceleration_series=result.acceleration_series,
        initial_capital=initial_capital,
        save_path=chart_path,
        title=f"Momentum Acceleration — {symbol} "
              f"(SL={stop_atr}ATR, TP={tp_atr}ATR, R:R=1:2)",
    )

    logger.info("All outputs saved to %s/", output_dir)


def run_momentum_demo(
    output_dir: str = "output",
    n_bars: int = 500,
) -> None:
    """Run a full demonstration of the Momentum Acceleration strategy."""
    logger.info("=" * 64)
    logger.info("  MOMENTUM ACCELERATION STRATEGY")
    logger.info("  Momentum Breakout + Acceleration Trading")
    logger.info("  Catch the fastest, strongest main trend")
    logger.info("  0.5 ATR stop, 1.0 ATR target (R:R = 1:2)")
    logger.info("  Holding period: 2-20 bars")
    logger.info("=" * 64)

    df = generate_synthetic_data(n_bars=n_bars, seed=42)
    logger.info("Generated %d bars of synthetic data", len(df))

    run_momentum_backtest(df, symbol="SYNTH-DEMO", output_dir=output_dir)


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Institutional Trend Engine — Professional Intraday Trading System"
    )
    parser.add_argument(
        "mode",
        choices=["backtest", "live", "demo", "swing", "swing2", "momentum"],
        help="Run mode: backtest, live, demo, swing (every swing low), "
             "swing2 (底部抬高 elevated swing low), "
             "or momentum (动量突破+加速度交易)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to CSV file with OHLCV data (backtest/swing/swing2 mode)",
    )
    parser.add_argument(
        "--symbol", type=str, default="SPY",
        help="Trading symbol (live mode)",
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--bars", type=int, default=500,
        help="Number of synthetic bars to generate (demo/swing/swing2 mode)",
    )
    parser.add_argument(
        "--stop-atr", type=float, default=1.0,
        help="Stop loss ATR multiplier (swing/swing2 mode, default 1.0)",
    )
    parser.add_argument(
        "--tp-atr", type=float, default=3.0,
        help="Take profit ATR multiplier (swing/swing2 mode, default 2.0)",
    )
    parser.add_argument(
        "--swing-window", type=int, default=3,
        help="Swing detection window (swing/swing2 mode, default 3)",
    )
    parser.add_argument(
        "--prev-lows", type=int, default=2,
        help="Number of previous swing lows to compare (swing2 mode, default 2)",
    )
    parser.add_argument(
        "--extrema-order", type=int, default=5,
        help="argrelextrema order for swing detection (momentum mode, default 5)",
    )
    parser.add_argument(
        "--consol-window", type=int, default=20,
        help="Consolidation detection window (momentum mode, default 20)",
    )
    parser.add_argument(
        "--consol-range", type=float, default=0.7,
        help="Consolidation range ratio (fraction of rolling median, momentum mode, default 0.7)",
    )
    parser.add_argument(
        "--vel-min-atr", type=float, default=0.3,
        help="Min velocity as ATR fraction (momentum mode, default 0.3)",
    )
    parser.add_argument(
        "--accel-min-atr", type=float, default=0.1,
        help="Min acceleration as ATR fraction (momentum mode, default 0.1)",
    )
    parser.add_argument(
        "--max-hold", type=int, default=20,
        help="Max holding bars before time exit (momentum mode, default 20)",
    )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.mode == "demo":
        run_demo(output_dir=args.output, n_bars=args.bars)

    elif args.mode == "swing":
        if args.data:
            df = pd.read_csv(args.data)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            logger.info("Loaded %d bars from %s", len(df), args.data)
        else:
            logger.info("No data file specified — using synthetic data")
            df = generate_synthetic_data(n_bars=args.bars)

        run_swing_backtest(
            df,
            symbol=args.symbol,
            output_dir=args.output,
            swing_window=args.swing_window,
            stop_atr=args.stop_atr,
            tp_atr=args.tp_atr,
        )

    elif args.mode == "swing2":
        if args.data:
            df = pd.read_csv(args.data)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            logger.info("Loaded %d bars from %s", len(df), args.data)
        else:
            logger.info("No data file specified — using synthetic data")
            df = generate_synthetic_data(n_bars=args.bars)

        run_swing2_backtest(
            df,
            symbol=args.symbol,
            output_dir=args.output,
            swing_window=args.swing_window,
            stop_atr=args.stop_atr,
            tp_atr=args.tp_atr,
            require_prev_lows=args.prev_lows,
        )

    elif args.mode == "backtest":
        if args.data:
            df = pd.read_csv(args.data)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            logger.info("Loaded %d bars from %s", len(df), args.data)
        else:
            logger.info("No data file specified — using synthetic data")
            df = generate_synthetic_data(n_bars=args.bars)

        run_backtest(df, symbol=args.symbol, output_dir=args.output)

    elif args.mode == "momentum":
        if args.data:
            df = pd.read_csv(args.data)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            logger.info("Loaded %d bars from %s", len(df), args.data)
        else:
            logger.info("No data file specified — using synthetic data")
            df = generate_synthetic_data(n_bars=args.bars)

        # Momentum mode uses different default stop/tp than swing modes
        # Only override if user didn't explicitly set them
        stop_atr = args.stop_atr if args.stop_atr != 1.0 else 0.5
        tp_atr = args.tp_atr if args.tp_atr != 3.0 else 1.0

        run_momentum_backtest(
            df,
            symbol=args.symbol,
            output_dir=args.output,
            extrema_order=args.extrema_order,
            consolidation_window=args.consol_window,
            consolidation_range_max=args.consol_range,
            velocity_min_atr=args.vel_min_atr,
            accel_min_atr=args.accel_min_atr,
            stop_atr=stop_atr,
            tp_atr=tp_atr,
            max_holding_bars=args.max_hold,
        )

    elif args.mode == "live":
        run_live()


if __name__ == "__main__":
    main()
