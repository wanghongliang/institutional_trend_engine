"""
Signal Engine — orchestrates all strategy components.

Takes raw OHLCV data and produces actionable trading signals by
combining:
  1. Market Structure Engine (Swing / BOS / CHoCH / Liquidity)
  2. Trend Score Engine (Turning Point Detection)
  3. State Machine (Market Regime)
  4. Breakout Strategy (Early Entry + Breakout)

The Signal Engine is the single entry point that the main loop
and backtest engine call.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import Config, get_config
from indicators.market_structure import MarketStructureEngine
from indicators.trend_score import TrendScoreEngine
from strategy.breakout import BreakoutStrategy, Signal
from strategy.state_machine import StateMachine
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SignalEngine:
    """
    The master signal-generation orchestrator.

    Usage
    -----
    >>> engine = SignalEngine()
    >>> signals = engine.run(df)
    >>> engine.signals_df  # DataFrame of all signals
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()

        # Initialize sub-engines
        self.structure_engine = MarketStructureEngine(
            swing_window=self.config.swing_window,
            min_atr=self.config.swing_min_atr,
        )
        self.score_engine = TrendScoreEngine(
            weights=self.config.score_weights,
            threshold=self.config.score_threshold,
        )
        self.state_machine = StateMachine()
        self.breakout_strategy = BreakoutStrategy(
            structure_engine=self.structure_engine,
            score_engine=self.score_engine,
            early_entry_atr_ratio=self.config.early_entry_atr_ratio,
        )

        self.signals: list[Signal] = []
        self.signals_df: pd.DataFrame = pd.DataFrame()
        self.scored_df: pd.DataFrame = pd.DataFrame()
        self.regime_df: pd.DataFrame = pd.DataFrame()
        self.structure_snapshots = []

    def run(self, df: pd.DataFrame) -> list[Signal]:
        """
        Run the full signal pipeline on *df*.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex.

        Returns
        -------
        list[Signal]
            Trading signals.
        """
        if len(df) < 50:
            logger.warning("Not enough data (%d bars) for signal generation", len(df))
            return []

        logger.info("Running signal engine on %d bars...", len(df))

        # 1. Compute scores
        self.scored_df = self.score_engine.compute(df)
        logger.info("Trend score computed. Turning points: %d",
                     self.scored_df["is_turning"].sum())

        # 2. Classify regimes
        regime_snapshots = self.state_machine.classify(df)
        self.regime_df = self.state_machine.to_dataframe(regime_snapshots)

        # 3. Generate signals
        self.signals = self.breakout_strategy.generate_signals(
            df, regime_df=self.regime_df
        )
        self.signals_df = self.breakout_strategy.signals_to_dataframe(self.signals)

        # 4. Keep structure snapshots for visualization
        self.structure_snapshots = self.structure_engine.snapshots

        logger.info("Signal engine complete: %d signals generated", len(self.signals))
        return self.signals

    def summary(self) -> dict:
        """Return a summary dict of the last run."""
        if self.signals_df.empty:
            return {"total_signals": 0}

        signal_counts = self.signals_df["signal_type"].value_counts().to_dict()
        return {
            "total_signals": len(self.signals_df),
            "signal_breakdown": signal_counts,
            "turning_points_detected": int(self.scored_df["is_turning"].sum()) if not self.scored_df.empty else 0,
            "regime_distribution": self.regime_df["regime"].value_counts().to_dict() if not self.regime_df.empty else {},
        }
