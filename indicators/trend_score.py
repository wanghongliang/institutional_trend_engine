"""
Trend Score / Turning Point Detection.

Implements the multi-factor turning-point scoring system described
in the project specification:

    Factor                  Weight
    ─────────────────────────────────
    Acceleration zero-cross    25
    OFI reversal               20
    Curvature increase         15
    Slope change               15
    VWAP reclaim               10
    Volume Delta               5
    RV expansion               5
    GEX support                5
    ─────────────────────────────────
    Total                     100

When Turning Score > threshold (default 80), the engine considers
a turning point to have begun.

Each factor is evaluated as a boolean (fires or doesn't) and the
score is the sum of weights for factors that fire.  This keeps the
system transparent and backtestable.
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from indicators.momentum import (
    velocity,
    acceleration,
    acceleration_zero_crossing,
    slope,
    slope_sign_change,
    curvature_indicator,
)
from indicators.vwap import vwap, vwap_reclaim_signal, vwap_slope
from indicators.volume_delta import volume_delta, delta_reversal_bull, delta_reversal_bear
from indicators.realized_vol import realized_volatility, rv_expansion


@dataclass
class TurningScoreResult:
    """Result of the turning-point evaluation for a single bar."""
    timestamp: pd.Timestamp
    score: float
    factors_fired: Dict[str, bool]
    direction: int          # +1 = bullish turn, -1 = bearish turn, 0 = neutral
    is_turning: bool        # score >= threshold AND direction != 0


class TrendScoreEngine:
    """
    Multi-factor turning-point scoring engine.

    Parameters
    ----------
    weights : dict
        Factor name → weight (should sum to 100).
    threshold : float
        Minimum score to declare a turning point.
    """

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        threshold: float = 80.0,
    ):
        self.weights = weights or {
            "acceleration": 25,
            "ofi": 20,
            "curvature": 15,
            "slope": 15,
            "vwap": 10,
            "volume_delta": 5,
            "rv": 5,
            "gex": 5,
        }
        self.threshold = threshold

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute turning score for every bar in *df*.

        Adds the following columns to a copy of df:
          - velocity, acceleration, accel_cross
          - slope, slope_cross
          - curvature
          - vwap, vwap_reclaim, vwap_slope_pos
          - vol_delta, delta_rev
          - rv, rv_exp
          - turning_score, turning_direction, is_turning

        Returns
        -------
        pd.DataFrame
            Enhanced DataFrame with all factor columns and the score.
        """
        out = df.copy()

        # ── Acceleration (weight: 25) ───────────────────────────
        out["velocity"] = velocity(out["close"])
        out["acceleration"] = acceleration(out["close"])
        accel_cross = acceleration_zero_crossing(out["acceleration"])
        out["accel_bull"] = accel_cross == 1
        out["accel_bear"] = accel_cross == -1

        # ── OFI / Volume Delta reversal (weight: 20) ────────────
        out["vol_delta"] = volume_delta(
            out["high"], out["low"], out["close"], out["volume"]
        )
        delta_rev_bull = delta_reversal_bull(out["vol_delta"], lookback=5)
        delta_rev_bear = delta_reversal_bear(out["vol_delta"], lookback=5)
        out["ofi_bull"] = delta_rev_bull
        out["ofi_bear"] = delta_rev_bear

        # ── Curvature increase (weight: 15) ─────────────────────
        out["curvature"] = curvature_indicator(out["close"], window=10)
        curv_increasing = out["curvature"] > out["curvature"].shift(1)
        out["curv_bull"] = curv_increasing & (out["curvature"] > 0)
        out["curv_bear"] = curv_increasing & (out["curvature"] < 0)

        # ── Slope sign change (weight: 15) ──────────────────────
        out["slope"] = slope(out["close"], window=20)
        slope_cross = slope_sign_change(out["slope"])
        out["slope_bull"] = slope_cross == 1
        out["slope_bear"] = slope_cross == -1

        # ── VWAP reclaim (weight: 10) ───────────────────────────
        out["vwap"] = vwap(
            out["high"], out["low"], out["close"], out["volume"]
        )
        vwap_reclaim = vwap_reclaim_signal(out["close"], out["vwap"])
        out["vwap_bull"] = vwap_reclaim
        out["vwap_bear"] = (
            (out["close"] < out["vwap"]) &
            (out["close"].shift(1) >= out["vwap"].shift(1))
        )

        # ── Volume Delta direction (weight: 5) ─────────────────
        out["vdelta_bull"] = out["vol_delta"] > 0
        out["vdelta_bear"] = out["vol_delta"] < 0

        # ── Realized Volatility expansion (weight: 5) ───────────
        out["rv"] = realized_volatility(out["close"], period=20)
        rv_exp = rv_expansion(out["rv"], lookback=50)
        out["rv_bull"] = rv_exp
        out["rv_bear"] = rv_exp  # expansion is direction-agnostic

        # ── GEX (weight: 5) ─────────────────────────────────────
        # GEX requires option-chain data; if not available, neutral.
        out["gex_bull"] = False
        out["gex_bear"] = False

        # ── Compute scores ──────────────────────────────────────
        bull_factors = {
            "acceleration": out["accel_bull"],
            "ofi": out["ofi_bull"],
            "curvature": out["curv_bull"],
            "slope": out["slope_bull"],
            "vwap": out["vwap_bull"],
            "volume_delta": out["vdelta_bull"],
            "rv": out["rv_bull"],
            "gex": out["gex_bull"],
        }
        bear_factors = {
            "acceleration": out["accel_bear"],
            "ofi": out["ofi_bear"],
            "curvature": out["curv_bear"],
            "slope": out["slope_bear"],
            "vwap": out["vwap_bear"],
            "volume_delta": out["vdelta_bear"],
            "rv": out["rv_bear"],
            "gex": out["gex_bear"],
        }

        bull_score = sum(
            self.weights.get(k, 0) * v.astype(float)
            for k, v in bull_factors.items()
        )
        bear_score = sum(
            self.weights.get(k, 0) * v.astype(float)
            for k, v in bear_factors.items()
        )

        out["bull_score"] = bull_score
        out["bear_score"] = bear_score
        out["turning_score"] = np.maximum(bull_score, bear_score)
        out["turning_direction"] = np.where(
            bull_score > bear_score, 1,
            np.where(bear_score > bull_score, -1, 0)
        )
        out["is_turning"] = (
            (out["turning_score"] >= self.threshold) &
            (out["turning_direction"] != 0)
        )

        return out

    def get_turning_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only the bars where a turning point was detected.
        """
        scored = self.compute(df) if "turning_score" not in df.columns else df
        return scored[scored["is_turning"]].copy()
