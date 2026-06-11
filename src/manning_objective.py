"""
Explicit CAF objective for simulation-based policy search (paper path 1).

Unlike gym rewards, this returns an interpretable scalar cost to minimize.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.manning_engine import CAFSimulation


@dataclass(frozen=True)
class ObjectiveWeights:
    wg_shortfall: float = 0.75
    fl_shortfall: float = 0.52
    ip_shortfall: float = 0.52
    manning_gap: float = 2.0
    deferral_burden: float = 0.05
    experience: float = 0.25


DEFAULT_TARGET_PILOTS = 3500


def caf_phase_means(history: pd.DataFrame) -> pd.DataFrame:
    """One row per (year, phase): CAF-wide mean of squadron snapshot columns."""
    if history.empty:
        return history
    group_cols = ["year", "phase"]
    numeric_cols = [
        "wg_rap_shortfall",
        "fl_rap_shortfall",
        "ip_rap_shortfall",
        "deferred_sortie_burden",
        "percent_manned",
        "exp_rat",
        "line_pilots",
    ]
    present = [c for c in numeric_cols if c in history.columns]
    agg = history.groupby(group_cols, as_index=False)[present].mean()
    return agg


def weighted_rap_shortfall(row, weights: ObjectiveWeights) -> float:
    wg = max(0.0, float(row.get("wg_rap_shortfall", 0.0)))
    fl = max(0.0, float(row.get("fl_rap_shortfall", 0.0)))
    ip = max(0.0, float(row.get("ip_rap_shortfall", 0.0)))
    return (
        weights.wg_shortfall * wg
        + weights.fl_shortfall * fl
        + weights.ip_shortfall * ip
    )


def objective_from_history(
    history: pd.DataFrame,
    *,
    target_pilots: int = DEFAULT_TARGET_PILOTS,
    weights: Optional[ObjectiveWeights] = None,
    terminal_active_pilots: Optional[int] = None,
) -> dict:
    """
    Return cost (lower is better) and component breakdown.

    Cost = mean phase RAP shortfall
         + manning gap penalty at horizon
         + mean deferral burden
         - small bonus for experience ratio (encourages IP/FL depth once RAP is met)
    """
    weights = weights or ObjectiveWeights()
    if history.empty:
        return {
            "cost": 1e6,
            "mean_shortfall": 1e6,
            "manning_penalty": 0.0,
            "deferral_penalty": 0.0,
            "experience_bonus": 0.0,
            "terminal_pilots": 0,
        }

    phase_df = caf_phase_means(history)
    shortfalls = phase_df.apply(lambda r: weighted_rap_shortfall(r, weights), axis=1)
    mean_shortfall = float(shortfalls.mean())

    deferral_penalty = weights.deferral_burden * float(
        phase_df.get("deferred_sortie_burden", pd.Series([0.0])).mean()
    )

    if terminal_active_pilots is None:
        terminal_active_pilots = int(history.iloc[-1].get("total_pilots", 0))
    manning_gap = max(0.0, target_pilots - terminal_active_pilots) / max(target_pilots, 1)
    manning_penalty = weights.manning_gap * manning_gap

    experience_bonus = weights.experience * float(phase_df.get("exp_rat", pd.Series([0.0])).mean())

    cost = mean_shortfall + manning_penalty + deferral_penalty - experience_bonus
    return {
        "cost": cost,
        "mean_shortfall": mean_shortfall,
        "manning_penalty": manning_penalty,
        "deferral_penalty": deferral_penalty,
        "experience_bonus": experience_bonus,
        "terminal_pilots": terminal_active_pilots,
    }


def terminal_metrics(sim: CAFSimulation, history: pd.DataFrame) -> dict:
    """Summary stats at end of rollout for logging / paper tables."""
    obj = objective_from_history(
        history,
        terminal_active_pilots=sim.total_active_pilot_count,
    )
    return {
        **obj,
        "wg_shortfall": sim.current_wg_shortfall,
        "fl_shortfall": sim.current_fl_shortfall,
        "ip_shortfall": sim.current_ip_shortfall,
        "experience_ratio": sim.experience_ratio,
        "line_pilots": sim.total_line_pilot_count,
        "staff_pilots": sim.total_staff_pilot_count,
    }
