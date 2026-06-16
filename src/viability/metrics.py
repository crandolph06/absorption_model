from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

from src.viability.config import ConstraintScalesConfig, RequirementsConfig


def compute_raw_metrics(history: pd.DataFrame, assessment_start_year: int) -> dict[str, float]:
    """Aggregate simulator history into force-level feasibility metrics.

    ``CAFSimulation.history`` contains one row per squadron per phase. Force inventory
    metrics are summed by phase. RAP values are signed margins from the model's
    ``target - observed_rate`` convention: positive is a shortfall, zero exactly
    meets RAP, and negative is slack above the RAP target.
    """
    required = {
        "year",
        "phase",
        "total_pilots",
        "line_pilots",
        "staff_ips",
        "staff_fls",
        "wg_rap_shortfall",
        "fl_rap_shortfall",
        "ip_rap_shortfall",
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"History is missing required metric columns: {missing}")
    if history.empty:
        raise ValueError("History is empty")

    frame = history.copy()

    phase_groups = frame.groupby(["year", "phase"], sort=True)
    per_phase = phase_groups.agg(
        total_pilots=("total_pilots", "sum"),
        line_pilots=("line_pilots", "sum"),
        staff_ips=("staff_ips", "sum"),
        staff_fls=("staff_fls", "sum"),
        wg_rap_shortfall=("wg_rap_shortfall", "mean"),
        fl_rap_shortfall=("fl_rap_shortfall", "mean"),
        ip_rap_shortfall=("ip_rap_shortfall", "mean"),
    )

    if {"fl_qty", "ip_qty"}.issubset(frame.columns):
        experienced = phase_groups[["fl_qty", "ip_qty"]].sum().sum(axis=1)
        line = per_phase["line_pilots"].replace(0, math.nan)
        per_phase.loc[:, "experience_ratio"] = (experienced / line).fillna(0.0)
    elif "exp_rat" in frame.columns:
        weighted = frame.assign(_weighted_exp=frame["exp_rat"] * frame["line_pilots"])
        weighted_groups = weighted.groupby(["year", "phase"], sort=True)
        per_phase.loc[:, "experience_ratio"] = (
            weighted_groups["_weighted_exp"].sum()
            / per_phase["line_pilots"].replace(0, math.nan)
        ).fillna(0.0)
    else:
        per_phase.loc[:, "experience_ratio"] = math.nan

    assessed = per_phase[per_phase.index.get_level_values("year") >= assessment_start_year]
    if assessed.empty:
        raise ValueError(
            "No history rows are at or after model.assessment_start_year="
            f"{assessment_start_year}"
        )

    final = per_phase.iloc[-1]
    return {
        "final_total_pilots": float(final["total_pilots"]),
        "final_line_pilots": float(final["line_pilots"]),
        "final_staff_ips": float(final["staff_ips"]),
        "final_staff_fls": float(final["staff_fls"]),
        "min_total_pilots_after_assessment_start": float(assessed["total_pilots"].min()),
        "min_line_pilots_after_assessment_start": float(assessed["line_pilots"].min()),
        "min_staff_ips_after_assessment_start": float(assessed["staff_ips"].min()),
        "min_staff_fls_after_assessment_start": float(assessed["staff_fls"].min()),
        "min_experience_ratio_after_assessment_start": float(assessed["experience_ratio"].min()),
        "max_wg_rap_shortfall_after_assessment_start": float(assessed["wg_rap_shortfall"].max()),
        "max_fl_rap_shortfall_after_assessment_start": float(assessed["fl_rap_shortfall"].max()),
        "max_ip_rap_shortfall_after_assessment_start": float(assessed["ip_rap_shortfall"].max()),
        "mean_wg_rap_shortfall_after_assessment_start": float(assessed["wg_rap_shortfall"].mean()),
        "mean_fl_rap_shortfall_after_assessment_start": float(assessed["fl_rap_shortfall"].mean()),
        "mean_ip_rap_shortfall_after_assessment_start": float(assessed["ip_rap_shortfall"].mean()),
    }


def compute_constraints(
    raw_metrics: Mapping[str, float], requirements: RequirementsConfig
) -> dict[str, float]:
    """Return constraints using ``g(x) <= 0`` as the satisfied convention."""
    constraints: dict[str, float] = {}

    if requirements.target_total_pilots is not None:
        constraints["total_pilots_final"] = (
            requirements.target_total_pilots - raw_metrics["final_total_pilots"]
        )
        constraints["total_pilots_window"] = (
            requirements.target_total_pilots
            - raw_metrics["min_total_pilots_after_assessment_start"]
        )

    if requirements.target_line_pilots is not None:
        constraints["line_pilots_window"] = (
            requirements.target_line_pilots
            - raw_metrics["min_line_pilots_after_assessment_start"]
        )

    if requirements.allowed_wg_rap_shortfall is not None:
        constraints["wg_rap"] = (
            raw_metrics["max_wg_rap_shortfall_after_assessment_start"]
            - requirements.allowed_wg_rap_shortfall
        )
    if requirements.allowed_fl_rap_shortfall is not None:
        constraints["fl_rap"] = (
            raw_metrics["max_fl_rap_shortfall_after_assessment_start"]
            - requirements.allowed_fl_rap_shortfall
        )
    if requirements.allowed_ip_rap_shortfall is not None:
        constraints["ip_rap"] = (
            raw_metrics["max_ip_rap_shortfall_after_assessment_start"]
            - requirements.allowed_ip_rap_shortfall
        )

    if requirements.target_staff_ips is not None:
        constraints["staff_ips"] = (
            requirements.target_staff_ips
            - raw_metrics["min_staff_ips_after_assessment_start"]
        )
    if requirements.target_staff_fls is not None:
        constraints["staff_fls"] = (
            requirements.target_staff_fls
            - raw_metrics["min_staff_fls_after_assessment_start"]
        )
    if requirements.min_experience_ratio is not None:
        constraints["experience_ratio"] = (
            requirements.min_experience_ratio
            - raw_metrics["min_experience_ratio_after_assessment_start"]
        )

    if not constraints:
        raise ValueError("No enabled constraints were configured")
    return {name: float(value) for name, value in constraints.items()}


def aggregate_violation(
    constraints: Mapping[str, float], scales: ConstraintScalesConfig
) -> tuple[float, str | None, float | None]:
    """Return phi, the binding constraint name, and its unnormalized value."""
    if not constraints:
        raise ValueError("Cannot aggregate an empty constraint set")

    normalized: dict[str, float] = {}
    for name, value in constraints.items():
        scale = scales.scale_for(name)
        if scale <= 0:
            raise ValueError(f"Constraint scale for {name!r} must be positive")
        normalized[name] = value / scale

    active_constraint = max(normalized, key=normalized.get)
    phi = float(normalized[active_constraint])
    return phi, active_constraint, float(constraints[active_constraint])
