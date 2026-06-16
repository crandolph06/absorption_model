from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.viability.config import ViabilityConfig
from src.viability.dynamic_analysis_common import (
    EvaluateSchedules,
    base_seed_offset,
    best_ok_row,
    clone_config_with_policy_highs,
    dedupe_candidates,
    format_optional,
    format_token,
    markdown_table,
    positive_constraints,
    raw_values_from_row,
    schedule_row_from_flat_values,
    summarize_best_by_experiment,
    sweep_values_for_variable,
)
from src.viability.dynamic_policy import dynamic_feature_names
from src.viability.evaluator import evaluate_schedules_parallel
from src.viability.io import write_config_resolved, write_table
from src.viability.surrogate import read_evaluations_table


DEFAULT_BOUND_EXTENSIONS: dict[str, tuple[float, ...]] = {
    "retention_rate": (0.75, 0.85, 0.95),
    "annual_intake": (400.0, 500.0, 650.0),
    "ute": (22.0, 25.0, 30.0),
    "paa": (35.0, 40.0),
    "max_manning_pct": (225.0, 250.0),
    "flug_quota_per_phase": (15.0, 20.0),
    "ipug_quota_per_phase": (15.0, 20.0),
}


@dataclass(frozen=True)
class DynamicBoundRelaxationResult:
    output_dir: Path
    candidates_path: Path
    evaluations_path: Path
    best_by_experiment_path: Path
    summary_path: Path
    report_path: Path
    evaluated_count: int
    feasible_count: int
    best_phi: float


def run_dynamic_bound_relaxation_study(
    *,
    config: ViabilityConfig,
    evaluations_path: str | Path,
    output_dir: str | Path,
    epoch_count: int,
    workers: int | None = None,
    checkpoint_every: int = 10,
    sweep_points: int = 5,
    bound_extensions: Mapping[str, Sequence[float]] | None = None,
    evaluator: EvaluateSchedules = evaluate_schedules_parallel,
) -> DynamicBoundRelaxationResult:
    """Run one-at-a-time fixed-shape sweeps with temporarily widened bounds."""
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    if sweep_points < 2:
        raise ValueError("sweep_points must be at least 2")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {
        name: tuple(float(value) for value in values)
        for name, values in (bound_extensions or DEFAULT_BOUND_EXTENSIONS).items()
    }
    high_overrides = max_high_overrides(config, extensions)
    widened_config = clone_config_with_policy_highs(config, high_overrides)
    write_config_resolved(widened_config, output_path)

    evaluations = read_evaluations_table(evaluations_path)
    best_row = best_ok_row(evaluations)
    candidates = generate_bound_relaxation_candidates(
        widened_config,
        best_row,
        epoch_count=epoch_count,
        bound_extensions=extensions,
        sweep_points=sweep_points,
    )
    candidates_path = write_table(
        candidates,
        output_path / "bound_relaxation_candidates.csv",
        prefer_parquet=False,
    )
    direct_evaluations = evaluator(
        candidates,
        widened_config,
        epoch_count=epoch_count,
        workers=workers,
        checkpoint_dir=output_path / "checkpoints",
        checkpoint_every=checkpoint_every,
    )
    evaluations_output_path = write_table(
        direct_evaluations,
        output_path / "bound_relaxation_evaluations.parquet",
    )
    best_by_experiment = summarize_best_by_experiment(direct_evaluations)
    best_by_experiment_path = write_table(
        best_by_experiment,
        output_path / "best_by_bound_experiment.csv",
        prefer_parquet=False,
    )
    summary = bound_relaxation_summary(
        direct_evaluations,
        best_by_experiment,
        base_schedule_id=str(best_row["schedule_id"]),
        high_overrides=high_overrides,
    )
    summary_path = output_path / "bound_relaxation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output_path / "bound_relaxation_report.md"
    report_path.write_text(
        render_bound_relaxation_report(summary, best_by_experiment),
        encoding="utf-8",
    )
    ok = direct_evaluations[direct_evaluations["status"] == "ok"]
    return DynamicBoundRelaxationResult(
        output_dir=output_path.resolve(),
        candidates_path=candidates_path.resolve(),
        evaluations_path=evaluations_output_path.resolve(),
        best_by_experiment_path=best_by_experiment_path.resolve(),
        summary_path=summary_path.resolve(),
        report_path=report_path.resolve(),
        evaluated_count=int(len(direct_evaluations)),
        feasible_count=int(ok["feasible"].astype(bool).sum()) if not ok.empty else 0,
        best_phi=float(ok["phi"].min()) if not ok.empty else float("inf"),
    )


def generate_bound_relaxation_candidates(
    config: ViabilityConfig,
    base_row: pd.Series,
    *,
    epoch_count: int,
    bound_extensions: Mapping[str, Sequence[float]],
    sweep_points: int,
) -> pd.DataFrame:
    feature_names = dynamic_feature_names(config.policy, epoch_count)
    base_values = raw_values_from_row(base_row, feature_names)
    seed_offset = base_seed_offset(base_row)
    total_phases = config.model.years_to_run * 3
    rows: list[dict[str, object]] = []

    rows.append(
        schedule_row_from_flat_values(
            config,
            base_values,
            epoch_count=epoch_count,
            total_phases=total_phases,
            schedule_id="bound_base",
            source="bound_base",
            sample_index=0,
            metadata={
                "experiment_id": "baseline",
                "base_schedule_id": str(base_row["schedule_id"]),
                "seed_offset": seed_offset,
                "counterfactual": "baseline",
                "analysis_note": "original best schedule under widened analysis config",
            },
        )
    )
    sample_index = 1
    for variable_name, highs in bound_extensions.items():
        if variable_name not in config.policy.variables:
            raise ValueError(f"Unknown policy variable {variable_name!r}")
        for high in sorted(set(float(value) for value in highs)):
            values = sweep_values_for_variable(
                config,
                base_values,
                variable_name,
                epoch_count=epoch_count,
                high=high,
                sweep_points=sweep_points,
            )
            for sweep_value in values:
                candidate = dict(base_values)
                for epoch_index in range(epoch_count):
                    candidate[f"epoch{epoch_index + 1}_{variable_name}"] = sweep_value
                rows.append(
                    schedule_row_from_flat_values(
                        config,
                        candidate,
                        epoch_count=epoch_count,
                        total_phases=total_phases,
                        schedule_id=f"bound_{sample_index:04d}",
                        source="bound_relaxation",
                        sample_index=sample_index,
                        metadata={
                            "experiment_id": f"{variable_name}_high_{format_token(high)}",
                            "base_schedule_id": str(base_row["schedule_id"]),
                            "seed_offset": seed_offset,
                            "relaxed_variable": variable_name,
                            "relaxed_high": float(high),
                            "sweep_value": float(sweep_value),
                            "counterfactual": "one_at_a_time_bound",
                        },
                    )
                )
                sample_index += 1
    return dedupe_candidates(pd.DataFrame(rows), feature_names)


def bound_relaxation_summary(
    evaluations: pd.DataFrame,
    best_by_experiment: pd.DataFrame,
    *,
    base_schedule_id: str,
    high_overrides: Mapping[str, float],
) -> dict[str, object]:
    ok = evaluations[evaluations["status"] == "ok"].copy()
    feasible = ok[ok["feasible"].astype(bool)] if not ok.empty else ok
    best = ok.sort_values(["phi", "schedule_id"]).head(1)
    best_row = best.iloc[0] if not best.empty else None
    return {
        "base_schedule_id": base_schedule_id,
        "evaluated_count": int(len(evaluations)),
        "ok_count": int(len(ok)),
        "feasible_count": int(len(feasible)),
        "best_phi": None if best_row is None else float(best_row["phi"]),
        "best_schedule_id": None if best_row is None else str(best_row["schedule_id"]),
        "best_experiment_id": None if best_row is None else str(best_row.get("experiment_id")),
        "best_active_constraint": None if best_row is None else str(best_row.get("active_constraint")),
        "high_overrides": {name: float(value) for name, value in high_overrides.items()},
        "best_positive_constraints": positive_constraints(best_row),
        "best_by_experiment_count": int(len(best_by_experiment)),
        "note": (
            "One-at-a-time fixed-shape input-bound relaxation evidence. "
            "This does not prove global feasibility or infeasibility under widened bounds."
        ),
    }


def render_bound_relaxation_report(
    summary: Mapping[str, object],
    best_by_experiment: pd.DataFrame,
) -> str:
    lines = [
        "# Dynamic Input-Bound Relaxation Study",
        "",
        "This study widens selected policy-input upper bounds and evaluates fixed-shape one-at-a-time sweeps around the best observed schedule.",
        "It is direct-physics evidence for local bound sensitivity, not a global proof under widened bounds.",
        "",
        "## Summary",
        "",
        f"- Base schedule: {summary['base_schedule_id']}",
        f"- Evaluated rows: {summary['evaluated_count']}",
        f"- Direct-feasible rows: {summary['feasible_count']}",
        f"- Best phi: {format_optional(summary['best_phi'])}",
        f"- Best experiment: {summary['best_experiment_id']}",
        f"- Best active constraint: {summary['best_active_constraint']}",
        "",
        "## Best Positive Constraints",
        "",
    ]
    positive = summary.get("best_positive_constraints") or {}
    if positive:
        for name, value in positive.items():
            lines.append(f"- {name}: {float(value):.6g}")
    else:
        lines.append("- none")
    lines.extend(["", "## Best By Experiment", ""])
    display_columns = [
        column
        for column in [
            "experiment_id",
            "relaxed_variable",
            "relaxed_high",
            "sweep_value",
            "schedule_id",
            "phi",
            "feasible",
            "active_constraint",
            "constraint_total_pilots_window",
            "constraint_wg_rap",
            "constraint_fl_rap",
            "constraint_ip_rap",
        ]
        if column in best_by_experiment.columns
    ]
    if display_columns:
        lines.extend(markdown_table(best_by_experiment[display_columns].head(40)))
    return "\n".join(lines) + "\n"


def max_high_overrides(
    config: ViabilityConfig,
    bound_extensions: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    overrides = {}
    for name, values in bound_extensions.items():
        if name not in config.policy.variables:
            raise ValueError(f"Unknown policy variable {name!r}")
        if not values:
            continue
        current_high = config.policy.variables[name].high
        new_high = max(float(value) for value in values)
        if new_high > current_high:
            overrides[name] = new_high
    return overrides
