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
    format_optional,
    format_token,
    markdown_table,
    positive_constraints,
    raw_values_from_row,
    schedule_row_from_flat_values,
)
from src.viability.dynamic_policy import dynamic_feature_names
from src.viability.evaluator import evaluate_schedules_parallel
from src.viability.io import write_config_resolved, write_table
from src.viability.surrogate import read_evaluations_table


@dataclass(frozen=True)
class DynamicIpugDiagnosticResult:
    output_dir: Path
    candidates_path: Path
    evaluations_path: Path
    summary_path: Path
    report_path: Path
    evaluated_count: int
    feasible_count: int
    best_phi: float


def run_dynamic_ipug_diagnostic(
    *,
    config: ViabilityConfig,
    evaluations_path: str | Path,
    output_dir: str | Path,
    epoch_count: int,
    ipug_values: Sequence[float] = (0, 2, 5, 8, 10, 15, 20),
    workers: int | None = None,
    checkpoint_every: int = 10,
    evaluator: EvaluateSchedules = evaluate_schedules_parallel,
) -> DynamicIpugDiagnosticResult:
    """Evaluate fixed-shape IPUG counterfactuals around the best schedule."""
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    if not ipug_values:
        raise ValueError("At least one IPUG value is required")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    max_ipug = max(float(value) for value in ipug_values)
    widened_config = clone_config_with_policy_highs(
        config,
        {"ipug_quota_per_phase": max(max_ipug, config.policy.variables["ipug_quota_per_phase"].high)},
    )
    write_config_resolved(widened_config, output_path)

    evaluations = read_evaluations_table(evaluations_path)
    best_row = best_ok_row(evaluations)
    candidates = generate_ipug_counterfactual_candidates(
        widened_config,
        best_row,
        epoch_count=epoch_count,
        ipug_values=ipug_values,
    )
    candidates_path = write_table(
        candidates,
        output_path / "ipug_counterfactual_candidates.csv",
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
        output_path / "ipug_counterfactual_evaluations.parquet",
    )
    summary = ipug_diagnostic_summary(direct_evaluations, str(best_row["schedule_id"]))
    summary_path = output_path / "ipug_counterfactual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output_path / "ipug_counterfactual_report.md"
    report_path.write_text(
        render_ipug_diagnostic_report(summary, direct_evaluations),
        encoding="utf-8",
    )
    ok = direct_evaluations[direct_evaluations["status"] == "ok"]
    return DynamicIpugDiagnosticResult(
        output_dir=output_path.resolve(),
        candidates_path=candidates_path.resolve(),
        evaluations_path=evaluations_output_path.resolve(),
        summary_path=summary_path.resolve(),
        report_path=report_path.resolve(),
        evaluated_count=int(len(direct_evaluations)),
        feasible_count=int(ok["feasible"].astype(bool).sum()) if not ok.empty else 0,
        best_phi=float(ok["phi"].min()) if not ok.empty else float("inf"),
    )


def generate_ipug_counterfactual_candidates(
    config: ViabilityConfig,
    base_row: pd.Series,
    *,
    epoch_count: int,
    ipug_values: Sequence[float],
) -> pd.DataFrame:
    feature_names = dynamic_feature_names(config.policy, epoch_count)
    base_values = raw_values_from_row(base_row, feature_names)
    seed_offset = base_seed_offset(base_row)
    total_phases = config.model.years_to_run * 3
    rows = []
    for index, value in enumerate(sorted(set(float(item) for item in ipug_values))):
        candidate = dict(base_values)
        for epoch_index in range(epoch_count):
            candidate[f"epoch{epoch_index + 1}_ipug_quota_per_phase"] = value
        rows.append(
            schedule_row_from_flat_values(
                config,
                candidate,
                epoch_count=epoch_count,
                total_phases=total_phases,
                schedule_id=f"ipug_{index:04d}",
                source="ipug_counterfactual",
                sample_index=index,
                metadata={
                    "experiment_id": f"ipug_all_epochs_{format_token(value)}",
                    "base_schedule_id": str(base_row["schedule_id"]),
                    "seed_offset": seed_offset,
                    "relaxed_variable": "ipug_quota_per_phase",
                    "relaxed_high": float(config.policy.variables["ipug_quota_per_phase"].high),
                    "sweep_value": float(value),
                    "counterfactual": "ipug_all_epochs",
                },
            )
        )
    return pd.DataFrame(rows)


def ipug_diagnostic_summary(evaluations: pd.DataFrame, base_schedule_id: str) -> dict[str, object]:
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
        "best_ipug_all_epochs": None if best_row is None else float(best_row.get("sweep_value")),
        "best_active_constraint": None if best_row is None else str(best_row.get("active_constraint")),
        "best_positive_constraints": positive_constraints(best_row),
        "note": "Fixed-shape IPUG counterfactual around the best observed schedule.",
    }


def render_ipug_diagnostic_report(
    summary: Mapping[str, object],
    evaluations: pd.DataFrame,
) -> str:
    lines = [
        "# Dynamic IPUG Counterfactual Diagnostic",
        "",
        "This study changes only the IPUG quota in every epoch of the best observed schedule and reruns direct physics.",
        "",
        "## Summary",
        "",
        f"- Base schedule: {summary['base_schedule_id']}",
        f"- Evaluated rows: {summary['evaluated_count']}",
        f"- Direct-feasible rows: {summary['feasible_count']}",
        f"- Best phi: {format_optional(summary['best_phi'])}",
        f"- Best all-epoch IPUG value: {format_optional(summary['best_ipug_all_epochs'])}",
        f"- Best active constraint: {summary['best_active_constraint']}",
        "",
        "## Evaluations",
        "",
    ]
    display_columns = [
        column
        for column in [
            "sweep_value",
            "phi",
            "feasible",
            "active_constraint",
            "constraint_total_pilots_window",
            "constraint_wg_rap",
            "constraint_fl_rap",
            "constraint_ip_rap",
            "metric_min_staff_ips_after_assessment_start",
            "metric_max_ip_rap_shortfall_after_assessment_start",
        ]
        if column in evaluations.columns
    ]
    if display_columns:
        lines.extend(markdown_table(evaluations[display_columns].sort_values("sweep_value")))
    return "\n".join(lines) + "\n"
