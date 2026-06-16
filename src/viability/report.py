from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.viability.config import ReportConfig, ViabilityConfig
from src.viability.surrogate import read_evaluations_table


@dataclass(frozen=True)
class ReportResult:
    report_path: Path
    feasible_count: int
    verified_count: int
    best_candidate_id: str


def write_viability_report_from_files(
    *,
    config: ViabilityConfig,
    evaluations_path: str | Path,
    verified_candidates_path: str | Path,
    search_summary_path: str | Path,
    verification_summary_path: str | Path,
    envelope_summary_path: str | Path,
    output_path: str | Path,
) -> ReportResult:
    evaluations = read_evaluations_table(evaluations_path)
    verified_candidates = read_evaluations_table(verified_candidates_path)
    search_summary = _read_json_object(search_summary_path)
    verification_summary = _read_json_object(verification_summary_path)
    envelope_summary = _read_json_object(envelope_summary_path)
    return write_viability_report(
        config=config,
        evaluations=evaluations,
        verified_candidates=verified_candidates,
        search_summary=search_summary,
        verification_summary=verification_summary,
        envelope_summary=envelope_summary,
        output_path=output_path,
    )


def write_viability_report(
    *,
    config: ViabilityConfig,
    evaluations: pd.DataFrame,
    verified_candidates: pd.DataFrame,
    search_summary: dict[str, Any],
    verification_summary: dict[str, Any],
    envelope_summary: dict[str, Any],
    output_path: str | Path,
) -> ReportResult:
    report_config = require_report_config(config)
    _require_verified_columns(verified_candidates)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    best = verified_candidates.sort_values(["phi", "candidate_id"]).iloc[0]
    feasible = verified_candidates.loc[verified_candidates["feasible"].astype(bool)].copy()
    infeasible = verified_candidates.loc[~verified_candidates["feasible"].astype(bool)].copy()
    near_boundary = feasible.assign(_abs_phi=feasible["phi"].astype(float).abs()).sort_values(
        ["_abs_phi", "phi", "candidate_id"]
    )
    top_candidates = verified_candidates.sort_values(["phi", "candidate_id"]).head(
        report_config.top_candidate_count
    )
    near_boundary = near_boundary.head(report_config.near_boundary_count)

    lines = [
        "# Viability Search Report",
        "",
        "## Summary",
        "",
        f"- Direct evaluation rows supplied: `{len(evaluations)}`",
        f"- Verified candidate rows: `{len(verified_candidates)}`",
        f"- Verified feasible candidates: `{len(feasible)}`",
        f"- Verified infeasible candidates: `{len(infeasible)}`",
        f"- Best verified candidate: `{best['candidate_id']}` / `{best['design_id']}`",
        f"- Best verified `phi`: `{float(best['phi']):.6g}`",
        f"- Best active constraint: `{best['active_constraint']}`",
        "",
        (
            "Direct verification is the source of truth in this report. "
            "Surrogate predictions are used for screening and plotting only."
        ),
        "",
        "## Requirements",
        "",
        _markdown_table(_requirements_rows(config), ["Requirement", "Value"]),
        "",
        "## Policy Bounds",
        "",
        _markdown_table(_policy_bound_rows(config), ["Lever", "Type", "Low", "High"]),
        "",
        "## Search And Verification",
        "",
        _markdown_table(
            _search_verification_rows(search_summary, verification_summary),
            ["Metric", "Value"],
        ),
        "",
        "## Best Verified Policies",
        "",
        _markdown_table(_candidate_rows(top_candidates), _candidate_headers()),
        "",
        "## Near-Boundary Feasible Policies",
        "",
        _markdown_table(_candidate_rows(near_boundary), _candidate_headers()),
        "",
        "## Binding Constraints",
        "",
        _markdown_table(
            _active_constraint_rows(verified_candidates),
            ["Active Constraint", "Count"],
        ),
        "",
        "## Envelope Plots",
        "",
        *_plot_lines(envelope_summary, output_file.parent),
        "",
        "## Caveats",
        "",
        "- The prototype currently evaluates constant policies only.",
        "- Feasible-envelope plots are surrogate-predicted explanatory views.",
        (
            "- Projected envelopes minimize hidden policy levers on the surrogate, "
            "then require direct verification before any policy claim."
        ),
        (
            "- The current report consumes one direct-evaluations table plus "
            "one verified-candidates table."
        ),
        "",
    ]
    output_file.write_text("\n".join(lines), encoding="utf-8")
    return ReportResult(
        report_path=output_file.resolve(),
        feasible_count=int(len(feasible)),
        verified_count=int(len(verified_candidates)),
        best_candidate_id=str(best["candidate_id"]),
    )


def require_report_config(config: ViabilityConfig) -> ReportConfig:
    if config.report is None:
        raise ValueError("Config must include a report section for make-report")
    return config.report


def _read_json_object(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {input_path}")
    return data


def _require_verified_columns(verified_candidates: pd.DataFrame) -> None:
    required = (
        "candidate_id",
        "design_id",
        "phi",
        "feasible",
        "active_constraint",
        "annual_intake",
        "retention_rate",
        "ute",
        "paa",
        "max_manning_pct",
        "flug_quota_per_phase",
        "ipug_quota_per_phase",
    )
    missing = [column for column in required if column not in verified_candidates.columns]
    if missing:
        raise ValueError(f"Verified candidates table is missing required columns: {missing}")


def _requirements_rows(config: ViabilityConfig) -> list[list[str]]:
    rows = []
    for name, value in config.requirements.__dict__.items():
        rows.append([name, _format_value(value)])
    return rows


def _policy_bound_rows(config: ViabilityConfig) -> list[list[str]]:
    rows = []
    for name, variable in config.policy.variables.items():
        rows.append(
            [
                name,
                variable.type,
                _format_value(variable.low),
                _format_value(variable.high),
            ]
        )
    return rows


def _search_verification_rows(
    search_summary: dict[str, Any],
    verification_summary: dict[str, Any],
) -> list[list[str]]:
    pairs = [
        ("Scored candidate count", search_summary["scored_count"]),
        ("Selected candidate count", search_summary["selected_count"]),
        ("Verified count", verification_summary["verified_count"]),
        ("Verified feasible count", verification_summary["verified_feasible_count"]),
        ("Predicted feasible count", verification_summary["predicted_feasible_count"]),
        (
            "Conservative predicted feasible count",
            verification_summary["conservative_predicted_feasible_count"],
        ),
        ("False feasible count", verification_summary["false_feasible_count"]),
        (
            "False conservative feasible count",
            verification_summary["false_conservative_feasible_count"],
        ),
        ("Best verified phi", verification_summary["best_verified_phi"]),
    ]
    return [[name, _format_value(value)] for name, value in pairs]


def _candidate_headers() -> list[str]:
    return [
        "Candidate",
        "Phi",
        "Feasible",
        "Active Constraint",
        "Annual Intake",
        "Retention",
        "UTE",
        "PAA",
        "Max Manning",
        "FLUG",
        "IPUG",
    ]


def _candidate_rows(frame: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                str(row["candidate_id"]),
                _format_value(row["phi"]),
                str(bool(row["feasible"])),
                str(row["active_constraint"]),
                _format_value(row["annual_intake"]),
                _format_value(row["retention_rate"]),
                _format_value(row["ute"]),
                _format_value(row["paa"]),
                _format_value(row["max_manning_pct"]),
                _format_value(row["flug_quota_per_phase"]),
                _format_value(row["ipug_quota_per_phase"]),
            ]
        )
    return rows


def _active_constraint_rows(verified_candidates: pd.DataFrame) -> list[list[str]]:
    counts = verified_candidates["active_constraint"].fillna("none").value_counts().sort_index()
    return [[str(name), str(int(count))] for name, count in counts.items()]


def _plot_lines(envelope_summary: dict[str, Any], report_dir: Path) -> list[str]:
    if "slices" not in envelope_summary:
        raise ValueError("Envelope summary is missing required key 'slices'")
    lines = []
    for item in envelope_summary["slices"]:
        x_name = item["x"]
        y_name = item["y"]
        fixed_path = _relative_path(item["fixed_plot_path"], report_dir)
        projected_path = _relative_path(item["projected_plot_path"], report_dir)
        lines.extend(
            [
                f"### {x_name} vs {y_name}",
                "",
                f"![Fixed {x_name} vs {y_name}]({fixed_path})",
                "",
                f"![Projected {x_name} vs {y_name}]({projected_path})",
                "",
            ]
        )
    return lines


def _markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(_escape_markdown(value) for value in row) + " |")
    return "\n".join(output)


def _format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if hasattr(value, "item"):
        return _format_value(value.item())
    return str(value)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")


def _relative_path(path: str, report_dir: Path) -> str:
    return os.path.relpath(Path(path), report_dir)
