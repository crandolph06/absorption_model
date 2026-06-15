from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.viability.config import ViabilityConfig
from src.viability.io import write_config_resolved, write_table
from src.viability.surrogate import read_evaluations_table


DEFAULT_RELAXATION_CONSTRAINTS = (
    "total_pilots_window",
    "wg_rap",
    "fl_rap",
    "ip_rap",
)


@dataclass(frozen=True)
class DynamicRelaxationStudyResult:
    output_dir: Path
    nearest_path: Path
    pareto_path: Path
    relaxation_sets_path: Path
    summary_path: Path
    report_path: Path
    evaluated_count: int
    feasible_count: int
    best_phi: float
    best_linf_relaxation: float


def run_dynamic_relaxation_study(
    *,
    config: ViabilityConfig,
    evaluation_paths: Sequence[str | Path],
    output_dir: str | Path,
    constraints: Sequence[str] = DEFAULT_RELAXATION_CONSTRAINTS,
    top_n: int = 50,
) -> DynamicRelaxationStudyResult:
    if not evaluation_paths:
        raise ValueError("At least one evaluations path is required")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_config_resolved(config, output_path)

    frames = []
    for path in evaluation_paths:
        frame = read_evaluations_table(path).copy()
        frame.loc[:, "source_path"] = str(path)
        frames.append(frame)
    evaluations = pd.concat(frames, ignore_index=True, sort=False)
    study = build_relaxation_study(
        evaluations,
        config,
        constraints=constraints,
        top_n=top_n,
    )

    nearest_path = write_table(
        study["nearest"],
        output_path / "nearest_under_relaxation.csv",
        prefer_parquet=False,
    )
    pareto_path = write_table(
        study["pareto"],
        output_path / "pareto_frontier.csv",
        prefer_parquet=False,
    )
    relaxation_sets_path = write_table(
        study["relaxation_sets"],
        output_path / "relaxation_sets.csv",
        prefer_parquet=False,
    )
    summary = study["summary"]
    summary.update(
        {
            "output_dir": str(output_path.resolve()),
            "evaluation_paths": [str(path) for path in evaluation_paths],
            "nearest_path": str(nearest_path.resolve()),
            "pareto_path": str(pareto_path.resolve()),
            "relaxation_sets_path": str(relaxation_sets_path.resolve()),
        }
    )
    summary_path = output_path / "relaxation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output_path / "relaxation_report.md"
    report_path.write_text(
        render_relaxation_report(summary, study["relaxation_sets"]),
        encoding="utf-8",
    )

    return DynamicRelaxationStudyResult(
        output_dir=output_path.resolve(),
        nearest_path=nearest_path.resolve(),
        pareto_path=pareto_path.resolve(),
        relaxation_sets_path=relaxation_sets_path.resolve(),
        summary_path=summary_path.resolve(),
        report_path=report_path.resolve(),
        evaluated_count=int(summary["evaluated_count"]),
        feasible_count=int(summary["feasible_count"]),
        best_phi=float(summary["best_phi"]),
        best_linf_relaxation=float(summary["best_linf_relaxation"]),
    )


def build_relaxation_study(
    evaluations: pd.DataFrame,
    config: ViabilityConfig,
    *,
    constraints: Sequence[str] = DEFAULT_RELAXATION_CONSTRAINTS,
    top_n: int = 50,
) -> dict[str, object]:
    ok = evaluations[evaluations["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No ok dynamic evaluations are available")
    constraint_names = list(constraints)
    constraint_columns = [f"constraint_{name}" for name in constraint_names]
    missing = [column for column in constraint_columns if column not in ok.columns]
    if missing:
        raise ValueError(f"Evaluations are missing relaxation constraint columns: {missing}")

    scored = add_relaxation_columns(ok, config, constraint_names)
    nearest = scored.sort_values(
        ["max_normalized_relaxation", "positive_normalized_sum", "phi", "schedule_id"]
    ).head(top_n)
    pareto = pareto_frontier(scored, [f"relax_{name}" for name in constraint_names])
    relaxation_sets = relaxation_set_table(scored, config, constraint_names)
    best_phi = scored.sort_values(["phi", "schedule_id"]).iloc[0]
    best_linf = nearest.iloc[0]
    feasible = scored[scored["feasible"].astype(bool)]
    summary = {
        "evaluated_count": int(len(evaluations)),
        "ok_count": int(len(scored)),
        "feasible_count": int(len(feasible)),
        "constraint_names": constraint_names,
        "best_phi": float(best_phi["phi"]),
        "best_phi_schedule_id": str(best_phi["schedule_id"]),
        "best_phi_active_constraint": str(best_phi["active_constraint"]),
        "best_linf_relaxation": float(best_linf["max_normalized_relaxation"]),
        "best_linf_schedule_id": str(best_linf["schedule_id"]),
        "pareto_count": int(len(pareto)),
        "note": (
            "Observed-search relaxation evidence only; this is not a proof of "
            "mathematical infeasibility."
        ),
    }
    for name in constraint_names:
        summary[f"best_phi_relax_{name}"] = float(best_phi[f"relax_{name}"])
        summary[f"best_linf_relax_{name}"] = float(best_linf[f"relax_{name}"])
        summary[f"min_relax_{name}"] = float(scored[f"relax_{name}"].min())
    return {
        "scored": scored,
        "nearest": nearest,
        "pareto": pareto,
        "relaxation_sets": relaxation_sets,
        "summary": summary,
    }


def add_relaxation_columns(
    evaluations: pd.DataFrame,
    config: ViabilityConfig,
    constraint_names: Sequence[str],
) -> pd.DataFrame:
    frame = evaluations.copy()
    normalized_columns = []
    raw_columns = []
    for name in constraint_names:
        column = f"constraint_{name}"
        relax_column = f"relax_{name}"
        normalized_column = f"normalized_relax_{name}"
        scale = config.constraint_scales.scale_for(name)
        frame.loc[:, relax_column] = frame[column].astype(float).clip(lower=0.0)
        frame.loc[:, normalized_column] = frame[relax_column] / float(scale)
        raw_columns.append(relax_column)
        normalized_columns.append(normalized_column)
    frame.loc[:, "max_normalized_relaxation"] = frame[normalized_columns].max(axis=1)
    frame.loc[:, "positive_normalized_sum"] = frame[normalized_columns].sum(axis=1)
    frame.loc[:, "positive_constraint_sum"] = frame[raw_columns].sum(axis=1)
    frame.loc[:, "violated_constraint_count"] = (frame[raw_columns] > 0.0).sum(axis=1)
    return frame


def pareto_frontier(frame: pd.DataFrame, relaxation_columns: Sequence[str]) -> pd.DataFrame:
    if not relaxation_columns:
        raise ValueError("At least one relaxation column is required")
    values = frame[list(relaxation_columns)].to_numpy(dtype=float)
    keep = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        if not keep[i]:
            continue
        dominated = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        if dominated.any():
            keep[i] = False
    return frame.loc[keep].sort_values(
        ["max_normalized_relaxation", "positive_normalized_sum", "phi", "schedule_id"]
    )


def relaxation_set_table(
    scored: pd.DataFrame,
    config: ViabilityConfig,
    constraint_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for size in range(1, len(constraint_names) + 1):
        for subset in itertools.combinations(constraint_names, size):
            normalized_columns = [f"normalized_relax_{name}" for name in subset]
            raw_columns = [f"relax_{name}" for name in subset]
            working = scored.copy()
            working.loc[:, "_set_max_normalized"] = working[normalized_columns].max(axis=1)
            working.loc[:, "_set_sum_normalized"] = working[normalized_columns].sum(axis=1)
            best = working.sort_values(
                ["_set_max_normalized", "_set_sum_normalized", "phi", "schedule_id"]
            ).iloc[0]
            row = {
                "constraint_set": "+".join(subset),
                "constraint_count": len(subset),
                "schedule_id": best["schedule_id"],
                "phi": float(best["phi"]),
                "max_normalized_relaxation": float(best["_set_max_normalized"]),
                "sum_normalized_relaxation": float(best["_set_sum_normalized"]),
            }
            for name, raw_column in zip(subset, raw_columns, strict=True):
                row[f"relax_{name}"] = float(best[raw_column])
                row[f"normalized_relax_{name}"] = float(best[f"normalized_relax_{name}"])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["constraint_count", "max_normalized_relaxation", "sum_normalized_relaxation"]
    )


def render_relaxation_report(
    summary: dict[str, object],
    relaxation_sets: pd.DataFrame,
) -> str:
    lines = [
        "# Dynamic Policy Requirement-Relaxation Study",
        "",
        "This report summarizes observed direct-physics dynamic evaluations.",
        "It is not a mathematical infeasibility proof.",
        "",
        "## Summary",
        "",
        f"- Evaluated rows: {summary['evaluated_count']}",
        f"- Successful rows: {summary['ok_count']}",
        f"- Direct-feasible rows: {summary['feasible_count']}",
        f"- Best phi: {summary['best_phi']:.6g} ({summary['best_phi_schedule_id']})",
        (
            "- Minimum L-infinity normalized relaxation: "
            f"{summary['best_linf_relaxation']:.6g} ({summary['best_linf_schedule_id']})"
        ),
        "",
        "## Best Observed Relaxations",
        "",
    ]
    for name in summary["constraint_names"]:
        lines.append(f"- {name}: minimum observed raw relaxation {summary[f'min_relax_{name}']:.6g}")
    lines.extend(["", "## Constraint Set Minima", ""])
    display_columns = [
        "constraint_set",
        "schedule_id",
        "max_normalized_relaxation",
        "sum_normalized_relaxation",
    ]
    lines.extend(_markdown_table(relaxation_sets[display_columns]))
    lines.append("")
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows
