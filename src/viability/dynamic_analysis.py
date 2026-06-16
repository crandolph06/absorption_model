from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.viability.config import PolicyConfig, VariableConfig, ViabilityConfig
from src.viability.dashboard import aggregate_history_trajectory
from src.viability.dynamic_policy import (
    EpochPolicySchedule,
    dynamic_feature_names,
)
from src.viability.evaluator import (
    evaluate_schedules_parallel,
    simulate_policy_schedule_history,
)
from src.viability.io import write_config_resolved, write_table
from src.viability.surrogate import read_evaluations_table


EvaluateSchedules = Callable[..., pd.DataFrame]
ScheduleHistoryRunner = Callable[[EpochPolicySchedule, ViabilityConfig], pd.DataFrame]


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


@dataclass(frozen=True)
class DynamicTrajectoryArtifactResult:
    output_dir: Path
    summary_path: Path
    selected_policies_path: Path
    trajectory_paths: dict[str, Path]
    figure_paths: dict[str, Path]


def clone_config_with_policy_highs(
    config: ViabilityConfig,
    high_overrides: Mapping[str, float],
) -> ViabilityConfig:
    """Return a config copy with explicit upper-bound overrides."""
    variables = dict(config.policy.variables)
    for name, new_high in high_overrides.items():
        if name not in variables:
            raise ValueError(f"Unknown policy variable {name!r}")
        variable = variables[name]
        if float(new_high) < variable.low:
            raise ValueError(
                f"New high for {name}={new_high} is below low bound {variable.low}"
            )
        variables[name] = replace(variable, high=float(new_high))
    return replace(
        config,
        policy=PolicyConfig(
            parameterization=config.policy.parameterization,
            variables=variables,
        ),
    )


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
    high_overrides = _max_high_overrides(config, extensions)
    widened_config = clone_config_with_policy_highs(config, high_overrides)
    write_config_resolved(widened_config, output_path)

    evaluations = read_evaluations_table(evaluations_path)
    best_row = _best_ok_row(evaluations)
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
    base_values = _raw_values_from_row(base_row, feature_names)
    seed_offset = _base_seed_offset(base_row)
    total_phases = config.model.years_to_run * 3
    rows: list[dict[str, object]] = []

    rows.append(
        _schedule_row_from_flat_values(
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
            values = _sweep_values_for_variable(
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
                    _schedule_row_from_flat_values(
                        config,
                        candidate,
                        epoch_count=epoch_count,
                        total_phases=total_phases,
                        schedule_id=f"bound_{sample_index:04d}",
                        source="bound_relaxation",
                        sample_index=sample_index,
                        metadata={
                            "experiment_id": f"{variable_name}_high_{_format_token(high)}",
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
    return _dedupe_candidates(pd.DataFrame(rows), feature_names)


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
    best_row = _best_ok_row(evaluations)
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
    base_values = _raw_values_from_row(base_row, feature_names)
    seed_offset = _base_seed_offset(base_row)
    total_phases = config.model.years_to_run * 3
    rows = []
    for index, value in enumerate(sorted(set(float(item) for item in ipug_values))):
        candidate = dict(base_values)
        for epoch_index in range(epoch_count):
            candidate[f"epoch{epoch_index + 1}_ipug_quota_per_phase"] = value
        rows.append(
            _schedule_row_from_flat_values(
                config,
                candidate,
                epoch_count=epoch_count,
                total_phases=total_phases,
                schedule_id=f"ipug_{index:04d}",
                source="ipug_counterfactual",
                sample_index=index,
                metadata={
                    "experiment_id": f"ipug_all_epochs_{_format_token(value)}",
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


def run_dynamic_trajectory_artifacts(
    *,
    config: ViabilityConfig,
    evaluation_specs: Sequence[tuple[str | Path, int, str]],
    output_dir: str | Path,
    history_runner: ScheduleHistoryRunner | None = None,
) -> DynamicTrajectoryArtifactResult:
    """Rerun selected schedules and write trajectories plus publication figures."""
    if not evaluation_specs:
        raise ValueError("At least one evaluation spec is required")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_config_resolved(config, output_path)
    runner = history_runner or _default_history_runner

    selected_rows = []
    trajectory_paths: dict[str, Path] = {}
    trajectories: dict[str, pd.DataFrame] = {}
    combined_evaluations = []
    for path, epoch_count, label in evaluation_specs:
        evaluations = read_evaluations_table(path)
        combined_evaluations.append(evaluations.assign(result_label=label))
        row = _best_ok_row(evaluations)
        schedule = _schedule_from_row(row, config, epoch_count=epoch_count)
        history = runner(schedule, config)
        trajectory = aggregate_history_trajectory(history, config)
        key = _safe_name(label)
        trajectory_path = output_path / f"trajectory_{key}.csv"
        trajectory.to_csv(trajectory_path, index=False)
        trajectory_paths[label] = trajectory_path.resolve()
        trajectories[label] = trajectory
        selected_rows.append(_selected_policy_summary(row, label, epoch_count))

    selected = pd.DataFrame(selected_rows)
    selected_path = write_table(
        selected,
        output_path / "selected_policies.csv",
        prefer_parquet=False,
    )
    all_evaluations = pd.concat(combined_evaluations, ignore_index=True, sort=False)
    figure_paths = write_dynamic_figures(
        trajectories=trajectories,
        selected_policies=selected,
        evaluations=all_evaluations,
        config=config,
        output_dir=output_path,
    )
    summary = {
        "output_dir": str(output_path.resolve()),
        "selected_count": int(len(selected)),
        "selected_policies_path": str(selected_path.resolve()),
        "trajectory_paths": {name: str(path) for name, path in trajectory_paths.items()},
        "figure_paths": {name: str(path) for name, path in figure_paths.items()},
    }
    summary_path = output_path / "trajectory_artifacts_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return DynamicTrajectoryArtifactResult(
        output_dir=output_path.resolve(),
        summary_path=summary_path.resolve(),
        selected_policies_path=selected_path.resolve(),
        trajectory_paths=trajectory_paths,
        figure_paths=figure_paths,
    )


def summarize_best_by_experiment(evaluations: pd.DataFrame) -> pd.DataFrame:
    ok = evaluations[evaluations["status"] == "ok"].copy()
    if ok.empty:
        return ok
    if "experiment_id" not in ok.columns:
        raise ValueError("Evaluations are missing experiment_id metadata")
    rows = []
    for experiment_id, group in ok.groupby("experiment_id", dropna=False):
        best = group.sort_values(["phi", "schedule_id"]).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).sort_values(["phi", "experiment_id"]).reset_index(drop=True)


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
        "best_positive_constraints": _positive_constraints(best_row),
        "best_by_experiment_count": int(len(best_by_experiment)),
        "note": (
            "One-at-a-time fixed-shape input-bound relaxation evidence. "
            "This does not prove global feasibility or infeasibility under widened bounds."
        ),
    }


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
        "best_positive_constraints": _positive_constraints(best_row),
        "note": "Fixed-shape IPUG counterfactual around the best observed schedule.",
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
        f"- Best phi: {_format_optional(summary['best_phi'])}",
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
        lines.extend(_markdown_table(best_by_experiment[display_columns].head(40)))
    return "\n".join(lines) + "\n"


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
        f"- Best phi: {_format_optional(summary['best_phi'])}",
        f"- Best all-epoch IPUG value: {_format_optional(summary['best_ipug_all_epochs'])}",
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
        lines.extend(_markdown_table(evaluations[display_columns].sort_values("sweep_value")))
    return "\n".join(lines) + "\n"


def write_dynamic_figures(
    *,
    trajectories: Mapping[str, pd.DataFrame],
    selected_policies: pd.DataFrame,
    evaluations: pd.DataFrame,
    config: ViabilityConfig,
    output_dir: Path,
) -> dict[str, Path]:
    _configure_matplotlib_cache(output_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}
    colors = ["#2563eb", "#7c3aed", "#059669", "#ea580c", "#111827"]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for idx, (label, trajectory) in enumerate(trajectories.items()):
        ax.plot(
            _trajectory_x(trajectory),
            trajectory["total_pilots"],
            label=label,
            color=colors[idx % len(colors)],
            linewidth=2.2,
        )
    if config.requirements.target_total_pilots is not None:
        ax.axhline(
            float(config.requirements.target_total_pilots),
            color="#b91c1c",
            linestyle="--",
            linewidth=1.4,
            label="3500 target",
        )
    _mark_assessment_start(ax, config)
    ax.set_xlabel("Simulation phase")
    ax.set_ylabel("Total pilots")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    paths["inventory"] = _save_figure(fig, output_dir / "trajectory_total_pilots.png", plt)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
    rap_columns = [
        ("wg_rap_margin", "WG RAP shortfall"),
        ("fl_rap_margin", "FL RAP shortfall"),
        ("ip_rap_margin", "IP RAP shortfall"),
    ]
    for axis, (column, ylabel) in zip(axes, rap_columns, strict=True):
        for idx, (label, trajectory) in enumerate(trajectories.items()):
            axis.plot(
                _trajectory_x(trajectory),
                trajectory[column],
                label=label,
                color=colors[idx % len(colors)],
                linewidth=1.9,
            )
        axis.axhline(0.0, color="#b91c1c", linestyle="--", linewidth=1.1)
        _mark_assessment_start(axis, config)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Simulation phase")
    axes[0].legend(loc="best", fontsize=8)
    paths["rap"] = _save_figure(fig, output_dir / "trajectory_rap_shortfalls.png", plt)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for idx, (label, trajectory) in enumerate(trajectories.items()):
        x_values = _trajectory_x(trajectory)
        ax.plot(
            x_values,
            trajectory["staff_ips"],
            label=f"{label} staff IPs",
            color=colors[idx % len(colors)],
            linewidth=2.0,
        )
        ax.plot(
            x_values,
            trajectory["staff_fls"],
            label=f"{label} staff FLs",
            color=colors[idx % len(colors)],
            linewidth=1.4,
            linestyle=":",
        )
    _mark_assessment_start(ax, config)
    ax.set_xlabel("Simulation phase")
    ax.set_ylabel("Staff counts")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    paths["staff"] = _save_figure(fig, output_dir / "trajectory_staff_counts.png", plt)

    fig, axes = plt.subplots(4, 2, figsize=(9.2, 8.2), sharex=True)
    axes_flat = axes.ravel()
    first_label = selected_policies.iloc[0]["label"]
    first_epoch_count = int(selected_policies.iloc[0]["epoch_count"])
    first_row = selected_policies.iloc[0]
    for axis, name in zip(axes_flat, config.policy.variables, strict=False):
        values = [float(first_row[f"epoch{epoch}_{name}"]) for epoch in range(1, first_epoch_count + 1)]
        axis.step(range(1, first_epoch_count + 1), values, where="mid", color="#2563eb", linewidth=2.0)
        axis.scatter(range(1, first_epoch_count + 1), values, color="#111827", s=18)
        axis.set_title(name.replace("_", " "), fontsize=9)
        axis.grid(True, alpha=0.25)
    for axis in axes_flat[len(config.policy.variables):]:
        axis.axis("off")
    axes_flat[0].set_ylabel(first_label)
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    paths["policy"] = _save_figure(fig, output_dir / "best_policy_epoch_controls.png", plt)

    ok = evaluations[evaluations["status"] == "ok"].copy()
    needed = {"constraint_total_pilots_window", "constraint_wg_rap", "constraint_fl_rap"}
    if needed.issubset(ok.columns):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        scatter = ax.scatter(
            ok["constraint_total_pilots_window"],
            ok["constraint_wg_rap"],
            c=ok["constraint_fl_rap"],
            cmap="viridis",
            s=20,
            alpha=0.75,
            edgecolor="none",
        )
        best = ok.sort_values(["phi", "schedule_id"]).iloc[0]
        ax.scatter(
            [best["constraint_total_pilots_window"]],
            [best["constraint_wg_rap"]],
            marker="*",
            s=160,
            color="#b91c1c",
            label="best observed",
            zorder=5,
        )
        ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Total-pilot-window violation")
        ax.set_ylabel("WG RAP violation")
        ax.legend(loc="best", fontsize=8)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("FL RAP violation")
        ax.grid(True, alpha=0.25)
        paths["trade_space"] = _save_figure(fig, output_dir / "trade_space_total_wg_fl.png", plt)

    return {name: path.resolve() for name, path in paths.items()}


def _max_high_overrides(
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


def _best_ok_row(evaluations: pd.DataFrame) -> pd.Series:
    required = ["schedule_id", "status", "phi"]
    missing = [column for column in required if column not in evaluations.columns]
    if missing:
        raise ValueError(f"Evaluations are missing required columns: {missing}")
    ok = evaluations[evaluations["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No successful dynamic evaluations are available")
    return ok.sort_values(["phi", "schedule_id"]).iloc[0]


def _raw_values_from_row(row: pd.Series, feature_names: Sequence[str]) -> dict[str, float]:
    values = {}
    for name in feature_names:
        raw_column = f"raw_{name}"
        if raw_column in row and pd.notna(row[raw_column]):
            values[name] = float(row[raw_column])
        elif name in row and pd.notna(row[name]):
            values[name] = float(row[name])
        else:
            raise ValueError(f"Dynamic row is missing schedule value {name!r}")
    return values


def _base_seed_offset(row: pd.Series) -> int:
    if "seed_offset" in row and pd.notna(row["seed_offset"]):
        return int(row["seed_offset"])
    if "sample_index" in row and pd.notna(row["sample_index"]):
        return int(row["sample_index"])
    schedule_id = str(row.get("schedule_id", ""))
    digits = "".join(reversed([char for char in reversed(schedule_id) if char.isdigit()]))
    if digits:
        return int(digits)
    return 0


def _schedule_from_row(
    row: pd.Series,
    config: ViabilityConfig,
    *,
    epoch_count: int,
) -> EpochPolicySchedule:
    feature_names = dynamic_feature_names(config.policy, epoch_count)
    values = {name: row[name] for name in feature_names}
    raw_values = {
        name: float(row[f"raw_{name}"])
        for name in feature_names
        if f"raw_{name}" in row and pd.notna(row[f"raw_{name}"])
    }
    return EpochPolicySchedule.from_flat_mapping(
        values,
        config.policy,
        epoch_count=epoch_count,
        total_phases=config.model.years_to_run * 3,
        raw_values=raw_values if raw_values else None,
    )


def _schedule_row_from_flat_values(
    config: ViabilityConfig,
    flat_values: Mapping[str, float],
    *,
    epoch_count: int,
    total_phases: int,
    schedule_id: str,
    source: str,
    sample_index: int,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    schedule = EpochPolicySchedule.from_flat_mapping(
        flat_values,
        config.policy,
        epoch_count=epoch_count,
        total_phases=total_phases,
        raw_values={name: float(value) for name, value in flat_values.items()},
    )
    raw = schedule.to_flat_dict(raw=True)
    applied = schedule.to_flat_dict(raw=False)
    row: dict[str, object] = {
        "schedule_id": schedule_id,
        "schedule_source": source,
        "sample_index": int(sample_index),
    }
    row.update(metadata)
    for name in dynamic_feature_names(config.policy, epoch_count):
        row[f"raw_{name}"] = raw[name]
        row[f"applied_{name}"] = applied[name]
        row[name] = applied[name]
    return row


def _sweep_values_for_variable(
    config: ViabilityConfig,
    base_values: Mapping[str, float],
    variable_name: str,
    *,
    epoch_count: int,
    high: float,
    sweep_points: int,
) -> list[float]:
    variable = config.policy.variables[variable_name]
    base_epoch_values = [
        float(base_values[f"epoch{epoch_index + 1}_{variable_name}"])
        for epoch_index in range(epoch_count)
    ]
    start = min(base_epoch_values)
    stop = float(high)
    if stop < start:
        start, stop = stop, start
    if variable.type == "int":
        raw_values = np.unique(np.rint(np.linspace(start, stop, sweep_points))).astype(float)
    else:
        raw_values = np.linspace(start, stop, sweep_points)
    clipped = [
        min(max(float(value), variable.low), variable.high)
        for value in raw_values
    ]
    return sorted(set(clipped))


def _dedupe_candidates(candidates: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    subset = [f"raw_{name}" for name in feature_names]
    deduped = candidates.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    deduped.loc[:, "sample_index"] = range(len(deduped))
    deduped.loc[:, "schedule_id"] = [f"bound_{index:04d}" for index in range(len(deduped))]
    deduped.loc[deduped["experiment_id"] == "baseline", "schedule_id"] = "bound_base"
    return deduped


def _positive_constraints(row: pd.Series | None) -> dict[str, float]:
    if row is None:
        return {}
    positive = {}
    for column, value in row.items():
        if str(column).startswith("constraint_"):
            numeric = float(value)
            if numeric > 0.0:
                positive[str(column).removeprefix("constraint_")] = numeric
    return positive


def _selected_policy_summary(row: pd.Series, label: str, epoch_count: int) -> dict[str, object]:
    output = {
        "label": label,
        "schedule_id": row["schedule_id"],
        "epoch_count": int(epoch_count),
        "phi": float(row["phi"]),
        "feasible": bool(row["feasible"]),
        "active_constraint": row.get("active_constraint"),
    }
    for column, value in row.items():
        if str(column).startswith("constraint_") or str(column).startswith("epoch"):
            output[str(column)] = value
    return output


def _default_history_runner(
    schedule: EpochPolicySchedule,
    config: ViabilityConfig,
) -> pd.DataFrame:
    return simulate_policy_schedule_history(schedule, config)


def _trajectory_x(trajectory: pd.DataFrame) -> np.ndarray:
    return np.arange(len(trajectory), dtype=float)


def _mark_assessment_start(axis, config: ViabilityConfig) -> None:
    phase_index = max(0, (config.model.assessment_start_year - config.model.start_year) * 3)
    axis.axvline(
        phase_index,
        color="#6b7280",
        linestyle=":",
        linewidth=1.0,
        label="assessment start",
    )


def _save_figure(fig, path: Path, plt_module) -> Path:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt_module.close(fig)
    return path


def _configure_matplotlib_cache(output_path: Path) -> None:
    cache_dir = output_path / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _safe_name(value: str) -> str:
    safe = []
    for char in str(value).lower():
        if char.isalnum():
            safe.append(char)
        elif char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "policy"


def _format_token(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p")
    return text.replace("-", "m")


def _format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, int)):
        return f"{float(value):.6g}"
    return str(value)


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
