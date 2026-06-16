from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.viability.config import PolicyConfig, ViabilityConfig
from src.viability.dynamic_policy import EpochPolicySchedule, dynamic_feature_names


EvaluateSchedules = Callable[..., pd.DataFrame]
ScheduleHistoryRunner = Callable[[EpochPolicySchedule, ViabilityConfig], pd.DataFrame]


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


def best_ok_row(evaluations: pd.DataFrame) -> pd.Series:
    required = ["schedule_id", "status", "phi"]
    missing = [column for column in required if column not in evaluations.columns]
    if missing:
        raise ValueError(f"Evaluations are missing required columns: {missing}")
    ok = evaluations[evaluations["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No successful dynamic evaluations are available")
    return ok.sort_values(["phi", "schedule_id"]).iloc[0]


def raw_values_from_row(row: pd.Series, feature_names: Sequence[str]) -> dict[str, float]:
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


def base_seed_offset(row: pd.Series) -> int:
    if "seed_offset" in row and pd.notna(row["seed_offset"]):
        return int(row["seed_offset"])
    if "sample_index" in row and pd.notna(row["sample_index"]):
        return int(row["sample_index"])
    schedule_id = str(row.get("schedule_id", ""))
    digits = "".join(reversed([char for char in reversed(schedule_id) if char.isdigit()]))
    if digits:
        return int(digits)
    return 0


def schedule_from_row(
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


def schedule_row_from_flat_values(
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


def sweep_values_for_variable(
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


def dedupe_candidates(candidates: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    subset = [f"raw_{name}" for name in feature_names]
    deduped = candidates.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    deduped.loc[:, "sample_index"] = range(len(deduped))
    deduped.loc[:, "schedule_id"] = [f"bound_{index:04d}" for index in range(len(deduped))]
    deduped.loc[deduped["experiment_id"] == "baseline", "schedule_id"] = "bound_base"
    return deduped


def positive_constraints(row: pd.Series | None) -> dict[str, float]:
    if row is None:
        return {}
    positive = {}
    for column, value in row.items():
        if str(column).startswith("constraint_"):
            numeric = float(value)
            if numeric > 0.0:
                positive[str(column).removeprefix("constraint_")] = numeric
    return positive


def selected_policy_summary(row: pd.Series, label: str, epoch_count: int) -> dict[str, object]:
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


def trajectory_x(trajectory: pd.DataFrame) -> np.ndarray:
    return np.arange(len(trajectory), dtype=float)


def mark_assessment_start(axis, config: ViabilityConfig) -> None:
    phase_index = max(0, (config.model.assessment_start_year - config.model.start_year) * 3)
    axis.axvline(
        phase_index,
        color="#6b7280",
        linestyle=":",
        linewidth=1.0,
        label="assessment start",
    )


def save_figure(fig, path: Path, plt_module, *, tight_layout: bool = True) -> Path:
    if tight_layout:
        fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt_module.close(fig)
    return path


def configure_matplotlib_cache(output_path: Path) -> None:
    cache_dir = output_path / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def safe_name(value: str) -> str:
    safe = []
    for char in str(value).lower():
        if char.isalnum():
            safe.append(char)
        elif char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "policy"


def format_token(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p")
    return text.replace("-", "m")


def format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, int)):
        return f"{float(value):.6g}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> list[str]:
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
