"""Structured dynamic-policy search as finite-horizon control.

The dynamic policy workflow keeps the repo's existing physics simulator as the
state transition map.  It optimizes open-loop epoch controls, then direct-checks
all reported candidates with the authoritative simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.viability.config import ViabilityConfig
from src.viability.doe import _sample_unit_cube
from src.viability.dynamic_policy import (
    dynamic_feature_names,
    schedule_from_unit_vector,
)
from src.viability.evaluator import evaluate_schedules_parallel
from src.viability.io import write_config_resolved, write_table
from src.viability.surrogate import read_evaluations_table

EvaluateSchedules = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class DynamicSearchResult:
    output_dir: Path
    initial_schedules_path: Path
    initial_evaluations_path: Path
    optimizer_candidates_path: Path
    optimizer_evaluations_path: Path
    all_evaluations_path: Path
    summary_path: Path
    evaluated_count: int
    feasible_count: int
    best_phi: float


@dataclass(frozen=True)
class DynamicDiagnosticResult:
    output_dir: Path
    diagnostic_schedules_path: Path
    diagnostic_evaluations_path: Path
    sensitivity_path: Path
    report_path: Path
    evaluated_count: int


@dataclass(frozen=True)
class DynamicRefinementResult:
    output_dir: Path
    refinement_candidates_path: Path
    refinement_evaluations_path: Path
    all_evaluations_path: Path
    summary_path: Path
    candidate_count: int
    evaluated_count: int
    feasible_count: int
    best_phi: float


@dataclass(frozen=True)
class DynamicSeedTemplate:
    """Named open-loop seed schedule used before surrogate optimization."""

    name: str
    epochs: tuple[dict[str, float], ...]


def run_dynamic_policy_search(
    *,
    config: ViabilityConfig,
    output_dir: str | Path,
    epoch_count: int = 3,
    initial_samples: int = 64,
    optimizer_pool_size: int = 4096,
    verify_top: int = 32,
    workers: int | None = None,
    checkpoint_every: int = 10,
    evaluator: EvaluateSchedules = evaluate_schedules_parallel,
) -> DynamicSearchResult:
    """Run surrogate-assisted finite-horizon control search with direct verification."""
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    if initial_samples < 0:
        raise ValueError("initial_samples must be non-negative")
    if optimizer_pool_size <= 0:
        raise ValueError("optimizer_pool_size must be positive")
    if verify_top <= 0:
        raise ValueError("verify_top must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_config_resolved(config, output_path)

    total_phases = config.model.years_to_run * 3
    feature_names = dynamic_feature_names(config.policy, epoch_count)
    dimension = len(feature_names)

    initial_schedules = generate_dynamic_schedules(
        config,
        epoch_count=epoch_count,
        n=initial_samples,
        start_index=0,
        source="initial_sobol",
        include_heuristics=True,
    )
    initial_schedules_path = write_table(
        initial_schedules,
        output_path / "initial_schedules.csv",
        prefer_parquet=False,
    )
    initial_evaluations = evaluator(
        initial_schedules,
        config,
        epoch_count=epoch_count,
        workers=workers,
        checkpoint_dir=output_path / "initial_checkpoints",
        checkpoint_every=checkpoint_every,
    )
    initial_evaluations_path = write_table(
        initial_evaluations,
        output_path / "initial_evaluations.parquet",
    )

    surrogate = fit_phi_gpr(initial_evaluations, config, epoch_count=epoch_count)
    optimizer_candidates = propose_optimizer_candidates(
        config,
        surrogate=surrogate,
        epoch_count=epoch_count,
        optimizer_pool_size=optimizer_pool_size,
        verify_top=verify_top,
        start_index=100_000,
    )
    optimizer_candidates_path = write_table(
        optimizer_candidates,
        output_path / "optimizer_candidates.csv",
        prefer_parquet=False,
    )
    optimizer_evaluations = evaluator(
        optimizer_candidates,
        config,
        epoch_count=epoch_count,
        workers=workers,
        checkpoint_dir=output_path / "optimizer_checkpoints",
        checkpoint_every=checkpoint_every,
    )
    optimizer_evaluations_path = write_table(
        optimizer_evaluations,
        output_path / "optimizer_evaluations.parquet",
    )

    all_evaluations = pd.concat([initial_evaluations, optimizer_evaluations], ignore_index=True)
    all_evaluations_path = write_table(
        all_evaluations,
        output_path / "all_evaluations.parquet",
    )
    summary = dynamic_search_summary(
        initial_evaluations=initial_evaluations,
        optimizer_evaluations=optimizer_evaluations,
        all_evaluations=all_evaluations,
        output_dir=output_path,
        total_phases=total_phases,
        epoch_count=epoch_count,
        workers=config.run.workers if workers is None else workers,
        surrogate=surrogate,
    )
    summary_path = output_path / "dynamic_search_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    ok = all_evaluations[all_evaluations["status"] == "ok"]
    best_phi = float(ok["phi"].min()) if not ok.empty else float("inf")
    feasible_count = int(ok["feasible"].sum()) if not ok.empty else 0
    return DynamicSearchResult(
        output_dir=output_path.resolve(),
        initial_schedules_path=initial_schedules_path.resolve(),
        initial_evaluations_path=initial_evaluations_path.resolve(),
        optimizer_candidates_path=optimizer_candidates_path.resolve(),
        optimizer_evaluations_path=optimizer_evaluations_path.resolve(),
        all_evaluations_path=all_evaluations_path.resolve(),
        summary_path=summary_path.resolve(),
        evaluated_count=int(len(all_evaluations)),
        feasible_count=feasible_count,
        best_phi=best_phi,
    )


def run_dynamic_policy_refinement(
    *,
    config: ViabilityConfig,
    previous_evaluations_path: str | Path,
    output_dir: str | Path,
    diagnostic_sensitivity_path: str | Path | None = None,
    epoch_count: int = 3,
    local_samples: int = 512,
    optimizer_pool_size: int = 4096,
    verify_top: int = 32,
    workers: int | None = None,
    checkpoint_every: int = 10,
    evaluator: EvaluateSchedules = evaluate_schedules_parallel,
) -> DynamicRefinementResult:
    """Refine around direct-physics nearest misses with surrogate-ranked candidates."""
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    if local_samples <= 0:
        raise ValueError("local_samples must be positive")
    if optimizer_pool_size <= 0:
        raise ValueError("optimizer_pool_size must be positive")
    if verify_top <= 0:
        raise ValueError("verify_top must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_config_resolved(config, output_path)

    previous_evaluations = read_evaluations_table(previous_evaluations_path)
    sensitivity = (
        read_evaluations_table(diagnostic_sensitivity_path)
        if diagnostic_sensitivity_path
        else None
    )
    candidates = generate_refinement_candidates(
        config,
        previous_evaluations,
        epoch_count=epoch_count,
        local_samples=local_samples,
        optimizer_pool_size=optimizer_pool_size,
        verify_top=verify_top,
        sensitivity=sensitivity,
    )
    candidates_path = write_table(
        candidates,
        output_path / "refinement_candidates.csv",
        prefer_parquet=False,
    )
    refinement_evaluations = evaluator(
        candidates,
        config,
        epoch_count=epoch_count,
        workers=workers,
        checkpoint_dir=output_path / "refinement_checkpoints",
        checkpoint_every=checkpoint_every,
    )
    refinement_evaluations_path = write_table(
        refinement_evaluations,
        output_path / "refinement_evaluations.parquet",
    )
    all_evaluations = pd.concat(
        [previous_evaluations, refinement_evaluations],
        ignore_index=True,
        sort=False,
    )
    all_evaluations_path = write_table(
        all_evaluations,
        output_path / "all_evaluations.parquet",
    )

    surrogate = fit_phi_gpr(previous_evaluations, config, epoch_count=epoch_count)
    summary = dynamic_refinement_summary(
        previous_evaluations=previous_evaluations,
        refinement_evaluations=refinement_evaluations,
        all_evaluations=all_evaluations,
        output_dir=output_path,
        total_phases=config.model.years_to_run * 3,
        epoch_count=epoch_count,
        workers=config.run.workers if workers is None else workers,
        candidate_count=len(candidates),
        surrogate=surrogate,
    )
    summary_path = output_path / "dynamic_refinement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    ok = all_evaluations[all_evaluations["status"] == "ok"]
    best_phi = float(ok["phi"].min()) if not ok.empty else float("inf")
    feasible_count = int(ok["feasible"].sum()) if not ok.empty else 0
    return DynamicRefinementResult(
        output_dir=output_path.resolve(),
        refinement_candidates_path=candidates_path.resolve(),
        refinement_evaluations_path=refinement_evaluations_path.resolve(),
        all_evaluations_path=all_evaluations_path.resolve(),
        summary_path=summary_path.resolve(),
        candidate_count=int(len(candidates)),
        evaluated_count=int(len(refinement_evaluations)),
        feasible_count=feasible_count,
        best_phi=best_phi,
    )


def run_dynamic_policy_diagnostic(
    *,
    config: ViabilityConfig,
    evaluations_path: str | Path,
    output_dir: str | Path,
    epoch_count: int = 3,
    perturbation_fraction: float = 0.05,
    workers: int | None = None,
    checkpoint_every: int = 10,
    evaluator: EvaluateSchedules = evaluate_schedules_parallel,
) -> DynamicDiagnosticResult:
    """Direct local linearization diagnostic around the best open-loop schedule."""
    if perturbation_fraction <= 0:
        raise ValueError("perturbation_fraction must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_config_resolved(config, output_path)

    evaluations = pd.read_parquet(evaluations_path)
    best = _best_row_for_diagnostics(evaluations)
    total_phases = config.model.years_to_run * 3
    schedules = generate_local_perturbation_schedules(
        config,
        best,
        epoch_count=epoch_count,
        total_phases=total_phases,
        perturbation_fraction=perturbation_fraction,
    )
    diagnostic_schedules_path = write_table(
        schedules,
        output_path / "diagnostic_schedules.csv",
        prefer_parquet=False,
    )
    diagnostic_evaluations = evaluator(
        schedules,
        config,
        epoch_count=epoch_count,
        workers=workers,
        checkpoint_dir=output_path / "diagnostic_checkpoints",
        checkpoint_every=checkpoint_every,
    )
    diagnostic_evaluations_path = write_table(
        diagnostic_evaluations,
        output_path / "diagnostic_evaluations.parquet",
    )
    sensitivity = compute_local_sensitivities(diagnostic_evaluations, config, epoch_count)
    sensitivity_path = write_table(
        sensitivity,
        output_path / "local_sensitivity.csv",
        prefer_parquet=False,
    )
    report_path = output_path / "dynamic_control_report.md"
    report_path.write_text(
        render_dynamic_control_report(
            search_evaluations=evaluations,
            diagnostic_evaluations=diagnostic_evaluations,
            sensitivity=sensitivity,
            epoch_count=epoch_count,
            total_phases=total_phases,
            perturbation_fraction=perturbation_fraction,
        ),
        encoding="utf-8",
    )
    return DynamicDiagnosticResult(
        output_dir=output_path.resolve(),
        diagnostic_schedules_path=diagnostic_schedules_path.resolve(),
        diagnostic_evaluations_path=diagnostic_evaluations_path.resolve(),
        sensitivity_path=sensitivity_path.resolve(),
        report_path=report_path.resolve(),
        evaluated_count=int(len(diagnostic_evaluations)),
    )


def generate_dynamic_schedules(
    config: ViabilityConfig,
    *,
    epoch_count: int,
    n: int,
    start_index: int,
    source: str,
    include_heuristics: bool,
) -> pd.DataFrame:
    total_phases = config.model.years_to_run * 3
    feature_names = dynamic_feature_names(config.policy, epoch_count)
    dimension = len(feature_names)
    rows: list[dict[str, object]] = []

    if include_heuristics:
        for index, template in enumerate(_heuristic_templates(epoch_count)):
            unit_values = _unit_vector_from_epoch_dicts(config, template.epochs)
            row = _schedule_row_from_unit_vector(
                config,
                unit_values,
                epoch_count=epoch_count,
                total_phases=total_phases,
                schedule_id=f"heuristic_{index:04d}",
                source="heuristic",
                sample_index=index,
            )
            row["template_name"] = template.name
            rows.append(row)

    samples = _sample_unit_cube(
        n=n,
        dimension=dimension,
        method="sobol",
        random_seed=config.run.random_seed,
        start_index=start_index,
        scramble=config.doe.scramble,
    )
    for offset, unit_values in enumerate(samples):
        sample_index = start_index + offset
        rows.append(
            _schedule_row_from_unit_vector(
                config,
                unit_values,
                epoch_count=epoch_count,
                total_phases=total_phases,
                schedule_id=f"{source}_{sample_index:06d}",
                source=source,
                sample_index=sample_index,
            )
        )

    deduped = _dedupe_schedule_rows(rows, feature_names)
    columns = ["schedule_id", "schedule_source", "sample_index", "template_name"]
    for name in feature_names:
        columns.extend([f"raw_{name}", f"applied_{name}", name])
    return pd.DataFrame(deduped, columns=columns)


def fit_phi_gpr(
    evaluations: pd.DataFrame,
    config: ViabilityConfig,
    *,
    epoch_count: int,
):
    feature_names = dynamic_feature_names(config.policy, epoch_count)
    ok = evaluations[
        (evaluations["status"] == "ok")
        & np.isfinite(evaluations["phi"].astype(float))
    ].copy()
    if len(ok) < 2:
        raise ValueError("Need at least two ok dynamic evaluations to fit surrogate")
    x = _unit_matrix_from_frame(ok, config, epoch_count=epoch_count)
    y = ok["phi"].to_numpy(dtype=float)

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except ModuleNotFoundError as exc:
        raise RuntimeError("Dynamic policy GPR search requires scikit-learn") from exc

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(x.shape[1]),
        length_scale_bounds=(1e-2, 1e5),
        nu=1.5,
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0,
    )
    model.fit(x, y)
    return model


def propose_optimizer_candidates(
    config: ViabilityConfig,
    *,
    surrogate,
    epoch_count: int,
    optimizer_pool_size: int,
    verify_top: int,
    start_index: int,
) -> pd.DataFrame:
    total_phases = config.model.years_to_run * 3
    feature_names = dynamic_feature_names(config.policy, epoch_count)
    dimension = len(feature_names)
    pool = _sample_unit_cube(
        n=optimizer_pool_size,
        dimension=dimension,
        method="sobol",
        random_seed=config.run.random_seed + 17,
        start_index=start_index,
        scramble=config.doe.scramble,
    )
    mean, sigma = surrogate.predict(pool, return_std=True)
    lcb = mean - sigma
    order = np.argsort(lcb)

    selected: list[np.ndarray] = []
    selected_sources: list[str] = []
    for idx in order[: max(verify_top * 4, verify_top)]:
        candidate = pool[int(idx)]
        if _is_diverse(candidate, selected, min_distance=0.12):
            selected.append(candidate)
            selected_sources.append("surrogate_lcb")
        if len(selected) >= verify_top:
            break

    de_candidate = _differential_evolution_candidate(surrogate, dimension)
    if de_candidate is not None and _is_diverse(de_candidate, selected, min_distance=0.04):
        selected.insert(0, de_candidate)
        selected_sources.insert(0, "surrogate_de")

    rows = []
    for index, (unit_values, source) in enumerate(zip(selected[:verify_top], selected_sources[:verify_top])):
        rows.append(
            _schedule_row_from_unit_vector(
                config,
                unit_values,
                epoch_count=epoch_count,
                total_phases=total_phases,
                schedule_id=f"optimizer_{index:04d}",
                source=source,
                sample_index=index,
            )
        )
    columns = ["schedule_id", "schedule_source", "sample_index"]
    for name in feature_names:
        columns.extend([f"raw_{name}", f"applied_{name}", name])
    return pd.DataFrame(rows, columns=columns)


def generate_refinement_candidates(
    config: ViabilityConfig,
    previous_evaluations: pd.DataFrame,
    *,
    epoch_count: int,
    local_samples: int,
    optimizer_pool_size: int,
    verify_top: int,
    sensitivity: pd.DataFrame | None = None,
    anchor_count: int = 8,
) -> pd.DataFrame:
    """Generate local refinement schedules around direct nearest misses."""
    if local_samples <= 0:
        raise ValueError("local_samples must be positive")
    if optimizer_pool_size <= 0:
        raise ValueError("optimizer_pool_size must be positive")
    if verify_top <= 0:
        raise ValueError("verify_top must be positive")

    total_phases = config.model.years_to_run * 3
    feature_names = dynamic_feature_names(config.policy, epoch_count)
    dimension = len(feature_names)
    anchors = _select_refinement_anchors(
        previous_evaluations,
        config,
        epoch_count=epoch_count,
        anchor_count=anchor_count,
    )
    anchor_units = _unit_matrix_from_frame(anchors, config, epoch_count=epoch_count)

    candidate_units: list[np.ndarray] = []
    candidate_sources: list[str] = []
    for unit_values in anchor_units:
        candidate_units.append(unit_values)
        candidate_sources.append("previous_best")

    local_units = _local_refinement_unit_vectors(
        config,
        anchor_units,
        epoch_count=epoch_count,
        n=local_samples,
        start_index=200_000,
        radius=0.10,
    )
    candidate_units.extend(local_units)
    candidate_sources.extend(["local_sobol"] * len(local_units))

    if sensitivity is not None and not sensitivity.empty:
        diagnostic_units = _diagnostic_refinement_unit_vectors(
            config,
            anchors.iloc[0],
            sensitivity,
            epoch_count=epoch_count,
        )
        candidate_units.extend(diagnostic_units)
        candidate_sources.extend(["diagnostic_move"] * len(diagnostic_units))

    global_pool = _sample_unit_cube(
        n=optimizer_pool_size,
        dimension=dimension,
        method="sobol",
        random_seed=config.run.random_seed + 31,
        start_index=300_000,
        scramble=config.doe.scramble,
    )
    surrogate = fit_phi_gpr(previous_evaluations, config, epoch_count=epoch_count)
    pool_mean, pool_sigma = surrogate.predict(global_pool, return_std=True)
    pool_order = np.argsort(pool_mean - pool_sigma)
    global_units = [global_pool[int(index)] for index in pool_order[: max(verify_top * 2, verify_top)]]
    candidate_units.extend(global_units)
    candidate_sources.extend(["global_lcb"] * len(global_units))

    ranked_units, ranked_sources = _rank_refinement_units(
        candidate_units,
        candidate_sources,
        surrogate=surrogate,
        verify_top=verify_top,
    )

    rows = []
    for index, (unit_values, source) in enumerate(zip(ranked_units, ranked_sources)):
        rows.append(
            _schedule_row_from_unit_vector(
                config,
                unit_values,
                epoch_count=epoch_count,
                total_phases=total_phases,
                schedule_id=f"refine_{index:04d}",
                source=source,
                sample_index=index,
            )
        )
    columns = ["schedule_id", "schedule_source", "sample_index"]
    for name in feature_names:
        columns.extend([f"raw_{name}", f"applied_{name}", name])
    return pd.DataFrame(rows, columns=columns)


def dynamic_refinement_summary(
    *,
    previous_evaluations: pd.DataFrame,
    refinement_evaluations: pd.DataFrame,
    all_evaluations: pd.DataFrame,
    output_dir: Path,
    total_phases: int,
    epoch_count: int,
    workers: int,
    candidate_count: int,
    surrogate,
) -> dict[str, object]:
    base_ok = previous_evaluations[previous_evaluations["status"] == "ok"].copy()
    refined_ok = refinement_evaluations[refinement_evaluations["status"] == "ok"].copy()
    all_ok = all_evaluations[all_evaluations["status"] == "ok"].copy()
    base_best = float(base_ok["phi"].min()) if not base_ok.empty else None
    refined_best = float(refined_ok["phi"].min()) if not refined_ok.empty else None
    all_best = all_ok.sort_values("phi").head(1)
    feasible = all_ok[all_ok["feasible"]] if not all_ok.empty else all_ok
    positive_constraints: dict[str, float] = {}
    if not all_best.empty:
        best_row = all_best.iloc[0]
        for column in [c for c in all_ok.columns if c.startswith("constraint_")]:
            value = float(best_row[column])
            if value > 0:
                positive_constraints[column.removeprefix("constraint_")] = value
    improvement = (
        None
        if base_best is None or refined_best is None
        else float(base_best - min(base_best, refined_best))
    )
    return {
        "output_dir": str(output_dir.resolve()),
        "phase_backend": _single_value(all_ok, "phase_backend"),
        "epoch_count": epoch_count,
        "total_phases": total_phases,
        "workers": workers,
        "candidate_count": int(candidate_count),
        "previous_evaluated_count": int(len(previous_evaluations)),
        "refinement_evaluated_count": int(len(refinement_evaluations)),
        "evaluated_count": int(len(all_evaluations)),
        "ok_count": int((all_evaluations["status"] == "ok").sum()),
        "feasible_count": int(len(feasible)),
        "previous_best_phi": base_best,
        "refinement_best_phi": refined_best,
        "best_phi": float(all_ok["phi"].min()) if not all_ok.empty else None,
        "best_phi_improvement": improvement,
        "best_schedule_id": None if all_best.empty else str(all_best.iloc[0]["schedule_id"]),
        "best_active_constraint": None if all_best.empty else str(all_best.iloc[0]["active_constraint"]),
        "best_positive_constraints": positive_constraints,
        "surrogate_kernel": str(getattr(surrogate, "kernel_", "")),
    }


def dynamic_search_summary(
    *,
    initial_evaluations: pd.DataFrame,
    optimizer_evaluations: pd.DataFrame,
    all_evaluations: pd.DataFrame,
    output_dir: Path,
    total_phases: int,
    epoch_count: int,
    workers: int,
    surrogate,
) -> dict[str, object]:
    ok = all_evaluations[all_evaluations["status"] == "ok"].copy()
    feasible = ok[ok["feasible"]]
    best = ok.sort_values("phi").head(1)
    constraint_columns = [c for c in ok.columns if c.startswith("constraint_")]
    positive_constraints: dict[str, float] = {}
    if not best.empty:
        best_row = best.iloc[0]
        for column in constraint_columns:
            value = float(best_row[column])
            if value > 0:
                positive_constraints[column.removeprefix("constraint_")] = value

    min_constraints = {
        column.removeprefix("constraint_"): float(ok[column].min())
        for column in constraint_columns
    } if not ok.empty else {}
    active_counts = (
        ok["active_constraint"].value_counts(dropna=False).to_dict()
        if not ok.empty
        else {}
    )

    return {
        "output_dir": str(output_dir.resolve()),
        "phase_backend": _single_value(ok, "phase_backend"),
        "epoch_count": epoch_count,
        "total_phases": total_phases,
        "workers": workers,
        "initial_evaluated_count": int(len(initial_evaluations)),
        "optimizer_evaluated_count": int(len(optimizer_evaluations)),
        "evaluated_count": int(len(all_evaluations)),
        "ok_count": int((all_evaluations["status"] == "ok").sum()),
        "feasible_count": int(len(feasible)),
        "best_phi": float(ok["phi"].min()) if not ok.empty else None,
        "best_schedule_id": None if best.empty else str(best.iloc[0]["schedule_id"]),
        "best_active_constraint": None if best.empty else str(best.iloc[0]["active_constraint"]),
        "best_positive_constraints": positive_constraints,
        "min_observed_constraints": min_constraints,
        "active_constraint_counts": {str(k): int(v) for k, v in active_counts.items()},
        "surrogate_kernel": str(getattr(surrogate, "kernel_", "")),
    }


def generate_local_perturbation_schedules(
    config: ViabilityConfig,
    best_row: pd.Series,
    *,
    epoch_count: int,
    total_phases: int,
    perturbation_fraction: float,
) -> pd.DataFrame:
    feature_names = dynamic_feature_names(config.policy, epoch_count)
    base_values = {
        name: float(best_row[f"raw_{name}"])
        for name in feature_names
    }
    rows = [
        _schedule_row_from_flat_values(
            config,
            base_values,
            epoch_count=epoch_count,
            total_phases=total_phases,
            schedule_id="diagnostic_base",
            source="diagnostic_base",
            sample_index=0,
        )
    ]

    sample_index = 1
    for epoch_index in range(epoch_count):
        prefix = f"epoch{epoch_index + 1}"
        for name, variable in config.policy.variables.items():
            column = f"{prefix}_{name}"
            span = variable.high - variable.low
            step = span * perturbation_fraction
            if variable.type == "int":
                step = max(1.0, round(step))
            for direction, sign in (("minus", -1.0), ("plus", 1.0)):
                perturbed = dict(base_values)
                perturbed[column] = min(
                    max(base_values[column] + sign * step, variable.low),
                    variable.high,
                )
                if perturbed[column] == base_values[column]:
                    continue
                rows.append(
                    _schedule_row_from_flat_values(
                        config,
                        perturbed,
                        epoch_count=epoch_count,
                        total_phases=total_phases,
                        schedule_id=f"diagnostic_{prefix}_{name}_{direction}",
                        source=f"diagnostic_{prefix}_{name}_{direction}",
                        sample_index=sample_index,
                    )
                )
                sample_index += 1

    columns = ["schedule_id", "schedule_source", "sample_index"]
    for feature_name in feature_names:
        columns.extend([f"raw_{feature_name}", f"applied_{feature_name}", feature_name])
    return pd.DataFrame(rows, columns=columns)


def compute_local_sensitivities(
    diagnostic_evaluations: pd.DataFrame,
    config: ViabilityConfig,
    epoch_count: int,
) -> pd.DataFrame:
    base = diagnostic_evaluations[
        diagnostic_evaluations["schedule_id"] == "diagnostic_base"
    ]
    if base.empty:
        raise ValueError("Diagnostic evaluations are missing diagnostic_base")
    base_row = base.iloc[0]
    response_columns = [
        "phi",
        *[column for column in diagnostic_evaluations.columns if column.startswith("constraint_")],
    ]
    rows: list[dict[str, object]] = []
    for epoch_index in range(epoch_count):
        prefix = f"epoch{epoch_index + 1}"
        for name in config.policy.variables:
            feature = f"{prefix}_{name}"
            plus = diagnostic_evaluations[
                diagnostic_evaluations["schedule_id"]
                == f"diagnostic_{feature}_plus"
            ]
            minus = diagnostic_evaluations[
                diagnostic_evaluations["schedule_id"]
                == f"diagnostic_{feature}_minus"
            ]
            for response in response_columns:
                if not plus.empty and not minus.empty:
                    dx = float(plus.iloc[0][f"raw_{feature}"]) - float(minus.iloc[0][f"raw_{feature}"])
                    dy = float(plus.iloc[0][response]) - float(minus.iloc[0][response])
                    method = "central"
                elif not plus.empty:
                    dx = float(plus.iloc[0][f"raw_{feature}"]) - float(base_row[f"raw_{feature}"])
                    dy = float(plus.iloc[0][response]) - float(base_row[response])
                    method = "forward"
                elif not minus.empty:
                    dx = float(base_row[f"raw_{feature}"]) - float(minus.iloc[0][f"raw_{feature}"])
                    dy = float(base_row[response]) - float(minus.iloc[0][response])
                    method = "backward"
                else:
                    continue
                if abs(dx) <= 1e-12:
                    continue
                rows.append(
                    {
                        "epoch": epoch_index + 1,
                        "control": name,
                        "response": response.removeprefix("constraint_"),
                        "sensitivity": dy / dx,
                        "abs_sensitivity": abs(dy / dx),
                        "method": method,
                        "base_value": float(base_row[f"raw_{feature}"]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "epoch",
            "control",
            "response",
            "sensitivity",
            "abs_sensitivity",
            "method",
            "base_value",
        ],
    )


def render_dynamic_control_report(
    *,
    search_evaluations: pd.DataFrame,
    diagnostic_evaluations: pd.DataFrame,
    sensitivity: pd.DataFrame,
    epoch_count: int,
    total_phases: int,
    perturbation_fraction: float,
) -> str:
    ok = search_evaluations[search_evaluations["status"] == "ok"].copy()
    feasible = ok[ok["feasible"]]
    best = _best_row_for_diagnostics(ok)
    positive = {
        column.removeprefix("constraint_"): float(best[column])
        for column in ok.columns
        if column.startswith("constraint_") and float(best[column]) > 0
    }
    authority_lines = []
    for response in ["phi", "total_pilots_window", "wg_rap", "fl_rap", "ip_rap"]:
        subset = sensitivity[sensitivity["response"] == response].sort_values(
            "abs_sensitivity",
            ascending=False,
        ).head(5)
        if subset.empty:
            continue
        authority_lines.append(f"### {response}")
        for _, row in subset.iterrows():
            authority_lines.append(
                "- epoch {epoch} {control}: sensitivity {sensitivity:.4g} per unit".format(
                    epoch=int(row["epoch"]),
                    control=row["control"],
                    sensitivity=float(row["sensitivity"]),
                )
            )

    best_controls = []
    for epoch_index in range(epoch_count):
        best_controls.append(f"### Epoch {epoch_index + 1}")
        prefix = f"epoch{epoch_index + 1}"
        for column in [c for c in best.index if c.startswith(prefix) and not c.startswith("raw_") and not c.startswith("applied_")]:
            best_controls.append(f"- {column.removeprefix(prefix + '_')}: {best[column]}")

    return "\n".join(
        [
            "# Dynamic Policy / Finite-Horizon Control Search",
            "",
            "This artifact frames the viability search as open-loop finite-horizon nonlinear optimal control over structured epoch controls.",
            "",
            f"- State `x_k`: compressed force/training state at phase `k`, represented by the simulator history: total/line pilots, qual mix, staff counts, upgrade/carryover burden, and RAP shortfalls.",
            f"- Control `u_k`: seven policy levers held constant within each of {epoch_count} epochs: annual intake, retention, UTE, PAA, max manning, FLUG quota, and IPUG quota.",
            "- Dynamics: `x_{k+1} = f_k(x_k, u_k)` through the existing physics-backed simulator.",
            "- Constraints: enabled viability constraints from the config, using `g(x) <= 0`.",
            "- Objective: minimize the maximum normalized constraint violation `phi`; direct physics remains the source of truth.",
            "",
            "## Search Result",
            "",
            f"- Total phases: {total_phases}",
            f"- Direct search evaluations: {len(search_evaluations)}",
            f"- Direct diagnostic evaluations: {len(diagnostic_evaluations)}",
            f"- Feasible search evaluations: {len(feasible)}",
            f"- Best schedule id: {best['schedule_id']}",
            f"- Best phi: {float(best['phi']):.6g}",
            f"- Best active constraint: {best['active_constraint']}",
            f"- Positive constraints at best point: {positive}",
            "",
            "## Best Schedule",
            "",
            *best_controls,
            "",
            "## Local Control Authority",
            "",
            f"Finite differences used perturbations of {perturbation_fraction:.3g} of each configured control range around the best schedule.",
            "",
            *(authority_lines if authority_lines else ["No local sensitivities were available."]),
            "",
            "## Interpretation",
            "",
            "This is diagnostic evidence only. It is not an MPC or LQR rewrite, and it does not replace direct verification. If no feasible policy is found, use the positive constraints and local authority table to decide whether the current structured policy class lacks enough control authority or whether requirements need relaxation.",
            "",
        ]
    )


def _best_row_for_diagnostics(evaluations: pd.DataFrame) -> pd.Series:
    ok = evaluations[evaluations["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No ok evaluations available")
    constraint_columns = [c for c in ok.columns if c.startswith("constraint_")]
    if constraint_columns:
        positive_sum = ok[constraint_columns].clip(lower=0.0).sum(axis=1)
        ok = ok.assign(_positive_constraint_sum=positive_sum)
        return ok.sort_values(["phi", "_positive_constraint_sum"]).iloc[0]
    return ok.sort_values("phi").iloc[0]


def _select_refinement_anchors(
    evaluations: pd.DataFrame,
    config: ViabilityConfig,
    *,
    epoch_count: int,
    anchor_count: int,
) -> pd.DataFrame:
    ok = evaluations[evaluations["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No ok evaluations available for dynamic refinement")
    feature_names = dynamic_feature_names(config.policy, epoch_count)
    missing = [f"raw_{name}" for name in feature_names if f"raw_{name}" not in ok.columns]
    if missing:
        raise ValueError(f"Previous evaluations are missing dynamic raw columns: {missing}")
    constraint_columns = [c for c in ok.columns if c.startswith("constraint_")]
    if constraint_columns:
        ok.loc[:, "_positive_constraint_sum"] = ok[constraint_columns].clip(lower=0.0).sum(axis=1)
        return ok.sort_values(["phi", "_positive_constraint_sum", "schedule_id"]).head(anchor_count)
    return ok.sort_values(["phi", "schedule_id"]).head(anchor_count)


def _local_refinement_unit_vectors(
    config: ViabilityConfig,
    anchor_units: np.ndarray,
    *,
    epoch_count: int,
    n: int,
    start_index: int,
    radius: float,
) -> list[np.ndarray]:
    if len(anchor_units) == 0:
        return []
    dimension = len(dynamic_feature_names(config.policy, epoch_count))
    samples = _sample_unit_cube(
        n=n,
        dimension=dimension,
        method="sobol",
        random_seed=config.run.random_seed + 23,
        start_index=start_index,
        scramble=config.doe.scramble,
    )
    vectors = []
    for index, sample in enumerate(samples):
        anchor = anchor_units[index % len(anchor_units)]
        delta = (sample - 0.5) * 2.0 * radius
        vectors.append(np.clip(anchor + delta, 0.0, 1.0))
    return vectors


def _diagnostic_refinement_unit_vectors(
    config: ViabilityConfig,
    anchor_row: pd.Series,
    sensitivity: pd.DataFrame,
    *,
    epoch_count: int,
) -> list[np.ndarray]:
    required = {"epoch", "control", "response", "sensitivity", "abs_sensitivity"}
    missing = sorted(required - set(sensitivity.columns))
    if missing:
        raise ValueError(f"Diagnostic sensitivity is missing required columns: {missing}")
    variable_names = list(config.policy.variables)
    unit = _unit_matrix_from_frame(
        pd.DataFrame([anchor_row]),
        config,
        epoch_count=epoch_count,
    )[0]
    responses = {"phi", "total_pilots_window", "wg_rap", "fl_rap", "ip_rap"}
    ranked = (
        sensitivity[sensitivity["response"].astype(str).isin(responses)]
        .sort_values("abs_sensitivity", ascending=False)
        .head(12)
    )
    vectors: list[np.ndarray] = []
    for _, row in ranked.iterrows():
        epoch = int(row["epoch"])
        control = str(row["control"])
        if epoch < 1 or epoch > epoch_count or control not in config.policy.variables:
            continue
        control_index = variable_names.index(control)
        feature_index = (epoch - 1) * len(variable_names) + control_index
        sign = -1.0 if float(row["sensitivity"]) > 0 else 1.0
        for step in (0.025, 0.05, 0.10, 0.20):
            moved = unit.copy()
            moved[feature_index] = np.clip(moved[feature_index] + sign * step, 0.0, 1.0)
            if moved[feature_index] != unit[feature_index]:
                vectors.append(moved)
    return vectors


def _rank_refinement_units(
    candidate_units: list[np.ndarray],
    candidate_sources: list[str],
    *,
    surrogate,
    verify_top: int,
) -> tuple[list[np.ndarray], list[str]]:
    if len(candidate_units) != len(candidate_sources):
        raise ValueError("candidate_units and candidate_sources must have the same length")
    selected: list[np.ndarray] = []
    sources: list[str] = []
    seen: set[tuple[float, ...]] = set()

    def add_candidate(unit_values: np.ndarray, source: str, *, min_distance: float) -> None:
        key = tuple(np.round(unit_values, 8))
        if key in seen:
            return
        if not _is_diverse(unit_values, selected, min_distance=min_distance):
            return
        seen.add(key)
        selected.append(unit_values)
        sources.append(source)

    for unit_values, source in zip(candidate_units, candidate_sources):
        if source == "previous_best":
            add_candidate(unit_values, source, min_distance=0.0)
            if len(selected) >= verify_top:
                return selected[:verify_top], sources[:verify_top]

    matrix = np.asarray(candidate_units, dtype=float)
    mean, sigma = surrogate.predict(matrix, return_std=True)
    order = np.argsort(mean - sigma)
    for index in order:
        add_candidate(candidate_units[int(index)], candidate_sources[int(index)], min_distance=0.03)
        if len(selected) >= verify_top:
            break
    return selected[:verify_top], sources[:verify_top]


def _schedule_row_from_unit_vector(
    config: ViabilityConfig,
    unit_values: np.ndarray,
    *,
    epoch_count: int,
    total_phases: int,
    schedule_id: str,
    source: str,
    sample_index: int,
) -> dict[str, object]:
    schedule = schedule_from_unit_vector(
        unit_values,
        config.policy,
        epoch_count=epoch_count,
        total_phases=total_phases,
    )
    raw = schedule.to_flat_dict(raw=True)
    applied = schedule.to_flat_dict(raw=False)
    row: dict[str, object] = {
        "schedule_id": schedule_id,
        "schedule_source": source,
        "sample_index": sample_index,
    }
    for name in dynamic_feature_names(config.policy, epoch_count):
        row[f"raw_{name}"] = float(raw[name])
        row[f"applied_{name}"] = applied[name]
        row[name] = applied[name]
    return row


def _schedule_row_from_flat_values(
    config: ViabilityConfig,
    flat_values: dict[str, float],
    *,
    epoch_count: int,
    total_phases: int,
    schedule_id: str,
    source: str,
    sample_index: int,
) -> dict[str, object]:
    from src.viability.dynamic_policy import EpochPolicySchedule

    schedule = EpochPolicySchedule.from_flat_mapping(
        flat_values,
        config.policy,
        epoch_count=epoch_count,
        total_phases=total_phases,
    )
    raw = schedule.to_flat_dict(raw=True)
    applied = schedule.to_flat_dict(raw=False)
    row: dict[str, object] = {
        "schedule_id": schedule_id,
        "schedule_source": source,
        "sample_index": sample_index,
    }
    for name in dynamic_feature_names(config.policy, epoch_count):
        row[f"raw_{name}"] = float(raw[name])
        row[f"applied_{name}"] = applied[name]
        row[name] = applied[name]
    return row


def _heuristic_templates(epoch_count: int) -> list[DynamicSeedTemplate]:
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    return [
        DynamicSeedTemplate(
            name=template.name,
            epochs=_interpolate_template_epochs(template.epochs, epoch_count),
        )
        for template in _base_heuristic_templates()
    ]


def _base_heuristic_templates() -> list[DynamicSeedTemplate]:
    return [
        DynamicSeedTemplate(
            name="static_near_miss_high_retention",
            epochs=(_epoch_policy(152, 0.565, 18.75, 29, 180, 2, 0),) * 3,
        ),
        DynamicSeedTemplate(
            name="static_low_upgrade_low_retention",
            epochs=(_epoch_policy(140, 0.535, 17.4, 27, 180, 1, 0),) * 3,
        ),
        DynamicSeedTemplate(
            name="static_best_current_miss",
            epochs=(_epoch_policy(150, 0.55, 20, 30, 180, 2, 0),) * 3,
        ),
        DynamicSeedTemplate(
            name="growth_then_rap_recovery",
            epochs=(
                _epoch_policy(350, 0.65, 20, 30, 200, 2, 0),
                _epoch_policy(220, 0.60, 20, 30, 180, 5, 0),
                _epoch_policy(80, 0.60, 20, 30, 160, 10, 0),
            ),
        ),
        DynamicSeedTemplate(
            name="smooth_growth_ipug_suppressed",
            epochs=(
                _epoch_policy(300, 0.60, 19, 28, 180, 1, 0),
                _epoch_policy(220, 0.60, 20, 30, 180, 4, 0),
                _epoch_policy(140, 0.60, 20, 30, 160, 8, 0),
            ),
        ),
        DynamicSeedTemplate(
            name="rap_first_reference",
            epochs=(
                _epoch_policy(40, 0.65, 20, 30, 160, 10, 0),
                _epoch_policy(80, 0.65, 20, 30, 160, 10, 0),
                _epoch_policy(120, 0.65, 20, 30, 160, 10, 0),
            ),
        ),
        DynamicSeedTemplate(
            name="growth_first_reference",
            epochs=(
                _epoch_policy(350, 0.65, 18, 28, 200, 0, 0),
                _epoch_policy(350, 0.65, 20, 30, 200, 2, 0),
                _epoch_policy(250, 0.65, 20, 30, 180, 6, 0),
            ),
        ),
    ]


def _interpolate_template_epochs(
    epochs: tuple[dict[str, float], ...],
    epoch_count: int,
) -> tuple[dict[str, float], ...]:
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    if len(epochs) == epoch_count:
        return epochs
    source_points = np.linspace(0.0, 1.0, len(epochs))
    target_points = np.linspace(0.0, 1.0, epoch_count)
    names = list(epochs[0])
    interpolated = []
    for target in target_points:
        row = {}
        for name in names:
            values = [float(epoch[name]) for epoch in epochs]
            row[name] = float(np.interp(target, source_points, values))
        interpolated.append(row)
    return tuple(interpolated)


def _epoch_policy(
    annual_intake: float,
    retention_rate: float,
    ute: float,
    paa: float,
    max_manning_pct: float,
    flug_quota_per_phase: float,
    ipug_quota_per_phase: float,
) -> dict[str, float]:
    return {
        "annual_intake": annual_intake,
        "retention_rate": retention_rate,
        "ute": ute,
        "paa": paa,
        "max_manning_pct": max_manning_pct,
        "flug_quota_per_phase": flug_quota_per_phase,
        "ipug_quota_per_phase": ipug_quota_per_phase,
    }


def _unit_vector_from_epoch_dicts(
    config: ViabilityConfig,
    epochs: tuple[dict[str, float], ...],
) -> np.ndarray:
    from src.viability.design_space import DesignSpace

    space = DesignSpace(config.policy)
    return np.concatenate([space.normalize(epoch) for epoch in epochs])


def _unit_matrix_from_frame(
    frame: pd.DataFrame,
    config: ViabilityConfig,
    *,
    epoch_count: int,
) -> np.ndarray:
    from src.viability.design_space import DesignSpace

    space = DesignSpace(config.policy)
    rows = []
    for _, row in frame.iterrows():
        vector = []
        for epoch_index in range(epoch_count):
            prefix = f"epoch{epoch_index + 1}"
            values = {
                name: float(row[f"raw_{prefix}_{name}"])
                for name in config.policy.variables
            }
            vector.append(space.normalize(values))
        rows.append(np.concatenate(vector))
    return np.asarray(rows, dtype=float)


def _differential_evolution_candidate(surrogate, dimension: int) -> np.ndarray | None:
    try:
        from scipy.optimize import differential_evolution
    except ModuleNotFoundError:
        return None

    def objective(x: np.ndarray) -> float:
        x2 = np.asarray(x, dtype=float).reshape(1, -1)
        return float(surrogate.predict(x2)[0])

    result = differential_evolution(
        objective,
        bounds=[(0.0, 1.0)] * dimension,
        maxiter=40,
        popsize=8,
        polish=False,
        seed=123,
        updating="immediate",
        workers=1,
    )
    return np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)


def _is_diverse(candidate: np.ndarray, selected: list[np.ndarray], *, min_distance: float) -> bool:
    if not selected:
        return True
    return all(float(np.linalg.norm(candidate - existing)) >= min_distance for existing in selected)


def _dedupe_schedule_rows(
    rows: list[dict[str, object]],
    feature_names: list[str],
) -> list[dict[str, object]]:
    seen: set[tuple[object, ...]] = set()
    deduped: list[dict[str, object]] = []
    for row in rows:
        key = tuple(row[name] for name in feature_names)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _single_value(df: pd.DataFrame, column: str) -> object:
    if df.empty or column not in df:
        return None
    values = df[column].dropna().unique()
    if len(values) == 1:
        return values[0].item() if hasattr(values[0], "item") else values[0]
    return [value.item() if hasattr(value, "item") else value for value in values]
