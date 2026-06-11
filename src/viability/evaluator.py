from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor
import random

import numpy as np
import pandas as pd

from src.manning_config import SQUADRON_DATA, get_initial_squadrons
from src.manning_engine import CAFSimulation
from src.models import PriorityMode
from src.viability.config import ViabilityConfig
from src.viability.io import write_evaluation_batch
from src.viability.metrics import (
    aggregate_violation,
    compute_constraints,
    compute_raw_metrics,
)
from src.viability.policy import PolicyDesign


@dataclass
class EvaluationResult:
    design: dict[str, Any]
    raw_design: dict[str, float]
    applied_design: dict[str, Any]
    raw_metrics: dict[str, float]
    constraints: dict[str, float]
    phi: float
    feasible: bool
    active_constraint: str | None
    active_constraint_value: float | None
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "raw_design": self.raw_design,
            "applied_design": self.applied_design,
            "raw_metrics": self.raw_metrics,
            "constraints": self.constraints,
            "phi": self.phi,
            "feasible": self.feasible,
            "active_constraint": self.active_constraint,
            "active_constraint_value": self.active_constraint_value,
            "status": self.status,
            "error": self.error,
        }


_BRAIN_CACHE: dict[str, Any] = {}


def evaluate_design(
    design: PolicyDesign, config: ViabilityConfig, seed: int | None = None
) -> EvaluationResult:
    """Evaluate one constant policy design with the existing long-horizon model."""
    run_seed = config.run.random_seed if seed is None else seed
    random.seed(run_seed)
    np.random.seed(run_seed)

    try:
        brain = _load_brain(config.model.brain_path)
        _validate_brain_output(brain, config.model.expected_brain_outputs)

        sim = CAFSimulation(
            annual_intake=design.annual_intake,
            retention_rate=design.retention_rate,
            round_robin=config.model.round_robin,
            brain=brain,
            max_manning_pct=design.max_manning_pct,
            staff_priority_mode=_parse_priority_mode(config.model.staff_priority_mode),
            use_upgrade_quotas=config.model.use_upgrade_quotas,
        )
        sim.current_year = config.model.start_year
        sim.current_phase = 1
        sim.sq_phase_flug_intake = design.flug_quota_per_phase
        sim.sq_phase_ipug_intake = design.ipug_quota_per_phase

        squadrons = get_initial_squadrons(config.model.start_year, SQUADRON_DATA)
        for sq in squadrons:
            sq.paa = design.paa
            sq.update_stats()

        history = sim.run_simulation(
            years_to_run=config.model.years_to_run,
            squadron_configs=squadrons,
            ute=design.ute,
        )
        raw_metrics = compute_raw_metrics(history, config.model.assessment_start_year)
        constraints = compute_constraints(raw_metrics, config.requirements)
        phi, active_constraint, active_constraint_value = aggregate_violation(
            constraints, config.constraint_scales
        )
        return EvaluationResult(
            design=design.to_dict(),
            raw_design=design.to_raw_dict(),
            applied_design=design.to_applied_dict(),
            raw_metrics=raw_metrics,
            constraints=constraints,
            phi=phi,
            feasible=phi <= 0.0,
            active_constraint=active_constraint,
            active_constraint_value=active_constraint_value,
            status="ok",
        )
    except Exception as exc:
        return EvaluationResult(
            design=design.to_dict(),
            raw_design=design.to_raw_dict(),
            applied_design=design.to_applied_dict(),
            raw_metrics={},
            constraints={},
            phi=float("inf"),
            feasible=False,
            active_constraint=None,
            active_constraint_value=None,
            status="failed",
            error=str(exc),
        )


def evaluate_designs_parallel(
    designs: pd.DataFrame,
    config: ViabilityConfig,
    workers: int | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """Evaluate a batch of constant-policy input combinations."""
    worker_count = config.run.workers if workers is None else workers
    variable_names = list(config.policy.variables)
    missing = sorted(set(variable_names) - set(designs.columns))
    if missing:
        raise ValueError(f"Design table is missing required columns: {missing}")

    jobs = []
    for index, row in designs.reset_index(drop=True).iterrows():
        design_id = row["design_id"] if "design_id" in row else index
        values = {name: row[name] for name in variable_names}
        raw_values = None
        if all(f"raw_{name}" in row for name in variable_names):
            raw_values = {name: float(row[f"raw_{name}"]) for name in variable_names}
        jobs.append(
            (
                design_id,
                values,
                raw_values,
                config,
                config.run.random_seed + int(index),
            )
        )

    flattened_rows: list[dict[str, Any]] = []
    checkpoint_buffer: list[dict[str, Any]] = []
    batch_index = 0

    def flush_checkpoint(force: bool = False) -> None:
        nonlocal batch_index, checkpoint_buffer
        if checkpoint_dir is None or not checkpoint_buffer:
            return
        if not force and len(checkpoint_buffer) < checkpoint_every:
            return
        batch_index += 1
        write_evaluation_batch(
            pd.DataFrame(checkpoint_buffer),
            checkpoint_dir,
            batch_index,
        )
        checkpoint_buffer = []

    def consume_result(design_id: Any, result: EvaluationResult) -> None:
        row = _flatten_result(design_id, result, variable_names)
        flattened_rows.append(row)
        checkpoint_buffer.append(row)
        flush_checkpoint()

    if worker_count <= 1:
        for job in jobs:
            design_id, result = _evaluate_design_job(job)
            consume_result(design_id, result)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for design_id, result in executor.map(_evaluate_design_job, jobs):
                consume_result(design_id, result)

    flush_checkpoint(force=True)
    return pd.DataFrame(flattened_rows)


def _evaluate_design_job(
    job: tuple[Any, dict[str, Any], dict[str, float] | None, ViabilityConfig, int]
) -> tuple[Any, EvaluationResult]:
    design_id, values, raw_values, config, seed = job
    design = PolicyDesign.from_mapping(values, config.policy, raw_values=raw_values)
    return design_id, evaluate_design(design, config, seed=seed)


def _flatten_result(
    design_id: Any,
    result: EvaluationResult,
    variable_names: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "design_id": design_id,
        "phi": result.phi,
        "feasible": result.feasible,
        "active_constraint": result.active_constraint,
        "active_constraint_value": result.active_constraint_value,
        "status": result.status,
        "error": result.error,
    }
    for name in variable_names:
        row[f"raw_{name}"] = result.raw_design.get(name)
        row[f"applied_{name}"] = result.applied_design.get(name)
        row[name] = result.applied_design.get(name)
    for name, value in result.raw_metrics.items():
        row[f"metric_{name}"] = value
    for name, value in result.constraints.items():
        row[f"constraint_{name}"] = value
    return row


def _load_brain(path: str) -> Any:
    brain_path = Path(path)
    if not brain_path.exists():
        raise FileNotFoundError(
            f"Configured brain_path does not exist: {brain_path}. "
            "Train or provide the sortie brain before running viability analysis."
        )
    cache_key = str(brain_path.resolve())
    if cache_key not in _BRAIN_CACHE:
        import joblib

        _BRAIN_CACHE[cache_key] = joblib.load(brain_path)
    return _BRAIN_CACHE[cache_key]


def _validate_brain_output(brain: Any, expected_outputs: int) -> None:
    columns = CAFSimulation._PREDICT_FEATURE_COLS
    probe = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    predicted = np.asarray(brain.predict(probe))
    if predicted.ndim != 2:
        raise ValueError(f"Brain predict() must return a 2D array; got shape {predicted.shape}")
    actual_outputs = predicted.shape[1]
    if actual_outputs != expected_outputs:
        raise ValueError(
            "Brain output layout is incompatible with the current manning engine: "
            f"expected {expected_outputs} outputs, got {actual_outputs}. "
            "The checked engine indexes the 16-output layout with sim rates at columns "
            "6-9 and deferrals at columns 10-15; do not run viability evaluation with "
            "a legacy 12-output brain without an explicit adapter or retrained brain."
        )


def _parse_priority_mode(value: str) -> PriorityMode:
    normalized = value.lower()
    for mode in PriorityMode:
        if normalized in {mode.value, mode.name.lower()}:
            return mode
    valid = ", ".join(mode.value for mode in PriorityMode)
    raise ValueError(f"Unknown staff_priority_mode {value!r}; expected one of {valid}")
