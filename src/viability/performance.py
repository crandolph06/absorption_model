from __future__ import annotations

import cProfile
import io
import json
import pstats
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pandas as pd

from src.viability.config import ViabilityConfig
from src.viability.doe import generate_doe
from src.viability.evaluator import evaluate_design, evaluate_designs_parallel
from src.viability.io import write_config_resolved, write_table
from src.viability.policy import PolicyDesign

PhaseBackendSelection = Literal["as-config", "brain", "physics"]


@dataclass(frozen=True)
class WorkerBenchmarkResult:
    workers: int
    designs: int
    elapsed_seconds: float
    evaluations_per_second: float
    ok_count: int
    error_count: int
    min_phi: float | None
    max_phi: float | None
    output_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "workers": self.workers,
            "designs": self.designs,
            "elapsed_seconds": self.elapsed_seconds,
            "evaluations_per_second": self.evaluations_per_second,
            "ok_count": self.ok_count,
            "error_count": self.error_count,
            "min_phi": self.min_phi,
            "max_phi": self.max_phi,
            "output_path": str(self.output_path),
        }


@dataclass(frozen=True)
class ProfileResult:
    elapsed_seconds: float
    stats_path: Path
    text_path: Path
    phi: float
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "stats_path": str(self.stats_path),
            "text_path": str(self.text_path),
            "phi": self.phi,
            "status": self.status,
        }


def parse_worker_counts(raw: str) -> list[int]:
    """Parse a comma-separated worker-count list, preserving order and removing duplicates."""
    counts: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            count = int(stripped)
        except ValueError as exc:
            raise ValueError(f"Worker count {stripped!r} is not an integer") from exc
        if count <= 0:
            raise ValueError("Worker counts must be positive")
        if count not in seen:
            counts.append(count)
            seen.add(count)
    if not counts:
        raise ValueError("At least one worker count is required")
    return counts


def prepare_benchmark_config(
    config: ViabilityConfig,
    *,
    phase_backend: PhaseBackendSelection = "physics",
    years_to_run: int | None = None,
) -> ViabilityConfig:
    """Return a config suitable for timing direct evaluation behavior."""
    model = config.model
    if phase_backend != "as-config":
        model = replace(
            model,
            phase_backend=phase_backend,
            brain_path=None if phase_backend == "physics" else model.brain_path,
            expected_brain_outputs=None
            if phase_backend == "physics"
            else model.expected_brain_outputs,
        )
    if years_to_run is not None:
        if years_to_run <= 0:
            raise ValueError("years_to_run must be positive")
        model = replace(
            model,
            years_to_run=years_to_run,
            assessment_start_year=model.start_year,
            target_year=model.start_year + years_to_run - 1,
        )
    benchmark_config = replace(config, model=model)
    benchmark_config.validate()
    return benchmark_config


def benchmark_evaluation_batch(
    *,
    config: ViabilityConfig,
    output_dir: str | Path,
    n: int,
    worker_counts: list[int],
    method: str | None = None,
    include_corners: bool = False,
    include_baselines: bool = False,
    checkpoint_every: int = 100,
) -> tuple[pd.DataFrame, list[WorkerBenchmarkResult]]:
    """Generate one DOE batch and time evaluation across worker counts."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not worker_counts:
        raise ValueError("At least one worker count is required")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_config_resolved(config, output_path)

    designs = generate_doe(
        config,
        n=n,
        method=method,
        include_corners=include_corners,
        include_baselines=include_baselines,
    )
    write_table(designs, output_path / "doe.csv", prefer_parquet=False)

    summaries: list[WorkerBenchmarkResult] = []
    for workers in worker_counts:
        if workers <= 0:
            raise ValueError("Worker counts must be positive")
        run_dir = output_path / f"workers_{workers:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        results = evaluate_designs_parallel(
            designs,
            config,
            workers=workers,
            checkpoint_dir=run_dir / "checkpoints",
            checkpoint_every=checkpoint_every,
        )
        elapsed = time.perf_counter() - start
        evaluations_path = write_table(results, run_dir / "evaluations.parquet")
        ok_count = int((results["status"] == "ok").sum()) if "status" in results else 0
        error_count = int(len(results) - ok_count)
        min_phi = float(results["phi"].min()) if "phi" in results and len(results) else None
        max_phi = float(results["phi"].max()) if "phi" in results and len(results) else None
        summaries.append(
            WorkerBenchmarkResult(
                workers=workers,
                designs=len(results),
                elapsed_seconds=elapsed,
                evaluations_per_second=(len(results) / elapsed) if elapsed > 0 else float("inf"),
                ok_count=ok_count,
                error_count=error_count,
                min_phi=min_phi,
                max_phi=max_phi,
                output_path=evaluations_path,
            )
        )

    summary_frame = pd.DataFrame([row.to_dict() for row in summaries])
    write_table(summary_frame, output_path / "benchmark_summary.csv", prefer_parquet=False)
    (output_path / "benchmark_summary.json").write_text(
        json.dumps([row.to_dict() for row in summaries], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return designs, summaries


def profile_first_design(
    *,
    config: ViabilityConfig,
    designs: pd.DataFrame,
    output_dir: str | Path,
    sort_by: str = "cumtime",
    limit: int = 60,
) -> ProfileResult:
    """Profile one direct design evaluation and write both machine and text stats."""
    if designs.empty:
        raise ValueError("Cannot profile an empty design table")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    variable_names = list(config.policy.variables)
    first = designs.iloc[0]
    values = {name: first[name] for name in variable_names}
    raw_values = None
    if all(f"raw_{name}" in first for name in variable_names):
        raw_values = {name: float(first[f"raw_{name}"]) for name in variable_names}
    design = PolicyDesign.from_mapping(values, config.policy, raw_values=raw_values)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    result = evaluate_design(design, config)
    profiler.disable()
    elapsed = time.perf_counter() - start

    stats_path = output_path / "single_design.prof"
    text_path = output_path / "single_design_profile.txt"
    profiler.dump_stats(stats_path)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(sort_by).print_stats(limit)
    text_path.write_text(stream.getvalue(), encoding="utf-8")
    return ProfileResult(
        elapsed_seconds=elapsed,
        stats_path=stats_path,
        text_path=text_path,
        phi=float(result.phi),
        status=result.status,
    )
