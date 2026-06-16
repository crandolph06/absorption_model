from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.viability.config import load_config
from src.viability.performance import (
    benchmark_evaluation_batch,
    parse_worker_counts,
    prepare_benchmark_config,
    profile_first_design,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark viability evaluation throughput for direct physics analysis."
    )
    parser.add_argument("--config", required=True, help="Path to viability YAML config")
    parser.add_argument(
        "--output-dir",
        default="outputs/viability/physics_benchmark",
        help="Directory for ignored benchmark outputs",
    )
    parser.add_argument("--n", type=int, default=8, help="DOE sample count")
    parser.add_argument("--method", default=None, help="Override DOE method")
    parser.add_argument(
        "--workers",
        default="1,4,8",
        help="Comma-separated worker counts to benchmark",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=1,
        help="Override model.years_to_run for the benchmark",
    )
    parser.add_argument(
        "--phase-backend",
        choices=("as-config", "brain", "physics"),
        default="physics",
        help="Evaluation backend to benchmark",
    )
    parser.add_argument(
        "--include-corners",
        action="store_true",
        help="Include configured corner designs in addition to --n samples",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Include configured baseline designs in addition to --n samples",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Flush evaluation checkpoint batches every N completed designs",
    )
    parser.add_argument(
        "--profile-single",
        action="store_true",
        help="Write cProfile output for the first generated design",
    )
    parser.add_argument(
        "--profile-limit",
        type=int,
        default=60,
        help="Number of cProfile rows to include in text output",
    )
    args = parser.parse_args()

    worker_counts = parse_worker_counts(args.workers)
    config = prepare_benchmark_config(
        load_config(args.config),
        phase_backend=args.phase_backend,
        years_to_run=args.years,
    )
    output_dir = Path(args.output_dir)
    designs, summaries = benchmark_evaluation_batch(
        config=config,
        output_dir=output_dir,
        n=args.n,
        worker_counts=worker_counts,
        method=args.method,
        include_corners=args.include_corners,
        include_baselines=args.include_baselines,
        checkpoint_every=args.checkpoint_every,
    )

    payload: dict[str, object] = {
        "output_dir": str(output_dir),
        "design_count": len(designs),
        "phase_backend": config.model.phase_backend,
        "years_to_run": config.model.years_to_run,
        "worker_results": [result.to_dict() for result in summaries],
    }
    if args.profile_single:
        profile = profile_first_design(
            config=config,
            designs=designs,
            output_dir=output_dir / "profile",
            limit=args.profile_limit,
        )
        payload["profile"] = profile.to_dict()

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
