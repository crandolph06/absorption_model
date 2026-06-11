from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hpc_sweepers.single_phase.hpc_single_phase_sweeper import (
    is_valid_config,
    process_single_config,
)
from hpc_train_brain_multi_output import TARGETS


DEFAULT_OUTPUT = "outputs/single_phase/parquet/batch_low_viability_bootstrap.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a small local single-phase training batch for the 16-output brain."
    )
    parser.add_argument("--n", type=int, required=True, help="Number of valid rows to write")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output parquet path, default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--method",
        choices=("sobol", "random"),
        default="sobol",
        help="Candidate sampler for single-phase inputs",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Sampler index to start from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--scramble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use scrambled Sobol sampling",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker count for single-phase evaluations",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Candidate chunk size used while searching for valid rows",
    )
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    indexed_configs = collect_valid_configs(
        n=args.n,
        method=args.method,
        start_index=args.start_index,
        seed=args.seed,
        scramble=args.scramble,
        chunk_size=args.chunk_size,
    )
    rows = evaluate_indexed_configs(indexed_configs, workers=args.workers)
    if len(rows) < args.n:
        raise RuntimeError(f"Only produced {len(rows)} successful rows out of requested {args.n}")

    frame = pd.DataFrame(rows[: args.n])
    missing_targets = sorted(set(TARGETS) - set(frame.columns))
    if missing_targets:
        raise RuntimeError(f"Generated data is missing required target columns: {missing_targets}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    print(f"Wrote {len(frame)} rows to {output_path}")
    print(f"Sample indices: {frame['sample_index'].min()}..{frame['sample_index'].max()}")
    print(f"Target columns: {len(TARGETS)}")
    return 0


def collect_valid_configs(
    *,
    n: int,
    method: str,
    start_index: int,
    seed: int,
    scramble: bool,
    chunk_size: int,
) -> list[tuple[int, tuple[int, int, float, int, int, int, int, int]]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    configs: list[tuple[int, tuple[int, int, float, int, int, int, int, int]]] = []
    next_index = start_index

    if method == "sobol":
        try:
            from scipy.stats import qmc
        except ImportError as exc:  # pragma: no cover - scipy is in project requirements
            raise RuntimeError("Sobol sampling requires scipy.stats.qmc") from exc
        sampler = qmc.Sobol(d=8, scramble=scramble, seed=seed)
        if start_index:
            sampler.fast_forward(start_index)

        def sample_chunk() -> np.ndarray:
            return sampler.random(chunk_size)

    elif method == "random":
        rng = np.random.default_rng(seed)
        if start_index:
            rng.random((start_index, 8))

        def sample_chunk() -> np.ndarray:
            return rng.random((chunk_size, 8))

    else:
        raise ValueError(f"Unsupported method {method!r}")

    while len(configs) < n:
        for unit_values in sample_chunk():
            config = single_phase_config_from_unit_values(unit_values)
            if is_valid_config(
                total=config[7],
                exp=config[2],
                ip_q=config[1],
                mqt=config[4],
                flug=config[5],
                ipug=config[6],
            ):
                configs.append((next_index, config))
                if len(configs) >= n:
                    break
            next_index += 1

    return configs


def single_phase_config_from_unit_values(
    unit_values: np.ndarray,
) -> tuple[int, int, float, int, int, int, int, int]:
    if len(unit_values) != 8:
        raise ValueError(f"Expected 8 unit values, got {len(unit_values)}")
    ute = _int_from_unit(unit_values[0], 6, 20)
    ip_qty = _int_from_unit(unit_values[1], 3, 9)
    exp_ratio = round(float(np.clip(unit_values[2], 0.0, 1.0)), 2)
    paa = _int_from_unit(unit_values[3], 18, 23)
    mqt = _int_from_unit(unit_values[4], 0, 14)
    flug = _int_from_unit(unit_values[5], 0, 14)
    ipug = _int_from_unit(unit_values[6], 0, 14)
    total_pilots = _int_from_unit(unit_values[7], 25, 49)
    return ute, ip_qty, exp_ratio, paa, mqt, flug, ipug, total_pilots


def evaluate_indexed_configs(
    indexed_configs: list[tuple[int, tuple[int, int, float, int, int, int, int, int]]],
    *,
    workers: int,
) -> list[dict[str, object]]:
    configs = [config for _, config in indexed_configs]
    if workers <= 1:
        results = [process_single_config(config) for config in configs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_single_config, configs))

    rows: list[dict[str, object]] = []
    for (sample_index, _), row in zip(indexed_configs, results):
        if row is None:
            continue
        row = dict(row)
        row["sample_source"] = "local_viability_bootstrap"
        row["sample_index"] = sample_index
        rows.append(row)
    return rows


def _int_from_unit(value: float, low: int, high: int) -> int:
    bounded = float(np.clip(value, 0.0, np.nextafter(1.0, 0.0)))
    return int(low + np.floor(bounded * (high - low + 1)))


if __name__ == "__main__":
    raise SystemExit(main())
