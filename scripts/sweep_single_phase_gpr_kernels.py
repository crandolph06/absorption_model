#!/usr/bin/env python3
"""Matched Sobol sweep for single-phase GPR kernel and target grouping choices."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from active_learn_single_phase_surrogate import (
    QUALITY_THRESHOLDS,
    fit_snapshot,
    json_ready,
    passes_quality,
    sobol_valid_unique_configs,
)
from audit_single_phase_surrogate import (
    FEATURES,
    TARGETS,
    evaluate_configs,
)
from src.simulation_config import DEFAULT_PHASE_LENGTH_DAYS


KERNEL_IDS = (
    "matern_nu0p5_ard",
    "matern_nu1p5_ard",
    "matern_nu2p5_ard",
    "rbf_ard",
)
MODEL_MODES = ("shared_ard", "grouped_ard", "per_target_ard")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    log(f"Generating matched Sobol pool with 2^{args.pool_power} candidates")
    raw_pool, pool_summary = sobol_valid_unique_configs(
        seed=args.seed,
        pool_power=args.pool_power,
    )
    required = args.n_holdout + args.n_train
    if len(raw_pool) < required:
        raise ValueError(
            f"Sobol pool has {len(raw_pool)} unique valid configs; need {required}"
        )

    holdout_raw = raw_pool.iloc[: args.n_holdout].reset_index(drop=True)
    train_raw = raw_pool.iloc[
        args.n_holdout : args.n_holdout + args.n_train
    ].reset_index(drop=True)
    train_raw.to_csv(args.output_dir / "train_raw_configs.csv", index=False)
    holdout_raw.to_csv(args.output_dir / "holdout_raw_configs.csv", index=False)

    write_json(
        args.output_dir / "sweep_config.json",
        {
            "seed": args.seed,
            "pool_power": args.pool_power,
            "pool_summary": pool_summary,
            "n_train": args.n_train,
            "n_holdout": args.n_holdout,
            "phase_length_days": args.phase_length_days,
            "allocation_noise": args.allocation_noise,
            "gpr_alpha": args.gpr_alpha,
            "gpr_restarts": args.gpr_restarts,
            "modes": args.modes,
            "kernels": args.kernels,
            "features": FEATURES,
            "targets": TARGETS,
            "quality_thresholds": QUALITY_THRESHOLDS,
        },
    )

    log(f"Evaluating {len(holdout_raw)} holdout points with direct solver")
    holdout_t0 = time.perf_counter()
    holdout_truth = evaluate_configs(
        holdout_raw,
        phase_length_days=args.phase_length_days,
        allocation_noise=args.allocation_noise,
        seed=args.seed + 100_000,
    )
    holdout_seconds = time.perf_counter() - holdout_t0
    holdout_truth.to_csv(args.output_dir / "holdout_truth.csv", index=False)

    log(f"Evaluating {len(train_raw)} training points with direct solver")
    train_t0 = time.perf_counter()
    train_truth = evaluate_configs(
        train_raw,
        phase_length_days=args.phase_length_days,
        allocation_noise=args.allocation_noise,
        seed=args.seed + 200_000,
    )
    train_seconds = time.perf_counter() - train_t0
    train_truth.to_csv(args.output_dir / "train_truth.csv", index=False)

    rows: list[dict[str, Any]] = []
    target_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for mode in args.modes:
        for kernel in args.kernels:
            model_label = f"{mode}_{kernel}"
            model_dir = models_dir / model_label
            log(f"Fitting {model_label}")
            try:
                snapshot = fit_snapshot(
                    iteration=0,
                    train_raw=train_raw,
                    train_truth=train_truth,
                    holdout_raw=holdout_raw,
                    holdout_truth=holdout_truth,
                    output_dir=model_dir,
                    seed=args.seed,
                    gpr_alpha=args.gpr_alpha,
                    gpr_restarts=args.gpr_restarts,
                    model_mode=mode,
                    kernel_id=kernel,
                    fixed_kernel_source=None,
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                log(f"{model_label} failed: {exc}")
                rows.append(
                    {
                        "status": "failed",
                        "model_mode": mode,
                        "kernel_id": kernel,
                        "error": str(exc),
                    }
                )
                write_tables(args.output_dir, rows, target_frames, group_frames)
                continue

            row = summarize_snapshot(
                snapshot=snapshot,
                model_mode=mode,
                kernel_id=kernel,
                model_dir=model_dir,
                holdout_solver_seconds=holdout_seconds,
                train_solver_seconds=train_seconds,
                elapsed_seconds=time.perf_counter() - t0,
            )
            rows.append(row)

            target_df = snapshot.target_metrics.copy()
            target_df.insert(0, "kernel_id", kernel)
            target_df.insert(0, "model_mode", mode)
            target_frames.append(target_df)

            group_df = snapshot.group_metrics.copy()
            group_df.insert(0, "kernel_id", kernel)
            group_df.insert(0, "model_mode", mode)
            group_frames.append(group_df)

            write_json(
                model_dir / "kernel_summary.json",
                {
                    "model_mode": mode,
                    "kernel_id": kernel,
                    "optimized_kernels": snapshot.bundle["kernels"],
                },
            )
            write_tables(args.output_dir, rows, target_frames, group_frames)
            log(
                f"{model_label} done: all_targets_r2={row.get('all_targets_r2'):.4f}, "
                f"gap_sum={row['quality_gap_sum']:.4f}, fit={row['fit_seconds']:.1f}s"
            )

    successful = best_rows(rows)
    best = successful[0] if successful else None
    write_json(
        args.output_dir / "best_model_summary.json",
        {
            "best": best,
            "ranked_successful_models": successful,
            "quality_thresholds": QUALITY_THRESHOLDS,
            "total_seconds": time.perf_counter() - t0,
        },
    )
    print((args.output_dir / "best_model_summary.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/single_phase/kernel_sweep"),
    )
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--pool-power", type=int, default=14)
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-holdout", type=int, default=1024)
    parser.add_argument("--phase-length-days", type=int, default=DEFAULT_PHASE_LENGTH_DAYS)
    parser.add_argument("--allocation-noise", type=float, default=0.0)
    parser.add_argument("--gpr-alpha", type=float, default=1e-8)
    parser.add_argument("--gpr-restarts", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODEL_MODES,
        default=["grouped_ard", "shared_ard"],
        help=(
            "Target grouping modes to sweep. per_target_ard is available but "
            "slow for large training sets."
        ),
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=KERNEL_IDS,
        default=list(KERNEL_IDS),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def summarize_snapshot(
    *,
    snapshot,
    model_mode: str,
    kernel_id: str,
    model_dir: Path,
    holdout_solver_seconds: float,
    train_solver_seconds: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    group_df = snapshot.group_metrics.set_index("target")
    target_df = snapshot.target_metrics
    row: dict[str, Any] = {
        "status": "ok",
        "model_mode": model_mode,
        "kernel_id": kernel_id,
        "train_rows": int(len(snapshot.train_raw)),
        "holdout_rows": int(len(snapshot.predictions)),
        "fit_seconds": float(snapshot.fit_seconds),
        "holdout_solver_seconds": float(holdout_solver_seconds),
        "train_solver_seconds": float(train_solver_seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "passes_quality": passes_quality(snapshot.group_metrics),
        "quality_gap_sum": quality_gap_sum(snapshot.group_metrics),
        "quality_gap_max": quality_gap_max(snapshot.group_metrics),
        "worst_target_r2": float(target_df["r2"].min()),
        "worst_target_rmse": float(target_df["rmse"].max()),
        "worst_target_max_abs_error": float(target_df["max_abs_error"].max()),
        "bundle_path": str((model_dir / "single_phase_gpr_bundle.joblib").resolve()),
        "group_metrics_csv": str((model_dir / "group_metrics.csv").resolve()),
        "target_metrics_csv": str((model_dir / "target_metrics.csv").resolve()),
    }
    for group in QUALITY_THRESHOLDS:
        if group not in group_df.index:
            continue
        row[f"{group}_r2"] = float(group_df.loc[group, "r2"])
        row[f"{group}_rmse"] = float(group_df.loc[group, "rmse"])
        row[f"{group}_max_abs_error"] = float(
            group_df.loc[group, "max_abs_error"]
        )
    return row


def quality_gap_sum(group_metrics_df: pd.DataFrame) -> float:
    groups = group_metrics_df.set_index("target")
    return float(
        sum(
            max(0.0, threshold - float(groups.loc[group, "r2"]))
            for group, threshold in QUALITY_THRESHOLDS.items()
            if group in groups.index
        )
    )


def quality_gap_max(group_metrics_df: pd.DataFrame) -> float:
    groups = group_metrics_df.set_index("target")
    gaps = [
        max(0.0, threshold - float(groups.loc[group, "r2"]))
        for group, threshold in QUALITY_THRESHOLDS.items()
        if group in groups.index
    ]
    return float(max(gaps, default=float("inf")))


def best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "ok"]
    return sorted(
        successful,
        key=lambda row: (
            float(row["quality_gap_sum"]),
            float(row["quality_gap_max"]),
            -float(row.get("all_targets_r2", float("-inf"))),
            -float(row.get("sortie_rates_r2", float("-inf"))),
            -float(row.get("blue_sortie_rates_r2", float("-inf"))),
            -float(row.get("sim_rates_r2", float("-inf"))),
            -float(row.get("deferrals_r2", float("-inf"))),
            float(row["fit_seconds"]),
        ),
    )


def write_tables(
    output_dir: Path,
    rows: list[dict[str, Any]],
    target_frames: list[pd.DataFrame],
    group_frames: list[pd.DataFrame],
) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "kernel_sweep_metrics.csv", index=False)
    if target_frames:
        pd.concat(target_frames, ignore_index=True).to_csv(
            output_dir / "kernel_sweep_target_metrics.csv",
            index=False,
        )
    if group_frames:
        pd.concat(group_frames, ignore_index=True).to_csv(
            output_dir / "kernel_sweep_group_metrics.csv",
            index=False,
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


if __name__ == "__main__":
    main()
