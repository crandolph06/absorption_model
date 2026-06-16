#!/usr/bin/env python3
"""Active-learning campaign for the single-phase sortie GPR surrogate."""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from audit_single_phase_surrogate import (
    FEATURES,
    RAW_SPECS,
    TARGETS,
    TARGET_GROUPS,
    evaluate_configs,
    feature_frame,
    group_metrics,
    is_valid_config,
    scale_raw,
    target_metrics,
)
from src.simulation_config import DEFAULT_PHASE_LENGTH_DAYS


QUALITY_THRESHOLDS = {
    "all_targets": 0.92,
    "sortie_rates": 0.88,
    "blue_sortie_rates": 0.88,
    "sim_rates": 0.90,
    "deferrals": 0.80,
}

GROUPED_TARGETS = {
    group: [target for target in targets if target in TARGETS]
    for group, targets in TARGET_GROUPS.items()
}


@dataclass(frozen=True)
class Snapshot:
    iteration: int
    train_raw: pd.DataFrame
    train_truth: pd.DataFrame
    bundle: dict[str, Any]
    predictions: np.ndarray
    target_metrics: pd.DataFrame
    group_metrics: pd.DataFrame
    fit_seconds: float


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    log(f"Generating Sobol pool with 2^{args.pool_power} candidates")
    raw_pool, pool_summary = sobol_valid_unique_configs(
        seed=args.seed,
        pool_power=args.pool_power,
    )
    required = args.n_holdout + args.n_initial + args.batch_size * args.iterations
    if len(raw_pool) < required:
        raise ValueError(
            f"Sobol pool has {len(raw_pool)} unique valid configs; need at least {required}"
        )
    log(
        "Using "
        f"{args.n_holdout} holdout, {args.n_initial} initial train, "
        f"{len(raw_pool) - args.n_holdout - args.n_initial} candidate rows"
    )

    holdout_raw = raw_pool.iloc[: args.n_holdout].reset_index(drop=True)
    train_raw = raw_pool.iloc[
        args.n_holdout : args.n_holdout + args.n_initial
    ].reset_index(drop=True)
    candidate_raw = raw_pool.iloc[args.n_holdout + args.n_initial :].reset_index(
        drop=True
    )

    write_json(
        output_dir / "campaign_config.json",
        {
            "seed": args.seed,
            "pool_power": args.pool_power,
            "pool_summary": pool_summary,
            "n_holdout": args.n_holdout,
            "n_initial": args.n_initial,
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            "candidate_rows_available": int(len(candidate_raw)),
            "phase_length_days": args.phase_length_days,
            "allocation_noise": args.allocation_noise,
            "features": FEATURES,
            "targets": TARGETS,
            "quality_thresholds": QUALITY_THRESHOLDS,
            "model_mode": args.model_mode,
            "kernel_id": args.kernel_id,
            "fixed_kernel_source": None
            if args.fixed_kernel_source is None
            else str(args.fixed_kernel_source),
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
    holdout_raw.to_csv(output_dir / "holdout_raw_configs.csv", index=False)
    holdout_truth.to_csv(output_dir / "holdout_truth.csv", index=False)
    log(f"Holdout direct solve finished in {holdout_seconds:.1f}s")

    log(f"Evaluating {len(train_raw)} initial training points with direct solver")
    train_t0 = time.perf_counter()
    train_truth = evaluate_configs(
        train_raw,
        phase_length_days=args.phase_length_days,
        allocation_noise=args.allocation_noise,
        seed=args.seed + 200_000,
    )
    train_seconds = time.perf_counter() - train_t0
    log(f"Initial direct solve finished in {train_seconds:.1f}s")

    metric_rows: list[dict[str, Any]] = []
    selection_frames: list[pd.DataFrame] = []
    log(
        "Fitting iteration 0 "
        f"({args.model_mode}, {args.kernel_id}, {len(train_raw)} rows)"
    )
    snapshot = fit_snapshot(
        iteration=0,
        train_raw=train_raw,
        train_truth=train_truth,
        holdout_raw=holdout_raw,
        holdout_truth=holdout_truth,
        output_dir=output_dir / "iteration_000",
        seed=args.seed,
        gpr_alpha=args.gpr_alpha,
        gpr_restarts=args.gpr_restarts,
        model_mode=args.model_mode,
        kernel_id=args.kernel_id,
        fixed_kernel_source=args.fixed_kernel_source,
    )
    log(
        f"Iteration 0 fit finished in {snapshot.fit_seconds:.1f}s; "
        f"passes_quality={passes_quality(snapshot.group_metrics)}"
    )
    metric_rows.append(
        iteration_summary(
            snapshot,
            selected_rows=0,
            selected_solver_seconds=0.0,
            holdout_solver_seconds=holdout_seconds,
            initial_solver_seconds=train_seconds,
            campaign_seconds=time.perf_counter() - t0,
        )
    )
    write_metrics(output_dir, metric_rows)

    for iteration in range(1, args.iterations + 1):
        if passes_quality(snapshot.group_metrics):
            break
        iteration_dir = output_dir / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        log(f"Scoring candidate pool for iteration {iteration}")
        scored = score_candidates(
            snapshot=snapshot,
            candidate_raw=candidate_raw,
            holdout_truth=holdout_truth,
            min_distance=args.min_normalized_distance,
        )
        scored.head(args.candidate_report_rows).to_csv(
            iteration_dir / "scored_candidates_top.csv",
            index=False,
        )
        selected, candidate_raw = select_batch(
            scored=scored,
            candidate_raw=candidate_raw,
            train_raw=snapshot.train_raw,
            batch_size=args.batch_size,
            min_distance=args.min_normalized_distance,
            rng=np.random.default_rng(args.seed + iteration),
        )
        selected.insert(0, "selection_iteration", iteration)
        selected.insert(1, "selection_rank", np.arange(1, len(selected) + 1))
        selected.to_csv(iteration_dir / "selected_candidates.csv", index=False)
        selection_frames.append(selected)

        log(f"Evaluating {len(selected)} selected points for iteration {iteration}")
        selected_solver_t0 = time.perf_counter()
        selected_truth = evaluate_configs(
            selected[raw_columns()],
            phase_length_days=args.phase_length_days,
            allocation_noise=args.allocation_noise,
            seed=args.seed + 300_000 + iteration * 10_000,
        )
        selected_solver_seconds = time.perf_counter() - selected_solver_t0
        selected_truth.to_csv(iteration_dir / "selected_truth.csv", index=False)
        log(
            f"Selected direct solves finished in {selected_solver_seconds:.1f}s; "
            f"training rows will be {len(snapshot.train_raw) + len(selected)}"
        )

        train_raw = pd.concat(
            [snapshot.train_raw, selected[raw_columns()]],
            ignore_index=True,
        )
        train_truth = pd.concat(
            [snapshot.train_truth, selected_truth],
            ignore_index=True,
        )
        snapshot = fit_snapshot(
            iteration=iteration,
            train_raw=train_raw,
            train_truth=train_truth,
            holdout_raw=holdout_raw,
            holdout_truth=holdout_truth,
            output_dir=iteration_dir,
            seed=args.seed + iteration,
            gpr_alpha=args.gpr_alpha,
            gpr_restarts=args.gpr_restarts,
            model_mode=args.model_mode,
            kernel_id=args.kernel_id,
            fixed_kernel_source=args.fixed_kernel_source,
        )
        log(
            f"Iteration {iteration} fit finished in {snapshot.fit_seconds:.1f}s; "
            f"passes_quality={passes_quality(snapshot.group_metrics)}"
        )
        metric_rows.append(
            iteration_summary(
                snapshot,
                selected_rows=len(selected),
                selected_solver_seconds=selected_solver_seconds,
                holdout_solver_seconds=holdout_seconds,
                initial_solver_seconds=train_seconds,
                campaign_seconds=time.perf_counter() - t0,
            )
        )
        write_metrics(output_dir, metric_rows)

    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    final_bundle_path = final_dir / "single_phase_gpr_bundle.joblib"
    joblib.dump(snapshot.bundle, final_bundle_path)
    snapshot.train_raw.to_csv(final_dir / "training_raw_configs.csv", index=False)
    snapshot.train_truth.to_csv(final_dir / "training_truth.csv", index=False)
    snapshot.target_metrics.to_csv(final_dir / "target_metrics.csv", index=False)
    snapshot.group_metrics.to_csv(final_dir / "group_metrics.csv", index=False)
    if selection_frames:
        pd.concat(selection_frames, ignore_index=True).to_csv(
            final_dir / "selected_candidates_all.csv",
            index=False,
        )
    write_json(
        final_dir / "summary.json",
        {
            "final_iteration": int(snapshot.iteration),
            "final_train_rows": int(len(snapshot.train_raw)),
            "bundle_path": str(final_bundle_path.resolve()),
            "passes_quality": passes_quality(snapshot.group_metrics),
            "quality_thresholds": QUALITY_THRESHOLDS,
            "total_seconds": time.perf_counter() - t0,
            "metrics_csv": str((output_dir / "active_learning_metrics.csv").resolve()),
            "target_metrics_csv": str((final_dir / "target_metrics.csv").resolve()),
            "group_metrics_csv": str((final_dir / "group_metrics.csv").resolve()),
        },
    )
    print((final_dir / "summary.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/single_phase/active_learning"),
    )
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--pool-power", type=int, default=15)
    parser.add_argument("--n-holdout", type=int, default=768)
    parser.add_argument("--n-initial", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--candidate-report-rows", type=int, default=500)
    parser.add_argument("--phase-length-days", type=int, default=DEFAULT_PHASE_LENGTH_DAYS)
    parser.add_argument("--allocation-noise", type=float, default=0.0)
    parser.add_argument("--gpr-alpha", type=float, default=1e-8)
    parser.add_argument("--gpr-restarts", type=int, default=0)
    parser.add_argument(
        "--model-mode",
        choices=("shared_ard", "grouped_ard", "per_target_ard"),
        default="grouped_ard",
    )
    parser.add_argument(
        "--kernel-id",
        choices=("matern_nu0p5_ard", "matern_nu1p5_ard", "matern_nu2p5_ard", "rbf_ard"),
        default="matern_nu1p5_ard",
    )
    parser.add_argument(
        "--fixed-kernel-source",
        type=Path,
        default=None,
        help=(
            "Optional prior bundle whose optimized kernels should be reused with "
            "optimizer=None for faster larger-N refits."
        ),
    )
    parser.add_argument("--min-normalized-distance", type=float, default=0.035)
    return parser.parse_args()


def sobol_valid_unique_configs(
    *, seed: int, pool_power: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = time.perf_counter()
    sampler = qmc.Sobol(d=len(RAW_SPECS), scramble=True, seed=seed)
    unit = sampler.random_base2(pool_power)
    raw = pd.DataFrame([scale_raw(row) for row in unit])
    valid = raw[raw.apply(is_valid_config, axis=1)].copy()
    before_dedupe = len(valid)
    valid = valid.drop_duplicates(subset=raw_columns()).reset_index(drop=True)
    return valid, {
        "pool_power": pool_power,
        "candidate_count": int(len(raw)),
        "valid_count_before_dedupe": int(before_dedupe),
        "valid_unique_count": int(len(valid)),
        "valid_unique_fraction": float(len(valid) / len(raw)),
        "generation_seconds": time.perf_counter() - start,
    }


def fit_snapshot(
    *,
    iteration: int,
    train_raw: pd.DataFrame,
    train_truth: pd.DataFrame,
    holdout_raw: pd.DataFrame,
    holdout_truth: pd.DataFrame,
    output_dir: Path,
    seed: int,
    gpr_alpha: float,
    gpr_restarts: int,
    model_mode: str,
    kernel_id: str,
    fixed_kernel_source: Path | None,
) -> Snapshot:
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_t0 = time.perf_counter()
    x_train = feature_frame(train_raw)
    x_holdout = feature_frame(holdout_raw)
    bundle = fit_gpr_bundle(
        x_train,
        train_truth[TARGETS],
        seed=seed,
        gpr_alpha=gpr_alpha,
        gpr_restarts=gpr_restarts,
        model_mode=model_mode,
        kernel_id=kernel_id,
        fixed_kernel_source=fixed_kernel_source,
    )
    predictions, std = predict_bundle(bundle, x_holdout, return_std=True)
    fit_seconds = time.perf_counter() - fit_t0
    target_metrics_df = target_metrics("active_gpr", holdout_truth[TARGETS], predictions)
    group_metrics_df = group_metrics("active_gpr", holdout_truth[TARGETS], predictions)

    target_metrics_df.to_csv(output_dir / "target_metrics.csv", index=False)
    group_metrics_df.to_csv(output_dir / "group_metrics.csv", index=False)
    holdout_prediction_frame(
        holdout_raw,
        holdout_truth,
        predictions,
        std,
    ).to_csv(output_dir / "holdout_predictions.csv", index=False)
    train_raw.to_csv(output_dir / "training_raw_configs.csv", index=False)
    train_truth.to_csv(output_dir / "training_truth.csv", index=False)
    joblib.dump(bundle, output_dir / "single_phase_gpr_bundle.joblib")
    return Snapshot(
        iteration=iteration,
        train_raw=train_raw.reset_index(drop=True),
        train_truth=train_truth.reset_index(drop=True),
        bundle=bundle,
        predictions=predictions,
        target_metrics=target_metrics_df,
        group_metrics=group_metrics_df,
        fit_seconds=fit_seconds,
    )


def fit_gpr_bundle(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    seed: int,
    gpr_alpha: float,
    gpr_restarts: int,
    model_mode: str,
    kernel_id: str,
    fixed_kernel_source: Path | None,
) -> dict[str, Any]:
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)
    model_specs = model_target_specs(model_mode)
    fixed_bundle = None
    if fixed_kernel_source is not None:
        fixed_bundle = joblib.load(fixed_kernel_source)
        validate_fixed_kernel_bundle(fixed_bundle, model_mode, kernel_id, model_specs)
    models = {}
    kernels = {}
    for index, (model_name, targets) in enumerate(model_specs.items()):
        kernel = make_kernel(kernel_id)
        optimizer = "fmin_l_bfgs_b"
        if fixed_bundle is not None:
            kernel = clone(fixed_bundle["models"][model_name].kernel_)
            optimizer = None
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=gpr_alpha,
            normalize_y=True,
            n_restarts_optimizer=gpr_restarts,
            optimizer=optimizer,
            random_state=seed + index,
        )
        y = y_train[targets].to_numpy(dtype=float)
        if y.shape[1] == 1:
            y = y.ravel()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(x_scaled, y)
        models[model_name] = model
        kernels[model_name] = str(model.kernel_)
    return {
        "model_type": "single_phase_gpr",
        "model_mode": model_mode,
        "kernel_id": kernel_id,
        "fixed_kernel_source": None if fixed_kernel_source is None else str(fixed_kernel_source),
        "feature_names": FEATURES,
        "target_names": TARGETS,
        "target_groups": TARGET_GROUPS,
        "model_target_specs": model_specs,
        "raw_specs": [
            {"name": name, "low": lo, "high": hi, "integer": integer}
            for name, lo, hi, integer in RAW_SPECS
        ],
        "x_scaler": scaler,
        "models": models,
        "kernels": kernels,
    }


def validate_fixed_kernel_bundle(
    bundle: dict[str, Any],
    model_mode: str,
    kernel_id: str,
    model_specs: dict[str, list[str]],
) -> None:
    if bundle.get("model_mode") != model_mode:
        raise ValueError(
            f"Fixed kernel bundle model_mode={bundle.get('model_mode')!r}; expected {model_mode!r}"
        )
    if bundle.get("kernel_id") != kernel_id:
        raise ValueError(
            f"Fixed kernel bundle kernel_id={bundle.get('kernel_id')!r}; expected {kernel_id!r}"
        )
    if bundle.get("feature_names") != FEATURES:
        raise ValueError("Fixed kernel bundle feature_names do not match this run")
    if bundle.get("model_target_specs") != model_specs:
        raise ValueError("Fixed kernel bundle target grouping does not match this run")
    if "models" not in bundle:
        raise ValueError("Fixed kernel bundle is missing models")
    missing_models = set(model_specs) - set(bundle["models"])
    if missing_models:
        raise ValueError(f"Fixed kernel bundle is missing models: {sorted(missing_models)}")


def model_target_specs(model_mode: str) -> dict[str, list[str]]:
    if model_mode == "shared_ard":
        return {"all_targets": list(TARGETS)}
    if model_mode == "grouped_ard":
        return {group: list(targets) for group, targets in GROUPED_TARGETS.items()}
    if model_mode == "per_target_ard":
        return {target: [target] for target in TARGETS}
    raise ValueError(f"Unsupported model_mode {model_mode!r}")


def make_kernel(kernel_id: str):
    constant = ConstantKernel(1.0, (1e-3, 1e3))
    length_scale = np.ones(len(FEATURES))
    length_bounds = (1e-3, 1e3)
    if kernel_id == "matern_nu0p5_ard":
        base = Matern(length_scale=length_scale, length_scale_bounds=length_bounds, nu=0.5)
    elif kernel_id == "matern_nu1p5_ard":
        base = Matern(length_scale=length_scale, length_scale_bounds=length_bounds, nu=1.5)
    elif kernel_id == "matern_nu2p5_ard":
        base = Matern(length_scale=length_scale, length_scale_bounds=length_bounds, nu=2.5)
    elif kernel_id == "rbf_ard":
        from sklearn.gaussian_process.kernels import RBF

        base = RBF(length_scale=length_scale, length_scale_bounds=length_bounds)
    else:
        raise ValueError(f"Unsupported kernel_id {kernel_id!r}")
    return constant * base + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))


def predict_bundle(
    bundle: dict[str, Any],
    x: pd.DataFrame,
    *,
    return_std: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    x_scaled = bundle["x_scaler"].transform(x[FEATURES])
    mean_by_target: dict[str, np.ndarray] = {}
    std_by_target: dict[str, np.ndarray] = {}
    for model_name, targets in bundle["model_target_specs"].items():
        model = bundle["models"][model_name]
        if return_std:
            mean, std = model.predict(x_scaled, return_std=True)
            mean_2d = ensure_2d(mean)
            std_2d = ensure_2d(std)
            for i, target in enumerate(targets):
                mean_by_target[target] = mean_2d[:, i]
                std_by_target[target] = std_2d[:, i]
        else:
            mean = ensure_2d(model.predict(x_scaled))
            for i, target in enumerate(targets):
                mean_by_target[target] = mean[:, i]
    mean_array = np.column_stack([mean_by_target[target] for target in TARGETS])
    if return_std:
        return mean_array, np.column_stack([std_by_target[target] for target in TARGETS])
    return mean_array


def ensure_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def score_candidates(
    *,
    snapshot: Snapshot,
    candidate_raw: pd.DataFrame,
    holdout_truth: pd.DataFrame,
    min_distance: float,
) -> pd.DataFrame:
    x_candidates = feature_frame(candidate_raw)
    means, stds = predict_bundle(snapshot.bundle, x_candidates, return_std=True)
    target_metrics_df = snapshot.target_metrics.set_index("target")
    target_stds = holdout_truth[TARGETS].std(ddof=0).replace(0.0, 1.0)

    weights = []
    for target in TARGETS:
        row = target_metrics_df.loc[target]
        norm_rmse = float(row["rmse"]) / max(float(row["truth_std"]), 1e-6)
        weight = 1.0 + min(4.0, 2.0 * norm_rmse)
        if target_group(target) == "deferrals":
            weight *= 1.35
        weights.append(weight)
    weights_arr = np.asarray(weights)
    std_norm = stds / target_stds.to_numpy(dtype=float)
    mean_norm = np.maximum(means, 0.0) / target_stds.to_numpy(dtype=float)

    deferral_indices = np.array(
        [i for i, target in enumerate(TARGETS) if target_group(target) == "deferrals"],
        dtype=int,
    )
    weak_indices = np.array(
        [
            i
            for i, target in enumerate(TARGETS)
            if float(target_metrics_df.loc[target, "r2"]) < 0.75
        ],
        dtype=int,
    )
    if weak_indices.size == 0:
        weak_indices = np.arange(len(TARGETS), dtype=int)

    scored = candidate_raw.reset_index(drop=True).copy()
    scored["candidate_index"] = np.arange(len(scored), dtype=int)
    scored["weighted_uncertainty"] = (std_norm * weights_arr).mean(axis=1)
    scored["weak_target_uncertainty"] = (std_norm[:, weak_indices] * weights_arr[weak_indices]).mean(axis=1)
    scored["deferral_uncertainty"] = (std_norm[:, deferral_indices] * weights_arr[deferral_indices]).mean(axis=1)
    scored["predicted_deferral_signal"] = mean_norm[:, deferral_indices].mean(axis=1)
    scored["distance_to_training"] = distance_to_training(candidate_raw, snapshot.train_raw)
    acquisition_score = (
        scored["weighted_uncertainty"]
        + 0.35 * scored["weak_target_uncertainty"]
        + 0.25 * scored["deferral_uncertainty"]
        + 0.10 * scored["predicted_deferral_signal"]
        + 0.05 * scored["distance_to_training"]
    )
    scored = scored.assign(acquisition_score=acquisition_score.to_numpy(dtype=float))
    # Keep the selected batch spatially useful; distances below the requested
    # threshold can still appear in the report but are deprioritized.
    near_training = scored["distance_to_training"] < min_distance
    scored.loc[near_training, "acquisition_score"] = (
        scored.loc[near_training, "acquisition_score"].to_numpy(dtype=float) * 0.5
    )
    return scored.sort_values("acquisition_score", ascending=False).reset_index(drop=True)


def select_batch(
    *,
    scored: pd.DataFrame,
    candidate_raw: pd.DataFrame,
    train_raw: pd.DataFrame,
    batch_size: int,
    min_distance: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_parts = []
    selected_keys: set[tuple[Any, ...]] = set()
    selected_vectors: list[np.ndarray] = []
    sources = [
        ("weighted_uncertainty", 0.55),
        ("weak_target_uncertainty", 0.25),
        ("deferral_uncertainty", 0.15),
        ("distance_to_training", 0.05),
    ]
    for source, fraction in sources:
        count = int(round(batch_size * fraction))
        if source == sources[-1][0]:
            count = batch_size - sum(len(part) for part in selected_parts)
        ordered = scored.sort_values(source, ascending=False).reset_index(drop=True)
        picked = pick_diverse(
            ordered,
            count=count,
            min_distance=min_distance,
            selected_keys=selected_keys,
            selected_vectors=selected_vectors,
        )
        if not picked.empty:
            picked["selection_source"] = source
            selected_parts.append(picked)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else scored.iloc[[]].copy()
    if len(selected) < batch_size:
        remainder = pick_diverse(
            scored.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))),
            count=batch_size - len(selected),
            min_distance=max(0.0, min_distance * 0.5),
            selected_keys=selected_keys,
            selected_vectors=selected_vectors,
        )
        remainder["selection_source"] = "diversity_fill"
        selected = pd.concat([selected, remainder], ignore_index=True)
    if len(selected) != batch_size:
        raise RuntimeError(
            f"Could only select {len(selected)} candidates for batch_size={batch_size}"
        )

    selected_candidate_indices = set(selected["candidate_index"].astype(int))
    remaining = candidate_raw.loc[
        ~candidate_raw.reset_index(drop=True).index.isin(selected_candidate_indices)
    ].reset_index(drop=True)
    return selected.reset_index(drop=True), remaining


def pick_diverse(
    ordered: pd.DataFrame,
    *,
    count: int,
    min_distance: float,
    selected_keys: set[tuple[Any, ...]],
    selected_vectors: list[np.ndarray],
) -> pd.DataFrame:
    if count <= 0:
        return ordered.iloc[[]].copy()
    picked_indices = []
    for row_index, row in ordered.iterrows():
        key = raw_key(row)
        if key in selected_keys:
            continue
        vector = raw_vector(row)
        if all(float(np.linalg.norm(vector - other)) >= min_distance for other in selected_vectors):
            picked_indices.append(row_index)
            selected_keys.add(key)
            selected_vectors.append(vector)
        if len(picked_indices) == count:
            break
    return ordered.loc[picked_indices].copy()


def distance_to_training(candidate_raw: pd.DataFrame, train_raw: pd.DataFrame) -> np.ndarray:
    candidate_vectors = np.vstack([raw_vector(row) for _, row in candidate_raw.iterrows()])
    train_vectors = np.vstack([raw_vector(row) for _, row in train_raw.iterrows()])
    return cdist(candidate_vectors, train_vectors).min(axis=1)


def iteration_summary(
    snapshot: Snapshot,
    *,
    selected_rows: int,
    selected_solver_seconds: float,
    holdout_solver_seconds: float,
    initial_solver_seconds: float,
    campaign_seconds: float,
) -> dict[str, Any]:
    groups = snapshot.group_metrics.set_index("target")
    targets = snapshot.target_metrics
    row = {
        "iteration": int(snapshot.iteration),
        "train_rows": int(len(snapshot.train_raw)),
        "selected_rows": int(selected_rows),
        "fit_seconds": float(snapshot.fit_seconds),
        "selected_solver_seconds": float(selected_solver_seconds),
        "holdout_solver_seconds": float(holdout_solver_seconds),
        "initial_solver_seconds": float(initial_solver_seconds),
        "campaign_seconds": float(campaign_seconds),
        "passes_quality": passes_quality(snapshot.group_metrics),
        "worst_target_r2": float(targets["r2"].min()),
        "worst_target_rmse": float(targets["rmse"].max()),
        "worst_target_max_abs_error": float(targets["max_abs_error"].max()),
    }
    for group in ["all_targets", "sortie_rates", "blue_sortie_rates", "sim_rates", "deferrals"]:
        if group in groups.index:
            row[f"{group}_r2"] = float(groups.loc[group, "r2"])
            row[f"{group}_mae"] = float(groups.loc[group, "mae"])
            row[f"{group}_rmse"] = float(groups.loc[group, "rmse"])
            row[f"{group}_max_abs_error"] = float(groups.loc[group, "max_abs_error"])
    return row


def passes_quality(group_metrics_df: pd.DataFrame) -> bool:
    groups = group_metrics_df.set_index("target")
    for group, threshold in QUALITY_THRESHOLDS.items():
        if group not in groups.index:
            return False
        if float(groups.loc[group, "r2"]) < threshold:
            return False
    return True


def write_metrics(output_dir: Path, metric_rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(metric_rows).to_csv(output_dir / "active_learning_metrics.csv", index=False)


def holdout_prediction_frame(
    holdout_raw: pd.DataFrame,
    holdout_truth: pd.DataFrame,
    predictions: np.ndarray,
    std: np.ndarray,
) -> pd.DataFrame:
    out = holdout_raw.reset_index(drop=True).copy()
    for i, target in enumerate(TARGETS):
        out[f"truth_{target}"] = holdout_truth[target].to_numpy(dtype=float)
        out[f"predicted_{target}"] = predictions[:, i]
        out[f"sigma_{target}"] = std[:, i]
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def target_group(target: str) -> str:
    for group, targets in TARGET_GROUPS.items():
        if target in targets:
            return group
    return "unknown"


def raw_columns() -> list[str]:
    return [name for name, *_ in RAW_SPECS]


def raw_key(row: pd.Series) -> tuple[Any, ...]:
    values = []
    for name, *_ in RAW_SPECS:
        value = row[name]
        if hasattr(value, "item"):
            value = value.item()
        values.append(value)
    return tuple(values)


def raw_vector(row: pd.Series) -> np.ndarray:
    values = []
    for name, low, high, _integer in RAW_SPECS:
        values.append((float(row[name]) - float(low)) / (float(high) - float(low)))
    return np.asarray(values, dtype=float)


if __name__ == "__main__":
    main()
