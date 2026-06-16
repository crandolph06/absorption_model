#!/usr/bin/env python3
"""Sobol holdout audit for the single-phase sortie surrogate."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.engine import create_pilots, phase_upgrade_metrics, run_phase_simulation
from src.models import SquadronConfig
from src.rap_state import (
    mqt_observed_sim_metrics,
    mqt_observed_sortie_metrics,
    rap_assess,
    sim_rap_metrics,
)
from src.simulation_config import DEFAULT_PHASE_LENGTH_DAYS, SimulationConfig


FEATURES = [
    "paa",
    "ute",
    "exp_ratio",
    "ip_ratio",
    "fl_congestion",
    "wg_crowding",
    "sorties_avail",
    "pilot_to_sortie",
    "ip_to_stud_ratio",
]

TARGETS = [
    "wg_monthly",
    "fl_monthly",
    "ip_monthly",
    "wg_blue_monthly",
    "fl_blue_monthly",
    "ip_blue_monthly",
    "mqt_sim_monthly",
    "wg_sim_monthly",
    "fl_sim_monthly",
    "ip_sim_monthly",
    "remaining_mqt_syllabi_mean",
    "remaining_flug_syllabi_mean",
    "remaining_ipug_syllabi_mean",
    "remaining_mqt_syllabi_sorties_only_mean",
    "remaining_flug_syllabi_sorties_only_mean",
    "remaining_ipug_syllabi_sorties_only_mean",
]

TARGET_GROUPS = {
    "sortie_rates": ["wg_monthly", "fl_monthly", "ip_monthly"],
    "blue_sortie_rates": [
        "wg_blue_monthly",
        "fl_blue_monthly",
        "ip_blue_monthly",
    ],
    "sim_rates": [
        "mqt_sim_monthly",
        "wg_sim_monthly",
        "fl_sim_monthly",
        "ip_sim_monthly",
    ],
    "deferrals": [
        "remaining_mqt_syllabi_mean",
        "remaining_flug_syllabi_mean",
        "remaining_ipug_syllabi_mean",
        "remaining_mqt_syllabi_sorties_only_mean",
        "remaining_flug_syllabi_sorties_only_mean",
        "remaining_ipug_syllabi_sorties_only_mean",
    ],
}

RAW_SPECS = [
    ("ute", 6.0, 20.0, False),
    ("ip_qty", 3, 9, True),
    ("exp_ratio", 0.0, 1.0, False),
    ("paa", 18, 23, True),
    ("mqt_qty", 0, 14, True),
    ("flug_qty", 0, 14, True),
    ("ipug_qty", 0, 14, True),
    ("total_pilots", 25, 49, True),
]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    raw_candidates, candidate_summary = sobol_valid_configs(
        required=args.n_train + args.n_test,
        seed=args.seed,
        pool_power=args.pool_power,
    )
    train_raw = raw_candidates.iloc[: args.n_train].reset_index(drop=True)
    test_raw = raw_candidates.iloc[args.n_train : args.n_train + args.n_test].reset_index(
        drop=True
    )

    train_solver_t0 = time.perf_counter()
    train_truth = evaluate_configs(
        train_raw,
        phase_length_days=args.phase_length_days,
        allocation_noise=args.allocation_noise,
        seed=args.seed,
    )
    train_solver_seconds = time.perf_counter() - train_solver_t0

    test_solver_t0 = time.perf_counter()
    test_truth = evaluate_configs(
        test_raw,
        phase_length_days=args.phase_length_days,
        allocation_noise=args.allocation_noise,
        seed=args.seed + args.n_train,
    )
    test_solver_seconds = time.perf_counter() - test_solver_t0

    x_train = feature_frame(train_raw)
    x_test = feature_frame(test_raw)

    mlp = joblib.load(args.mlp_path)
    mlp_output_count = int(mlp.predict(x_test.iloc[:1]).shape[1])
    if mlp_output_count != len(TARGETS):
        raise ValueError(
            f"Expected {len(TARGETS)} MLP outputs, got {mlp_output_count}"
        )

    mlp_seconds, mlp_pred = timed_predict(
        lambda x: mlp.predict(x),
        x_test,
        repeats=args.mlp_timing_repeats,
    )

    gpr_train_t0 = time.perf_counter()
    x_scaler = StandardScaler().fit(x_train)
    x_train_scaled = x_scaler.transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(
            length_scale=np.ones(len(FEATURES)),
            length_scale_bounds=(1e-2, 1e2),
            nu=1.5,
        )
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=args.gpr_alpha,
        normalize_y=True,
        n_restarts_optimizer=args.gpr_restarts,
        random_state=args.seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gpr.fit(x_train_scaled, train_truth[TARGETS])
    gpr_train_seconds = time.perf_counter() - gpr_train_t0

    gpr_seconds, gpr_pred = timed_predict(
        lambda x: gpr.predict(x),
        x_test_scaled,
        repeats=args.gpr_timing_repeats,
    )

    metrics = pd.concat(
        [
            target_metrics("mlp", test_truth[TARGETS], mlp_pred),
            target_metrics("gpr", test_truth[TARGETS], gpr_pred),
        ],
        ignore_index=True,
    )
    group_metrics_df = pd.concat(
        [
            group_metrics("mlp", test_truth[TARGETS], mlp_pred),
            group_metrics("gpr", test_truth[TARGETS], gpr_pred),
        ],
        ignore_index=True,
    )

    predictions = prediction_frame(test_raw, test_truth, mlp_pred, gpr_pred)
    metrics.to_csv(output_dir / "sobol_mlp_gpr_target_metrics.csv", index=False)
    group_metrics_df.to_csv(output_dir / "sobol_mlp_gpr_group_metrics.csv", index=False)
    predictions.to_csv(output_dir / "sobol_mlp_gpr_predictions.csv", index=False)
    train_raw.to_csv(output_dir / "sobol_train_raw_configs.csv", index=False)
    test_raw.to_csv(output_dir / "sobol_test_raw_configs.csv", index=False)

    timing = {
        "candidate_generation_seconds": candidate_summary["generation_seconds"],
        "train_direct_solver_seconds": train_solver_seconds,
        "test_direct_solver_seconds": test_solver_seconds,
        "direct_solver_seconds_per_point": (
            train_solver_seconds + test_solver_seconds
        )
        / (args.n_train + args.n_test),
        "mlp_predict_seconds_total": mlp_seconds,
        "mlp_predict_seconds_per_point": mlp_seconds
        / max(1, args.n_test * args.mlp_timing_repeats),
        "gpr_train_seconds": gpr_train_seconds,
        "gpr_predict_seconds_total": gpr_seconds,
        "gpr_predict_seconds_per_point": gpr_seconds
        / max(1, args.n_test * args.gpr_timing_repeats),
        "total_script_seconds": time.perf_counter() - t0,
    }
    summary = {
        "mlp_path": str(args.mlp_path),
        "mlp_output_count": mlp_output_count,
        "features": FEATURES,
        "targets": TARGETS,
        "raw_specs": [
            {"name": name, "low": lo, "high": hi, "integer": integer}
            for name, lo, hi, integer in RAW_SPECS
        ],
        "n_train": args.n_train,
        "n_test": args.n_test,
        "phase_length_days": args.phase_length_days,
        "allocation_noise": args.allocation_noise,
        "candidate_summary": candidate_summary,
        "timing": timing,
        "gpr_kernel": str(gpr.kernel_),
        "metrics_csv": str(output_dir / "sobol_mlp_gpr_target_metrics.csv"),
        "group_metrics_csv": str(output_dir / "sobol_mlp_gpr_group_metrics.csv"),
        "predictions_csv": str(output_dir / "sobol_mlp_gpr_predictions.csv"),
    }
    (output_dir / "sobol_mlp_gpr_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mlp-path",
        type=Path,
        default=Path("outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/single_phase/audit"),
    )
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--pool-power", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-length-days", type=int, default=DEFAULT_PHASE_LENGTH_DAYS)
    parser.add_argument("--allocation-noise", type=float, default=0.0)
    parser.add_argument("--gpr-alpha", type=float, default=1e-6)
    parser.add_argument("--gpr-restarts", type=int, default=0)
    parser.add_argument("--mlp-timing-repeats", type=int, default=20)
    parser.add_argument("--gpr-timing-repeats", type=int, default=5)
    return parser.parse_args()


def sobol_valid_configs(
    *, required: int, seed: int, pool_power: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = time.perf_counter()
    sampler = qmc.Sobol(d=len(RAW_SPECS), scramble=True, seed=seed)
    unit = sampler.random_base2(pool_power)
    rows = [scale_raw(row) for row in unit]
    candidates = pd.DataFrame(rows)
    valid_mask = candidates.apply(is_valid_config, axis=1)
    valid = candidates[valid_mask].reset_index(drop=True)
    if len(valid) < required:
        raise ValueError(
            f"Sobol pool 2**{pool_power} yielded {len(valid)} valid configs; "
            f"need {required}. Increase --pool-power."
        )
    summary = {
        "pool_power": pool_power,
        "candidate_count": int(len(candidates)),
        "valid_count": int(len(valid)),
        "valid_fraction": float(len(valid) / len(candidates)),
        "used_count": int(required),
        "generation_seconds": time.perf_counter() - start,
    }
    return valid.iloc[:required].copy(), summary


def scale_raw(unit_row: np.ndarray) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for value, (name, low, high, integer) in zip(unit_row, RAW_SPECS):
        if integer:
            scaled = int(math.floor(low + value * (high - low + 1)))
            values[name] = min(int(high), max(int(low), scaled))
        else:
            values[name] = float(low + value * (high - low))
    return values


def is_valid_config(row: pd.Series) -> bool:
    total = int(row["total_pilots"])
    exp = float(row["exp_ratio"])
    ip_q = int(row["ip_qty"])
    mqt = int(row["mqt_qty"])
    flug = int(row["flug_qty"])
    ipug = int(row["ipug_qty"])
    experienced = int(total * exp)
    wg_count = total - experienced
    fl_count = experienced - ip_q

    if ip_q > experienced:
        return False
    if experienced > total:
        return False
    if (mqt + flug + ipug + ip_q) > total:
        return False
    if (mqt + flug) > wg_count:
        return False
    if ipug > fl_count:
        return False
    return True


def feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in raw.iterrows():
        total = int(row["total_pilots"])
        exp = float(row["exp_ratio"])
        ip_q = int(row["ip_qty"])
        mqt = int(row["mqt_qty"])
        flug = int(row["flug_qty"])
        ipug = int(row["ipug_qty"])
        paa = int(row["paa"])
        ute = float(row["ute"])
        experienced = int(total * exp)
        wg_qty = total - experienced
        fl_qty = experienced - ip_q
        total_students = mqt + flug + ipug
        sorties_avail = paa * ute
        records.append(
            {
                "paa": paa,
                "ute": ute,
                "exp_ratio": exp,
                "ip_ratio": ip_q / total if total else 0.0,
                "fl_congestion": (ipug + flug) / fl_qty if fl_qty else 0.0,
                "wg_crowding": total_students / wg_qty if wg_qty else 0.0,
                "sorties_avail": sorties_avail,
                "pilot_to_sortie": total / sorties_avail if sorties_avail else 0.0,
                "ip_to_stud_ratio": ip_q / (total_students if total_students else 0.1),
            }
        )
    return pd.DataFrame(records, columns=FEATURES).replace([np.inf, -np.inf], 0).fillna(0)


def evaluate_configs(
    raw: pd.DataFrame,
    *,
    phase_length_days: int,
    allocation_noise: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    sim_config = SimulationConfig(
        phase_length_days=phase_length_days,
        allocation_noise=allocation_noise,
    )
    for idx, row in raw.reset_index(drop=True).iterrows():
        random.seed(seed + idx)
        np.random.seed(seed + idx)
        rows.append(evaluate_one(row, sim_config))
    return pd.DataFrame(rows, columns=TARGETS)


def evaluate_one(row: pd.Series, sim_config: SimulationConfig) -> dict[str, float]:
    cfg = SquadronConfig(
        paa=int(row["paa"]),
        ute=float(row["ute"]),
        experience_ratio=float(row["exp_ratio"]),
        ip_qty=int(row["ip_qty"]),
        mqt_students=int(row["mqt_qty"]),
        flug_students=int(row["flug_qty"]),
        ipug_students=int(row["ipug_qty"]),
        total_pilots=int(row["total_pilots"]),
        id=99,
    )
    pilots = create_pilots(cfg)
    final_pilots = run_phase_simulation(cfg, pilots, sim_config=sim_config)
    rap, blue_rap, _red = rap_assess(final_pilots)
    sim_metrics = sim_rap_metrics(final_pilots)
    mqt_sims = mqt_observed_sim_metrics(final_pilots)
    _mqt_sorties = mqt_observed_sortie_metrics(final_pilots)
    upgrade_metrics = phase_upgrade_metrics(final_pilots)
    return {
        "wg_monthly": float(rap["WG"][1]),
        "fl_monthly": float(rap["FL"][1]),
        "ip_monthly": float(rap["IP"][1]),
        "wg_blue_monthly": float(blue_rap["WG"][1]),
        "fl_blue_monthly": float(blue_rap["FL"][1]),
        "ip_blue_monthly": float(blue_rap["IP"][1]),
        "mqt_sim_monthly": float(mqt_sims["sim_mo"]),
        "wg_sim_monthly": float(sim_metrics["WG"]["sim_mo"]),
        "fl_sim_monthly": float(sim_metrics["FL"]["sim_mo"]),
        "ip_sim_monthly": float(sim_metrics["IP"]["sim_mo"]),
        "remaining_mqt_syllabi_mean": float(upgrade_metrics["remaining_mqt_syllabi"]),
        "remaining_flug_syllabi_mean": float(upgrade_metrics["remaining_flug_syllabi"]),
        "remaining_ipug_syllabi_mean": float(upgrade_metrics["remaining_ipug_syllabi"]),
        "remaining_mqt_syllabi_sorties_only_mean": float(
            upgrade_metrics["remaining_mqt_syllabi_sorties_only"]
        ),
        "remaining_flug_syllabi_sorties_only_mean": float(
            upgrade_metrics["remaining_flug_syllabi_sorties_only"]
        ),
        "remaining_ipug_syllabi_sorties_only_mean": float(
            upgrade_metrics["remaining_ipug_syllabi_sorties_only"]
        ),
    }


def timed_predict(func, x, *, repeats: int) -> tuple[float, np.ndarray]:
    pred = func(x)
    start = time.perf_counter()
    for _ in range(repeats):
        pred = func(x)
    return time.perf_counter() - start, np.asarray(pred)


def target_metrics(model: str, truth: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    pred_df = pd.DataFrame(pred, columns=TARGETS)
    for target in TARGETS:
        y = truth[target].to_numpy()
        p = pred_df[target].to_numpy()
        rows.append(metric_row(model, target, target_group(target), y, p))
    return pd.DataFrame(rows)


def group_metrics(model: str, truth: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    pred_df = pd.DataFrame(pred, columns=TARGETS)
    rows = []
    for group, targets in TARGET_GROUPS.items():
        y = truth[targets].to_numpy().ravel()
        p = pred_df[targets].to_numpy().ravel()
        rows.append(metric_row(model, group, group, y, p))
    rows.append(
        metric_row(
            model,
            "all_targets",
            "all_targets",
            truth[TARGETS].to_numpy().ravel(),
            pred_df[TARGETS].to_numpy().ravel(),
        )
    )
    return pd.DataFrame(rows)


def metric_row(
    model: str,
    target: str,
    group: str,
    truth: np.ndarray,
    pred: np.ndarray,
) -> dict[str, Any]:
    err = pred - truth
    return {
        "model": model,
        "target": target,
        "group": group,
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(mean_squared_error(truth, pred) ** 0.5),
        "r2": float(r2_score(truth, pred)),
        "bias": float(np.mean(err)),
        "max_abs_error": float(np.max(np.abs(err))),
        "truth_mean": float(np.mean(truth)),
        "truth_std": float(np.std(truth)),
    }


def target_group(target: str) -> str:
    for group, targets in TARGET_GROUPS.items():
        if target in targets:
            return group
    return "unknown"


def prediction_frame(
    raw: pd.DataFrame,
    truth: pd.DataFrame,
    mlp_pred: np.ndarray,
    gpr_pred: np.ndarray,
) -> pd.DataFrame:
    out = raw.reset_index(drop=True).copy()
    for prefix, values in [
        ("truth", truth[TARGETS].to_numpy()),
        ("mlp", mlp_pred),
        ("gpr", gpr_pred),
    ]:
        for i, target in enumerate(TARGETS):
            out[f"{prefix}_{target}"] = values[:, i]
    return out


if __name__ == "__main__":
    main()
