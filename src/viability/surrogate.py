from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.viability.config import ViabilityConfig
from src.viability.design_space import DesignSpace


@dataclass(frozen=True)
class SurrogateFitResult:
    metrics_path: Path
    model_paths: dict[str, Path]
    metrics: dict[str, Any]


def fit_surrogates_from_file(
    path: str | Path,
    config: ViabilityConfig,
    output_dir: str | Path,
    *,
    boundary_threshold: float = 0.1,
    fit_gpr: bool = True,
    max_gpr_rows: int = 2000,
) -> SurrogateFitResult:
    evaluations = read_evaluations_table(path)
    return fit_surrogates(
        evaluations,
        config,
        output_dir,
        boundary_threshold=boundary_threshold,
        fit_gpr=fit_gpr,
        max_gpr_rows=max_gpr_rows,
    )


def fit_surrogates(
    evaluations: pd.DataFrame,
    config: ViabilityConfig,
    output_dir: str | Path,
    *,
    boundary_threshold: float = 0.1,
    fit_gpr: bool = True,
    max_gpr_rows: int = 2000,
) -> SurrogateFitResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_training_frame(evaluations, config)
    frame = prepared["frame"]
    x = prepared["x"]
    feature_names = prepared["feature_names"]
    constraint_columns = prepared["constraint_columns"]

    if len(frame) < 2:
        raise ValueError("At least two successful evaluation rows are required to fit a surrogate")

    train_idx, test_idx, evaluation_mode = _split_indices(len(frame), config.run.random_seed)
    x_train = x[train_idx]
    x_test = x[test_idx]

    y_phi = frame["phi"].to_numpy(dtype=float)
    phi_model = _fit_ridge(x_train, y_phi[train_idx])
    model_paths: dict[str, Path] = {}
    model_paths["phi_ridge"] = _dump_model_bundle(
        output_path / "surrogate_phi_ridge.joblib",
        model=phi_model,
        target="phi",
        feature_names=feature_names,
        config=config,
    )

    phi_pred = phi_model.predict(x_test)
    constraint_predictions: dict[str, np.ndarray] = {}
    for column in constraint_columns:
        target_name = column.removeprefix("constraint_")
        y_constraint = frame[column].to_numpy(dtype=float)
        constraint_model = _fit_ridge(x_train, y_constraint[train_idx])
        constraint_predictions[column] = constraint_model.predict(x_test)
        model_paths[f"constraint_{target_name}_ridge"] = _dump_model_bundle(
            output_path / f"surrogate_constraints_{target_name}_ridge.joblib",
            model=constraint_model,
            target=column,
            feature_names=feature_names,
            config=config,
        )

    gpr_status = "disabled"
    if fit_gpr:
        if len(train_idx) < 2:
            gpr_status = "skipped_not_enough_rows"
        else:
            gpr_model = _fit_gpr(
                x_train,
                y_phi[train_idx],
                random_seed=config.run.random_seed,
                max_rows=max_gpr_rows,
            )
            model_paths["phi_gpr"] = _dump_model_bundle(
                output_path / "surrogate_phi_gpr.joblib",
                model=gpr_model,
                target="phi",
                feature_names=feature_names,
                config=config,
            )
            gpr_status = "fit"

    metrics = _compute_metrics(
        y_true_phi=y_phi[test_idx],
        y_pred_phi=phi_pred,
        y_true_constraints=frame.loc[test_idx, constraint_columns],
        y_pred_constraints=constraint_predictions,
        boundary_threshold=boundary_threshold,
    )
    metrics.update(
        {
            "n_rows_total": int(len(evaluations)),
            "n_rows_used": int(len(frame)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "evaluation_mode": evaluation_mode,
            "feature_names": feature_names,
            "target_columns": ["phi", *constraint_columns],
            "boundary_threshold": float(boundary_threshold),
            "gpr_status": gpr_status,
            "model_paths": {name: str(path) for name, path in model_paths.items()},
        }
    )
    metrics = _json_ready(metrics)
    metrics_path = output_path / "surrogate_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return SurrogateFitResult(metrics_path=metrics_path, model_paths=model_paths, metrics=metrics)


def read_evaluations_table(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists() and input_path.suffix == ".parquet":
        csv_fallback = input_path.with_suffix(".csv")
        if csv_fallback.exists():
            input_path = csv_fallback
    if not input_path.exists():
        raise FileNotFoundError(f"Evaluations file does not exist: {input_path}")
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    return pd.read_csv(input_path)


def _prepare_training_frame(evaluations: pd.DataFrame, config: ViabilityConfig) -> dict[str, Any]:
    frame = evaluations.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"] == "ok"].copy()
    if "phi" not in frame.columns:
        raise ValueError("Evaluations table must include a phi column")
    frame = frame[np.isfinite(frame["phi"].astype(float))].reset_index(drop=True)

    constraint_columns = [
        column
        for column in frame.columns
        if column.startswith("constraint_") and np.isfinite(frame[column].astype(float)).all()
    ]
    if not constraint_columns:
        raise ValueError("Evaluations table must include at least one finite constraint_* column")

    space = DesignSpace(config.policy)
    missing = sorted(set(space.variable_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluations table is missing policy columns: {missing}")

    normalized_rows = [space.normalize(row.to_dict()) for _, row in frame.iterrows()]
    x = np.vstack(normalized_rows)
    return {
        "frame": frame,
        "x": x,
        "feature_names": space.variable_names,
        "constraint_columns": constraint_columns,
    }


def _split_indices(n_rows: int, random_seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    indices = np.arange(n_rows)
    if n_rows < 5:
        return indices, indices, "train_only_small_n"
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=random_seed,
    )
    return np.asarray(train_idx), np.asarray(test_idx), "holdout"


def _fit_ridge(x: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(x, y)
    return model


def _fit_gpr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    random_seed: int,
    max_rows: int,
) -> Pipeline:
    if len(x) > max_rows:
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(len(x), size=max_rows, replace=False)
        x = x[keep]
        y = y[keep]
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(length_scale=np.ones(x.shape[1]), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    random_state=random_seed,
                    n_restarts_optimizer=0,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def _dump_model_bundle(
    path: Path,
    *,
    model: Any,
    target: str,
    feature_names: list[str],
    config: ViabilityConfig,
) -> Path:
    bundle = {
        "model": model,
        "target": target,
        "feature_names": feature_names,
        "input_space": "normalized_policy_variables",
        "policy_parameterization": config.policy.parameterization,
    }
    joblib.dump(bundle, path)
    return path


def _compute_metrics(
    *,
    y_true_phi: np.ndarray,
    y_pred_phi: np.ndarray,
    y_true_constraints: pd.DataFrame,
    y_pred_constraints: dict[str, np.ndarray],
    boundary_threshold: float,
) -> dict[str, Any]:
    true_feasible = y_true_phi <= 0.0
    pred_feasible = y_pred_phi <= 0.0
    boundary_mask = np.abs(y_true_phi) <= boundary_threshold

    metrics: dict[str, Any] = {
        "MAE_phi": float(mean_absolute_error(y_true_phi, y_pred_phi)),
        "RMSE_phi": float(mean_squared_error(y_true_phi, y_pred_phi) ** 0.5),
        "R2_phi": _safe_r2(y_true_phi, y_pred_phi),
        "feasible_class_accuracy": float(np.mean(true_feasible == pred_feasible)),
        "false_feasible_rate": float(np.mean(pred_feasible & ~true_feasible)),
        "false_infeasible_rate": float(np.mean(~pred_feasible & true_feasible)),
        "boundary_MAE_phi": None,
    }
    if boundary_mask.any():
        metrics["boundary_MAE_phi"] = float(
            mean_absolute_error(y_true_phi[boundary_mask], y_pred_phi[boundary_mask])
        )

    sign_scores = []
    for column, predicted in y_pred_constraints.items():
        truth = y_true_constraints[column].to_numpy(dtype=float)
        sign_scores.append(np.mean((truth <= 0.0) == (predicted <= 0.0)))
    metrics["constraint_sign_accuracy"] = (
        float(np.mean(sign_scores)) if sign_scores else None
    )
    return metrics


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2:
        return None
    return float(r2_score(y_true, y_pred))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
