from __future__ import annotations

from dataclasses import dataclass
import json
import os
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

from src.viability.config import ViabilityConfig, load_config
from src.viability.design_space import DesignSpace


@dataclass(frozen=True)
class SurrogateFitResult:
    metrics_path: Path
    model_paths: dict[str, Path]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class SurrogateConvergenceResult:
    metrics_path: Path
    metrics_table_path: Path
    model_path: Path
    plot_paths: dict[str, Path]
    metrics_table: pd.DataFrame
    converged: bool


@dataclass(frozen=True)
class GPRPredictionOverlayResult:
    plot_path: Path
    point_counts: dict[str, int]


@dataclass(frozen=True)
class HoldoutSelectionResult:
    holdout_path: Path
    n_rows_total: int
    holdout_size: int


@dataclass(frozen=True)
class _PredictionScatterData:
    label: str
    true_phi: np.ndarray
    predicted_phi: np.ndarray


@dataclass(frozen=True)
class ConstraintSurrogatePrediction:
    mu: pd.DataFrame
    sigma: pd.DataFrame
    predicted_phi: np.ndarray
    sigma_phi: np.ndarray
    conservative_phi: np.ndarray
    active_constraint: np.ndarray


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
    gpr_phi_metrics = None
    if fit_gpr:
        if len(train_idx) < 2:
            gpr_status = "skipped_not_enough_rows"
        else:
            normalized_constraints = normalized_constraint_frame(frame, constraint_columns, config)
            gpr_bundle = fit_constraint_gpr_bundle(
                x=x_train,
                normalized_constraints=normalized_constraints.iloc[train_idx].reset_index(drop=True),
                feature_names=feature_names,
                config=config,
                max_rows=max_gpr_rows,
            )
            model_paths["constraints_gpr"] = dump_constraint_gpr_bundle(
                output_path / "surrogate_constraints_gpr.joblib",
                gpr_bundle,
            )
            gpr_prediction = predict_constraint_surrogate(gpr_bundle, x_test)
            gpr_phi_metrics = _compute_phi_metrics(
                y_true_phi=y_phi[test_idx],
                y_pred_phi=gpr_prediction.predicted_phi,
                boundary_threshold=boundary_threshold,
            )
            gpr_phi_metrics["constraint_sign_accuracy"] = _constraint_sign_accuracy(
                normalized_constraints.iloc[test_idx].reset_index(drop=True),
                gpr_prediction.mu,
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
    if gpr_phi_metrics is not None:
        metrics["gpr_phi"] = gpr_phi_metrics
    metrics = _json_ready(metrics)
    metrics_path = output_path / "surrogate_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return SurrogateFitResult(metrics_path=metrics_path, model_paths=model_paths, metrics=metrics)


def run_gpr_convergence_from_file(
    path: str | Path,
    config: ViabilityConfig,
    output_dir: str | Path,
    *,
    train_sizes: list[int] | None = None,
    holdout_fraction: float = 0.2,
    target_r2: float | None = None,
    target_normalized_mae: float | None = None,
    target_normalized_rmse: float | None = None,
    boundary_threshold: float = 0.1,
    holdout_path: str | Path | None = None,
) -> SurrogateConvergenceResult:
    evaluations = read_evaluations_table(path)
    holdout_evaluations = read_evaluations_table(holdout_path) if holdout_path else None
    return run_gpr_convergence(
        evaluations,
        config,
        output_dir,
        train_sizes=train_sizes,
        holdout_fraction=holdout_fraction,
        target_r2=target_r2,
        target_normalized_mae=target_normalized_mae,
        target_normalized_rmse=target_normalized_rmse,
        boundary_threshold=boundary_threshold,
        holdout_evaluations=holdout_evaluations,
        holdout_path=holdout_path,
    )


def run_gpr_convergence(
    evaluations: pd.DataFrame,
    config: ViabilityConfig,
    output_dir: str | Path,
    *,
    train_sizes: list[int] | None = None,
    holdout_fraction: float = 0.2,
    target_r2: float | None = None,
    target_normalized_mae: float | None = None,
    target_normalized_rmse: float | None = None,
    boundary_threshold: float = 0.1,
    holdout_evaluations: pd.DataFrame | None = None,
    holdout_path: str | Path | None = None,
) -> SurrogateConvergenceResult:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_training_frame(evaluations, config)
    frame = _sort_for_convergence(prepared["frame"])
    x = prepared["x"][frame["_prepared_index"].to_numpy(dtype=int)]
    y_phi = frame["phi"].to_numpy(dtype=float)
    feature_names = prepared["feature_names"]
    constraint_columns = prepared["constraint_columns"]
    normalized_constraints = normalized_constraint_frame(frame, constraint_columns, config)

    if len(frame) < 8:
        raise ValueError("At least 8 successful rows are required for a convergence study")

    if holdout_evaluations is None:
        train_pool_idx, holdout_idx = _fixed_holdout_indices(
            len(frame),
            holdout_fraction=holdout_fraction,
            random_seed=config.run.random_seed,
        )
        holdout_source = "generated_split"
        holdout_id_columns: list[str] = []
        x_train_source = x
        normalized_train_source = normalized_constraints
        x_holdout = x[holdout_idx]
        y_holdout = y_phi[holdout_idx]
        normalized_holdout = normalized_constraints.iloc[holdout_idx].reset_index(drop=True)
    else:
        prepared_holdout = _prepare_training_frame(holdout_evaluations, config)
        holdout_frame = _sort_for_convergence(prepared_holdout["frame"])
        x_holdout = prepared_holdout["x"][
            holdout_frame["_prepared_index"].to_numpy(dtype=int)
        ]
        y_holdout = holdout_frame["phi"].to_numpy(dtype=float)
        normalized_holdout = normalized_constraint_frame(
            holdout_frame,
            constraint_columns,
            config,
        ).reset_index(drop=True)
        holdout_id_columns = _common_holdout_id_columns(frame, holdout_frame)
        frame, x, y_phi = _exclude_holdout_rows(
            frame=frame,
            x=x,
            y_phi=y_phi,
            holdout_frame=holdout_frame,
            id_columns=holdout_id_columns,
        )
        holdout_source = "external_table"
        train_pool_idx = np.arange(len(frame), dtype=int)
        x_train_source = x
        normalized_train_source = normalized_constraint_frame(frame, constraint_columns, config)
    if len(y_holdout) < 2:
        raise ValueError("Holdout set must contain at least two rows")
    if len(train_pool_idx) < 2:
        raise ValueError("Training pool must contain at least two rows after holdout exclusion")

    selected_train_sizes = _resolve_train_sizes(train_sizes, max_train_size=len(train_pool_idx))
    rows: list[dict[str, Any]] = []
    last_model: dict[str, Any] | None = None
    last_predictions: np.ndarray | None = None

    for train_size in selected_train_sizes:
        current_train_idx = train_pool_idx[:train_size]
        model = fit_constraint_gpr_bundle(
            x=x_train_source[current_train_idx],
            normalized_constraints=normalized_train_source.iloc[current_train_idx].reset_index(drop=True),
            feature_names=feature_names,
            config=config,
            max_rows=train_size,
        )
        prediction = predict_constraint_surrogate(model, x_holdout)
        predictions = prediction.predicted_phi
        row = {
            "train_size": int(train_size),
            "holdout_size": int(len(y_holdout)),
            **_compute_phi_metrics(
                y_true_phi=y_holdout,
                y_pred_phi=predictions,
                boundary_threshold=boundary_threshold,
            ),
            **_compute_normalized_phi_metrics(y_holdout, predictions),
            "constraint_sign_accuracy": _constraint_sign_accuracy(
                normalized_holdout,
                prediction.mu,
            ),
        }
        rows.append(row)
        last_model = model
        last_predictions = predictions

    if last_model is None or last_predictions is None:
        raise RuntimeError("No GPR convergence models were fit")

    metrics_table = pd.DataFrame(rows)
    metrics_table_path = output_path / "gpr_convergence_metrics.csv"
    metrics_table.to_csv(metrics_table_path, index=False)

    final_metrics = metrics_table.iloc[-1].to_dict()
    converged = _meets_convergence_targets(
        final_metrics,
        target_r2=target_r2,
        target_normalized_mae=target_normalized_mae,
        target_normalized_rmse=target_normalized_rmse,
    )
    model_path = dump_constraint_gpr_bundle(
        output_path / "surrogate_constraints_gpr.joblib",
        last_model,
    )
    plot_paths = _write_convergence_plots(
        output_path=output_path,
        metrics_table=metrics_table,
        y_holdout=y_holdout,
        y_pred=last_predictions,
    )

    metrics_payload = _json_ready(
        {
            "converged": converged,
            "targets": {
                "target_r2": target_r2,
                "target_normalized_mae": target_normalized_mae,
                "target_normalized_rmse": target_normalized_rmse,
            },
            "final_metrics": final_metrics,
            "n_rows_total": int(len(evaluations)),
            "n_rows_used": int(len(frame)),
            "holdout_fraction": float(holdout_fraction),
            "holdout_size": int(len(y_holdout)),
            "holdout_source": holdout_source,
            "holdout_path": str(holdout_path) if holdout_path is not None else None,
            "holdout_id_columns": holdout_id_columns,
            "train_sizes": selected_train_sizes,
            "metrics_table": str(metrics_table_path),
            "model_path": str(model_path),
            "plot_paths": {name: str(path) for name, path in plot_paths.items()},
        }
    )
    metrics_path = output_path / "gpr_convergence_summary.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    return SurrogateConvergenceResult(
        metrics_path=metrics_path,
        metrics_table_path=metrics_table_path,
        model_path=model_path,
        plot_paths=plot_paths,
        metrics_table=metrics_table,
        converged=converged,
    )


def write_holdout_selection_from_file(
    path: str | Path,
    config: ViabilityConfig,
    output_path: str | Path,
    *,
    holdout_fraction: float = 0.2,
) -> HoldoutSelectionResult:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    evaluations = read_evaluations_table(path)
    prepared = _prepare_training_frame(evaluations, config)
    frame = _sort_for_convergence(prepared["frame"])
    _, holdout_idx = _fixed_holdout_indices(
        len(frame),
        holdout_fraction=holdout_fraction,
        random_seed=config.run.random_seed,
    )
    holdout = frame.iloc[holdout_idx].drop(columns=["_prepared_index"], errors="ignore")
    output_file = _write_dataframe_table(holdout, output_path)
    return HoldoutSelectionResult(
        holdout_path=output_file,
        n_rows_total=int(len(frame)),
        holdout_size=int(len(holdout)),
    )


def write_gpr_prediction_overlay_plot(
    run_dirs: list[str | Path],
    output_path: str | Path,
    *,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    alphas: list[float] | None = None,
    zorders: list[float] | None = None,
) -> GPRPredictionOverlayResult:
    if len(run_dirs) < 1:
        raise ValueError("At least one convergence run directory is required")

    case_labels = _resolve_style_values(
        values=labels,
        defaults=[Path(run_dir).name for run_dir in run_dirs],
        name="labels",
    )
    case_colors = _resolve_style_values(
        values=colors,
        defaults=[f"C{index}" for index in range(len(run_dirs))],
        name="colors",
    )
    case_alphas = [
        float(value)
        for value in _resolve_style_values(
            values=alphas,
            defaults=[0.75] * len(run_dirs),
            name="alphas",
        )
    ]
    case_zorders = [
        float(value)
        for value in _resolve_style_values(
            values=zorders,
            defaults=list(range(1, len(run_dirs) + 1)),
            name="zorders",
        )
    ]

    cases = [
        _load_prediction_scatter_data(Path(run_dir), label=case_labels[index])
        for index, run_dir in enumerate(run_dirs)
    ]
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _write_prediction_overlay_plot(
        output_path=output_file,
        cases=cases,
        colors=case_colors,
        alphas=case_alphas,
        zorders=case_zorders,
    )
    return GPRPredictionOverlayResult(
        plot_path=output_file,
        point_counts={case.label: int(len(case.true_phi)) for case in cases},
    )


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
    frame["_prepared_index"] = np.arange(len(frame), dtype=int)

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


def normalized_constraint_frame(
    frame: pd.DataFrame,
    constraint_columns: list[str],
    config: ViabilityConfig,
    *,
    verify_phi: bool = True,
) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for column in constraint_columns:
        name = _constraint_name_from_column(column)
        scale = config.constraint_scales.scale_for(name)
        if scale <= 0.0:
            raise ValueError(f"Constraint scale for {name!r} must be positive")
        data[name] = frame[column].to_numpy(dtype=float) / scale

    normalized = pd.DataFrame(data, index=frame.index)
    if verify_phi:
        _verify_phi_reconstruction(normalized, frame["phi"].to_numpy(dtype=float))
    return normalized


def fit_constraint_gpr_bundle(
    *,
    x: np.ndarray,
    normalized_constraints: pd.DataFrame,
    feature_names: list[str],
    config: ViabilityConfig,
    max_rows: int,
) -> dict[str, Any]:
    models_by_constraint = {}
    for constraint_name in normalized_constraints.columns:
        y = normalized_constraints[constraint_name].to_numpy(dtype=float)
        models_by_constraint[constraint_name] = _fit_gpr(
            x,
            y,
            random_seed=config.run.random_seed,
            max_rows=max_rows,
        )
    return {
        "models_by_constraint": models_by_constraint,
        "constraint_names": list(normalized_constraints.columns),
        "target": "normalized_constraints",
        "feature_names": feature_names,
        "input_space": "normalized_policy_variables",
        "policy_parameterization": config.policy.parameterization,
    }


def predict_constraint_surrogate(
    bundle: dict[str, Any],
    x_values: np.ndarray,
    *,
    conservative_sigma: float = 0.0,
) -> ConstraintSurrogatePrediction:
    _validate_constraint_bundle(bundle)
    constraint_names = list(bundle["constraint_names"])
    mu_columns: dict[str, np.ndarray] = {}
    sigma_columns: dict[str, np.ndarray] = {}
    for constraint_name in constraint_names:
        model = bundle["models_by_constraint"][constraint_name]
        mu, sigma = predict_with_uncertainty(model, x_values)
        mu_columns[constraint_name] = mu
        sigma_columns[constraint_name] = sigma

    mu_frame = pd.DataFrame(mu_columns)
    sigma_frame = pd.DataFrame(sigma_columns)
    mu_values = mu_frame.to_numpy(dtype=float)
    sigma_values = sigma_frame.to_numpy(dtype=float)
    active_indices = np.argmax(mu_values, axis=1)
    conservative_values = mu_values + float(conservative_sigma) * sigma_values
    return ConstraintSurrogatePrediction(
        mu=mu_frame,
        sigma=sigma_frame,
        predicted_phi=np.max(mu_values, axis=1),
        sigma_phi=np.max(sigma_values, axis=1),
        conservative_phi=np.max(conservative_values, axis=1),
        active_constraint=np.asarray([constraint_names[index] for index in active_indices]),
    )


def predict_with_uncertainty(model: Any, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "named_steps"):
        named_steps = model.named_steps
        if "scaler" in named_steps and "gpr" in named_steps:
            x_scaled = named_steps["scaler"].transform(x_values)
            mu, sigma = named_steps["gpr"].predict(x_scaled, return_std=True)
            return np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    try:
        mu, sigma = model.predict(x_values, return_std=True)
    except TypeError as exc:
        raise TypeError("Expected surrogate model with predict(..., return_std=True)") from exc
    return np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)


def dump_constraint_gpr_bundle(path: Path, bundle: dict[str, Any]) -> Path:
    _validate_constraint_bundle(bundle)
    joblib.dump(bundle, path)
    return path


def _constraint_name_from_column(column: str) -> str:
    if not column.startswith("constraint_"):
        raise ValueError(f"Expected constraint column to start with 'constraint_': {column!r}")
    return column.removeprefix("constraint_")


def _verify_phi_reconstruction(normalized_constraints: pd.DataFrame, phi: np.ndarray) -> None:
    reconstructed = normalized_constraints.max(axis=1).to_numpy(dtype=float)
    if not np.allclose(reconstructed, phi, rtol=1e-8, atol=1e-8):
        max_error = float(np.max(np.abs(reconstructed - phi)))
        raise ValueError(
            "Stored phi does not match max normalized constraint values; "
            f"maximum absolute mismatch is {max_error}"
        )


def _validate_constraint_bundle(bundle: dict[str, Any]) -> None:
    if not isinstance(bundle, dict):
        raise ValueError("Constraint surrogate bundle must be a dictionary")
    required = ("models_by_constraint", "constraint_names", "target", "feature_names")
    for key in required:
        if key not in bundle:
            raise ValueError(f"Constraint surrogate bundle is missing required key {key!r}")
    if bundle["target"] != "normalized_constraints":
        raise ValueError("Constraint surrogate bundle target must be 'normalized_constraints'")
    if not bundle["constraint_names"]:
        raise ValueError("Constraint surrogate bundle must include at least one constraint")
    for constraint_name in bundle["constraint_names"]:
        if constraint_name not in bundle["models_by_constraint"]:
            raise ValueError(f"Constraint surrogate bundle is missing model for {constraint_name!r}")


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


def _fixed_holdout_indices(
    n_rows: int,
    *,
    holdout_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    n_holdout = max(2, int(round(n_rows * holdout_fraction)))
    n_holdout = min(n_holdout, n_rows - 2)
    holdout_idx = np.sort(indices[:n_holdout])
    train_pool_idx = np.sort(indices[n_holdout:])
    return train_pool_idx, holdout_idx


def _resolve_train_sizes(train_sizes: list[int] | None, *, max_train_size: int) -> list[int]:
    if max_train_size < 2:
        raise ValueError("Need at least two training rows after holdout split")
    if train_sizes:
        sizes = sorted({int(size) for size in train_sizes if int(size) > 1})
        sizes = [size for size in sizes if size <= max_train_size]
    else:
        sizes = []
        size = 16
        while size <= max_train_size:
            sizes.append(size)
            size *= 2
    if max_train_size not in sizes:
        sizes.append(max_train_size)
    if not sizes:
        raise ValueError("No valid train sizes are available")
    return sizes


def _sort_for_convergence(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ["sample_index", "design_id"] if column in frame.columns]
    if not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns).reset_index(drop=True)


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
    metrics: dict[str, Any] = _compute_phi_metrics(
        y_true_phi=y_true_phi,
        y_pred_phi=y_pred_phi,
        boundary_threshold=boundary_threshold,
    )

    sign_scores = []
    for column, predicted in y_pred_constraints.items():
        truth = y_true_constraints[column].to_numpy(dtype=float)
        sign_scores.append(np.mean((truth <= 0.0) == (predicted <= 0.0)))
    metrics["constraint_sign_accuracy"] = (
        float(np.mean(sign_scores)) if sign_scores else None
    )
    return metrics


def _constraint_sign_accuracy(
    true_normalized_constraints: pd.DataFrame,
    predicted_normalized_constraints: pd.DataFrame,
) -> float:
    scores = []
    for constraint_name in true_normalized_constraints.columns:
        truth = true_normalized_constraints[constraint_name].to_numpy(dtype=float)
        predicted = predicted_normalized_constraints[constraint_name].to_numpy(dtype=float)
        scores.append(np.mean((truth <= 0.0) == (predicted <= 0.0)))
    if not scores:
        raise ValueError("At least one normalized constraint is required for sign accuracy")
    return float(np.mean(scores))


def _compute_phi_metrics(
    *,
    y_true_phi: np.ndarray,
    y_pred_phi: np.ndarray,
    boundary_threshold: float,
) -> dict[str, Any]:
    true_feasible = y_true_phi <= 0.0
    pred_feasible = y_pred_phi <= 0.0
    boundary_mask = np.abs(y_true_phi) <= boundary_threshold

    metrics: dict[str, Any] = {
        "MAE_phi": float(mean_absolute_error(y_true_phi, y_pred_phi)),
        "MSE_phi": float(mean_squared_error(y_true_phi, y_pred_phi)),
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
    return metrics


def _compute_normalized_phi_metrics(y_true_phi: np.ndarray, y_pred_phi: np.ndarray) -> dict[str, Any]:
    true_norm, pred_norm = _normalize_prediction_pair(y_true_phi, y_pred_phi)
    return {
        "MAE_phi_normalized": float(mean_absolute_error(true_norm, pred_norm)),
        "MSE_phi_normalized": float(mean_squared_error(true_norm, pred_norm)),
        "RMSE_phi_normalized": float(mean_squared_error(true_norm, pred_norm) ** 0.5),
        "R2_phi_normalized": _safe_r2(true_norm, pred_norm),
    }


def _normalize_prediction_pair(
    y_true_phi: np.ndarray,
    y_pred_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ymin = float(np.min(y_true_phi))
    ymax = float(np.max(y_true_phi))
    span = ymax - ymin
    if span <= 0.0:
        return np.zeros_like(y_true_phi), np.zeros_like(y_pred_phi)
    return (y_true_phi - ymin) / span, (y_pred_phi - ymin) / span


def _meets_convergence_targets(
    metrics: dict[str, Any],
    *,
    target_r2: float | None,
    target_normalized_mae: float | None,
    target_normalized_rmse: float | None,
) -> bool:
    checks = []
    if target_r2 is not None:
        checks.append(metrics["R2_phi"] is not None and metrics["R2_phi"] >= target_r2)
    if target_normalized_mae is not None:
        checks.append(metrics["MAE_phi_normalized"] <= target_normalized_mae)
    if target_normalized_rmse is not None:
        checks.append(metrics["RMSE_phi_normalized"] <= target_normalized_rmse)
    return bool(checks and all(checks))


def _write_convergence_plots(
    *,
    output_path: Path,
    metrics_table: pd.DataFrame,
    y_holdout: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Path]:
    cache_dir = output_path / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: dict[str, Path] = {}
    true_norm, pred_norm = _normalize_prediction_pair(y_holdout, y_pred)

    scatter_path = output_path / "gpr_predict_vs_truth_normalized.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_norm, pred_norm, s=20, alpha=0.75, edgecolor="none")
    lo = float(min(np.min(true_norm), np.min(pred_norm), 0.0))
    hi = float(max(np.max(true_norm), np.max(pred_norm), 1.0))
    pad = max((hi - lo) * 0.05, 0.02)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Truth phi, normalized")
    ax.set_ylabel("Predicted phi, normalized")
    ax.set_title("GPR Holdout Prediction")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)
    plot_paths["predict_vs_truth_normalized"] = scatter_path

    metrics_path = output_path / "gpr_convergence_metrics.png"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(
        metrics_table["train_size"],
        _positive_for_log(metrics_table["MAE_phi_normalized"].to_numpy(dtype=float)),
        marker="o",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Normalized MAE")
    axes[1].plot(
        metrics_table["train_size"],
        _positive_for_log(metrics_table["MSE_phi_normalized"].to_numpy(dtype=float)),
        marker="o",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Normalized MSE")
    axes[2].plot(metrics_table["train_size"], metrics_table["R2_phi"], marker="o")
    axes[2].set_ylabel("R2")
    for ax in axes:
        ax.set_xlabel("Training rows")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(metrics_path, dpi=180)
    plt.close(fig)
    plot_paths["convergence_metrics"] = metrics_path
    return plot_paths


def _load_prediction_scatter_data(run_dir: Path, *, label: str) -> _PredictionScatterData:
    config_path = run_dir / "config_resolved.yaml"
    evaluations_path = run_dir / "evaluations.parquet"
    summary_path = run_dir / "gpr_convergence_summary.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run directory is missing config_resolved.yaml: {run_dir}")
    config = load_config(config_path)
    summary = _read_convergence_summary(summary_path)
    if "holdout_fraction" not in summary:
        raise ValueError(f"Convergence summary is missing required key 'holdout_fraction': {summary_path}")
    holdout_fraction = float(summary["holdout_fraction"])
    model_path = _resolve_convergence_model_path(run_dir, summary)

    holdout_path = summary.get("holdout_path")
    if holdout_path:
        resolved_holdout_path = _resolve_existing_path(run_dir, str(holdout_path))
        holdout_evaluations = read_evaluations_table(resolved_holdout_path)
        prepared_holdout = _prepare_training_frame(holdout_evaluations, config)
        holdout_frame = _sort_for_convergence(prepared_holdout["frame"])
        x_holdout = prepared_holdout["x"][
            holdout_frame["_prepared_index"].to_numpy(dtype=int)
        ]
        y_holdout = holdout_frame["phi"].to_numpy(dtype=float)
    else:
        if not evaluations_path.exists() and not evaluations_path.with_suffix(".csv").exists():
            raise FileNotFoundError(f"Run directory is missing evaluations parquet or CSV: {run_dir}")
        evaluations = read_evaluations_table(evaluations_path)
        prepared = _prepare_training_frame(evaluations, config)
        frame = _sort_for_convergence(prepared["frame"])
        x = prepared["x"][frame["_prepared_index"].to_numpy(dtype=int)]
        y_phi = frame["phi"].to_numpy(dtype=float)
        _, holdout_idx = _fixed_holdout_indices(
            len(frame),
            holdout_fraction=holdout_fraction,
            random_seed=config.run.random_seed,
        )
        x_holdout = x[holdout_idx]
        y_holdout = y_phi[holdout_idx]
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "models_by_constraint" in bundle:
        predictions = predict_constraint_surrogate(bundle, x_holdout).predicted_phi
    else:
        model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
        predictions = np.asarray(model.predict(x_holdout), dtype=float)
    return _PredictionScatterData(
        label=label,
        true_phi=y_holdout,
        predicted_phi=predictions,
    )


def _read_convergence_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_convergence_model_path(run_dir: Path, summary: dict[str, Any]) -> Path:
    model_value = summary.get("model_path")
    if model_value:
        model_path = Path(str(model_value))
        if model_path.exists():
            return model_path
        local_model_path = run_dir / model_path.name
        if local_model_path.exists():
            return local_model_path
    constraints_path = run_dir / "surrogate_constraints_gpr.joblib"
    if constraints_path.exists():
        return constraints_path
    phi_path = run_dir / "surrogate_phi_gpr_converged.joblib"
    if phi_path.exists():
        return phi_path
    raise FileNotFoundError(f"Run directory is missing a converged GPR model: {run_dir}")


def _resolve_existing_path(run_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    local_path = run_dir / path
    if local_path.exists():
        return local_path
    sibling_path = run_dir / path.name
    if sibling_path.exists():
        return sibling_path
    raise FileNotFoundError(f"Referenced path does not exist: {value}")


def _common_holdout_id_columns(frame: pd.DataFrame, holdout_frame: pd.DataFrame) -> list[str]:
    for column in ["design_id", "sample_index"]:
        if column in frame.columns and column in holdout_frame.columns:
            return [column]
    return []


def _exclude_holdout_rows(
    *,
    frame: pd.DataFrame,
    x: np.ndarray,
    y_phi: np.ndarray,
    holdout_frame: pd.DataFrame,
    id_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not id_columns:
        return frame, x, y_phi
    holdout_keys = {
        tuple(row[column] for column in id_columns)
        for _, row in holdout_frame[id_columns].iterrows()
    }
    keep_mask = np.array(
        [
            tuple(row[column] for column in id_columns) not in holdout_keys
            for _, row in frame[id_columns].iterrows()
        ],
        dtype=bool,
    )
    return (
        frame.loc[keep_mask].reset_index(drop=True),
        x[keep_mask],
        y_phi[keep_mask],
    )


def _write_dataframe_table(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    return output_path


def _write_prediction_overlay_plot(
    *,
    output_path: Path,
    cases: list[_PredictionScatterData],
    colors: list[str],
    alphas: list[float],
    zorders: list[float],
) -> None:
    cache_dir = output_path.parent / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_values = np.concatenate([case.true_phi for case in cases])
    ymin = float(np.min(true_values))
    ymax = float(np.max(true_values))
    span = ymax - ymin
    if span <= 0.0:
        span = 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    lo = 0.0
    hi = 1.0
    for case, color, alpha, zorder in zip(cases, colors, alphas, zorders, strict=True):
        true_norm = (case.true_phi - ymin) / span
        pred_norm = (case.predicted_phi - ymin) / span
        lo = float(min(lo, np.min(true_norm), np.min(pred_norm)))
        hi = float(max(hi, np.max(true_norm), np.max(pred_norm)))
        ax.scatter(
            true_norm,
            pred_norm,
            s=22,
            alpha=alpha,
            color=color,
            edgecolor="none",
            label=f"{case.label} (n={len(case.true_phi)})",
            zorder=zorder,
        )

    pad = max((hi - lo) * 0.05, 0.02)
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="0.25",
        linewidth=1,
        zorder=max(zorders) + 1,
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Truth phi, normalized")
    ax.set_ylabel("Predicted phi, normalized")
    ax.set_title("GPR Holdout Prediction Overlay")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _resolve_style_values(
    *,
    values: list[Any] | None,
    defaults: list[Any],
    name: str,
) -> list[Any]:
    if values is None:
        return defaults
    if len(values) != len(defaults):
        raise ValueError(f"{name} must have exactly {len(defaults)} values")
    return values


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0.0)]
    floor = float(np.min(positive) * 0.5) if len(positive) else 1e-12
    return np.where(values > 0.0, values, floor)


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
