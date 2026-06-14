from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from src.viability.config import ActiveLearningConfig, ViabilityConfig
from src.viability.design_space import DesignSpace
from src.viability.doe import generate_doe
from src.viability.evaluator import evaluate_designs_parallel
from src.viability.io import write_config_resolved, write_table
from src.viability.surrogate import (
    _compute_normalized_phi_metrics,
    _compute_phi_metrics,
    _constraint_sign_accuracy,
    dump_constraint_gpr_bundle,
    fit_constraint_gpr_bundle,
    _json_ready,
    _positive_for_log,
    _prepare_training_frame,
    normalized_constraint_frame,
    predict_constraint_surrogate,
    predict_with_uncertainty,
    read_evaluations_table,
)

_STATE_KEYS = (
    "completed_iteration",
    "next_candidate_start_index",
    "config_hash",
    "latest_training_path",
    "latest_model_path",
)


class EvaluateBatch(Protocol):
    def __call__(
        self,
        designs: pd.DataFrame,
        config: ViabilityConfig,
        *,
        workers: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 50,
    ) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class ActiveLearningResult:
    output_dir: Path
    metrics_path: Path
    state_path: Path
    latest_training_path: Path
    latest_model_path: Path
    plot_paths: dict[str, Path]
    metrics_table: pd.DataFrame


@dataclass(frozen=True)
class SurrogateSnapshot:
    model: Any
    model_path: Path
    training_path: Path
    metrics: dict[str, Any]


def run_active_learning_from_files(
    *,
    evaluations_path: str | Path,
    holdout_path: str | Path,
    config: ViabilityConfig,
    output_dir: str | Path,
    resume: bool = False,
    boundary_threshold: float = 0.1,
    evaluator: EvaluateBatch = evaluate_designs_parallel,
) -> ActiveLearningResult:
    evaluations = read_evaluations_table(evaluations_path)
    holdout = read_evaluations_table(holdout_path)
    return run_active_learning(
        evaluations=evaluations,
        holdout=holdout,
        config=config,
        output_dir=output_dir,
        resume=resume,
        boundary_threshold=boundary_threshold,
        evaluator=evaluator,
    )


def run_active_learning(
    *,
    evaluations: pd.DataFrame,
    holdout: pd.DataFrame,
    config: ViabilityConfig,
    output_dir: str | Path,
    resume: bool = False,
    boundary_threshold: float = 0.1,
    evaluator: EvaluateBatch = evaluate_designs_parallel,
) -> ActiveLearningResult:
    active_config = require_active_learning_config(config)
    output_path = Path(output_dir)
    state_path = output_path / "state.json"
    config_hash = _config_hash(config)

    if state_path.exists():
        if not resume:
            raise FileExistsError(
                f"Active-learning state already exists at {state_path}; rerun with --resume to continue"
            )
        state = _read_state(state_path)
        _validate_state(state, config_hash)
        completed_iteration = int(state["completed_iteration"])
        next_candidate_start_index = int(state["next_candidate_start_index"])
        training = read_evaluations_table(Path(str(state["latest_training_path"])))
        latest_model_path = Path(str(state["latest_model_path"]))
        if not latest_model_path.exists():
            raise FileNotFoundError(f"Latest active-learning model is missing: {latest_model_path}")
        latest_model = _load_surrogate_bundle(latest_model_path)
        metrics_path = output_path / "active_learning_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Active-learning metrics are missing: {metrics_path}")
        metrics_table = pd.read_csv(metrics_path)
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume because active-learning state is missing: {state_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        write_config_resolved(config, output_path)
        training = exclude_holdout_rows(evaluations, holdout, config)
        if training.empty:
            raise ValueError("No training rows remain after fixed-holdout exclusion")
        baseline_dir = output_path / "baseline"
        baseline = fit_surrogate_snapshot(
            training=training,
            holdout=holdout,
            config=config,
            output_dir=baseline_dir,
            boundary_threshold=boundary_threshold,
            iteration=0,
            selected_rows=0,
            candidate_start_index=None,
        )
        completed_iteration = 0
        next_candidate_start_index = active_config.candidate_start_index
        latest_model = baseline.model
        metrics_table = pd.DataFrame([baseline.metrics])
        _write_metrics_table(metrics_table, output_path / "active_learning_metrics.csv")
        _write_state(
            state_path,
            completed_iteration=completed_iteration,
            next_candidate_start_index=next_candidate_start_index,
            config_hash=config_hash,
            latest_training_path=baseline.training_path,
            latest_model_path=baseline.model_path,
        )

    while completed_iteration < active_config.iterations:
        iteration = completed_iteration + 1
        iteration_dir = output_path / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        candidates = generate_candidate_pool(
            config=config,
            active_config=active_config,
            start_index=next_candidate_start_index,
        )
        candidates = remove_existing_designs(candidates, [training, holdout], config)
        scored = score_candidates(latest_model, candidates, config)
        top_scored = sort_scored_candidates(scored, active_config).head(active_config.candidate_report_rows)
        top_scored.to_csv(iteration_dir / "scored_candidates_top.csv", index=False)

        selected = select_candidate_batch(
            scored,
            config,
            active_config,
        )
        selected.insert(0, "selection_iteration", iteration)
        selected.insert(1, "selection_rank", np.arange(1, len(selected) + 1, dtype=int))
        selected.to_csv(iteration_dir / "selected_candidates.csv", index=False)

        selected_evaluations = evaluator(
            selected,
            config,
            workers=config.run.workers,
            checkpoint_dir=iteration_dir / "checkpoints",
            checkpoint_every=active_config.batch_size,
        )
        selected_evaluations_path = write_table(
            selected_evaluations,
            iteration_dir / "selected_evaluations.parquet",
        )
        training = pd.concat([training, selected_evaluations], ignore_index=True)
        snapshot = fit_surrogate_snapshot(
            training=training,
            holdout=holdout,
            config=config,
            output_dir=iteration_dir,
            boundary_threshold=boundary_threshold,
            iteration=iteration,
            selected_rows=len(selected),
            candidate_start_index=next_candidate_start_index,
        )
        latest_model = snapshot.model
        metrics_table = pd.concat(
            [metrics_table, pd.DataFrame([snapshot.metrics])],
            ignore_index=True,
        )
        _write_metrics_table(metrics_table, output_path / "active_learning_metrics.csv")

        completed_iteration = iteration
        next_candidate_start_index += active_config.candidate_pool_size
        _write_state(
            state_path,
            completed_iteration=completed_iteration,
            next_candidate_start_index=next_candidate_start_index,
            config_hash=config_hash,
            latest_training_path=snapshot.training_path,
            latest_model_path=snapshot.model_path,
        )
        _write_iteration_manifest(
            iteration_dir / "iteration_manifest.json",
            iteration=iteration,
            candidate_start_index=int(snapshot.metrics["candidate_start_index"]),
            candidate_pool_size=active_config.candidate_pool_size,
            selected_evaluations_path=selected_evaluations_path,
            training_path=snapshot.training_path,
            model_path=snapshot.model_path,
        )

    plot_paths = write_active_learning_plots(output_path, metrics_table)
    latest_state = _read_state(state_path)
    return ActiveLearningResult(
        output_dir=output_path,
        metrics_path=output_path / "active_learning_metrics.csv",
        state_path=state_path,
        latest_training_path=Path(str(latest_state["latest_training_path"])),
        latest_model_path=Path(str(latest_state["latest_model_path"])),
        plot_paths=plot_paths,
        metrics_table=metrics_table,
    )


def require_active_learning_config(config: ViabilityConfig) -> ActiveLearningConfig:
    if config.active_learning is None:
        raise ValueError("Config must include an active_learning section for active-learn")
    return config.active_learning


def generate_candidate_pool(
    *,
    config: ViabilityConfig,
    active_config: ActiveLearningConfig,
    start_index: int,
) -> pd.DataFrame:
    return generate_doe(
        config,
        n=active_config.candidate_pool_size,
        method=active_config.candidate_method,
        start_index=start_index,
        include_corners=False,
        include_baselines=False,
    )


def exclude_holdout_rows(
    evaluations: pd.DataFrame,
    holdout: pd.DataFrame,
    config: ViabilityConfig,
) -> pd.DataFrame:
    holdout_keys = set(policy_keys(holdout, config))
    keep_mask = [
        key not in holdout_keys
        for key in policy_keys(evaluations, config)
    ]
    return evaluations.loc[keep_mask].reset_index(drop=True)


def remove_existing_designs(
    candidates: pd.DataFrame,
    existing_frames: list[pd.DataFrame],
    config: ViabilityConfig,
) -> pd.DataFrame:
    existing_keys: set[tuple[Any, ...]] = set()
    for frame in existing_frames:
        existing_keys.update(policy_keys(frame, config))
    keep_mask = [
        key not in existing_keys
        for key in policy_keys(candidates, config)
    ]
    return candidates.loc[keep_mask].reset_index(drop=True)


def policy_keys(frame: pd.DataFrame, config: ViabilityConfig) -> list[tuple[Any, ...]]:
    space = DesignSpace(config.policy)
    columns = []
    for name in space.variable_names:
        applied_column = f"applied_{name}"
        if applied_column in frame.columns:
            columns.append(applied_column)
        elif name in frame.columns:
            columns.append(name)
        else:
            raise ValueError(f"Design table is missing policy column {name!r} and {applied_column!r}")
    keys = []
    for _, row in frame.loc[:, columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                raise ValueError(f"Design table has null policy value in column {column!r}")
            if hasattr(value, "item"):
                value = value.item()
            values.append(value)
        keys.append(tuple(values))
    return keys


def score_candidates(
    model: Any,
    candidates: pd.DataFrame,
    config: ViabilityConfig,
) -> pd.DataFrame:
    if candidates.empty:
        raise ValueError("Candidate pool is empty before scoring")
    x_candidates = design_matrix(candidates, config)
    scored = candidates.reset_index(drop=True).copy()
    if isinstance(model, dict) and "models_by_constraint" in model:
        prediction = predict_constraint_surrogate(model, x_candidates)
        mu_phi = prediction.predicted_phi
        sigma_phi = prediction.sigma_phi
        scored["predicted_active_constraint"] = prediction.active_constraint
        scored["conservative_phi"] = prediction.conservative_phi
        for constraint_name in prediction.mu.columns:
            scored[f"mu_constraint_{constraint_name}"] = prediction.mu[constraint_name].to_numpy(dtype=float)
            scored[f"sigma_constraint_{constraint_name}"] = prediction.sigma[constraint_name].to_numpy(dtype=float)
    else:
        mu_phi, sigma_phi = predict_with_uncertainty(model, x_candidates)
        scored["predicted_active_constraint"] = None
        scored["conservative_phi"] = mu_phi
    scored["candidate_pool_index"] = np.arange(len(scored), dtype=int)
    scored["mu_phi"] = mu_phi
    scored["sigma_phi"] = sigma_phi
    scored["abs_mu_phi"] = np.abs(mu_phi)
    scored["acquisition_score"] = sigma_phi
    scored.loc[:, "boundary_score"] = -scored["abs_mu_phi"].to_numpy(dtype=float)
    return scored


def sort_scored_candidates(
    scored: pd.DataFrame,
    active_config: ActiveLearningConfig | None = None,
) -> pd.DataFrame:
    if active_config is not None and active_config.acquisition == "boundary_stratified_uncertainty":
        return sort_boundary_candidates(scored)
    return scored.sort_values(
        ["acquisition_score", "abs_mu_phi", "design_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def sort_boundary_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.sort_values(
        ["abs_mu_phi", "acquisition_score", "design_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def select_candidate_batch(
    scored: pd.DataFrame,
    config: ViabilityConfig,
    active_config: ActiveLearningConfig,
) -> pd.DataFrame:
    if active_config.acquisition == "uncertainty":
        ordered = sort_scored_candidates(scored, active_config)
        return _select_diverse_candidates(
            ordered,
            config,
            batch_size=active_config.batch_size,
            min_normalized_distance=active_config.min_normalized_distance,
            selection_source="uncertainty",
        )
    if active_config.acquisition == "boundary_stratified_uncertainty":
        boundary_count = int(round(active_config.batch_size * active_config.boundary_batch_fraction))
        boundary_count = min(boundary_count, active_config.batch_size)
        uncertainty_count = active_config.batch_size - boundary_count
        selected = _select_diverse_candidates(
            sort_boundary_candidates(scored),
            config,
            batch_size=boundary_count,
            min_normalized_distance=active_config.min_normalized_distance,
            selection_source="boundary",
        )
        selected_keys = set(policy_keys(selected, config))
        uncertainty_candidates = sort_scored_candidates(scored)
        keep_uncertainty = [
            key not in selected_keys
            for key in policy_keys(uncertainty_candidates, config)
        ]
        uncertainty_selected = _select_diverse_candidates(
            uncertainty_candidates.loc[keep_uncertainty].reset_index(drop=True),
            config,
            batch_size=uncertainty_count,
            min_normalized_distance=active_config.min_normalized_distance,
            selection_source="uncertainty",
            selected_vectors=[design_vector(row, config) for _, row in selected.iterrows()],
        )
        return pd.concat([selected, uncertainty_selected], ignore_index=True)
    raise ValueError(f"Unsupported active-learning acquisition {active_config.acquisition!r}")


def _select_diverse_candidates(
    ordered: pd.DataFrame,
    config: ViabilityConfig,
    *,
    batch_size: int,
    min_normalized_distance: float,
    selection_source: str,
    selected_vectors: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    if batch_size == 0:
        selected = ordered.iloc[[]].copy()
        selected["selection_source"] = []
        return selected
    selected_indices = []
    vectors = [] if selected_vectors is None else list(selected_vectors)
    for row_index, row in ordered.iterrows():
        candidate_vector = design_vector(row, config)
        if all(
            float(np.linalg.norm(candidate_vector - selected_vector)) >= min_normalized_distance
            for selected_vector in vectors
        ):
            selected_indices.append(row_index)
            vectors.append(candidate_vector)
        if len(selected_indices) == batch_size:
            break
    if len(selected_indices) != batch_size:
        raise RuntimeError(
            "Candidate pool could not fill the requested active-learning batch after "
            "dedupe and diversity filtering; increase active_learning.candidate_pool_size "
            "or lower active_learning.min_normalized_distance"
        )
    selected = ordered.iloc[selected_indices].reset_index(drop=True).copy()
    selected["selection_source"] = selection_source
    return selected


def fit_surrogate_snapshot(
    *,
    training: pd.DataFrame,
    holdout: pd.DataFrame,
    config: ViabilityConfig,
    output_dir: str | Path,
    boundary_threshold: float,
    iteration: int,
    selected_rows: int,
    candidate_start_index: int | None,
) -> SurrogateSnapshot:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prepared_train = _prepare_training_frame(training, config)
    prepared_holdout = _prepare_training_frame(holdout, config)
    x_train = prepared_train["x"]
    y_train_phi = prepared_train["frame"]["phi"].to_numpy(dtype=float)
    normalized_train = normalized_constraint_frame(
        prepared_train["frame"],
        prepared_train["constraint_columns"],
        config,
    )
    x_holdout = prepared_holdout["x"]
    y_holdout = prepared_holdout["frame"]["phi"].to_numpy(dtype=float)
    normalized_holdout = normalized_constraint_frame(
        prepared_holdout["frame"],
        prepared_train["constraint_columns"],
        config,
    )

    if len(y_train_phi) < 2:
        raise ValueError("At least two successful training rows are required for active learning")
    if len(y_holdout) < 2:
        raise ValueError("At least two holdout rows are required for active learning")

    model = fit_constraint_gpr_bundle(
        x=x_train,
        normalized_constraints=normalized_train,
        feature_names=list(prepared_train["feature_names"]),
        config=config,
        max_rows=len(y_train_phi),
    )
    prediction = predict_constraint_surrogate(model, x_holdout)
    predictions = prediction.predicted_phi
    metrics = {
        "iteration": int(iteration),
        "train_rows": int(len(y_train_phi)),
        "holdout_rows": int(len(y_holdout)),
        "selected_rows": int(selected_rows),
        "candidate_start_index": candidate_start_index,
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
    metrics = _json_ready(metrics)

    training_path = write_table(training, output_path / "training_evaluations.parquet")
    model_path = output_path / "surrogate_constraints_gpr.joblib"
    dump_constraint_gpr_bundle(
        model_path,
        model,
    )
    _write_holdout_predictions(
        output_path / "holdout_predictions.csv",
        prepared_holdout["frame"],
        predictions,
    )
    (output_path / "holdout_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return SurrogateSnapshot(
        model=model,
        model_path=model_path.resolve(),
        training_path=training_path.resolve(),
        metrics=metrics,
    )


def design_matrix(frame: pd.DataFrame, config: ViabilityConfig) -> np.ndarray:
    return np.vstack([design_vector(row, config) for _, row in frame.iterrows()])


def design_vector(row: pd.Series, config: ViabilityConfig) -> np.ndarray:
    space = DesignSpace(config.policy)
    values: dict[str, Any] = {}
    for name in space.variable_names:
        if name not in row:
            raise ValueError(f"Design row is missing policy column {name!r}")
        values[name] = row[name]
    return space.normalize(values)


def write_active_learning_plots(output_dir: str | Path, metrics_table: pd.DataFrame) -> dict[str, Path]:
    output_path = Path(output_dir)
    cache_dir = output_path / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: dict[str, Path] = {}
    metrics_plot = output_path / "active_learning_holdout_metrics.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metric_specs = [
        ("MAE_phi", "Holdout MAE", True),
        ("MSE_phi", "Holdout MSE", True),
        ("RMSE_phi", "Holdout RMSE", True),
        ("R2_phi", "Holdout R2", False),
    ]
    iterations = metrics_table["iteration"].to_numpy(dtype=int)
    for ax, (column, title, log_y) in zip(axes.ravel(), metric_specs, strict=True):
        values = metrics_table[column].to_numpy(dtype=float)
        ax.plot(iterations, _positive_for_log(values) if log_y else values, marker="o")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Active-learning iteration")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(metrics_plot, dpi=180)
    plt.close(fig)
    plot_paths["holdout_metrics"] = metrics_plot

    overlay_plot = output_path / "active_learning_predict_vs_truth_normalized.png"
    _write_prediction_overlay(output_path, overlay_plot, plt)
    plot_paths["predict_vs_truth_normalized"] = overlay_plot

    selected_plot = output_path / "active_learning_selected_mu_sigma.png"
    selected_frames = _read_selected_candidate_frames(output_path)
    if selected_frames:
        selected = pd.concat(selected_frames, ignore_index=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(
            selected["mu_phi"],
            selected["sigma_phi"],
            c=selected["selection_iteration"],
            cmap="viridis",
            s=28,
            alpha=0.8,
            edgecolor="none",
        )
        ax.axvline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.set_xlabel("Predicted phi")
        ax.set_ylabel("Predictive sigma")
        ax.set_title("Selected Active-Learning Candidates")
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="Iteration")
        fig.tight_layout()
        fig.savefig(selected_plot, dpi=180)
        plt.close(fig)
        plot_paths["selected_mu_sigma"] = selected_plot
    return {name: path.resolve() for name, path in plot_paths.items()}


def _write_prediction_overlay(output_path: Path, overlay_plot: Path, plt: Any) -> None:
    prediction_frames = _read_prediction_frames(output_path)
    if not prediction_frames:
        raise FileNotFoundError(f"No holdout_predictions.csv files found under {output_path}")
    true_values = np.concatenate([frame["true_phi"].to_numpy(dtype=float) for _, frame in prediction_frames])
    ymin = float(np.min(true_values))
    ymax = float(np.max(true_values))
    span = ymax - ymin
    if span <= 0.0:
        span = 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["black", "firebrick", "steelblue", "darkgreen", "darkorange", "purple"]
    for index, (label, frame) in enumerate(prediction_frames):
        true_norm = (frame["true_phi"].to_numpy(dtype=float) - ymin) / span
        pred_norm = (frame["predicted_phi"].to_numpy(dtype=float) - ymin) / span
        color = colors[index % len(colors)]
        alpha = 0.25 if index == 0 else 0.65
        zorder = 1 + index
        ax.scatter(true_norm, pred_norm, s=18, color=color, alpha=alpha, edgecolor="none", label=label, zorder=zorder)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1, zorder=20)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Truth phi, normalized")
    ax.set_ylabel("Predicted phi, normalized")
    ax.set_title("Fixed-Holdout Prediction Across Active Learning")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(overlay_plot, dpi=180)
    plt.close(fig)


def _read_prediction_frames(output_path: Path) -> list[tuple[str, pd.DataFrame]]:
    frames = []
    baseline_path = output_path / "baseline" / "holdout_predictions.csv"
    if baseline_path.exists():
        frames.append(("baseline", pd.read_csv(baseline_path)))
    for iteration_dir in sorted(output_path.glob("iteration_*")):
        predictions_path = iteration_dir / "holdout_predictions.csv"
        if predictions_path.exists():
            frames.append((iteration_dir.name, pd.read_csv(predictions_path)))
    return frames


def _read_selected_candidate_frames(output_path: Path) -> list[pd.DataFrame]:
    frames = []
    for iteration_dir in sorted(output_path.glob("iteration_*")):
        selected_path = iteration_dir / "selected_candidates.csv"
        if selected_path.exists():
            frames.append(pd.read_csv(selected_path))
    return frames


def _write_holdout_predictions(path: Path, holdout_frame: pd.DataFrame, predictions: np.ndarray) -> None:
    prediction_frame = holdout_frame.drop(columns=["_prepared_index"], errors="ignore").copy()
    prediction_frame["true_phi"] = holdout_frame["phi"].to_numpy(dtype=float)
    prediction_frame["predicted_phi"] = predictions
    prediction_frame.to_csv(path, index=False)


def _load_surrogate_bundle(path: Path) -> Any:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected model bundle dict at {path}")
    if "models_by_constraint" in bundle:
        return bundle
    if "model" not in bundle:
        raise ValueError(f"Model bundle is missing required key 'model': {path}")
    return bundle["model"]


def _write_metrics_table(metrics_table: pd.DataFrame, path: Path) -> None:
    metrics_table.to_csv(path, index=False)


def _config_hash(config: ViabilityConfig) -> str:
    data = config.to_dict()
    if "active_learning" in data:
        active_data = dict(data["active_learning"])
        active_data["iterations"] = "<resume-control>"
        data["active_learning"] = active_data
    payload = json.dumps(data, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"Active-learning state must be a JSON object: {path}")
    for key in _STATE_KEYS:
        if key not in state:
            raise ValueError(f"Active-learning state is missing required key {key!r}: {path}")
    return state


def _validate_state(state: dict[str, Any], config_hash: str) -> None:
    if state["config_hash"] != config_hash:
        raise ValueError("Active-learning state config_hash does not match the current config")


def _write_state(
    path: Path,
    *,
    completed_iteration: int,
    next_candidate_start_index: int,
    config_hash: str,
    latest_training_path: Path,
    latest_model_path: Path,
) -> None:
    payload = {
        "completed_iteration": int(completed_iteration),
        "next_candidate_start_index": int(next_candidate_start_index),
        "config_hash": config_hash,
        "latest_training_path": str(latest_training_path.resolve()),
        "latest_model_path": str(latest_model_path.resolve()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_iteration_manifest(
    path: Path,
    *,
    iteration: int,
    candidate_start_index: int,
    candidate_pool_size: int,
    selected_evaluations_path: Path,
    training_path: Path,
    model_path: Path,
) -> None:
    payload = {
        "iteration": int(iteration),
        "candidate_start_index": int(candidate_start_index),
        "candidate_pool_size": int(candidate_pool_size),
        "selected_evaluations_path": str(selected_evaluations_path.resolve()),
        "training_path": str(training_path.resolve()),
        "model_path": str(model_path.resolve()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
