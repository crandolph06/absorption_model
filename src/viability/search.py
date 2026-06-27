from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Protocol, Sequence

import joblib
import numpy as np
import pandas as pd

from src.viability.config import SearchConfig, ViabilityConfig
from src.viability.design_space import DesignSpace
from src.viability.doe import generate_doe
from src.viability.evaluator import evaluate_designs_parallel
from src.viability.io import write_config_resolved, write_table
from src.viability.surrogate import predict_constraint_surrogate

_CANDIDATE_METADATA_COLUMNS = (
    "design_id",
    "candidate_id",
    "selection_rank",
    "selection_source",
    "predicted_phi",
    "predicted_sigma_phi",
    "conservative_phi",
    "predicted_feasible",
    "conservative_predicted_feasible",
    "predicted_active_constraint",
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
class SearchResult:
    output_dir: Path
    candidates_path: Path
    scored_path: Path
    summary_path: Path
    plot_paths: dict[str, Path]
    candidate_count: int
    scored_count: int


@dataclass(frozen=True)
class VerificationResult:
    output_dir: Path
    verified_path: Path
    summary_path: Path
    plot_paths: dict[str, Path]
    verified_count: int
    feasible_count: int


def run_surrogate_search_from_files(
    *,
    surrogate_path: str | Path,
    config: ViabilityConfig,
    output_dir: str | Path,
) -> SearchResult:
    surrogate = load_signed_constraint_surrogate(surrogate_path)
    return run_surrogate_search(
        surrogate=surrogate,
        surrogate_path=surrogate_path,
        config=config,
        output_dir=output_dir,
    )


def run_surrogate_search(
    *,
    surrogate: dict[str, Any],
    surrogate_path: str | Path,
    config: ViabilityConfig,
    output_dir: str | Path,
) -> SearchResult:
    search_config = require_search_config(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_config_resolved(config, output_path)

    candidates = generate_search_candidate_pool(config, search_config)
    scored = score_search_candidates(
        surrogate,
        candidates,
        config,
        conservative_sigma=search_config.conservative_sigma,
    )
    ranked = rank_scored_candidates(scored)
    scored_path = output_path / "scored_candidates_top.csv"
    ranked.head(search_config.candidate_report_rows).to_csv(scored_path, index=False)

    selected = select_candidates_to_verify(scored, config, search_config)
    candidates_path = output_path / "candidate_policies.csv"
    selected.to_csv(candidates_path, index=False)

    plot_paths = write_search_plots(output_path, scored, selected)
    summary_path = output_path / "search_summary.json"
    summary = _search_summary(
        surrogate_path=surrogate_path,
        search_config=search_config,
        scored=scored,
        selected=selected,
        candidates_path=candidates_path,
        scored_path=scored_path,
        plot_paths=plot_paths,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return SearchResult(
        output_dir=output_path.resolve(),
        candidates_path=candidates_path.resolve(),
        scored_path=scored_path.resolve(),
        summary_path=summary_path.resolve(),
        plot_paths={name: path.resolve() for name, path in plot_paths.items()},
        candidate_count=int(len(selected)),
        scored_count=int(len(scored)),
    )


def verify_candidates_from_file(
    *,
    candidates_path: str | Path,
    config: ViabilityConfig,
    output_dir: str | Path,
    workers: int | None = None,
    checkpoint_every: int = 50,
    evaluator: EvaluateBatch = evaluate_designs_parallel,
) -> VerificationResult:
    candidates = pd.read_csv(candidates_path)
    return verify_candidates(
        candidates=candidates,
        config=config,
        output_dir=output_dir,
        workers=workers,
        checkpoint_every=checkpoint_every,
        evaluator=evaluator,
    )


def verify_candidates(
    *,
    candidates: pd.DataFrame,
    config: ViabilityConfig,
    output_dir: str | Path,
    workers: int | None = None,
    checkpoint_every: int = 50,
    evaluator: EvaluateBatch = evaluate_designs_parallel,
) -> VerificationResult:
    search_config = require_search_config(config)
    if candidates.empty:
        raise ValueError("Candidate policy table is empty")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_config_resolved(config, output_path)

    submitted_count = int(len(candidates))
    candidates, skipped_predicted_count = filter_candidates_for_verify(
        candidates,
        search_config.required_constraints_for_verify,
    )
    if candidates.empty:
        required = ", ".join(search_config.required_constraints_for_verify)
        raise ValueError(
            "No candidate policies satisfy search.required_constraints_for_verify "
            f"({required}) before direct verification"
        )

    results = evaluator(
        candidates,
        config,
        workers=workers,
        checkpoint_dir=output_path / "checkpoints",
        checkpoint_every=checkpoint_every,
    )
    verified = attach_candidate_predictions(results, candidates)
    verified, dropped_after_verify_count = filter_verified_constraints(
        verified,
        search_config.required_constraints_for_verify,
    )
    if verified.empty:
        required = ", ".join(search_config.required_constraints_for_verify)
        raise ValueError(
            "No verified candidate policies satisfied search.required_constraints_for_verify "
            f"({required}) after direct evaluation"
        )
    verified_path = write_table(verified, output_path / "verified_candidates.parquet")
    plot_paths = write_verification_plots(output_path, verified)
    summary_path = output_path / "verification_summary.json"
    summary = verification_summary(
        verified,
        verified_path,
        plot_paths,
        submitted_count=submitted_count,
        skipped_predicted_count=skipped_predicted_count,
        dropped_after_verify_count=dropped_after_verify_count,
        required_constraints_for_verify=search_config.required_constraints_for_verify,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return VerificationResult(
        output_dir=output_path.resolve(),
        verified_path=verified_path.resolve(),
        summary_path=summary_path.resolve(),
        plot_paths={name: path.resolve() for name, path in plot_paths.items()},
        verified_count=int(len(verified)),
        feasible_count=int(summary["verified_feasible_count"]),
    )


def require_search_config(config: ViabilityConfig) -> SearchConfig:
    if config.search is None:
        raise ValueError("Config must include a search section for search and verify-candidates")
    return config.search


def load_signed_constraint_surrogate(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Signed-RAP surrogate does not exist: {model_path}")
    bundle = joblib.load(model_path)
    validate_signed_constraint_surrogate(bundle)
    return bundle


def validate_signed_constraint_surrogate(bundle: Any) -> None:
    if not isinstance(bundle, dict):
        raise ValueError("Signed-RAP surrogate must be a dictionary bundle")
    required_keys = ("models_by_constraint", "constraint_names", "target", "feature_names")
    for key in required_keys:
        if key not in bundle:
            raise ValueError(f"Signed-RAP surrogate is missing required key {key!r}")
    if bundle["target"] != "normalized_constraints":
        raise ValueError("Signed-RAP surrogate target must be 'normalized_constraints'")
    if not bundle["constraint_names"]:
        raise ValueError("Signed-RAP surrogate must include at least one constraint")
    for constraint_name in bundle["constraint_names"]:
        if constraint_name not in bundle["models_by_constraint"]:
            raise ValueError(f"Signed-RAP surrogate is missing model for {constraint_name!r}")


def generate_search_candidate_pool(
    config: ViabilityConfig,
    search_config: SearchConfig,
) -> pd.DataFrame:
    return generate_doe(
        config,
        n=search_config.candidate_pool_size,
        method=search_config.candidate_method,
        start_index=search_config.candidate_start_index,
        include_corners=False,
        include_baselines=False,
    )


def score_search_candidates(
    surrogate: dict[str, Any],
    candidates: pd.DataFrame,
    config: ViabilityConfig,
    *,
    conservative_sigma: float,
) -> pd.DataFrame:
    validate_signed_constraint_surrogate(surrogate)
    if candidates.empty:
        raise ValueError("Candidate pool is empty before scoring")
    x_values = design_matrix(candidates, config)
    prediction = predict_constraint_surrogate(
        surrogate,
        x_values,
        conservative_sigma=conservative_sigma,
    )
    scored = candidates.reset_index(drop=True).copy()
    scored["candidate_pool_index"] = np.arange(len(scored), dtype=int)
    scored["predicted_phi"] = prediction.predicted_phi
    scored["predicted_sigma_phi"] = prediction.sigma_phi
    scored["conservative_phi"] = prediction.conservative_phi
    scored["predicted_feasible"] = prediction.predicted_phi <= 0.0
    scored["conservative_predicted_feasible"] = prediction.conservative_phi <= 0.0
    scored["predicted_active_constraint"] = prediction.active_constraint
    scored["abs_predicted_phi"] = np.abs(prediction.predicted_phi)
    for constraint_name in prediction.mu.columns:
        scored[f"mu_constraint_{constraint_name}"] = prediction.mu[
            constraint_name
        ].to_numpy(dtype=float)
        scored[f"sigma_constraint_{constraint_name}"] = prediction.sigma[
            constraint_name
        ].to_numpy(dtype=float)
    return scored


def rank_scored_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.sort_values(
        [
            "conservative_phi",
            "predicted_phi",
            "abs_predicted_phi",
            "predicted_sigma_phi",
            "design_id",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)


def select_candidates_to_verify(
    scored: pd.DataFrame,
    config: ViabilityConfig,
    search_config: SearchConfig,
) -> pd.DataFrame:
    if scored.empty:
        raise ValueError("Scored candidate table is empty")
    viable_scored, _ = filter_scored_for_verify(
        scored,
        search_config.required_constraints_for_verify,
    )
    if viable_scored.empty:
        required = ", ".join(search_config.required_constraints_for_verify)
        raise RuntimeError(
            "Candidate pool has no policies predicted to satisfy "
            f"search.required_constraints_for_verify ({required}); "
            "increase search.candidate_pool_size or relax the requirement list"
        )
    quotas = selection_quotas(search_config.n_candidates_to_verify)
    selected_frames: list[pd.DataFrame] = []
    selected_keys: set[tuple[Any, ...]] = set()
    selected_vectors: list[np.ndarray] = []

    for category, count in quotas.items():
        ordered = ordered_candidates_for_category(viable_scored, category)
        selected = _select_from_ordered(
            ordered,
            config,
            count=count,
            min_normalized_distance=search_config.min_normalized_distance,
            selected_keys=selected_keys,
            selected_vectors=selected_vectors,
            selection_source=category,
        )
        if not selected.empty:
            selected_frames.append(selected)

    selected_count = sum(len(frame) for frame in selected_frames)
    if selected_count < search_config.n_candidates_to_verify:
        backfill = _select_from_ordered(
            rank_scored_candidates(viable_scored),
            config,
            count=search_config.n_candidates_to_verify - selected_count,
            min_normalized_distance=search_config.min_normalized_distance,
            selected_keys=selected_keys,
            selected_vectors=selected_vectors,
            selection_source="backfill",
        )
        if not backfill.empty:
            selected_frames.append(backfill)

    if not selected_frames:
        selected = viable_scored.iloc[[]].copy()
    else:
        selected = pd.concat(selected_frames, ignore_index=True)
    if len(selected) != search_config.n_candidates_to_verify:
        raise RuntimeError(
            "Candidate pool could not fill the requested verification batch after "
            "required-constraint filtering, dedupe, and diversity filtering; "
            "increase search.candidate_pool_size or lower search.min_normalized_distance"
        )
    selected = selected.reset_index(drop=True)
    selected.insert(0, "candidate_id", [f"candidate_{index:04d}" for index in range(len(selected))])
    selected.insert(1, "selection_rank", np.arange(1, len(selected) + 1, dtype=int))
    return selected


def filter_scored_for_verify(
    scored: pd.DataFrame,
    required_constraints: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """Keep scored rows predicted to satisfy every required constraint (margin <= 0)."""
    if not required_constraints:
        return scored.copy(), 0
    mask = _predicted_constraint_mask(scored, required_constraints)
    filtered = scored.loc[mask].copy()
    return filtered.reset_index(drop=True), int(len(scored) - len(filtered))


def filter_candidates_for_verify(
    candidates: pd.DataFrame,
    required_constraints: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """Drop candidates that fail the surrogate pre-check before direct verification."""
    if not required_constraints:
        return candidates.copy(), 0
    if _has_predicted_constraint_columns(candidates, required_constraints):
        mask = _predicted_constraint_mask(candidates, required_constraints)
        filtered = candidates.loc[mask].copy()
        return filtered.reset_index(drop=True), int(len(candidates) - len(filtered))
    return candidates.copy(), 0


def filter_verified_constraints(
    verified: pd.DataFrame,
    required_constraints: Sequence[str],
    *,
    tolerance: float = 0.0,
) -> tuple[pd.DataFrame, int]:
    """Keep only direct-evaluation rows that satisfy required constraints."""
    if not required_constraints:
        return verified.copy(), 0
    mask = _verified_constraint_mask(verified, required_constraints, tolerance=tolerance)
    filtered = verified.loc[mask].copy()
    return filtered.reset_index(drop=True), int(len(verified) - len(filtered))


def selection_quotas(n_candidates: int) -> dict[str, int]:
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    fractions = {
        "conservative_feasible": 0.40,
        "predicted_feasible_margin": 0.20,
        "near_boundary": 0.20,
        "minimum_predicted_violation": 0.10,
        "uncertainty_near_boundary": 0.10,
    }
    quotas = {name: int(np.floor(n_candidates * fraction)) for name, fraction in fractions.items()}
    remaining = n_candidates - sum(quotas.values())
    for name in fractions:
        if remaining == 0:
            break
        quotas[name] += 1
        remaining -= 1
    return quotas


def ordered_candidates_for_category(scored: pd.DataFrame, category: str) -> pd.DataFrame:
    if category == "conservative_feasible":
        subset = scored.loc[scored["conservative_predicted_feasible"]].copy()
        return subset.sort_values(
            ["conservative_phi", "predicted_phi", "design_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    if category == "predicted_feasible_margin":
        subset = scored.loc[scored["predicted_feasible"]].copy()
        return subset.sort_values(
            ["predicted_phi", "conservative_phi", "design_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    if category == "near_boundary":
        return scored.sort_values(
            ["abs_predicted_phi", "predicted_sigma_phi", "design_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    if category == "minimum_predicted_violation":
        return scored.sort_values(
            ["predicted_phi", "conservative_phi", "design_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    if category == "uncertainty_near_boundary":
        near_count = max(1, int(np.ceil(len(scored) * 0.25)))
        near = scored.sort_values(
            ["abs_predicted_phi", "design_id"],
            ascending=[True, True],
        ).head(near_count)
        return near.sort_values(
            ["predicted_sigma_phi", "abs_predicted_phi", "design_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    raise ValueError(f"Unknown search candidate category {category!r}")


def attach_candidate_predictions(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if "design_id" not in results.columns or "design_id" not in candidates.columns:
        raise ValueError("Both candidate and verification tables must include design_id")
    existing_columns = [
        column
        for column in _CANDIDATE_METADATA_COLUMNS
        if column in candidates.columns
    ]
    metadata = candidates.loc[:, existing_columns].copy()
    verified = results.merge(metadata, on="design_id", how="left", validate="one_to_one")
    if verified["candidate_id"].isna().any():
        missing = verified.loc[verified["candidate_id"].isna(), "design_id"].tolist()
        raise ValueError(f"Verified rows could not be matched to candidate metadata: {missing}")
    return verified.sort_values("selection_rank").reset_index(drop=True)


def verification_summary(
    verified: pd.DataFrame,
    verified_path: Path,
    plot_paths: dict[str, Path],
    *,
    submitted_count: int | None = None,
    skipped_predicted_count: int = 0,
    dropped_after_verify_count: int = 0,
    required_constraints_for_verify: Sequence[str] = (),
) -> dict[str, Any]:
    _require_verification_columns(verified)
    verified_feasible = verified["feasible"].astype(bool).to_numpy()
    predicted_feasible = verified["predicted_feasible"].astype(bool).to_numpy()
    conservative_feasible = verified["conservative_predicted_feasible"].astype(bool).to_numpy()
    false_feasible = predicted_feasible & ~verified_feasible
    false_conservative_feasible = conservative_feasible & ~verified_feasible
    best_index = int(verified["phi"].astype(float).idxmin())
    active_counts = verified["active_constraint"].fillna("none").value_counts().sort_index()
    summary = {
        "verified_path": str(Path(verified_path).resolve()),
        "verified_count": int(len(verified)),
        "submitted_candidate_count": int(submitted_count if submitted_count is not None else len(verified)),
        "skipped_predicted_infeasible_count": int(skipped_predicted_count),
        "dropped_after_verify_count": int(dropped_after_verify_count),
        "required_constraints_for_verify": list(required_constraints_for_verify),
        "verified_feasible_count": int(np.sum(verified_feasible)),
        "predicted_feasible_count": int(np.sum(predicted_feasible)),
        "conservative_predicted_feasible_count": int(np.sum(conservative_feasible)),
        "false_feasible_count": int(np.sum(false_feasible)),
        "false_conservative_feasible_count": int(np.sum(false_conservative_feasible)),
        "best_candidate_id": str(verified.loc[best_index, "candidate_id"]),
        "best_design_id": str(verified.loc[best_index, "design_id"]),
        "best_verified_phi": float(verified.loc[best_index, "phi"]),
        "best_predicted_phi": float(verified.loc[best_index, "predicted_phi"]),
        "best_conservative_phi": float(verified.loc[best_index, "conservative_phi"]),
        "active_constraint_counts": {
            str(name): int(count)
            for name, count in active_counts.items()
        },
        "plot_paths": {name: str(path.resolve()) for name, path in plot_paths.items()},
    }
    return summary


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


def policy_key(row: pd.Series, config: ViabilityConfig) -> tuple[Any, ...]:
    values = []
    for name in DesignSpace(config.policy).variable_names:
        applied_column = f"applied_{name}"
        if applied_column in row:
            value = row[applied_column]
        elif name in row:
            value = row[name]
        else:
            raise ValueError(f"Design row is missing policy column {name!r} and {applied_column!r}")
        if pd.isna(value):
            raise ValueError(f"Design row has null policy value for {name!r}")
        if hasattr(value, "item"):
            value = value.item()
        values.append(value)
    return tuple(values)


def write_search_plots(
    output_dir: str | Path,
    scored: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    _configure_matplotlib_cache(output_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: dict[str, Path] = {}
    mu_sigma_path = output_path / "search_mu_sigma.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        scored["predicted_phi"],
        scored["predicted_sigma_phi"],
        s=8,
        color="0.65",
        alpha=0.25,
        edgecolor="none",
        label="candidate pool",
    )
    ax.scatter(
        selected["predicted_phi"],
        selected["predicted_sigma_phi"],
        s=32,
        color="firebrick",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
        label="selected",
    )
    ax.axvline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("Predicted phi")
    ax.set_ylabel("Predictive sigma")
    ax.set_title("Surrogate Search Candidates")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(mu_sigma_path, dpi=180)
    plt.close(fig)
    plot_paths["mu_sigma"] = mu_sigma_path

    phi_path = output_path / "search_selected_phi.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ranks = selected["selection_rank"].to_numpy(dtype=int)
    ax.plot(ranks, selected["predicted_phi"].to_numpy(dtype=float), marker="o", label="predicted")
    ax.plot(
        ranks,
        selected["conservative_phi"].to_numpy(dtype=float),
        marker="o",
        label="conservative",
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("Selection rank")
    ax.set_ylabel("Phi")
    ax.set_title("Selected Candidate Phi")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(phi_path, dpi=180)
    plt.close(fig)
    plot_paths["selected_phi"] = phi_path
    return plot_paths


def write_verification_plots(
    output_dir: str | Path,
    verified: pd.DataFrame,
) -> dict[str, Path]:
    _require_verification_columns(verified)
    output_path = Path(output_dir)
    _configure_matplotlib_cache(output_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: dict[str, Path] = {}
    compare_path = output_path / "verified_predicted_vs_truth_phi.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    truth = verified["phi"].to_numpy(dtype=float)
    predicted = verified["predicted_phi"].to_numpy(dtype=float)
    conservative = verified["conservative_phi"].to_numpy(dtype=float)
    ax.scatter(
        truth,
        predicted,
        s=34,
        color="firebrick",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
        label="predicted",
    )
    ax.scatter(
        truth,
        conservative,
        s=22,
        color="0.25",
        alpha=0.45,
        edgecolor="none",
        label="conservative",
    )
    min_value = float(min(np.min(truth), np.min(predicted), np.min(conservative), 0.0))
    max_value = float(max(np.max(truth), np.max(predicted), np.max(conservative), 0.0))
    span = max(max_value - min_value, 1.0)
    padding = 0.05 * span
    ax.plot(
        [min_value - padding, max_value + padding],
        [min_value - padding, max_value + padding],
        color="black",
        linewidth=1,
        alpha=0.7,
    )
    ax.axvline(0.0, color="black", linewidth=1, alpha=0.25)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.25)
    ax.set_xlim(min_value - padding, max_value + padding)
    ax.set_ylim(min_value - padding, max_value + padding)
    ax.set_xlabel("Verified phi")
    ax.set_ylabel("Surrogate phi")
    ax.set_title("Candidate Verification")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(compare_path, dpi=180)
    plt.close(fig)
    plot_paths["predicted_vs_verified_phi"] = compare_path
    return plot_paths


def _select_from_ordered(
    ordered: pd.DataFrame,
    config: ViabilityConfig,
    *,
    count: int,
    min_normalized_distance: float,
    selected_keys: set[tuple[Any, ...]],
    selected_vectors: list[np.ndarray],
    selection_source: str,
) -> pd.DataFrame:
    if count == 0 or ordered.empty:
        selected = ordered.iloc[[]].copy()
        selected["selection_source"] = []
        return selected
    selected_indices = []
    for row_index, row in ordered.iterrows():
        key = policy_key(row, config)
        if key in selected_keys:
            continue
        candidate_vector = design_vector(row, config)
        if any(
            float(np.linalg.norm(candidate_vector - selected_vector)) < min_normalized_distance
            for selected_vector in selected_vectors
        ):
            continue
        selected_keys.add(key)
        selected_vectors.append(candidate_vector)
        selected_indices.append(row_index)
        if len(selected_indices) == count:
            break
    selected = ordered.loc[selected_indices].reset_index(drop=True).copy()
    selected["selection_source"] = selection_source
    return selected


def _search_summary(
    *,
    surrogate_path: str | Path,
    search_config: SearchConfig,
    scored: pd.DataFrame,
    selected: pd.DataFrame,
    candidates_path: Path,
    scored_path: Path,
    plot_paths: dict[str, Path],
) -> dict[str, Any]:
    viable_scored, predicted_infeasible_count = filter_scored_for_verify(
        scored,
        search_config.required_constraints_for_verify,
    )
    source_counts = selected["selection_source"].value_counts().sort_index()
    return {
        "surrogate_path": str(Path(surrogate_path).resolve()),
        "candidate_method": search_config.candidate_method,
        "candidate_start_index": int(search_config.candidate_start_index),
        "candidate_pool_size": int(search_config.candidate_pool_size),
        "n_candidates_to_verify": int(search_config.n_candidates_to_verify),
        "conservative_sigma": float(search_config.conservative_sigma),
        "min_normalized_distance": float(search_config.min_normalized_distance),
        "candidate_report_rows": int(search_config.candidate_report_rows),
        "required_constraints_for_verify": list(search_config.required_constraints_for_verify),
        "scored_count": int(len(scored)),
        "predicted_required_constraint_count": int(len(viable_scored)),
        "predicted_required_constraint_infeasible_count": int(predicted_infeasible_count),
        "selected_count": int(len(selected)),
        "predicted_feasible_count": int(scored["predicted_feasible"].sum()),
        "conservative_predicted_feasible_count": int(
            scored["conservative_predicted_feasible"].sum()
        ),
        "selected_predicted_feasible_count": int(selected["predicted_feasible"].sum()),
        "selected_conservative_predicted_feasible_count": int(
            selected["conservative_predicted_feasible"].sum()
        ),
        "selection_source_counts": {
            str(name): int(count)
            for name, count in source_counts.items()
        },
        "candidate_policies_path": str(candidates_path.resolve()),
        "scored_candidates_top_path": str(scored_path.resolve()),
        "plot_paths": {name: str(path.resolve()) for name, path in plot_paths.items()},
    }


def _require_verification_columns(verified: pd.DataFrame) -> None:
    required = (
        "candidate_id",
        "design_id",
        "phi",
        "feasible",
        "predicted_phi",
        "conservative_phi",
        "predicted_feasible",
        "conservative_predicted_feasible",
    )
    missing = [column for column in required if column not in verified.columns]
    if missing:
        raise ValueError(f"Verified candidate table is missing required columns: {missing}")


def _configure_matplotlib_cache(output_path: Path) -> None:
    cache_dir = output_path / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)


def _predicted_constraint_column(constraint_name: str) -> str:
    return f"mu_constraint_{constraint_name}"


def _verified_constraint_column(constraint_name: str) -> str:
    return f"constraint_{constraint_name}"


def _has_predicted_constraint_columns(
    frame: pd.DataFrame,
    required_constraints: Sequence[str],
) -> bool:
    return all(_predicted_constraint_column(name) in frame.columns for name in required_constraints)


def _predicted_constraint_mask(
    frame: pd.DataFrame,
    required_constraints: Sequence[str],
) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for name in required_constraints:
        column = _predicted_constraint_column(name)
        if column not in frame.columns:
            raise ValueError(
                "Candidate table is missing surrogate prediction column "
                f"{column!r} required by search.required_constraints_for_verify"
            )
        mask &= frame[column].astype(float) <= 0.0
    return mask


def _verified_constraint_mask(
    frame: pd.DataFrame,
    required_constraints: Sequence[str],
    *,
    tolerance: float,
) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for name in required_constraints:
        column = _verified_constraint_column(name)
        if column not in frame.columns:
            raise ValueError(
                "Verified candidate table is missing direct-evaluation column "
                f"{column!r} required by search.required_constraints_for_verify"
            )
        mask &= frame[column].astype(float) <= tolerance
    return mask
