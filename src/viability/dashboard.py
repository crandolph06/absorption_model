from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.viability.config import VariableConfig, ViabilityConfig, load_config
from src.viability.dynamic_policy import EpochPolicySchedule, dynamic_feature_names
from src.viability.evaluator import (
    EvaluationResult,
    simulate_design_history,
    simulate_policy_schedule_history,
)
from src.viability.metrics import (
    UTC_CONSTRAINT_SPECS,
    aggregate_violation,
    compute_constraints,
    compute_raw_metrics,
)
from src.viability.plots import apply_policy_value, predict_policy_frame
from src.viability.policy import PolicyDesign
from src.viability.search import load_signed_constraint_surrogate
from src.viability.surrogate import read_evaluations_table


POLICY_LABELS = {
    "annual_intake": "Annual B-course intake",
    "retention_rate": "Retention rate",
    "ute": "UTE",
    "paa": "PAA",
    "max_manning_pct": "Maximum manning %",
    "flug_quota_per_phase": "FLUG quota / phase",
    "ipug_quota_per_phase": "IPUG quota / phase",
}

STATIC_RAP_OPTIONS = ("a", "b", "c")
STATIC_CONSTRAINT_OPTIONS = ("current", "pragmatic", "optimistic", "ideal")
STATIC_SCOPE_OPTIONS = ("unit", "enterprise")
STATIC_DOE_DIR = "doe_128"
STATIC_CONSTRAINT_LABELS = {
    "current": "Current",
    "pragmatic": "Pragmatic",
    "optimistic": "Optimistic",
    "ideal": "Ideal",
}


@dataclass(frozen=True)
class DashboardArtifactPaths:
    config: Path
    surrogate: Path
    evaluations: Path
    verified_candidates: Path
    search_summary: Path
    verification_summary: Path
    envelope_summary: Path
    report: Path | None = None


@dataclass(frozen=True)
class DashboardArtifacts:
    paths: DashboardArtifactPaths
    config: ViabilityConfig
    surrogate: dict[str, Any]
    evaluations: pd.DataFrame
    verified_candidates: pd.DataFrame
    search_summary: dict[str, Any]
    verification_summary: dict[str, Any]
    envelope_summary: dict[str, Any]


@dataclass(frozen=True)
class DynamicDashboardArtifactPaths:
    config: Path
    evaluations: Path
    summary: Path
    sensitivity: Path | None = None
    report: Path | None = None
    relaxation_dir: Path | None = None
    bound_relaxation_dir: Path | None = None
    ipug_diagnostic_dir: Path | None = None
    paper_artifacts_dir: Path | None = None


@dataclass(frozen=True)
class DynamicDashboardArtifacts:
    paths: DynamicDashboardArtifactPaths
    config: ViabilityConfig
    evaluations: pd.DataFrame
    summary: dict[str, Any]
    epoch_count: int
    sensitivity: pd.DataFrame | None = None
    relaxation_summary: dict[str, Any] | None = None
    relaxation_nearest: pd.DataFrame | None = None
    relaxation_pareto: pd.DataFrame | None = None
    relaxation_sets: pd.DataFrame | None = None
    relaxation_report: str | None = None
    bound_relaxation_summary: dict[str, Any] | None = None
    bound_relaxation_best_by_experiment: pd.DataFrame | None = None
    ipug_summary: dict[str, Any] | None = None
    ipug_evaluations: pd.DataFrame | None = None
    paper_figure_paths: dict[str, Path] | None = None


@dataclass(frozen=True)
class SliderInterval:
    lever: str
    low: float
    high: float
    n_points: int


@dataclass(frozen=True)
class SliderSweepResult:
    lever: str
    sweep: pd.DataFrame
    intervals: list[SliderInterval]


@dataclass(frozen=True)
class DirectPolicyResult:
    evaluation: EvaluationResult
    history: pd.DataFrame
    trajectory: pd.DataFrame


def phase_backend_label(config: ViabilityConfig) -> str:
    if config.model.phase_backend == "physics":
        return "physics backend"
    return "brain backend"


def direct_verification_label(config: ViabilityConfig) -> str:
    return f"Direct long-horizon verification ({phase_backend_label(config)})"


def direct_verification_caveat(config: ViabilityConfig) -> str:
    if config.model.phase_backend == "physics":
        return (
            "This bypasses the outer signed-RAP surrogate and the internal sortie "
            "brain by using direct single-phase physics inside the long-horizon model."
        )
    return (
        "This bypasses the outer signed-RAP surrogate, but it still uses the "
        "configured internal sortie brain for each simulated phase."
    )


def policy_variable_is_fixed(variable: VariableConfig) -> bool:
    if variable.type == "int":
        return int(variable.low) == int(variable.high)
    return abs(float(variable.high) - float(variable.low)) <= 1e-12


def static_scenario_slug(*, rap: str, constraint: str, scope: str) -> str:
    rap_key = rap.lower()
    constraint_key = constraint.lower()
    scope_key = scope.lower()
    if rap_key not in STATIC_RAP_OPTIONS:
        raise ValueError(f"Unsupported RAP scenario {rap!r}; expected one of {STATIC_RAP_OPTIONS}")
    if constraint_key not in STATIC_CONSTRAINT_OPTIONS:
        raise ValueError(
            f"Unsupported constraint scenario {constraint!r}; "
            f"expected one of {STATIC_CONSTRAINT_OPTIONS}"
        )
    if scope_key not in STATIC_SCOPE_OPTIONS:
        raise ValueError(f"Unsupported scope {scope!r}; expected one of {STATIC_SCOPE_OPTIONS}")
    return f"{rap_key}_{constraint_key}_{scope_key}"


def static_scenario_output_dir(
    root: str | Path,
    *,
    rap: str,
    constraint: str,
    scope: str,
) -> Path:
    slug = static_scenario_slug(rap=rap, constraint=constraint, scope=scope)
    return Path(root) / "outputs" / "viability" / f"rap_{slug}"


def _rebase_onto_local(stored_path: Path, active_learn_dir: Path) -> Path | None:
    # state.json records an absolute path from the machine that ran the pipeline
    # (e.g. an HPC /p/work1/... path). After downloading, that prefix is wrong, so
    # re-attach the portion at and after "active_learn" to the local directory.
    parts = stored_path.parts
    if "active_learn" in parts:
        tail = parts[parts.index("active_learn") + 1 :]
        candidate = active_learn_dir.joinpath(*tail)
        if candidate.exists():
            return candidate
    local_guess = active_learn_dir / stored_path.name
    if local_guess.exists():
        return local_guess
    return None


def expected_active_learn_surrogate_path(active_learn_dir: str | Path) -> Path:
    directory = Path(active_learn_dir)
    state_path = directory / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        stored_path = Path(str(state["latest_model_path"]))
        if stored_path.exists():
            return stored_path
        rebased = _rebase_onto_local(stored_path, directory)
        if rebased is not None:
            return rebased
    iteration_models = sorted(directory.glob("iteration_*/surrogate_constraints_gpr.joblib"))
    if iteration_models:
        return iteration_models[-1]
    baseline = directory / "baseline" / "surrogate_constraints_gpr.joblib"
    if baseline.exists():
        return baseline
    return directory / "iteration_003" / "surrogate_constraints_gpr.joblib"


def static_artifact_paths_for_scenario(
    *,
    rap: str,
    constraint: str,
    scope: str,
    root: str | Path = ".",
    doe_dir: str = STATIC_DOE_DIR,
) -> DashboardArtifactPaths:
    base = Path(root)
    slug = static_scenario_slug(rap=rap, constraint=constraint, scope=scope)
    scenario_root = base / "outputs" / "viability" / f"rap_{slug}"
    active_learn_dir = scenario_root / "active_learn"
    report_path = scenario_root / "report.md"
    return DashboardArtifactPaths(
        config=base / "configs" / "viability" / f"{slug}.yaml",
        surrogate=expected_active_learn_surrogate_path(active_learn_dir),
        evaluations=scenario_root / doe_dir / "evaluations.parquet",
        verified_candidates=scenario_root / "verify" / "verified_candidates.parquet",
        search_summary=scenario_root / "search" / "search_summary.json",
        verification_summary=scenario_root / "verify" / "verification_summary.json",
        envelope_summary=scenario_root / "envelope" / "envelope_summary.json",
        report=report_path if report_path.exists() else None,
    )


def static_artifact_path_status(paths: DashboardArtifactPaths) -> list[tuple[str, Path, bool]]:
    items: list[tuple[str, Path]] = [
        ("config", paths.config),
        ("surrogate", paths.surrogate),
        ("evaluations", paths.evaluations),
        ("verified_candidates", paths.verified_candidates),
        ("search_summary", paths.search_summary),
        ("verification_summary", paths.verification_summary),
        ("envelope_summary", paths.envelope_summary),
    ]
    if paths.report is not None:
        items.append(("report", paths.report))
    return [(name, path, path.exists()) for name, path in items]


def default_artifact_paths(root: str | Path = ".") -> DashboardArtifactPaths:
    base = Path(root)
    search_dir = base / "outputs" / "viability" / "runs" / "search"
    return DashboardArtifactPaths(
        config=base / "configs" / "viability.example.yaml",
        surrogate=(
            base
            / "outputs"
            / "viability"
            / "runs"
            / "rap_signed"
            / "gpr"
            / "surrogate_constraints_gpr.joblib"
        ),
        evaluations=(
            base
            / "outputs"
            / "viability"
            / "runs"
            / "rap_signed"
            / "evaluations.parquet"
        ),
        verified_candidates=search_dir / "verified_candidates.parquet",
        search_summary=search_dir / "search_summary.json",
        verification_summary=search_dir / "verification_summary.json",
        envelope_summary=search_dir / "envelope" / "envelope_summary.json",
        report=search_dir / "report.md",
    )


def default_dynamic_artifact_paths(root: str | Path = ".") -> DynamicDashboardArtifactPaths:
    base = Path(root)
    dynamic_root = base / "outputs" / "viability" / "dynamic_policy_search"
    summary_candidates = [
        *dynamic_root.glob("*/dynamic_search_summary.json"),
        *dynamic_root.glob("*/dynamic_refinement_summary.json"),
    ]
    summary_path = _best_dynamic_summary_path(summary_candidates)
    run_dir = summary_path.parent if summary_path is not None else dynamic_root / "run_3epoch_512_32768_096"
    return DynamicDashboardArtifactPaths(
        config=run_dir / "config_resolved.yaml",
        evaluations=run_dir / "all_evaluations.parquet",
        summary=summary_path if summary_path is not None else run_dir / "dynamic_search_summary.json",
        sensitivity=run_dir / "diagnostic" / "local_sensitivity.csv",
        report=run_dir / "diagnostic" / "dynamic_control_report.md",
        relaxation_dir=dynamic_root / "relaxation_study_v1",
        bound_relaxation_dir=dynamic_root / "bound_relaxation_v2",
        ipug_diagnostic_dir=dynamic_root / "ipug_counterfactual_v2",
        paper_artifacts_dir=dynamic_root / "paper_artifacts_v1",
    )


def _best_dynamic_summary_path(summary_candidates: list[Path]) -> Path | None:
    if not summary_candidates:
        return None

    def score(path: Path) -> tuple[float, float]:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            best_phi = float(summary.get("best_phi"))
        except Exception:
            best_phi = float("inf")
        return best_phi, -float(path.stat().st_mtime)

    return min(summary_candidates, key=score)


def load_dashboard_artifacts(paths: DashboardArtifactPaths) -> DashboardArtifacts:
    config = load_config(paths.config)
    surrogate = load_signed_constraint_surrogate(paths.surrogate)
    evaluations = read_evaluations_table(paths.evaluations)
    _require_columns(
        evaluations,
        ["phi", *config.policy.variables],
        "evaluations",
    )
    verified_candidates = read_evaluations_table(paths.verified_candidates)
    _require_columns(
        verified_candidates,
        [
            "candidate_id",
            "design_id",
            "phi",
            "feasible",
            "active_constraint",
            *config.policy.variables,
        ],
        "verified candidates",
    )
    search_summary = _read_json_object(paths.search_summary)
    verification_summary = _read_json_object(paths.verification_summary)
    envelope_summary = _read_json_object(paths.envelope_summary)
    _validate_envelope_paths(envelope_summary)
    if paths.report is not None and not paths.report.exists():
        raise FileNotFoundError(f"Report artifact does not exist: {paths.report}")
    return DashboardArtifacts(
        paths=paths,
        config=config,
        surrogate=surrogate,
        evaluations=evaluations,
        verified_candidates=verified_candidates,
        search_summary=search_summary,
        verification_summary=verification_summary,
        envelope_summary=envelope_summary,
    )


def load_dynamic_dashboard_artifacts(
    paths: DynamicDashboardArtifactPaths,
) -> DynamicDashboardArtifacts:
    config = load_config(paths.config)
    evaluations = read_evaluations_table(paths.evaluations)
    summary = _read_json_object(paths.summary)
    epoch_count = int(summary.get("epoch_count") or infer_dynamic_epoch_count(evaluations, config))
    if epoch_count <= 0:
        raise ValueError("Dynamic search summary must identify a positive epoch_count")
    _require_columns(
        evaluations,
        [
            "schedule_id",
            "phi",
            "feasible",
            "active_constraint",
            "status",
            "phase_backend",
            *dynamic_feature_names(config.policy, epoch_count),
        ],
        "dynamic evaluations",
    )
    sensitivity = None
    if paths.sensitivity is not None and paths.sensitivity.exists():
        sensitivity = read_evaluations_table(paths.sensitivity)
        _require_columns(
            sensitivity,
            ["epoch", "control", "response", "sensitivity", "abs_sensitivity"],
            "dynamic sensitivity",
        )
    if paths.report is not None and not paths.report.exists():
        raise FileNotFoundError(f"Dynamic report artifact does not exist: {paths.report}")
    relaxation_summary = None
    relaxation_nearest = None
    relaxation_pareto = None
    relaxation_sets = None
    relaxation_report = None
    bound_relaxation_summary = None
    bound_relaxation_best = None
    ipug_summary = None
    ipug_evaluations = None
    paper_figure_paths = None
    if paths.relaxation_dir is not None:
        relaxation_dir = paths.relaxation_dir
        if relaxation_dir.exists():
            relaxation_summary = _read_json_object(relaxation_dir / "relaxation_summary.json")
            relaxation_nearest = read_evaluations_table(relaxation_dir / "nearest_under_relaxation.csv")
            relaxation_pareto = read_evaluations_table(relaxation_dir / "pareto_frontier.csv")
            relaxation_sets = read_evaluations_table(relaxation_dir / "relaxation_sets.csv")
            _require_columns(
                relaxation_nearest,
                ["schedule_id", "max_normalized_relaxation", "positive_normalized_sum"],
                "dynamic relaxation nearest policies",
            )
            _require_columns(
                relaxation_pareto,
                ["schedule_id", "max_normalized_relaxation", "positive_normalized_sum"],
                "dynamic relaxation Pareto frontier",
            )
            _require_columns(
                relaxation_sets,
                ["constraint_set", "schedule_id", "max_normalized_relaxation"],
                "dynamic relaxation constraint sets",
            )
            report_path = relaxation_dir / "relaxation_report.md"
            if report_path.exists():
                relaxation_report = report_path.read_text(encoding="utf-8")
        elif str(relaxation_dir).strip():
            raise FileNotFoundError(f"Dynamic relaxation artifact directory does not exist: {relaxation_dir}")
    if paths.bound_relaxation_dir is not None:
        bound_dir = paths.bound_relaxation_dir
        if bound_dir.exists():
            bound_relaxation_summary = _read_json_object(bound_dir / "bound_relaxation_summary.json")
            bound_relaxation_best = read_evaluations_table(bound_dir / "best_by_bound_experiment.csv")
            _require_columns(
                bound_relaxation_best,
                ["experiment_id", "phi", "feasible", "active_constraint"],
                "dynamic bound-relaxation best-by-experiment table",
            )
        elif str(bound_dir).strip():
            raise FileNotFoundError(f"Dynamic bound-relaxation artifact directory does not exist: {bound_dir}")
    if paths.ipug_diagnostic_dir is not None:
        ipug_dir = paths.ipug_diagnostic_dir
        if ipug_dir.exists():
            ipug_summary = _read_json_object(ipug_dir / "ipug_counterfactual_summary.json")
            ipug_evaluations = read_evaluations_table(ipug_dir / "ipug_counterfactual_evaluations.parquet")
            _require_columns(
                ipug_evaluations,
                ["sweep_value", "phi", "feasible", "active_constraint"],
                "dynamic IPUG diagnostic evaluations",
            )
        elif str(ipug_dir).strip():
            raise FileNotFoundError(f"Dynamic IPUG diagnostic artifact directory does not exist: {ipug_dir}")
    if paths.paper_artifacts_dir is not None:
        paper_dir = paths.paper_artifacts_dir
        if paper_dir.exists():
            figure_names = {
                "inventory": "trajectory_total_pilots.png",
                "rap": "trajectory_rap_shortfalls.png",
                "staff": "trajectory_staff_counts.png",
                "policy": "best_policy_epoch_controls.png",
                "trade_space": "trade_space_total_wg_fl.png",
            }
            paper_figure_paths = {
                name: paper_dir / filename
                for name, filename in figure_names.items()
                if (paper_dir / filename).exists()
            }
        elif str(paper_dir).strip():
            raise FileNotFoundError(f"Dynamic paper-artifacts directory does not exist: {paper_dir}")
    return DynamicDashboardArtifacts(
        paths=paths,
        config=config,
        evaluations=evaluations,
        summary=summary,
        epoch_count=epoch_count,
        sensitivity=sensitivity,
        relaxation_summary=relaxation_summary,
        relaxation_nearest=relaxation_nearest,
        relaxation_pareto=relaxation_pareto,
        relaxation_sets=relaxation_sets,
        relaxation_report=relaxation_report,
        bound_relaxation_summary=bound_relaxation_summary,
        bound_relaxation_best_by_experiment=bound_relaxation_best,
        ipug_summary=ipug_summary,
        ipug_evaluations=ipug_evaluations,
        paper_figure_paths=paper_figure_paths,
    )


def infer_dynamic_epoch_count(evaluations: pd.DataFrame, config: ViabilityConfig) -> int:
    control = next(iter(config.policy.variables))
    prefix = "epoch"
    suffix = f"_{control}"
    epochs = []
    for column in evaluations.columns:
        if not column.startswith(prefix) or not column.endswith(suffix):
            continue
        epoch_text = column[len(prefix): -len(suffix)]
        if epoch_text.isdigit():
            epochs.append(int(epoch_text))
    return max(epochs) if epochs else 0


def select_dynamic_schedule(
    evaluations: pd.DataFrame,
    *,
    mode: str,
    schedule_id: str | None = None,
) -> pd.Series:
    _require_columns(evaluations, ["schedule_id", "phi", "feasible", "status"], "dynamic evaluations")
    ok = evaluations[evaluations["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No successful dynamic evaluations are available")
    if mode == "best_phi":
        return ok.sort_values(["phi", "schedule_id"]).iloc[0]
    if mode == "best_feasible":
        feasible = ok[ok["feasible"].astype(bool)].copy()
        if feasible.empty:
            raise ValueError("No direct-feasible dynamic schedules are available")
        return feasible.sort_values(["phi", "schedule_id"]).iloc[0]
    if mode == "schedule_id":
        if schedule_id is None:
            raise ValueError("schedule_id must be supplied when mode='schedule_id'")
        matches = ok[ok["schedule_id"].astype(str) == str(schedule_id)]
        if matches.empty:
            raise ValueError(f"No dynamic evaluation has schedule_id={schedule_id!r}")
        return matches.iloc[0]
    raise ValueError("mode must be one of 'best_phi', 'best_feasible', or 'schedule_id'")


def dynamic_schedule_from_row(
    row: pd.Series | Mapping[str, Any],
    config: ViabilityConfig,
    *,
    epoch_count: int,
) -> EpochPolicySchedule:
    total_phases = config.model.years_to_run * 3
    values = {}
    raw_values = {}
    for name in dynamic_feature_names(config.policy, epoch_count):
        if name not in row:
            raise ValueError(f"Dynamic row is missing schedule value {name!r}")
        values[name] = row[name]
        raw_column = f"raw_{name}"
        if raw_column in row:
            raw_values[name] = float(row[raw_column])
    return EpochPolicySchedule.from_flat_mapping(
        values,
        config.policy,
        epoch_count=epoch_count,
        total_phases=total_phases,
        raw_values=raw_values if raw_values else None,
    )


def dynamic_epoch_table(
    row: pd.Series | Mapping[str, Any],
    config: ViabilityConfig,
    *,
    epoch_count: int,
) -> pd.DataFrame:
    rows = []
    for epoch_index in range(epoch_count):
        prefix = f"epoch{epoch_index + 1}"
        table_row: dict[str, Any] = {"epoch": epoch_index + 1}
        for name in config.policy.variables:
            column = f"{prefix}_{name}"
            if column not in row:
                raise ValueError(f"Dynamic row is missing schedule value {column!r}")
            table_row[name] = row[column]
        rows.append(table_row)
    return pd.DataFrame(rows)


def nearest_dynamic_misses(evaluations: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    _require_columns(evaluations, ["schedule_id", "phi", "feasible", "status"], "dynamic evaluations")
    ok = evaluations[evaluations["status"] == "ok"].copy()
    if ok.empty:
        return ok
    constraint_columns = [column for column in ok.columns if column.startswith("constraint_")]
    if constraint_columns:
        ok.loc[:, "positive_constraint_sum"] = ok[constraint_columns].clip(lower=0.0).sum(axis=1)
    else:
        ok.loc[:, "positive_constraint_sum"] = np.nan
    return ok.sort_values(["phi", "positive_constraint_sum", "schedule_id"]).head(top_n)


def constraint_relaxation_table(row: pd.Series | Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for column, value in row.items():
        if not str(column).startswith("constraint_"):
            continue
        numeric = float(value)
        if numeric > 0.0:
            rows.append(
                {
                    "constraint": str(column).removeprefix("constraint_"),
                    "required_relaxation": numeric,
                }
            )
    return pd.DataFrame(rows, columns=["constraint", "required_relaxation"]).sort_values(
        "required_relaxation",
        ascending=False,
    )


def select_dashboard_candidate(
    verified_candidates: pd.DataFrame,
    *,
    mode: str,
    candidate_id: str | None = None,
) -> pd.Series:
    _require_columns(
        verified_candidates,
        ["candidate_id", "phi", "feasible"],
        "verified candidates",
    )
    if mode == "near_boundary_feasible":
        feasible = _feasible_candidates(verified_candidates)
        ranked = feasible.assign(_abs_phi=feasible["phi"].astype(float).abs())
        return ranked.sort_values(["_abs_phi", "phi", "candidate_id"]).iloc[0].drop(
            labels=["_abs_phi"]
        )
    if mode == "best_verified":
        sort_keys = ["phi", "design_id"]
        if "candidate_id" in verified_candidates.columns:
            sort_keys = ["phi", "candidate_id", "design_id"]
        return verified_candidates.sort_values(sort_keys).iloc[0]
    if mode == "best_margin_feasible":
        feasible = _feasible_candidates(verified_candidates)
        return feasible.sort_values(["phi", "candidate_id"]).iloc[0]
    if mode == "candidate_id":
        if candidate_id is None:
            raise ValueError("candidate_id must be supplied when mode='candidate_id'")
        matches = verified_candidates.loc[
            verified_candidates["candidate_id"].astype(str) == str(candidate_id)
        ]
        if matches.empty:
            raise ValueError(f"No verified candidate has candidate_id={candidate_id!r}")
        return matches.iloc[0]
    raise ValueError(
        "mode must be one of 'near_boundary_feasible', "
        "'best_verified', 'best_margin_feasible', or 'candidate_id'"
    )


def policy_values_from_row(
    row: pd.Series | Mapping[str, Any], config: ViabilityConfig
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in config.policy.variables:
        if name in row:
            values[name] = row[name]
        elif f"applied_{name}" in row:
            values[name] = row[f"applied_{name}"]
        else:
            raise ValueError(f"Candidate row is missing policy value {name!r}")
    return PolicyDesign.from_mapping(values, config.policy).to_dict()


def policy_frame_from_values(
    values: Mapping[str, Any], config: ViabilityConfig
) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for name in config.policy.variables:
        if name not in values:
            raise ValueError(f"Policy values are missing {name!r}")
        raw_value = float(values[name])
        applied = apply_policy_value(raw_value, config, name)
        row[f"raw_{name}"] = raw_value
        row[f"applied_{name}"] = applied
        row[name] = applied
    return pd.DataFrame([row])


def score_policy_values(
    surrogate: dict[str, Any],
    config: ViabilityConfig,
    values: Mapping[str, Any],
    *,
    conservative_sigma: float,
) -> pd.Series:
    frame = policy_frame_from_values(values, config)
    prediction = predict_policy_frame(
        surrogate,
        frame,
        config,
        conservative_sigma=conservative_sigma,
        chunk_size=1,
    )
    return prediction.iloc[0]


def one_lever_sweep(
    config: ViabilityConfig,
    base_values: Mapping[str, Any],
    lever: str,
    *,
    max_points: int = 121,
) -> pd.DataFrame:
    if lever not in config.policy.variables:
        raise ValueError(f"Unknown policy lever {lever!r}")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")

    variable = config.policy.variables[lever]
    if variable.type == "int":
        low = int(np.ceil(variable.low))
        high = int(np.floor(variable.high))
        raw_values = np.arange(low, high + 1, dtype=float)
        if len(raw_values) > max_points:
            raw_values = np.unique(np.rint(np.linspace(low, high, max_points))).astype(float)
    else:
        raw_values = np.linspace(variable.low, variable.high, max_points)

    rows = []
    for raw_value in raw_values:
        values = dict(base_values)
        values[lever] = float(raw_value)
        policy_frame = policy_frame_from_values(values, config)
        rows.append(policy_frame.to_dict(orient="records")[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def local_feasible_sweep(
    surrogate: dict[str, Any],
    config: ViabilityConfig,
    base_values: Mapping[str, Any],
    lever: str,
    *,
    conservative_sigma: float,
    max_points: int = 121,
) -> SliderSweepResult:
    sweep = one_lever_sweep(config, base_values, lever, max_points=max_points)
    prediction = predict_policy_frame(
        surrogate,
        sweep,
        config,
        conservative_sigma=conservative_sigma,
        chunk_size=max_points,
    )
    scored = pd.concat([sweep.reset_index(drop=True), prediction], axis=1)
    intervals = feasible_intervals(
        scored,
        lever,
        feasible_column="conservative_phi",
        threshold=0.0,
    )
    return SliderSweepResult(lever=lever, sweep=scored, intervals=intervals)


def feasible_intervals(
    scored_sweep: pd.DataFrame,
    lever: str,
    *,
    feasible_column: str,
    threshold: float,
) -> list[SliderInterval]:
    value_column = f"raw_{lever}" if f"raw_{lever}" in scored_sweep.columns else lever
    _require_columns(scored_sweep, [value_column, feasible_column], "scored sweep")
    ordered = scored_sweep.sort_values(value_column).reset_index(drop=True)
    feasible_mask = ordered[feasible_column].astype(float) <= float(threshold)
    intervals: list[SliderInterval] = []
    start_index: int | None = None
    for index, is_feasible in enumerate(feasible_mask.to_list()):
        if is_feasible and start_index is None:
            start_index = index
        next_infeasible = not is_feasible
        last_row = index == len(ordered) - 1
        if start_index is not None and (next_infeasible or last_row):
            end_index = index if is_feasible and last_row else index - 1
            section = ordered.iloc[start_index:end_index + 1]
            intervals.append(
                SliderInterval(
                    lever=lever,
                    low=float(section[value_column].min()),
                    high=float(section[value_column].max()),
                    n_points=int(len(section)),
                )
            )
            start_index = None
    return intervals


def aggregate_history_trajectory(
    history: pd.DataFrame,
    config: ViabilityConfig,
) -> pd.DataFrame:
    required = [
        "year",
        "phase",
        "total_pilots",
        "line_pilots",
        "staff_ips",
        "staff_fls",
        "wg_rap_shortfall",
        "fl_rap_shortfall",
        "ip_rap_shortfall",
    ]
    _require_columns(history, required, "simulation history")
    if history.empty:
        raise ValueError("simulation history is empty")

    frame = history.copy()
    groups = frame.groupby(["year", "phase"], sort=True)
    agg_spec: dict[str, tuple[str, str]] = {
        "total_pilots": ("total_pilots", "sum"),
        "line_pilots": ("line_pilots", "sum"),
        "staff_ips": ("staff_ips", "sum"),
        "staff_fls": ("staff_fls", "sum"),
        "wg_rap_margin": ("wg_rap_shortfall", "mean"),
        "fl_rap_margin": ("fl_rap_shortfall", "mean"),
        "ip_rap_margin": ("ip_rap_shortfall", "mean"),
    }
    for constraint_name, history_column, _requirement_field in UTC_CONSTRAINT_SPECS:
        if history_column in frame.columns:
            agg_spec[f"{constraint_name}_margin"] = (history_column, "mean")
    if "unallocated_iron" in frame.columns:
        agg_spec["caf_unallocated_iron"] = ("unallocated_iron", "sum")
    trajectory = groups.agg(**agg_spec).reset_index()

    if {"fl_qty", "ip_qty"}.issubset(frame.columns):
        experienced = groups[["fl_qty", "ip_qty"]].sum().sum(axis=1).reset_index(drop=True)
        line = trajectory["line_pilots"].replace(0, np.nan)
        trajectory.loc[:, "experience_ratio"] = (experienced / line).fillna(0.0)
    elif "exp_rat" in frame.columns:
        weighted = frame.assign(_weighted_exp=frame["exp_rat"] * frame["line_pilots"])
        weighted_groups = weighted.groupby(["year", "phase"], sort=True)
        numerator = weighted_groups["_weighted_exp"].sum().reset_index(drop=True)
        line = trajectory["line_pilots"].replace(0, np.nan)
        trajectory.loc[:, "experience_ratio"] = (numerator / line).fillna(0.0)
    else:
        trajectory.loc[:, "experience_ratio"] = np.nan

    active_names = []
    active_values = []
    phi_values = []
    feasible_values = []
    for _, row in trajectory.iterrows():
        constraints = per_phase_constraints(row, config)
        phi, active_name, active_value = aggregate_violation(
            constraints,
            config.constraint_scales,
        )
        active_names.append(active_name)
        active_values.append(active_value)
        phi_values.append(phi)
        feasible_values.append(phi <= 0.0)

    trajectory.loc[:, "phi"] = phi_values
    trajectory.loc[:, "feasible"] = feasible_values
    trajectory.loc[:, "active_constraint"] = active_names
    trajectory.loc[:, "active_constraint_value"] = active_values
    trajectory.loc[:, "timeline"] = (
        trajectory["year"].astype(str) + " P" + trajectory["phase"].astype(str)
    )
    return trajectory


def per_phase_constraints(row: pd.Series, config: ViabilityConfig) -> dict[str, float]:
    requirements = config.requirements
    constraints: dict[str, float] = {}
    if requirements.target_total_pilots is not None:
        constraints["total_pilots_window"] = (
            requirements.target_total_pilots - float(row["total_pilots"])
        )
    if requirements.target_line_pilots is not None:
        constraints["line_pilots_window"] = (
            requirements.target_line_pilots - float(row["line_pilots"])
        )
    if requirements.allowed_wg_rap_shortfall is not None:
        constraints["wg_rap"] = (
            float(row["wg_rap_margin"]) - requirements.allowed_wg_rap_shortfall
        )
    if requirements.allowed_fl_rap_shortfall is not None:
        constraints["fl_rap"] = (
            float(row["fl_rap_margin"]) - requirements.allowed_fl_rap_shortfall
        )
    if requirements.allowed_ip_rap_shortfall is not None:
        constraints["ip_rap"] = (
            float(row["ip_rap_margin"]) - requirements.allowed_ip_rap_shortfall
        )
    for constraint_name, _history_column, requirement_field in UTC_CONSTRAINT_SPECS:
        allowed = getattr(requirements, requirement_field)
        margin_column = f"{constraint_name}_margin"
        if allowed is not None:
            constraints[constraint_name] = float(row[margin_column]) - allowed
    if requirements.target_staff_ips is not None:
        constraints["staff_ips"] = requirements.target_staff_ips - float(row["staff_ips"])
    if requirements.target_staff_fls is not None:
        constraints["staff_fls"] = requirements.target_staff_fls - float(row["staff_fls"])
    if requirements.min_experience_ratio is not None:
        constraints["experience_ratio"] = (
            requirements.min_experience_ratio - float(row["experience_ratio"])
        )
    if requirements.allowed_unallocated_iron is not None:
        constraints["unallocated_iron"] = (
            float(row["caf_unallocated_iron"]) - requirements.allowed_unallocated_iron
        )
    return constraints


def run_direct_policy(
    values: Mapping[str, Any],
    config: ViabilityConfig,
    *,
    seed: int | None = None,
) -> DirectPolicyResult:
    design = PolicyDesign.from_mapping(values, config.policy)
    try:
        history = simulate_design_history(design, config, seed=seed)
        raw_metrics = compute_raw_metrics(history, config.model.assessment_start_year)
        constraints = compute_constraints(raw_metrics, config.requirements)
        phi, active_constraint, active_constraint_value = aggregate_violation(
            constraints,
            config.constraint_scales,
        )
        evaluation = EvaluationResult(
            phase_backend=config.model.phase_backend,
            design=design.to_dict(),
            raw_design=design.to_raw_dict(),
            applied_design=design.to_applied_dict(),
            raw_metrics=raw_metrics,
            constraints=constraints,
            phi=phi,
            feasible=phi <= 0.0,
            active_constraint=active_constraint,
            active_constraint_value=active_constraint_value,
            status="ok",
        )
        trajectory = aggregate_history_trajectory(history, config)
        return DirectPolicyResult(
            evaluation=evaluation,
            history=history,
            trajectory=trajectory,
        )
    except Exception as exc:
        evaluation = EvaluationResult(
            phase_backend=config.model.phase_backend,
            design=design.to_dict(),
            raw_design=design.to_raw_dict(),
            applied_design=design.to_applied_dict(),
            raw_metrics={},
            constraints={},
            phi=float("inf"),
            feasible=False,
            active_constraint=None,
            active_constraint_value=None,
            status="failed",
            error=str(exc),
        )
        return DirectPolicyResult(
            evaluation=evaluation,
            history=pd.DataFrame(),
            trajectory=pd.DataFrame(),
        )


def run_direct_dynamic_schedule(
    row: pd.Series | Mapping[str, Any],
    config: ViabilityConfig,
    *,
    epoch_count: int,
    seed: int | None = None,
) -> DirectPolicyResult:
    schedule = dynamic_schedule_from_row(row, config, epoch_count=epoch_count)
    try:
        history = simulate_policy_schedule_history(schedule, config, seed=seed)
        raw_metrics = compute_raw_metrics(history, config.model.assessment_start_year)
        constraints = compute_constraints(raw_metrics, config.requirements)
        phi, active_constraint, active_constraint_value = aggregate_violation(
            constraints,
            config.constraint_scales,
        )
        evaluation = EvaluationResult(
            phase_backend=config.model.phase_backend,
            design=schedule.to_flat_dict(raw=False),
            raw_design=schedule.to_flat_dict(raw=True),
            applied_design=schedule.to_flat_dict(raw=False),
            raw_metrics=raw_metrics,
            constraints=constraints,
            phi=phi,
            feasible=phi <= 0.0,
            active_constraint=active_constraint,
            active_constraint_value=active_constraint_value,
            status="ok",
        )
        trajectory = aggregate_history_trajectory(history, config)
        return DirectPolicyResult(
            evaluation=evaluation,
            history=history,
            trajectory=trajectory,
        )
    except Exception as exc:
        evaluation = EvaluationResult(
            phase_backend=config.model.phase_backend,
            design=schedule.to_flat_dict(raw=False),
            raw_design=schedule.to_flat_dict(raw=True),
            applied_design=schedule.to_flat_dict(raw=False),
            raw_metrics={},
            constraints={},
            phi=float("inf"),
            feasible=False,
            active_constraint=None,
            active_constraint_value=None,
            status="failed",
            error=str(exc),
        )
        return DirectPolicyResult(
            evaluation=evaluation,
            history=pd.DataFrame(),
            trajectory=pd.DataFrame(),
        )


def envelope_plot_paths(envelope_summary: Mapping[str, Any]) -> list[tuple[str, Path, Path]]:
    slices = envelope_summary.get("slices")
    if not isinstance(slices, list):
        raise ValueError("envelope_summary must include a list of slices")
    paths = []
    for slice_summary in slices:
        x_name = str(slice_summary["x"])
        y_name = str(slice_summary["y"])
        fixed = Path(str(slice_summary["fixed_plot_path"]))
        projected = Path(str(slice_summary["projected_plot_path"]))
        paths.append((f"{x_name} vs {y_name}", fixed, projected))
    return paths


def _read_json_object(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Required JSON artifact does not exist: {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {input_path}")
    return data


def _validate_envelope_paths(envelope_summary: Mapping[str, Any]) -> None:
    missing_paths = []
    for _label, fixed, projected in envelope_plot_paths(envelope_summary):
        for path in (fixed, projected):
            if not path.exists():
                missing_paths.append(str(path))
    if missing_paths:
        raise FileNotFoundError(
            "Envelope summary references missing plot files: "
            + ", ".join(missing_paths)
        )


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _feasible_candidates(verified_candidates: pd.DataFrame) -> pd.DataFrame:
    feasible = verified_candidates.loc[verified_candidates["feasible"].astype(bool)].copy()
    if feasible.empty:
        raise ValueError("No verified feasible candidates are available")
    return feasible


CONSTRAINT_COLUMN_PREFIX = "constraint_"


def constraint_name_from_column(column: str) -> str:
    if column.startswith(CONSTRAINT_COLUMN_PREFIX):
        return column[len(CONSTRAINT_COLUMN_PREFIX) :]
    return column


def available_constraint_columns(verified_candidates: pd.DataFrame) -> list[str]:
    """Return the per-constraint margin columns (``constraint_<name>``) in the table."""
    return sorted(
        column
        for column in verified_candidates.columns
        if column.startswith(CONSTRAINT_COLUMN_PREFIX)
    )


@dataclass(frozen=True)
class ConstraintGateResult:
    """Outcome of filtering candidates by a set of must-meet constraints.

    ``filtered`` only contains candidates that satisfy every must-meet constraint.
    It carries three added columns describing the worst-violated constraint *among
    the constraints that were not gated*, so the user can see the next binding
    requirement instead of the always-dominant gated ones:
    ``gated_binding_constraint``, ``gated_binding_value`` (raw margin), and
    ``gated_binding_normalized`` (margin / constraint scale).
    """

    filtered: pd.DataFrame
    must_meet: tuple[str, ...]
    total_count: int
    passed_count: int
    fully_feasible_count: int
    remaining_infeasible_count: int
    binding_counts: dict[str, int]


def apply_constraint_gate(
    verified_candidates: pd.DataFrame,
    *,
    must_meet: Sequence[str],
    config: ViabilityConfig,
    tolerance: float = 0.0,
) -> ConstraintGateResult:
    """Keep only candidates meeting every must-meet constraint and re-rank the rest.

    A constraint uses the ``g(x) <= 0`` satisfied convention, so a candidate
    passes the gate when every selected ``constraint_<name>`` is ``<= tolerance``.
    Among the remaining (non-gated) constraints, the worst normalized margin
    (``margin / constraint_scale``) is reported as the binding constraint.
    """
    all_columns = available_constraint_columns(verified_candidates)
    must_meet = tuple(must_meet)
    must_meet_columns = [f"{CONSTRAINT_COLUMN_PREFIX}{name}" for name in must_meet]
    missing = [column for column in must_meet_columns if column not in all_columns]
    if missing:
        raise ValueError(
            "Verified candidates are missing must-meet constraint columns: "
            + ", ".join(missing)
        )

    total_count = int(len(verified_candidates))
    if must_meet_columns:
        gate_pass = (verified_candidates[must_meet_columns] <= tolerance).all(axis=1)
    else:
        gate_pass = pd.Series(True, index=verified_candidates.index)
    filtered = verified_candidates.loc[gate_pass].copy()

    remaining_columns = [
        column for column in all_columns if column not in must_meet_columns
    ]
    if filtered.empty:
        return ConstraintGateResult(
            filtered=filtered,
            must_meet=must_meet,
            total_count=total_count,
            passed_count=0,
            fully_feasible_count=0,
            remaining_infeasible_count=0,
            binding_counts={},
        )

    if remaining_columns:
        scales = {
            column: config.constraint_scales.scale_for(
                constraint_name_from_column(column)
            )
            for column in remaining_columns
        }
        normalized = filtered[remaining_columns].div(pd.Series(scales), axis=1)
        binding_column = normalized.idxmax(axis=1)
        binding_normalized = normalized.max(axis=1)
        binding_name = binding_column.map(constraint_name_from_column)
        binding_value = pd.Series(
            [filtered.at[index, column] for index, column in binding_column.items()],
            index=filtered.index,
            dtype=float,
        )
        remaining_feasible = (filtered[remaining_columns] <= 0.0).all(axis=1)
    else:
        binding_name = pd.Series("none", index=filtered.index)
        binding_value = pd.Series(np.nan, index=filtered.index)
        binding_normalized = pd.Series(np.nan, index=filtered.index)
        remaining_feasible = pd.Series(True, index=filtered.index)

    filtered["gated_binding_constraint"] = binding_name
    filtered["gated_binding_value"] = binding_value
    filtered["gated_binding_normalized"] = binding_normalized
    filtered["gated_feasible"] = remaining_feasible

    fully_feasible_count = int(remaining_feasible.sum())
    binding_counts = (
        binding_name.loc[~remaining_feasible].value_counts().sort_index().to_dict()
    )
    binding_counts = {str(name): int(count) for name, count in binding_counts.items()}

    return ConstraintGateResult(
        filtered=filtered,
        must_meet=must_meet,
        total_count=total_count,
        passed_count=int(len(filtered)),
        fully_feasible_count=fully_feasible_count,
        remaining_infeasible_count=int(len(filtered) - fully_feasible_count),
        binding_counts=binding_counts,
    )
