from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.simulation_config import SimulationConfig

STANDARD_POLICY_VARIABLES = frozenset(
    {
        "annual_intake",
        "retention_rate",
        "ute",
        "paa",
        "max_manning_pct",
        "flug_quota_per_phase",
        "ipug_quota_per_phase",
        "upgrade_sortie_fraction",
        "flug_window_start",
        "ipug_window_start",
    }
)

_TOP_LEVEL_SECTIONS = ("run", "model", "requirements", "constraint_scales", "policy", "doe")

_RUN_FIELDS = ("name", "random_seed", "output_dir", "workers")
_MODEL_FIELDS = (
    "years_to_run",
    "start_year",
    "assessment_start_year",
    "target_year",
    "round_robin",
    "use_upgrade_quotas",
    "staff_priority_mode",
    "n_replications",
)
_MODEL_OPTIONAL_FIELDS = ("phase_backend", "brain_path", "expected_brain_outputs", "simulation")
_SIMULATION_FIELDS = (
    "phase_length_days",
    "allocation_noise",
    "utc_wise_allocation",
)
_REQUIREMENTS_FIELDS = (
    "target_total_pilots",
    "target_line_pilots",
    "min_experience_ratio",
    "allowed_wg_rap_shortfall",
    "allowed_fl_rap_shortfall",
    "allowed_ip_rap_shortfall",
    "allowed_utc_1_wg_shortfall",
    "allowed_utc_1_fl_shortfall",
    "allowed_utc_2_wg_shortfall",
    "allowed_utc_2_fl_shortfall",
    "target_staff_ips",
    "target_staff_fls",
    "allowed_unallocated_iron",
)
_CONSTRAINT_SCALE_FIELDS = (
    "total_pilots",
    "line_pilots",
    "wg_rap",
    "fl_rap",
    "ip_rap",
    "utc_1_wg",
    "utc_1_fl",
    "utc_2_wg",
    "utc_2_fl",
    "staff_ips",
    "staff_fls",
    "experience_ratio",
    "unallocated_iron",
)
_POLICY_FIELDS = ("parameterization", "variables")
_DOE_FIELDS = (
    "method",
    "n_initial",
    "start_index",
    "scramble",
    "include_corners",
    "include_baselines",
    "include_corners_on_resume",
    "include_baselines_on_resume",
    "baselines",
)
_ACTIVE_LEARNING_FIELDS = (
    "candidate_method",
    "candidate_start_index",
    "candidate_pool_size",
    "iterations",
    "batch_size",
    "acquisition",
    "boundary_batch_fraction",
    "min_normalized_distance",
    "candidate_report_rows",
)
_SEARCH_FIELDS = (
    "candidate_method",
    "candidate_start_index",
    "candidate_pool_size",
    "n_candidates_to_verify",
    "conservative_sigma",
    "min_normalized_distance",
    "candidate_report_rows",
)
_ENVELOPE_FIELDS = (
    "anchor",
    "conservative_sigma",
    "grid_size",
    "prediction_chunk_size",
    "sobol_hidden_samples",
    "sobol_hidden_start_index",
    "de_compare_enabled",
    "de_compare_points_per_slice",
    "de_maxiter",
    "de_popsize",
    "de_polish",
    "slices",
)
_ENVELOPE_SLICE_FIELDS = ("x", "y")
_REPORT_FIELDS = ("top_candidate_count", "near_boundary_count")
_VARIABLE_FIELDS = ("type", "low", "high")


def _require_mapping(data: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if path not in data:
        raise ValueError(f"Missing required config key: {path}")
    value = data[path]
    if not isinstance(value, Mapping):
        raise ValueError(f"Config key {path} must be a mapping")
    return value


def _require_key(section: Mapping[str, Any], section_name: str, key: str) -> Any:
    dotted = f"{section_name}.{key}"
    if key not in section:
        raise ValueError(f"Missing required config key: {dotted}")
    return section[key]


@dataclass(frozen=True)
class RunConfig:
    name: str
    random_seed: int
    output_dir: str
    workers: int

    def __post_init__(self) -> None:
        if self.workers <= 0:
            raise ValueError("run.workers must be positive")


@dataclass(frozen=True)
class ModelConfig:
    phase_backend: str
    years_to_run: int
    start_year: int
    assessment_start_year: int
    target_year: int
    brain_path: str | None
    expected_brain_outputs: int | None
    round_robin: bool
    use_upgrade_quotas: bool
    staff_priority_mode: str
    n_replications: int
    simulation: SimulationConfig

    def __post_init__(self) -> None:
        if self.phase_backend not in {"brain", "physics"}:
            raise ValueError("model.phase_backend must be either 'brain' or 'physics'")


@dataclass(frozen=True)
class RequirementsConfig:
    target_total_pilots: float | None
    target_line_pilots: float | None
    min_experience_ratio: float | None
    allowed_wg_rap_shortfall: float | None
    allowed_fl_rap_shortfall: float | None
    allowed_ip_rap_shortfall: float | None
    allowed_utc_1_wg_shortfall: float | None
    allowed_utc_1_fl_shortfall: float | None
    allowed_utc_2_wg_shortfall: float | None
    allowed_utc_2_fl_shortfall: float | None
    target_staff_ips: float | None
    target_staff_fls: float | None
    allowed_unallocated_iron: float | None


@dataclass(frozen=True)
class ConstraintScalesConfig:
    total_pilots: float
    line_pilots: float
    wg_rap: float
    fl_rap: float
    ip_rap: float
    utc_1_wg: float
    utc_1_fl: float
    utc_2_wg: float
    utc_2_fl: float
    staff_ips: float
    staff_fls: float
    experience_ratio: float
    unallocated_iron: float

    def scale_for(self, constraint_name: str) -> float:
        if constraint_name.startswith("total_pilots"):
            return self.total_pilots
        if constraint_name.startswith("line_pilots"):
            return self.line_pilots
        if constraint_name == "wg_rap":
            return self.wg_rap
        if constraint_name == "fl_rap":
            return self.fl_rap
        if constraint_name == "ip_rap":
            return self.ip_rap
        if constraint_name == "utc_1_wg":
            return self.utc_1_wg
        if constraint_name == "utc_1_fl":
            return self.utc_1_fl
        if constraint_name == "utc_2_wg":
            return self.utc_2_wg
        if constraint_name == "utc_2_fl":
            return self.utc_2_fl
        if constraint_name == "staff_ips":
            return self.staff_ips
        if constraint_name == "staff_fls":
            return self.staff_fls
        if constraint_name == "experience_ratio":
            return self.experience_ratio
        if constraint_name == "unallocated_iron":
            return self.unallocated_iron
        raise KeyError(f"No constraint scale configured for {constraint_name!r}")


@dataclass(frozen=True)
class VariableConfig:
    type: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.type not in {"int", "float"}:
            raise ValueError(f"Unsupported variable type {self.type!r}")
        if self.low > self.high:
            raise ValueError(f"Variable lower bound {self.low} exceeds upper bound {self.high}")


@dataclass(frozen=True)
class PolicyConfig:
    parameterization: str
    variables: dict[str, VariableConfig]

    def __post_init__(self) -> None:
        if self.parameterization != "constant":
            raise ValueError("Only constant policy parameterization is implemented in this prototype")


@dataclass(frozen=True)
class DoeConfig:
    method: str
    n_initial: int
    start_index: int
    scramble: bool
    include_corners: bool
    include_baselines: bool
    include_corners_on_resume: bool
    include_baselines_on_resume: bool
    baselines: list[dict[str, float]]

    def __post_init__(self) -> None:
        if self.n_initial < 0:
            raise ValueError("doe.n_initial must be non-negative")
        if self.start_index < 0:
            raise ValueError("doe.start_index must be non-negative")
        if self.method not in {"random", "sobol", "latin_hypercube", "lhs"}:
            raise ValueError(f"Unsupported DOE method {self.method!r}")
        if self.start_index > 0 and self.method in {"latin_hypercube", "lhs"}:
            raise ValueError(
                "doe.start_index resume semantics are implemented for sobol and random only"
            )


@dataclass(frozen=True)
class ActiveLearningConfig:
    candidate_method: str
    candidate_start_index: int
    candidate_pool_size: int
    iterations: int
    batch_size: int
    acquisition: str
    boundary_batch_fraction: float
    min_normalized_distance: float
    candidate_report_rows: int

    def __post_init__(self) -> None:
        if self.candidate_method != "sobol":
            raise ValueError("active_learning.candidate_method currently supports only 'sobol'")
        if self.candidate_start_index < 0:
            raise ValueError("active_learning.candidate_start_index must be non-negative")
        if self.candidate_pool_size <= 0:
            raise ValueError("active_learning.candidate_pool_size must be positive")
        if self.iterations < 0:
            raise ValueError("active_learning.iterations must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("active_learning.batch_size must be positive")
        if self.acquisition not in {"uncertainty", "boundary_stratified_uncertainty"}:
            raise ValueError(
                "active_learning.acquisition supports only 'uncertainty' "
                "and 'boundary_stratified_uncertainty'"
            )
        if not 0.0 <= self.boundary_batch_fraction <= 1.0:
            raise ValueError("active_learning.boundary_batch_fraction must be between 0 and 1")
        if self.acquisition == "uncertainty" and self.boundary_batch_fraction != 0.0:
            raise ValueError(
                "active_learning.boundary_batch_fraction must be 0.0 when acquisition='uncertainty'"
            )
        if self.acquisition == "boundary_stratified_uncertainty" and self.boundary_batch_fraction <= 0.0:
            raise ValueError(
                "active_learning.boundary_batch_fraction must be positive when "
                "acquisition='boundary_stratified_uncertainty'"
            )
        if self.min_normalized_distance < 0.0:
            raise ValueError("active_learning.min_normalized_distance must be non-negative")
        if self.candidate_report_rows <= 0:
            raise ValueError("active_learning.candidate_report_rows must be positive")


@dataclass(frozen=True)
class SearchConfig:
    candidate_method: str
    candidate_start_index: int
    candidate_pool_size: int
    n_candidates_to_verify: int
    conservative_sigma: float
    min_normalized_distance: float
    candidate_report_rows: int

    def __post_init__(self) -> None:
        if self.candidate_method != "sobol":
            raise ValueError("search.candidate_method currently supports only 'sobol'")
        if self.candidate_start_index < 0:
            raise ValueError("search.candidate_start_index must be non-negative")
        if self.candidate_pool_size <= 0:
            raise ValueError("search.candidate_pool_size must be positive")
        if self.n_candidates_to_verify <= 0:
            raise ValueError("search.n_candidates_to_verify must be positive")
        if self.conservative_sigma < 0.0:
            raise ValueError("search.conservative_sigma must be non-negative")
        if self.min_normalized_distance < 0.0:
            raise ValueError("search.min_normalized_distance must be non-negative")
        if self.candidate_report_rows <= 0:
            raise ValueError("search.candidate_report_rows must be positive")


@dataclass(frozen=True)
class EnvelopeSliceConfig:
    x: str
    y: str

    def __post_init__(self) -> None:
        if self.x == self.y:
            raise ValueError("envelope.slices entries must use distinct x and y variables")


@dataclass(frozen=True)
class EnvelopeConfig:
    anchor: str
    conservative_sigma: float
    grid_size: int
    prediction_chunk_size: int
    sobol_hidden_samples: int
    sobol_hidden_start_index: int
    de_compare_enabled: bool
    de_compare_points_per_slice: int
    de_maxiter: int
    de_popsize: int
    de_polish: bool
    slices: list[EnvelopeSliceConfig]

    def __post_init__(self) -> None:
        if self.anchor != "near_boundary_feasible":
            raise ValueError("envelope.anchor currently supports only 'near_boundary_feasible'")
        if self.conservative_sigma < 0.0:
            raise ValueError("envelope.conservative_sigma must be non-negative")
        if self.grid_size < 2:
            raise ValueError("envelope.grid_size must be at least 2")
        if self.prediction_chunk_size <= 0:
            raise ValueError("envelope.prediction_chunk_size must be positive")
        if self.sobol_hidden_samples <= 0:
            raise ValueError("envelope.sobol_hidden_samples must be positive")
        if self.sobol_hidden_start_index < 0:
            raise ValueError("envelope.sobol_hidden_start_index must be non-negative")
        if self.de_compare_points_per_slice < 0:
            raise ValueError("envelope.de_compare_points_per_slice must be non-negative")
        if self.de_compare_enabled and self.de_compare_points_per_slice <= 0:
            raise ValueError(
                "envelope.de_compare_points_per_slice must be positive when "
                "envelope.de_compare_enabled is true"
            )
        if self.de_maxiter <= 0:
            raise ValueError("envelope.de_maxiter must be positive")
        if self.de_popsize <= 0:
            raise ValueError("envelope.de_popsize must be positive")
        if not self.slices:
            raise ValueError("envelope.slices must include at least one slice")


@dataclass(frozen=True)
class ReportConfig:
    top_candidate_count: int
    near_boundary_count: int

    def __post_init__(self) -> None:
        if self.top_candidate_count <= 0:
            raise ValueError("report.top_candidate_count must be positive")
        if self.near_boundary_count <= 0:
            raise ValueError("report.near_boundary_count must be positive")


@dataclass(frozen=True)
class ViabilityConfig:
    run: RunConfig
    model: ModelConfig
    requirements: RequirementsConfig
    constraint_scales: ConstraintScalesConfig
    policy: PolicyConfig
    doe: DoeConfig
    active_learning: ActiveLearningConfig | None = None
    search: SearchConfig | None = None
    envelope: EnvelopeConfig | None = None
    report: ReportConfig | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ViabilityConfig":
        for section in _TOP_LEVEL_SECTIONS:
            _require_mapping(data, section)

        run_data = _require_mapping(data, "run")
        model_data = _require_mapping(data, "model")
        requirements_data = _require_mapping(data, "requirements")
        scales_data = _require_mapping(data, "constraint_scales")
        policy_data = _require_mapping(data, "policy")
        doe_data = _require_mapping(data, "doe")

        run = RunConfig(
            **{key: _require_key(run_data, "run", key) for key in _RUN_FIELDS}
        )
        simulation_data = model_data.get("simulation", {})
        if simulation_data is None:
            simulation_data = {}
        if not isinstance(simulation_data, Mapping):
            raise ValueError("Config key model.simulation must be a mapping")
        simulation_defaults = asdict(SimulationConfig())
        simulation_values = {}
        for key in _SIMULATION_FIELDS:
            simulation_values[key] = simulation_data.get(key, simulation_defaults[key])

        model_values = {key: _require_key(model_data, "model", key) for key in _MODEL_FIELDS}
        for key in _MODEL_OPTIONAL_FIELDS:
            if key == "simulation":
                continue
            model_values[key] = model_data.get(key)
        if model_values["phase_backend"] is None:
            model_values["phase_backend"] = "brain"
        model = ModelConfig(
            **model_values,
            simulation=SimulationConfig(**simulation_values),
        )
        requirements = RequirementsConfig(
            **{
                key: _require_key(requirements_data, "requirements", key)
                for key in _REQUIREMENTS_FIELDS
            }
        )
        constraint_scales = ConstraintScalesConfig(
            **{
                key: _require_key(scales_data, "constraint_scales", key)
                for key in _CONSTRAINT_SCALE_FIELDS
            }
        )

        parameterization = _require_key(policy_data, "policy", "parameterization")
        variables_raw = _require_key(policy_data, "policy", "variables")
        if not isinstance(variables_raw, Mapping):
            raise ValueError("policy.variables must be a mapping")
        variables = {}
        for name, value in variables_raw.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"policy.variables.{name} must be a mapping")
            variables[name] = VariableConfig(
                **{
                    key: _require_key(value, f"policy.variables.{name}", key)
                    for key in _VARIABLE_FIELDS
                }
            )
        policy = PolicyConfig(parameterization=parameterization, variables=variables)

        baselines = _require_key(doe_data, "doe", "baselines")
        if not isinstance(baselines, list):
            raise ValueError("doe.baselines must be a list")
        doe = DoeConfig(
            method=_require_key(doe_data, "doe", "method"),
            n_initial=_require_key(doe_data, "doe", "n_initial"),
            start_index=_require_key(doe_data, "doe", "start_index"),
            scramble=_require_key(doe_data, "doe", "scramble"),
            include_corners=_require_key(doe_data, "doe", "include_corners"),
            include_baselines=_require_key(doe_data, "doe", "include_baselines"),
            include_corners_on_resume=_require_key(
                doe_data,
                "doe",
                "include_corners_on_resume",
            ),
            include_baselines_on_resume=_require_key(
                doe_data,
                "doe",
                "include_baselines_on_resume",
            ),
            baselines=baselines,
        )

        active_learning = None
        if "active_learning" in data:
            active_learning_data = _require_mapping(data, "active_learning")
            active_learning = ActiveLearningConfig(
                **{
                    key: _require_key(active_learning_data, "active_learning", key)
                    for key in _ACTIVE_LEARNING_FIELDS
                }
            )

        search = None
        if "search" in data:
            search_data = _require_mapping(data, "search")
            search = SearchConfig(
                **{
                    key: _require_key(search_data, "search", key)
                    for key in _SEARCH_FIELDS
                }
            )

        envelope = None
        if "envelope" in data:
            envelope_data = _require_mapping(data, "envelope")
            envelope_slices_raw = _require_key(envelope_data, "envelope", "slices")
            if not isinstance(envelope_slices_raw, list):
                raise ValueError("envelope.slices must be a list")
            envelope_slices = []
            for index, value in enumerate(envelope_slices_raw):
                if not isinstance(value, Mapping):
                    raise ValueError(f"envelope.slices[{index}] must be a mapping")
                envelope_slices.append(
                    EnvelopeSliceConfig(
                        **{
                            key: _require_key(value, f"envelope.slices[{index}]", key)
                            for key in _ENVELOPE_SLICE_FIELDS
                        }
                    )
                )
            envelope_values = {
                key: _require_key(envelope_data, "envelope", key)
                for key in _ENVELOPE_FIELDS
                if key != "slices"
            }
            envelope = EnvelopeConfig(**envelope_values, slices=envelope_slices)

        report = None
        if "report" in data:
            report_data = _require_mapping(data, "report")
            report = ReportConfig(
                **{
                    key: _require_key(report_data, "report", key)
                    for key in _REPORT_FIELDS
                }
            )

        config = cls(
            run=run,
            model=model,
            requirements=requirements,
            constraint_scales=constraint_scales,
            policy=policy,
            doe=doe,
            active_learning=active_learning,
            search=search,
            envelope=envelope,
            report=report,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model.years_to_run <= 0:
            raise ValueError("model.years_to_run must be positive")
        if self.model.n_replications != 1:
            raise ValueError("Only model.n_replications=1 is implemented in this first slice")
        if self.model.phase_backend == "brain":
            if not self.model.brain_path:
                raise ValueError(
                    "model.brain_path is required when model.phase_backend='brain'"
                )
            if self.model.expected_brain_outputs is None:
                raise ValueError(
                    "model.expected_brain_outputs is required when model.phase_backend='brain'"
                )
            if self.model.expected_brain_outputs < 16:
                raise ValueError(
                    "The current CAFSimulation brain path requires a 16-output "
                    "internal surrogate layout"
                )
        if self.model.simulation.phase_length_days <= 0:
            raise ValueError("model.simulation.phase_length_days must be positive")
        if self.model.simulation.allocation_noise < 0.0:
            raise ValueError("model.simulation.allocation_noise must be non-negative")

        horizon_end = self.model.start_year + self.model.years_to_run - 1
        if not (
            self.model.assessment_start_year
            <= self.model.target_year
            <= horizon_end
        ):
            raise ValueError(
                "model.target_year must satisfy "
                "assessment_start_year <= target_year <= start_year + years_to_run - 1"
            )

        if set(self.policy.variables) != STANDARD_POLICY_VARIABLES:
            raise ValueError(
                "policy.variables must exactly match the prototype constant-policy levers: "
                f"{sorted(STANDARD_POLICY_VARIABLES)}"
            )

        if self.doe.include_baselines and not self.doe.baselines:
            raise ValueError("doe.include_baselines is true but doe.baselines is empty")

        self._validate_envelope_slices()
        self._validate_enabled_constraint_scales()

    def _validate_envelope_slices(self) -> None:
        if self.envelope is None:
            return
        variable_names = set(self.policy.variables)
        for slice_config in self.envelope.slices:
            missing = [name for name in (slice_config.x, slice_config.y) if name not in variable_names]
            if missing:
                raise ValueError(f"envelope.slices references unknown policy variables: {missing}")

    def _validate_enabled_constraint_scales(self) -> None:
        req = self.requirements
        enabled: list[tuple[str, str]] = []
        if req.target_total_pilots is not None:
            enabled.append(("total_pilots_final", "total_pilots"))
            enabled.append(("total_pilots_window", "total_pilots"))
        if req.target_line_pilots is not None:
            enabled.append(("line_pilots_window", "line_pilots"))
        if req.allowed_wg_rap_shortfall is not None:
            enabled.append(("wg_rap", "wg_rap"))
        if req.allowed_fl_rap_shortfall is not None:
            enabled.append(("fl_rap", "fl_rap"))
        if req.allowed_ip_rap_shortfall is not None:
            enabled.append(("ip_rap", "ip_rap"))
        if req.allowed_utc_1_wg_shortfall is not None:
            enabled.append(("utc_1_wg", "utc_1_wg"))
        if req.allowed_utc_1_fl_shortfall is not None:
            enabled.append(("utc_1_fl", "utc_1_fl"))
        if req.allowed_utc_2_wg_shortfall is not None:
            enabled.append(("utc_2_wg", "utc_2_wg"))
        if req.allowed_utc_2_fl_shortfall is not None:
            enabled.append(("utc_2_fl", "utc_2_fl"))
        if req.target_staff_ips is not None:
            enabled.append(("staff_ips", "staff_ips"))
        if req.target_staff_fls is not None:
            enabled.append(("staff_fls", "staff_fls"))
        if req.min_experience_ratio is not None:
            enabled.append(("experience_ratio", "experience_ratio"))
        if req.allowed_unallocated_iron is not None:
            enabled.append(("unallocated_iron", "unallocated_iron"))

        if not enabled:
            raise ValueError("At least one requirement constraint must be enabled (non-null)")

        for _constraint_name, scale_key in enabled:
            scale_value = getattr(self.constraint_scales, scale_key)
            if scale_value <= 0:
                raise ValueError(
                    f"constraint_scales.{scale_key} must be positive when the "
                    f"matching requirement is enabled"
                )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "run": asdict(self.run),
            "model": asdict(self.model),
            "requirements": asdict(self.requirements),
            "constraint_scales": asdict(self.constraint_scales),
            "policy": {
                "parameterization": self.policy.parameterization,
                "variables": {
                    name: asdict(variable)
                    for name, variable in self.policy.variables.items()
                },
            },
            "doe": asdict(self.doe),
        }
        if self.active_learning is not None:
            data["active_learning"] = asdict(self.active_learning)
        if self.search is not None:
            data["search"] = asdict(self.search)
        if self.envelope is not None:
            data["envelope"] = asdict(self.envelope)
        if self.report is not None:
            data["report"] = asdict(self.report)
        return data

    def dump_resolved_config(self, path: str | Path) -> Path:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyYAML is required to write viability YAML config files") from exc

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        return output_path


def load_config(path: str | Path) -> ViabilityConfig:
    config_path = Path(path)
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load viability YAML config files") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError(f"Config file {config_path} is empty")
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping")
    return ViabilityConfig.from_dict(data)
