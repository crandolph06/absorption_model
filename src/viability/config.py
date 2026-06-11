from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

STANDARD_POLICY_VARIABLES = frozenset(
    {
        "annual_intake",
        "retention_rate",
        "ute",
        "paa",
        "max_manning_pct",
        "flug_quota_per_phase",
        "ipug_quota_per_phase",
    }
)

_TOP_LEVEL_SECTIONS = ("run", "model", "requirements", "constraint_scales", "policy", "doe")

_RUN_FIELDS = ("name", "random_seed", "output_dir", "workers")
_MODEL_FIELDS = (
    "years_to_run",
    "start_year",
    "assessment_start_year",
    "target_year",
    "brain_path",
    "expected_brain_outputs",
    "round_robin",
    "use_upgrade_quotas",
    "staff_priority_mode",
    "n_replications",
)
_REQUIREMENTS_FIELDS = (
    "target_total_pilots",
    "target_line_pilots",
    "min_experience_ratio",
    "allowed_wg_rap_shortfall",
    "allowed_fl_rap_shortfall",
    "allowed_ip_rap_shortfall",
    "target_staff_ips",
    "target_staff_fls",
)
_CONSTRAINT_SCALE_FIELDS = (
    "total_pilots",
    "line_pilots",
    "wg_rap",
    "fl_rap",
    "ip_rap",
    "staff_ips",
    "staff_fls",
    "experience_ratio",
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


@dataclass(frozen=True)
class ModelConfig:
    years_to_run: int
    start_year: int
    assessment_start_year: int
    target_year: int
    brain_path: str
    expected_brain_outputs: int
    round_robin: bool
    use_upgrade_quotas: bool
    staff_priority_mode: str
    n_replications: int


@dataclass(frozen=True)
class RequirementsConfig:
    target_total_pilots: float | None
    target_line_pilots: float | None
    min_experience_ratio: float | None
    allowed_wg_rap_shortfall: float | None
    allowed_fl_rap_shortfall: float | None
    allowed_ip_rap_shortfall: float | None
    target_staff_ips: float | None
    target_staff_fls: float | None


@dataclass(frozen=True)
class ConstraintScalesConfig:
    total_pilots: float
    line_pilots: float
    wg_rap: float
    fl_rap: float
    ip_rap: float
    staff_ips: float
    staff_fls: float
    experience_ratio: float

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
        if constraint_name == "staff_ips":
            return self.staff_ips
        if constraint_name == "staff_fls":
            return self.staff_fls
        if constraint_name == "experience_ratio":
            return self.experience_ratio
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
class ViabilityConfig:
    run: RunConfig
    model: ModelConfig
    requirements: RequirementsConfig
    constraint_scales: ConstraintScalesConfig
    policy: PolicyConfig
    doe: DoeConfig

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
        model = ModelConfig(
            **{key: _require_key(model_data, "model", key) for key in _MODEL_FIELDS}
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

        config = cls(
            run=run,
            model=model,
            requirements=requirements,
            constraint_scales=constraint_scales,
            policy=policy,
            doe=doe,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model.years_to_run <= 0:
            raise ValueError("model.years_to_run must be positive")
        if self.model.n_replications != 1:
            raise ValueError("Only model.n_replications=1 is implemented in this first slice")
        if self.model.expected_brain_outputs < 16:
            raise ValueError("The current CAFSimulation path requires a 16-output brain layout")

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

        self._validate_enabled_constraint_scales()

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
        if req.target_staff_ips is not None:
            enabled.append(("staff_ips", "staff_ips"))
        if req.target_staff_fls is not None:
            enabled.append(("staff_fls", "staff_fls"))
        if req.min_experience_ratio is not None:
            enabled.append(("experience_ratio", "experience_ratio"))

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
        return {
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
