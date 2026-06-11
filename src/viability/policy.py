from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.viability.config import PolicyConfig


@dataclass(frozen=True)
class PolicyDesign:
    """Typed constant-policy struct wired to CAFSimulation until piecewise policies land."""

    annual_intake: int
    retention_rate: float
    ute: float
    paa: int
    max_manning_pct: float
    flug_quota_per_phase: int
    ipug_quota_per_phase: int
    raw: dict[str, float]
    applied: dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        policy_config: PolicyConfig | None = None,
        raw_values: Mapping[str, float] | None = None,
    ) -> "PolicyDesign":
        converted = dict(values)
        raw: dict[str, float] = {}

        if policy_config is not None:
            missing = set(policy_config.variables) - set(converted)
            if missing:
                raise ValueError(f"Missing policy values: {sorted(missing)}")
            for name, variable in policy_config.variables.items():
                value = float(converted[name])
                if value < variable.low or value > variable.high:
                    raise ValueError(
                        f"{name}={value} outside configured bounds "
                        f"[{variable.low}, {variable.high}]"
                    )
                if raw_values is not None and name in raw_values:
                    raw[name] = float(raw_values[name])
                else:
                    raw[name] = value
                if variable.type == "int":
                    converted[name] = int(round(value))
                else:
                    converted[name] = float(value)
        else:
            for name, value in converted.items():
                raw[name] = float(raw_values[name]) if raw_values and name in raw_values else float(value)

        applied = {
            "annual_intake": int(converted["annual_intake"]),
            "retention_rate": float(converted["retention_rate"]),
            "ute": float(converted["ute"]),
            "paa": int(converted["paa"]),
            "max_manning_pct": float(converted["max_manning_pct"]),
            "flug_quota_per_phase": int(converted["flug_quota_per_phase"]),
            "ipug_quota_per_phase": int(converted["ipug_quota_per_phase"]),
        }
        return cls(
            annual_intake=applied["annual_intake"],
            retention_rate=applied["retention_rate"],
            ute=applied["ute"],
            paa=applied["paa"],
            max_manning_pct=applied["max_manning_pct"],
            flug_quota_per_phase=applied["flug_quota_per_phase"],
            ipug_quota_per_phase=applied["ipug_quota_per_phase"],
            raw=raw,
            applied=applied,
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.applied)

    def to_raw_dict(self) -> dict[str, float]:
        return dict(self.raw)

    def to_applied_dict(self) -> dict[str, Any]:
        return dict(self.applied)
