from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.viability.config import PolicyConfig
from src.viability.design_space import DesignSpace
from src.viability.policy import PolicyDesign


def dynamic_feature_names(policy: PolicyConfig, epoch_count: int) -> list[str]:
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    names: list[str] = []
    for epoch_index in range(epoch_count):
        prefix = f"epoch{epoch_index + 1}"
        names.extend(f"{prefix}_{name}" for name in policy.variables)
    return names


def epoch_for_phase_index(phase_index: int, total_phases: int, epoch_count: int) -> int:
    if phase_index < 0:
        raise ValueError("phase_index must be non-negative")
    if total_phases <= 0:
        raise ValueError("total_phases must be positive")
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    if phase_index >= total_phases:
        raise ValueError(f"phase_index={phase_index} outside total phases {total_phases}")
    return min(epoch_count - 1, int(phase_index * epoch_count / total_phases))


@dataclass(frozen=True)
class EpochPolicySchedule:
    """Open-loop piecewise-constant controls over evenly spaced horizon epochs."""

    epoch_designs: tuple[PolicyDesign, ...]
    total_phases: int

    def __post_init__(self) -> None:
        if not self.epoch_designs:
            raise ValueError("EpochPolicySchedule requires at least one epoch")
        if self.total_phases <= 0:
            raise ValueError("total_phases must be positive")

    @property
    def epoch_count(self) -> int:
        return len(self.epoch_designs)

    def policy_for_phase_index(self, phase_index: int) -> PolicyDesign:
        return self.epoch_designs[
            epoch_for_phase_index(phase_index, self.total_phases, self.epoch_count)
        ]

    def to_flat_dict(self, *, raw: bool = False) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for epoch_index, design in enumerate(self.epoch_designs):
            prefix = f"epoch{epoch_index + 1}"
            source = design.to_raw_dict() if raw else design.to_applied_dict()
            for name, value in source.items():
                values[f"{prefix}_{name}"] = value
        return values

    @classmethod
    def from_flat_mapping(
        cls,
        values: Mapping[str, Any],
        policy_config: PolicyConfig,
        *,
        epoch_count: int,
        total_phases: int,
        raw_values: Mapping[str, float] | None = None,
    ) -> "EpochPolicySchedule":
        if epoch_count <= 0:
            raise ValueError("epoch_count must be positive")
        designs: list[PolicyDesign] = []
        for epoch_index in range(epoch_count):
            prefix = f"epoch{epoch_index + 1}"
            epoch_values = {}
            epoch_raw = {}
            for name in policy_config.variables:
                column = f"{prefix}_{name}"
                if column not in values:
                    raise ValueError(f"Missing dynamic policy value {column!r}")
                epoch_values[name] = values[column]
                if raw_values is not None and column in raw_values:
                    epoch_raw[name] = float(raw_values[column])
            designs.append(
                PolicyDesign.from_mapping(
                    epoch_values,
                    policy_config,
                    raw_values=epoch_raw if epoch_raw else None,
                )
            )
        return cls(tuple(designs), total_phases=total_phases)


def schedule_from_unit_vector(
    unit_values: Sequence[float],
    policy_config: PolicyConfig,
    *,
    epoch_count: int,
    total_phases: int,
) -> EpochPolicySchedule:
    expected = len(policy_config.variables) * epoch_count
    if len(unit_values) != expected:
        raise ValueError(f"Expected {expected} unit values, got {len(unit_values)}")
    space = DesignSpace(policy_config)
    values = np.asarray(unit_values, dtype=float).reshape(epoch_count, space.dimension)
    designs: list[PolicyDesign] = []
    for epoch_values in values:
        raw, applied = space.denormalize_with_raw(epoch_values)
        designs.append(space.to_policy_design(applied, raw_values=raw))
    return EpochPolicySchedule(tuple(designs), total_phases=total_phases)


def unit_vector_from_schedule(
    schedule: EpochPolicySchedule,
    policy_config: PolicyConfig,
) -> np.ndarray:
    space = DesignSpace(policy_config)
    vectors = [space.normalize(design.to_raw_dict()) for design in schedule.epoch_designs]
    return np.concatenate(vectors)
