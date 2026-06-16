from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.viability.config import PolicyConfig, VariableConfig
from src.viability.policy import PolicyDesign


@dataclass(frozen=True)
class DesignSpace:
    """Normalized/physical conversion for constant policy input combinations."""

    policy: PolicyConfig

    @property
    def variable_names(self) -> list[str]:
        return list(self.policy.variables)

    @property
    def dimension(self) -> int:
        return len(self.variable_names)

    def normalize(self, design: Mapping[str, float]) -> np.ndarray:
        return np.array(
            [
                self._normalize_value(float(design[name]), self.policy.variables[name])
                for name in self.variable_names
            ],
            dtype=float,
        )

    def denormalize(self, values: Sequence[float]) -> dict[str, float | int]:
        _, applied = self.denormalize_with_raw(values)
        return applied

    def denormalize_with_raw(
        self, values: Sequence[float]
    ) -> tuple[dict[str, float], dict[str, float | int]]:
        if len(values) != self.dimension:
            raise ValueError(f"Expected {self.dimension} normalized values, got {len(values)}")

        raw: dict[str, float] = {}
        applied: dict[str, float | int] = {}
        for name, unit_value in zip(self.variable_names, values):
            variable = self.policy.variables[name]
            bounded = min(max(float(unit_value), 0.0), 1.0)
            physical = variable.low + bounded * (variable.high - variable.low)
            raw[name] = float(physical)
            if variable.type == "int":
                applied[name] = int(round(physical))
            else:
                applied[name] = float(physical)
        return raw, applied

    def to_policy_design(
        self,
        values: Mapping[str, float | int],
        raw_values: Mapping[str, float] | None = None,
    ) -> PolicyDesign:
        return PolicyDesign.from_mapping(values, self.policy, raw_values=raw_values)

    def validate_design(self, values: Mapping[str, float]) -> None:
        self.to_policy_design(values)

    def corner_designs(self) -> list[dict[str, float | int]]:
        corners = []
        for unit_values in product([0.0, 1.0], repeat=self.dimension):
            _, applied = self.denormalize_with_raw(unit_values)
            corners.append(applied)
        return corners

    def baseline_designs(self, baselines: Iterable[Mapping[str, float]] | None) -> list[dict[str, float | int]]:
        if baselines is None:
            return []
        return [self.to_policy_design(baseline).to_dict() for baseline in baselines]

    def _normalize_value(self, value: float, variable: VariableConfig) -> float:
        if value < variable.low or value > variable.high:
            raise ValueError(f"Value {value} outside bounds [{variable.low}, {variable.high}]")
        span = variable.high - variable.low
        if span == 0:
            return 0.0
        return (value - variable.low) / span
