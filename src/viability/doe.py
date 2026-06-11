from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from src.viability.config import ViabilityConfig
from src.viability.design_space import DesignSpace


def generate_doe(
    config: ViabilityConfig,
    n: int | None = None,
    method: str | None = None,
    include_corners: bool | None = None,
    include_baselines: bool | None = None,
) -> pd.DataFrame:
    """Generate constant-policy input combinations from configured bounds."""
    space = DesignSpace(config.policy)
    requested_n = config.doe.n_initial if n is None else n
    if requested_n < 0:
        raise ValueError("DOE sample count must be non-negative")

    selected_method = (config.doe.method if method is None else method).lower()
    unit_samples = _sample_unit_cube(
        n=requested_n,
        dimension=space.dimension,
        method=selected_method,
        random_seed=config.run.random_seed,
    )

    rows: list[dict[str, object]] = []
    for unit_values in unit_samples:
        raw, applied = space.denormalize_with_raw(unit_values)
        row: dict[str, object] = {}
        for name in space.variable_names:
            row[f"raw_{name}"] = raw[name]
            row[f"applied_{name}"] = applied[name]
            row[name] = applied[name]
        rows.append(row)

    use_corners = config.doe.include_corners if include_corners is None else include_corners
    if use_corners:
        for applied in space.corner_designs():
            row = {}
            for name in space.variable_names:
                row[f"raw_{name}"] = float(applied[name])
                row[f"applied_{name}"] = applied[name]
                row[name] = applied[name]
            rows.append(row)

    use_baselines = config.doe.include_baselines if include_baselines is None else include_baselines
    if use_baselines:
        for applied in space.baseline_designs(config.doe.baselines):
            row = {}
            for name in space.variable_names:
                row[f"raw_{name}"] = float(applied[name])
                row[f"applied_{name}"] = applied[name]
                row[name] = applied[name]
            rows.append(row)

    deduped = _dedupe_designs(rows, space.variable_names)
    columns = ["design_id"]
    for name in space.variable_names:
        columns.extend([f"raw_{name}", f"applied_{name}", name])
    df = pd.DataFrame(deduped, columns=columns[1:])
    df.insert(0, "design_id", range(len(df)))
    return df


def dataframe_to_design_records(df: pd.DataFrame, config: ViabilityConfig) -> list[dict[str, object]]:
    space = DesignSpace(config.policy)
    records = []
    for _, row in df.iterrows():
        values = {name: row[name] for name in space.variable_names}
        raw_values = None
        if all(f"raw_{name}" in row for name in space.variable_names):
            raw_values = {name: float(row[f"raw_{name}"]) for name in space.variable_names}
        records.append(space.to_policy_design(values, raw_values=raw_values).to_dict())
    return records


def _sample_unit_cube(n: int, dimension: int, method: str, random_seed: int) -> np.ndarray:
    if n == 0:
        return np.empty((0, dimension), dtype=float)
    if dimension <= 0:
        raise ValueError("DOE dimension must be positive")

    if method == "random":
        rng = np.random.default_rng(random_seed)
        return rng.random((n, dimension))

    if method in {"sobol", "latin_hypercube", "lhs"}:
        try:
            from scipy.stats import qmc
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"DOE method {method!r} requires scipy.stats.qmc; install scipy or use method='random'"
            ) from exc

        if method == "sobol":
            sampler = qmc.Sobol(d=dimension, scramble=True, seed=random_seed)
            return sampler.random(n)
        sampler = qmc.LatinHypercube(d=dimension, seed=random_seed)
        return sampler.random(n)

    raise ValueError(f"Unsupported DOE method {method!r}")


def _dedupe_designs(
    rows: list[Mapping[str, object]], variable_names: list[str]
) -> list[dict[str, object]]:
    seen: set[tuple[object, ...]] = set()
    deduped: list[dict[str, object]] = []
    for row in rows:
        key = tuple(row[f"applied_{name}"] for name in variable_names)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped
