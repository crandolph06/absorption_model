from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from src.viability.config import EnvelopeConfig, EnvelopeSliceConfig, ViabilityConfig
from src.viability.design_space import DesignSpace
from src.viability.io import write_config_resolved, write_table
from src.viability.search import load_signed_constraint_surrogate
from src.viability.surrogate import predict_constraint_surrogate, read_evaluations_table


@dataclass(frozen=True)
class EnvelopeResult:
    output_dir: Path
    summary_path: Path
    plot_paths: dict[str, Path]
    grid_paths: dict[str, Path]
    de_comparison_paths: dict[str, Path]
    anchor_design_id: str
    anchor_phi: float
    plots_skipped: bool = False
    plots_skipped_reason: str | None = None


def run_envelope_plots_from_files(
    *,
    surrogate_path: str | Path,
    evaluations_path: str | Path,
    verified_candidates_path: str | Path,
    config: ViabilityConfig,
    output_dir: str | Path,
) -> EnvelopeResult:
    surrogate = load_signed_constraint_surrogate(surrogate_path)
    evaluations = read_evaluations_table(evaluations_path)
    verified_candidates = read_evaluations_table(verified_candidates_path)
    return run_envelope_plots(
        surrogate=surrogate,
        surrogate_path=surrogate_path,
        evaluations=evaluations,
        verified_candidates=verified_candidates,
        config=config,
        output_dir=output_dir,
    )


def run_envelope_plots(
    *,
    surrogate: dict[str, Any],
    surrogate_path: str | Path,
    evaluations: pd.DataFrame,
    verified_candidates: pd.DataFrame,
    config: ViabilityConfig,
    output_dir: str | Path,
) -> EnvelopeResult:
    envelope_config = require_envelope_config(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_config_resolved(config, output_path)

    feasible_count = _verified_feasible_count(verified_candidates)
    if feasible_count == 0:
        best = select_best_verified_policy(verified_candidates)
        plots_skipped_reason = (
            "No verified feasible candidates; near-boundary feasible "
            "envelope anchor is unavailable."
        )
        summary = _envelope_summary(
            surrogate_path=surrogate_path,
            output_path=output_path,
            envelope_config=envelope_config,
            anchor=best,
            slice_summaries=[],
            plots_skipped=True,
            plots_skipped_reason=plots_skipped_reason,
            verified_feasible_count=0,
        )
        summary_path = output_path / "envelope_summary.json"
        summary_path.write_text(
            json.dumps(_json_ready(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return EnvelopeResult(
            output_dir=output_path.resolve(),
            summary_path=summary_path.resolve(),
            plot_paths={},
            grid_paths={},
            de_comparison_paths={},
            anchor_design_id=str(best["design_id"]),
            anchor_phi=float(best["phi"]),
            plots_skipped=True,
            plots_skipped_reason=plots_skipped_reason,
        )

    anchor = select_anchor_policy(verified_candidates, envelope_config)
    plot_paths: dict[str, Path] = {}
    grid_paths: dict[str, Path] = {}
    de_paths: dict[str, Path] = {}
    slice_summaries = []

    for slice_config in envelope_config.slices:
        fixed_grid = fixed_slice_grid(
            surrogate,
            anchor,
            slice_config,
            config,
            envelope_config,
        )
        fixed_grid_path = write_table(
            fixed_grid,
            output_path / _slice_filename(slice_config, "fixed_grid.parquet"),
        )
        fixed_plot = write_slice_plot(
            output_path=output_path,
            grid=fixed_grid,
            evaluations=evaluations,
            verified_candidates=verified_candidates,
            slice_config=slice_config,
            phi_column="predicted_phi",
            conservative_column="conservative_phi",
            title=f"Fixed Slice: {slice_config.x} vs {slice_config.y}",
            filename=_slice_filename(slice_config, "fixed.png"),
        )

        projected_grid = projected_sobol_grid(
            surrogate,
            slice_config,
            config,
            envelope_config,
        )
        projected_grid_path = write_table(
            projected_grid,
            output_path / _slice_filename(slice_config, "projected_grid.parquet"),
        )
        projected_plot = write_slice_plot(
            output_path=output_path,
            grid=projected_grid,
            evaluations=evaluations,
            verified_candidates=verified_candidates,
            slice_config=slice_config,
            phi_column="projected_phi",
            conservative_column="projected_conservative_phi",
            title=f"Projected Envelope: {slice_config.x} vs {slice_config.y}",
            filename=_slice_filename(slice_config, "projected.png"),
        )

        plot_paths[f"{slice_config.x}_vs_{slice_config.y}_fixed"] = fixed_plot
        plot_paths[f"{slice_config.x}_vs_{slice_config.y}_projected"] = projected_plot
        grid_paths[f"{slice_config.x}_vs_{slice_config.y}_fixed"] = fixed_grid_path
        grid_paths[f"{slice_config.x}_vs_{slice_config.y}_projected"] = projected_grid_path

        de_path = None
        if envelope_config.de_compare_enabled:
            de_comparison = differential_evolution_comparison(
                surrogate,
                projected_grid,
                slice_config,
                config,
                envelope_config,
            )
            de_path = output_path / _slice_filename(slice_config, "de_comparison.csv")
            de_comparison.to_csv(de_path, index=False)
            de_paths[f"{slice_config.x}_vs_{slice_config.y}"] = de_path

        slice_summaries.append(
            _slice_summary(
                slice_config=slice_config,
                fixed_grid=fixed_grid,
                projected_grid=projected_grid,
                fixed_grid_path=fixed_grid_path,
                projected_grid_path=projected_grid_path,
                fixed_plot=fixed_plot,
                projected_plot=projected_plot,
                de_path=de_path,
            )
        )

    summary = _envelope_summary(
        surrogate_path=surrogate_path,
        output_path=output_path,
        envelope_config=envelope_config,
        anchor=anchor,
        slice_summaries=slice_summaries,
        verified_feasible_count=feasible_count,
    )
    summary_path = output_path / "envelope_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return EnvelopeResult(
        output_dir=output_path.resolve(),
        summary_path=summary_path.resolve(),
        plot_paths={name: path.resolve() for name, path in plot_paths.items()},
        grid_paths={name: path.resolve() for name, path in grid_paths.items()},
        de_comparison_paths={name: path.resolve() for name, path in de_paths.items()},
        anchor_design_id=str(anchor["design_id"]),
        anchor_phi=float(anchor["phi"]),
        plots_skipped=False,
        plots_skipped_reason=None,
    )


def require_envelope_config(config: ViabilityConfig) -> EnvelopeConfig:
    if config.envelope is None:
        raise ValueError("Config must include an envelope section for plot-envelope")
    return config.envelope


def select_anchor_policy(
    verified_candidates: pd.DataFrame,
    envelope_config: EnvelopeConfig,
) -> pd.Series:
    if envelope_config.anchor != "near_boundary_feasible":
        raise ValueError(f"Unsupported envelope anchor {envelope_config.anchor!r}")
    required = ("feasible", "phi")
    missing = [column for column in required if column not in verified_candidates.columns]
    if missing:
        raise ValueError(f"Verified candidates table is missing required columns: {missing}")
    feasible = verified_candidates.loc[verified_candidates["feasible"].astype(bool)].copy()
    if feasible.empty:
        raise ValueError(
            "Cannot select near-boundary feasible anchor because no verified "
            "candidates are feasible"
        )
    feasible.loc[:, "_abs_phi"] = feasible["phi"].astype(float).abs()
    return feasible.sort_values(["_abs_phi", "phi", "design_id"]).iloc[0].drop(labels=["_abs_phi"])


def select_best_verified_policy(verified_candidates: pd.DataFrame) -> pd.Series:
    required = ("phi", "design_id")
    missing = [column for column in required if column not in verified_candidates.columns]
    if missing:
        raise ValueError(f"Verified candidates table is missing required columns: {missing}")
    if verified_candidates.empty:
        raise ValueError("Verified candidates table is empty")
    sort_keys = ["phi", "design_id"]
    if "candidate_id" in verified_candidates.columns:
        sort_keys = ["phi", "candidate_id", "design_id"]
    return verified_candidates.sort_values(sort_keys).iloc[0]


def _verified_feasible_count(verified_candidates: pd.DataFrame) -> int:
    if "feasible" not in verified_candidates.columns:
        return 0
    return int(verified_candidates["feasible"].astype(bool).sum())


def fixed_slice_grid(
    surrogate: dict[str, Any],
    anchor: pd.Series,
    slice_config: EnvelopeSliceConfig,
    config: ViabilityConfig,
    envelope_config: EnvelopeConfig,
) -> pd.DataFrame:
    rows = slice_design_rows(
        slice_config=slice_config,
        config=config,
        grid_size=envelope_config.grid_size,
        hidden_anchor=anchor,
    )
    prediction = predict_policy_frame(
        surrogate,
        rows,
        config,
        conservative_sigma=envelope_config.conservative_sigma,
        chunk_size=envelope_config.prediction_chunk_size,
    )
    scored = pd.concat([rows.reset_index(drop=True), prediction], axis=1)
    scored["slice_type"] = "fixed"
    return scored


def projected_sobol_grid(
    surrogate: dict[str, Any],
    slice_config: EnvelopeSliceConfig,
    config: ViabilityConfig,
    envelope_config: EnvelopeConfig,
) -> pd.DataFrame:
    space = DesignSpace(config.policy)
    hidden_names = [
        name
        for name in space.variable_names
        if name not in {slice_config.x, slice_config.y}
    ]
    hidden_values = sobol_hidden_values(
        hidden_names,
        config,
        n_samples=envelope_config.sobol_hidden_samples,
        start_index=envelope_config.sobol_hidden_start_index,
    )
    base_rows = slice_design_rows(
        slice_config=slice_config,
        config=config,
        grid_size=envelope_config.grid_size,
        hidden_anchor=None,
    )
    result_rows = []
    for grid_index, base_row in base_rows.iterrows():
        candidates = hidden_projection_candidates(base_row, hidden_values, hidden_names)
        prediction = predict_policy_frame(
            surrogate,
            candidates,
            config,
            conservative_sigma=envelope_config.conservative_sigma,
            chunk_size=envelope_config.prediction_chunk_size,
        )
        predicted_values = prediction["predicted_phi"].to_numpy(dtype=float)
        conservative_values = prediction["conservative_phi"].to_numpy(dtype=float)
        best_predicted_index = int(np.argmin(predicted_values))
        best_conservative_index = int(np.argmin(conservative_values))
        row = base_row.to_dict()
        row["grid_index"] = int(grid_index)
        row["slice_type"] = "projected"
        row["projected_phi"] = float(predicted_values[best_predicted_index])
        row["projected_sigma_phi"] = float(
            prediction.loc[best_predicted_index, "predicted_sigma_phi"]
        )
        row["projected_conservative_phi"] = float(conservative_values[best_conservative_index])
        row["projected_active_constraint"] = str(
            prediction.loc[best_predicted_index, "predicted_active_constraint"]
        )
        best_design = candidates.iloc[best_predicted_index]
        for hidden_name in hidden_names:
            row[f"best_{hidden_name}"] = best_design[hidden_name]
        result_rows.append(row)
    return pd.DataFrame(result_rows)


def differential_evolution_comparison(
    surrogate: dict[str, Any],
    projected_grid: pd.DataFrame,
    slice_config: EnvelopeSliceConfig,
    config: ViabilityConfig,
    envelope_config: EnvelopeConfig,
) -> pd.DataFrame:
    points = select_de_comparison_points(
        projected_grid,
        envelope_config.de_compare_points_per_slice,
    )
    rows = []
    hidden_names = [
        name
        for name in DesignSpace(config.policy).variable_names
        if name not in {slice_config.x, slice_config.y}
    ]
    if not hidden_names:
        return points.copy()
    for compare_index, (_, point) in enumerate(points.iterrows()):
        result = differential_evolution(
            lambda unit_values: _hidden_objective(
                unit_values,
                surrogate,
                point,
                hidden_names,
                config,
                envelope_config,
            ),
            bounds=[(0.0, 1.0)] * len(hidden_names),
            maxiter=envelope_config.de_maxiter,
            popsize=envelope_config.de_popsize,
            polish=envelope_config.de_polish,
            seed=config.run.random_seed + compare_index,
            updating="immediate",
            workers=1,
        )
        de_row = point.to_dict()
        design = hidden_design_from_units(point, hidden_names, result.x, config)
        prediction = predict_policy_frame(
            surrogate,
            design,
            config,
            conservative_sigma=envelope_config.conservative_sigma,
            chunk_size=1,
        )
        de_row["de_predicted_phi"] = float(prediction.loc[0, "predicted_phi"])
        de_row["de_conservative_phi"] = float(prediction.loc[0, "conservative_phi"])
        de_row["de_success"] = bool(result.success)
        de_row["de_message"] = str(result.message)
        de_row["de_nit"] = int(result.nit)
        de_row["de_nfev"] = int(result.nfev)
        for hidden_name in hidden_names:
            de_row[f"de_{hidden_name}"] = design.loc[0, hidden_name]
        rows.append(de_row)
    return pd.DataFrame(rows)


def select_de_comparison_points(projected_grid: pd.DataFrame, n_points: int) -> pd.DataFrame:
    if n_points <= 0:
        return projected_grid.iloc[[]].copy()
    selected_indices: list[int] = []
    boundary = projected_grid.assign(
        _abs_phi=projected_grid["projected_phi"].astype(float).abs()
    ).sort_values(["_abs_phi", "grid_index"])
    best = projected_grid.sort_values(["projected_phi", "grid_index"])
    for ordered in [boundary, best]:
        for row_index in ordered.index:
            if row_index not in selected_indices:
                selected_indices.append(int(row_index))
            if len(selected_indices) == n_points:
                break
        if len(selected_indices) == n_points:
            break
    return projected_grid.loc[selected_indices].reset_index(drop=True)


def slice_design_rows(
    *,
    slice_config: EnvelopeSliceConfig,
    config: ViabilityConfig,
    grid_size: int,
    hidden_anchor: pd.Series | None,
) -> pd.DataFrame:
    space = DesignSpace(config.policy)
    x_raw = np.linspace(
        config.policy.variables[slice_config.x].low,
        config.policy.variables[slice_config.x].high,
        grid_size,
    )
    y_raw = np.linspace(
        config.policy.variables[slice_config.y].low,
        config.policy.variables[slice_config.y].high,
        grid_size,
    )
    rows = []
    for y_index, y_value in enumerate(y_raw):
        for x_index, x_value in enumerate(x_raw):
            values: dict[str, Any] = {}
            raw_values: dict[str, float] = {}
            for name in space.variable_names:
                if name == slice_config.x:
                    raw_value = float(x_value)
                elif name == slice_config.y:
                    raw_value = float(y_value)
                elif hidden_anchor is not None:
                    raw_value = _raw_value_from_row(hidden_anchor, name)
                else:
                    raw_value = config.policy.variables[name].low
                applied = apply_policy_value(raw_value, config, name)
                raw_values[name] = raw_value
                values[name] = applied
            row: dict[str, Any] = {
                "grid_index": len(rows),
                "x_index": int(x_index),
                "y_index": int(y_index),
                "x_variable": slice_config.x,
                "y_variable": slice_config.y,
                "x_value": float(x_value),
                "y_value": float(y_value),
            }
            for name in space.variable_names:
                row[f"raw_{name}"] = raw_values[name]
                row[f"applied_{name}"] = values[name]
                row[name] = values[name]
            rows.append(row)
    return pd.DataFrame(rows)


def sobol_hidden_values(
    hidden_names: list[str],
    config: ViabilityConfig,
    *,
    n_samples: int,
    start_index: int,
) -> pd.DataFrame:
    if not hidden_names:
        return pd.DataFrame([{}])
    sampler = qmc.Sobol(
        d=len(hidden_names),
        scramble=config.doe.scramble,
        seed=config.run.random_seed,
    )
    if start_index > 0:
        sampler.fast_forward(start_index)
    unit_samples = sampler.random(n_samples)
    rows = []
    for unit_row in unit_samples:
        row: dict[str, Any] = {}
        for index, name in enumerate(hidden_names):
            variable = config.policy.variables[name]
            raw_value = variable.low + float(unit_row[index]) * (variable.high - variable.low)
            row[f"raw_{name}"] = float(raw_value)
            row[f"applied_{name}"] = apply_policy_value(raw_value, config, name)
            row[name] = row[f"applied_{name}"]
        rows.append(row)
    return pd.DataFrame(rows)


def hidden_projection_candidates(
    base_row: pd.Series,
    hidden_values: pd.DataFrame,
    hidden_names: list[str],
) -> pd.DataFrame:
    candidates = pd.DataFrame([base_row.to_dict()] * len(hidden_values))
    for hidden_name in hidden_names:
        candidates[f"raw_{hidden_name}"] = hidden_values[f"raw_{hidden_name}"].to_numpy()
        candidates[f"applied_{hidden_name}"] = hidden_values[f"applied_{hidden_name}"].to_numpy()
        candidates[hidden_name] = hidden_values[hidden_name].to_numpy()
    return candidates.reset_index(drop=True)


def hidden_design_from_units(
    base_row: pd.Series,
    hidden_names: list[str],
    unit_values: np.ndarray,
    config: ViabilityConfig,
) -> pd.DataFrame:
    row = base_row.to_dict()
    for index, hidden_name in enumerate(hidden_names):
        variable = config.policy.variables[hidden_name]
        raw_value = variable.low + float(unit_values[index]) * (variable.high - variable.low)
        row[f"raw_{hidden_name}"] = float(raw_value)
        row[f"applied_{hidden_name}"] = apply_policy_value(raw_value, config, hidden_name)
        row[hidden_name] = row[f"applied_{hidden_name}"]
    return pd.DataFrame([row])


def predict_policy_frame(
    surrogate: dict[str, Any],
    frame: pd.DataFrame,
    config: ViabilityConfig,
    *,
    conservative_sigma: float,
    chunk_size: int,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Cannot predict an empty policy frame")
    chunks = []
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start:start + chunk_size]
        x_values = design_matrix(chunk, config)
        prediction = predict_constraint_surrogate(
            surrogate,
            x_values,
            conservative_sigma=conservative_sigma,
        )
        chunks.append(
            pd.DataFrame(
                {
                    "predicted_phi": prediction.predicted_phi,
                    "predicted_sigma_phi": prediction.sigma_phi,
                    "conservative_phi": prediction.conservative_phi,
                    "predicted_active_constraint": prediction.active_constraint,
                }
            )
        )
    return pd.concat(chunks, ignore_index=True)


def design_matrix(frame: pd.DataFrame, config: ViabilityConfig) -> np.ndarray:
    space = DesignSpace(config.policy)
    rows = []
    for _, row in frame.iterrows():
        values = {name: row[name] for name in space.variable_names}
        rows.append(space.normalize(values))
    return np.vstack(rows)


def apply_policy_value(raw_value: float, config: ViabilityConfig, name: str) -> float | int:
    variable = config.policy.variables[name]
    bounded = min(max(float(raw_value), variable.low), variable.high)
    if variable.type == "int":
        return int(round(bounded))
    return float(bounded)


def write_slice_plot(
    *,
    output_path: Path,
    grid: pd.DataFrame,
    evaluations: pd.DataFrame,
    verified_candidates: pd.DataFrame,
    slice_config: EnvelopeSliceConfig,
    phi_column: str,
    conservative_column: str,
    title: str,
    filename: str,
) -> Path:
    _configure_matplotlib_cache(output_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_size = _grid_size_from_frame(grid)
    x_values = grid["x_value"].to_numpy(dtype=float).reshape(grid_size, grid_size)
    y_values = grid["y_value"].to_numpy(dtype=float).reshape(grid_size, grid_size)
    phi_values = grid[phi_column].to_numpy(dtype=float).reshape(grid_size, grid_size)
    conservative_values = grid[conservative_column].to_numpy(dtype=float).reshape(
        grid_size,
        grid_size,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    level_min = float(np.nanmin(phi_values))
    level_max = float(np.nanmax(phi_values))
    if level_min == level_max:
        level_min -= 1.0
        level_max += 1.0
    levels = np.linspace(level_min, level_max, 25)
    contour = ax.contourf(x_values, y_values, phi_values, levels=levels, cmap="RdBu_r", alpha=0.80)
    fig.colorbar(contour, ax=ax, label="Predicted phi")
    _draw_zero_contour(
        ax,
        x_values,
        y_values,
        phi_values,
        color="black",
        linestyle="-",
        label="phi = 0",
    )
    _draw_zero_contour(
        ax,
        x_values,
        y_values,
        conservative_values,
        color="black",
        linestyle="--",
        label="conservative phi = 0",
    )
    _overlay_points(ax, evaluations, slice_config, label_prefix="evaluated", alpha=0.25, size=14)
    _overlay_points(
        ax,
        verified_candidates,
        slice_config,
        label_prefix="verified",
        alpha=0.85,
        size=36,
    )
    ax.set_xlabel(slice_config.x)
    ax.set_ylabel(slice_config.y)
    ax.set_title(title)
    ax.grid(True, alpha=0.20)
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    fig.tight_layout()
    plot_path = output_path / filename
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _hidden_objective(
    unit_values: np.ndarray,
    surrogate: dict[str, Any],
    base_row: pd.Series,
    hidden_names: list[str],
    config: ViabilityConfig,
    envelope_config: EnvelopeConfig,
) -> float:
    design = hidden_design_from_units(base_row, hidden_names, unit_values, config)
    prediction = predict_policy_frame(
        surrogate,
        design,
        config,
        conservative_sigma=envelope_config.conservative_sigma,
        chunk_size=1,
    )
    return float(prediction.loc[0, "predicted_phi"])


def _raw_value_from_row(row: pd.Series, name: str) -> float:
    raw_column = f"raw_{name}"
    if raw_column in row and pd.notna(row[raw_column]):
        return float(row[raw_column])
    if name in row and pd.notna(row[name]):
        return float(row[name])
    raise ValueError(f"Anchor row is missing policy value for {name!r}")


def _grid_size_from_frame(grid: pd.DataFrame) -> int:
    grid_size = int(round(float(np.sqrt(len(grid)))))
    if grid_size * grid_size != len(grid):
        raise ValueError("Slice grid row count is not a square")
    return grid_size


def _draw_zero_contour(
    ax: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    *,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    z_min = float(np.nanmin(z_values))
    z_max = float(np.nanmax(z_values))
    if z_min <= 0.0 <= z_max:
        ax.contour(
            x_values,
            y_values,
            z_values,
            levels=[0.0],
            colors=color,
            linestyles=linestyle,
            linewidths=1.2,
        )
        ax.plot([], [], color=color, linestyle=linestyle, label=label)


def _overlay_points(
    ax: Any,
    frame: pd.DataFrame,
    slice_config: EnvelopeSliceConfig,
    *,
    label_prefix: str,
    alpha: float,
    size: float,
) -> None:
    required = (slice_config.x, slice_config.y, "feasible")
    if any(column not in frame.columns for column in required):
        return
    feasible = frame.loc[frame["feasible"].astype(bool)]
    infeasible = frame.loc[~frame["feasible"].astype(bool)]
    if not feasible.empty:
        ax.scatter(
            feasible[slice_config.x],
            feasible[slice_config.y],
            s=size,
            color="darkgreen" if label_prefix == "verified" else "0.15",
            alpha=alpha,
            edgecolor="none",
            label=f"{label_prefix} feasible",
        )
    if not infeasible.empty:
        ax.scatter(
            infeasible[slice_config.x],
            infeasible[slice_config.y],
            s=size,
            color="firebrick" if label_prefix == "verified" else "0.65",
            alpha=alpha,
            edgecolor="none",
            label=f"{label_prefix} infeasible",
        )


def _slice_filename(slice_config: EnvelopeSliceConfig, suffix: str) -> str:
    return f"{slice_config.x}_vs_{slice_config.y}_{suffix}"


def _slice_summary(
    *,
    slice_config: EnvelopeSliceConfig,
    fixed_grid: pd.DataFrame,
    projected_grid: pd.DataFrame,
    fixed_grid_path: Path,
    projected_grid_path: Path,
    fixed_plot: Path,
    projected_plot: Path,
    de_path: Path | None,
) -> dict[str, Any]:
    summary = {
        "x": slice_config.x,
        "y": slice_config.y,
        "fixed_grid_path": str(fixed_grid_path.resolve()),
        "projected_grid_path": str(projected_grid_path.resolve()),
        "fixed_plot_path": str(fixed_plot.resolve()),
        "projected_plot_path": str(projected_plot.resolve()),
        "fixed_min_phi": float(fixed_grid["predicted_phi"].min()),
        "projected_min_phi": float(projected_grid["projected_phi"].min()),
        "projected_feasible_grid_points": int((projected_grid["projected_phi"] <= 0.0).sum()),
    }
    if de_path is not None:
        summary["de_comparison_path"] = str(de_path.resolve())
    return summary


def _envelope_summary(
    *,
    surrogate_path: str | Path,
    output_path: Path,
    envelope_config: EnvelopeConfig,
    anchor: pd.Series,
    slice_summaries: list[dict[str, Any]],
    plots_skipped: bool = False,
    plots_skipped_reason: str | None = None,
    verified_feasible_count: int | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "surrogate_path": str(Path(surrogate_path).resolve()),
        "output_dir": str(output_path.resolve()),
        "anchor": envelope_config.anchor,
        "conservative_sigma": float(envelope_config.conservative_sigma),
        "grid_size": int(envelope_config.grid_size),
        "sobol_hidden_samples": int(envelope_config.sobol_hidden_samples),
        "de_compare_enabled": bool(envelope_config.de_compare_enabled),
        "plots_skipped": bool(plots_skipped),
        "slices": slice_summaries,
    }
    if plots_skipped:
        summary["plots_skipped_reason"] = plots_skipped_reason
        summary["verified_feasible_count"] = int(
            0 if verified_feasible_count is None else verified_feasible_count
        )
        summary["best_verified_candidate_id"] = (
            str(anchor["candidate_id"])
            if "candidate_id" in anchor.index
            else str(anchor["design_id"])
        )
        summary["best_verified_design_id"] = str(anchor["design_id"])
        summary["best_verified_phi"] = float(anchor["phi"])
        summary["anchor_design_id"] = None
        summary["anchor_phi"] = None
        return summary

    summary["anchor_design_id"] = str(anchor["design_id"])
    summary["anchor_phi"] = float(anchor["phi"])
    if verified_feasible_count is not None:
        summary["verified_feasible_count"] = int(verified_feasible_count)
    return summary


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _configure_matplotlib_cache(output_path: Path) -> None:
    cache_dir = output_path / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
