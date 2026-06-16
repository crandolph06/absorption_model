from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.viability.config import ViabilityConfig
from src.viability.dashboard import aggregate_history_trajectory
from src.viability.dynamic_analysis_common import (
    ScheduleHistoryRunner,
    best_ok_row,
    configure_matplotlib_cache,
    mark_assessment_start,
    safe_name,
    save_figure,
    schedule_from_row,
    selected_policy_summary,
    trajectory_x,
)
from src.viability.dynamic_policy import EpochPolicySchedule
from src.viability.evaluator import simulate_policy_schedule_history
from src.viability.io import write_config_resolved, write_table
from src.viability.surrogate import read_evaluations_table


@dataclass(frozen=True)
class DynamicTrajectoryArtifactResult:
    output_dir: Path
    summary_path: Path
    selected_policies_path: Path
    trajectory_paths: dict[str, Path]
    figure_paths: dict[str, Path]


def run_dynamic_trajectory_artifacts(
    *,
    config: ViabilityConfig,
    evaluation_specs: Sequence[tuple[str | Path, int, str]],
    output_dir: str | Path,
    history_runner: ScheduleHistoryRunner | None = None,
) -> DynamicTrajectoryArtifactResult:
    """Rerun selected schedules and write trajectories plus publication figures."""
    if not evaluation_specs:
        raise ValueError("At least one evaluation spec is required")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_config_resolved(config, output_path)
    runner = history_runner or _default_history_runner

    selected_rows = []
    trajectory_paths: dict[str, Path] = {}
    trajectories: dict[str, pd.DataFrame] = {}
    combined_evaluations = []
    for path, epoch_count, label in evaluation_specs:
        evaluations = read_evaluations_table(path)
        combined_evaluations.append(evaluations.assign(result_label=label))
        row = best_ok_row(evaluations)
        schedule = schedule_from_row(row, config, epoch_count=epoch_count)
        history = runner(schedule, config)
        trajectory = aggregate_history_trajectory(history, config)
        key = safe_name(label)
        trajectory_path = output_path / f"trajectory_{key}.csv"
        trajectory.to_csv(trajectory_path, index=False)
        trajectory_paths[label] = trajectory_path.resolve()
        trajectories[label] = trajectory
        selected_rows.append(selected_policy_summary(row, label, epoch_count))

    selected = pd.DataFrame(selected_rows)
    selected_path = write_table(
        selected,
        output_path / "selected_policies.csv",
        prefer_parquet=False,
    )
    all_evaluations = pd.concat(combined_evaluations, ignore_index=True, sort=False)
    figure_paths = write_dynamic_figures(
        trajectories=trajectories,
        selected_policies=selected,
        evaluations=all_evaluations,
        config=config,
        output_dir=output_path,
    )
    summary = {
        "output_dir": str(output_path.resolve()),
        "selected_count": int(len(selected)),
        "selected_policies_path": str(selected_path.resolve()),
        "trajectory_paths": {name: str(path) for name, path in trajectory_paths.items()},
        "figure_paths": {name: str(path) for name, path in figure_paths.items()},
    }
    summary_path = output_path / "trajectory_artifacts_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return DynamicTrajectoryArtifactResult(
        output_dir=output_path.resolve(),
        summary_path=summary_path.resolve(),
        selected_policies_path=selected_path.resolve(),
        trajectory_paths=trajectory_paths,
        figure_paths=figure_paths,
    )


def write_dynamic_figures(
    *,
    trajectories: Mapping[str, pd.DataFrame],
    selected_policies: pd.DataFrame,
    evaluations: pd.DataFrame,
    config: ViabilityConfig,
    output_dir: Path,
) -> dict[str, Path]:
    configure_matplotlib_cache(output_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}
    colors = ["#2563eb", "#7c3aed", "#059669", "#ea580c", "#111827"]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for idx, (label, trajectory) in enumerate(trajectories.items()):
        ax.plot(
            trajectory_x(trajectory),
            trajectory["total_pilots"],
            label=label,
            color=colors[idx % len(colors)],
            linewidth=2.2,
        )
    if config.requirements.target_total_pilots is not None:
        ax.axhline(
            float(config.requirements.target_total_pilots),
            color="#b91c1c",
            linestyle="--",
            linewidth=1.4,
            label="3500 target",
        )
    mark_assessment_start(ax, config)
    ax.set_xlabel("Simulation phase")
    ax.set_ylabel("Total pilots")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    paths["inventory"] = save_figure(fig, output_dir / "trajectory_total_pilots.png", plt)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
    rap_columns = [
        ("wg_rap_margin", "WG RAP shortfall"),
        ("fl_rap_margin", "FL RAP shortfall"),
        ("ip_rap_margin", "IP RAP shortfall"),
    ]
    for axis, (column, ylabel) in zip(axes, rap_columns, strict=True):
        for idx, (label, trajectory) in enumerate(trajectories.items()):
            axis.plot(
                trajectory_x(trajectory),
                trajectory[column],
                label=label,
                color=colors[idx % len(colors)],
                linewidth=1.9,
            )
        axis.axhline(0.0, color="#b91c1c", linestyle="--", linewidth=1.1)
        mark_assessment_start(axis, config)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Simulation phase")
    axes[0].legend(loc="best", fontsize=8)
    paths["rap"] = save_figure(fig, output_dir / "trajectory_rap_shortfalls.png", plt)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for idx, (label, trajectory) in enumerate(trajectories.items()):
        x_values = trajectory_x(trajectory)
        ax.plot(
            x_values,
            trajectory["staff_ips"],
            label=f"{label} staff IPs",
            color=colors[idx % len(colors)],
            linewidth=2.0,
        )
        ax.plot(
            x_values,
            trajectory["staff_fls"],
            label=f"{label} staff FLs",
            color=colors[idx % len(colors)],
            linewidth=1.4,
            linestyle=":",
        )
    mark_assessment_start(ax, config)
    ax.set_xlabel("Simulation phase")
    ax.set_ylabel("Staff counts")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    paths["staff"] = save_figure(fig, output_dir / "trajectory_staff_counts.png", plt)

    fig, axes = plt.subplots(4, 2, figsize=(9.2, 8.2), sharex=True)
    axes_flat = axes.ravel()
    first_label = selected_policies.iloc[0]["label"]
    first_epoch_count = int(selected_policies.iloc[0]["epoch_count"])
    first_row = selected_policies.iloc[0]
    for axis, name in zip(axes_flat, config.policy.variables, strict=False):
        values = [float(first_row[f"epoch{epoch}_{name}"]) for epoch in range(1, first_epoch_count + 1)]
        axis.step(range(1, first_epoch_count + 1), values, where="mid", color="#2563eb", linewidth=2.0)
        axis.scatter(range(1, first_epoch_count + 1), values, color="#111827", s=18)
        axis.set_title(name.replace("_", " "), fontsize=9)
        axis.grid(True, alpha=0.25)
    for axis in axes_flat[len(config.policy.variables):]:
        axis.axis("off")
    axes_flat[0].set_ylabel(first_label)
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    paths["policy"] = save_figure(fig, output_dir / "best_policy_epoch_controls.png", plt)

    ok = evaluations[evaluations["status"] == "ok"].copy()
    needed = {"constraint_total_pilots_window", "constraint_wg_rap", "constraint_fl_rap"}
    if needed.issubset(ok.columns):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        scatter = ax.scatter(
            ok["constraint_total_pilots_window"],
            ok["constraint_wg_rap"],
            c=ok["constraint_fl_rap"],
            cmap="viridis",
            s=20,
            alpha=0.75,
            edgecolor="none",
        )
        best = ok.sort_values(["phi", "schedule_id"]).iloc[0]
        ax.scatter(
            [best["constraint_total_pilots_window"]],
            [best["constraint_wg_rap"]],
            marker="*",
            s=160,
            color="#b91c1c",
            label="best observed",
            zorder=5,
        )
        ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Total-pilot-window violation")
        ax.set_ylabel("WG RAP violation")
        ax.legend(loc="best", fontsize=8)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("FL RAP violation")
        ax.grid(True, alpha=0.25)
        paths["trade_space"] = save_figure(fig, output_dir / "trade_space_total_wg_fl.png", plt)

        fig = plt.figure(figsize=(8.8, 7.3), constrained_layout=False)
        grid = fig.add_gridspec(
            2,
            5,
            width_ratios=[1.0, 0.045, 0.18, 1.0, 0.045],
            height_ratios=[1.0, 1.08],
            wspace=0.55,
            hspace=0.48,
        )
        axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 3]),
            fig.add_subplot(grid[1, 1:4]),
        ]
        color_axes = [
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 4]),
            fig.add_subplot(grid[1, 4]),
        ]
        front_mask = _nondominated_mask(
            ok,
            [
                "constraint_total_pilots_window",
                "constraint_wg_rap",
                "constraint_fl_rap",
            ],
        )
        front = ok.loc[front_mask]
        panels = [
            (
                axes[0],
                "constraint_total_pilots_window",
                "constraint_wg_rap",
                "constraint_fl_rap",
                "Inventory window vs WG RAP",
                "Total-pilot-window violation",
                "WG RAP violation",
                "FL RAP violation",
            ),
            (
                axes[1],
                "constraint_total_pilots_window",
                "constraint_fl_rap",
                "constraint_wg_rap",
                "Inventory window vs FL RAP",
                "Total-pilot-window violation",
                "FL RAP violation",
                "WG RAP violation",
            ),
            (
                axes[2],
                "constraint_wg_rap",
                "constraint_fl_rap",
                "constraint_total_pilots_window",
                "WG RAP vs FL RAP",
                "WG RAP violation",
                "FL RAP violation",
                "Total-pilot-window violation",
            ),
        ]
        for panel_index, (
            axis,
            x_col,
            y_col,
            color_col,
            title,
            xlabel,
            ylabel,
            color_label,
        ) in enumerate(panels):
            scatter = axis.scatter(
                ok[x_col],
                ok[y_col],
                c=ok[color_col],
                cmap="viridis",
                s=15,
                alpha=0.62,
                edgecolor="none",
            )
            axis.scatter(
                front[x_col],
                front[y_col],
                facecolors="none",
                edgecolors="#111827",
                s=34,
                linewidth=0.7,
                label="nondominated",
                zorder=4,
            )
            axis.scatter(
                [best[x_col]],
                [best[y_col]],
                marker="*",
                s=150,
                color="#b91c1c",
                label="best observed",
                zorder=5,
            )
            axis.axvline(0.0, color="#111827", linestyle="--", linewidth=0.9)
            axis.axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
            axis.set_title(title, fontsize=10)
            axis.set_xlabel(xlabel, fontsize=9)
            axis.set_ylabel(ylabel, fontsize=9)
            axis.grid(True, alpha=0.25)
            axis.tick_params(axis="both", labelsize=8)
            cbar = fig.colorbar(scatter, cax=color_axes[panel_index])
            cbar.set_label(color_label, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        axes[0].legend(loc="best", fontsize=7)
        paths["constraint_trade_space"] = save_figure(
            fig,
            output_dir / "trade_space_constraint_views.png",
            plt,
            tight_layout=False,
        )

    return {name: path.resolve() for name, path in paths.items()}


def _nondominated_mask(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Return rows that are not dominated when all listed columns are minimized."""
    values = frame.loc[:, list(columns)].to_numpy()
    mask = []
    for row in values:
        dominated = ((values <= row).all(axis=1) & (values < row).any(axis=1)).any()
        mask.append(not dominated)
    return pd.Series(mask, index=frame.index)


def _default_history_runner(
    schedule: EpochPolicySchedule,
    config: ViabilityConfig,
) -> pd.DataFrame:
    return simulate_policy_schedule_history(schedule, config)
