from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.viability.dashboard import (
    DynamicDashboardArtifactPaths,
    POLICY_LABELS,
    DashboardArtifactPaths,
    constraint_relaxation_table,
    default_artifact_paths,
    default_dynamic_artifact_paths,
    direct_verification_caveat,
    direct_verification_label,
    dynamic_epoch_table,
    envelope_plot_paths,
    load_dynamic_dashboard_artifacts,
    load_dashboard_artifacts,
    local_feasible_sweep,
    nearest_dynamic_misses,
    policy_values_from_row,
    run_direct_dynamic_schedule,
    run_direct_policy,
    score_policy_values,
    select_dashboard_candidate,
    select_dynamic_schedule,
)


st.set_page_config(page_title="Viability Dashboard", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_artifacts(
    config: str,
    surrogate: str,
    evaluations: str,
    verified_candidates: str,
    search_summary: str,
    verification_summary: str,
    envelope_summary: str,
    report: str,
):
    return load_dashboard_artifacts(
        DashboardArtifactPaths(
            config=Path(config),
            surrogate=Path(surrogate),
            evaluations=Path(evaluations),
            verified_candidates=Path(verified_candidates),
            search_summary=Path(search_summary),
            verification_summary=Path(verification_summary),
            envelope_summary=Path(envelope_summary),
            report=Path(report) if report else None,
        )
    )


@st.cache_resource(show_spinner=False)
def _load_dynamic_artifacts(
    config: str,
    evaluations: str,
    summary: str,
    sensitivity: str,
    report: str,
    relaxation_dir: str,
    bound_relaxation_dir: str,
    ipug_diagnostic_dir: str,
    paper_artifacts_dir: str,
):
    return load_dynamic_dashboard_artifacts(
        DynamicDashboardArtifactPaths(
            config=Path(config),
            evaluations=Path(evaluations),
            summary=Path(summary),
            sensitivity=Path(sensitivity) if sensitivity else None,
            report=Path(report) if report else None,
            relaxation_dir=Path(relaxation_dir) if relaxation_dir else None,
            bound_relaxation_dir=Path(bound_relaxation_dir) if bound_relaxation_dir else None,
            ipug_diagnostic_dir=Path(ipug_diagnostic_dir) if ipug_diagnostic_dir else None,
            paper_artifacts_dir=Path(paper_artifacts_dir) if paper_artifacts_dir else None,
        )
    )


def _path_input(label: str, path: Path | None, *, container=None) -> str:
    target = st.sidebar if container is None else container
    return target.text_input(label, value="" if path is None else str(path))


def _current_policy_key(row: pd.Series) -> str:
    return f"{row['candidate_id']}::{row['design_id']}"


def _slider_step(variable_type: str, name: str) -> float | int:
    if variable_type == "int":
        return 1
    if name == "retention_rate":
        return 0.001
    if name == "max_manning_pct":
        return 1.0
    return 0.1


def _decimal_places(value: float) -> int:
    text = f"{value:.10f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", maxsplit=1)[1])


def _float_slider_options(low: float, high: float, step: float) -> list[float]:
    decimals = _decimal_places(step)
    count = int(round((float(high) - float(low)) / step))
    options = [round(float(low) + index * step, decimals) for index in range(count + 1)]
    high_value = round(float(high), decimals)
    if options[-1] != high_value:
        options.append(high_value)
    return options


def _nearest_option(value: float, options: list[float]) -> float:
    return min(options, key=lambda option: abs(option - float(value)))


def _format_slider_option(name: str, step: float):
    decimals = _decimal_places(step)
    if name == "retention_rate":
        decimals = max(decimals, 3)
    return lambda value: f"{float(value):.{decimals}f}"


def _policy_slider(name: str, variable, stored_value):
    label = POLICY_LABELS.get(name, name)
    if variable.type == "int":
        return st.slider(
            label,
            min_value=int(variable.low),
            max_value=int(variable.high),
            value=int(round(float(stored_value))),
            step=1,
        )
    step = float(_slider_step(variable.type, name))
    options = _float_slider_options(variable.low, variable.high, step)
    return float(
        st.select_slider(
            label,
            options=options,
            value=_nearest_option(float(stored_value), options),
            format_func=_format_slider_option(name, step),
            width="stretch",
        )
    )


def _format_number(value: float) -> str:
    return f"{value:.3g}"


def _interval_text(intervals) -> str:
    if not intervals:
        return "No conservative-feasible interval at current context."
    parts = []
    for interval in intervals:
        parts.append(f"{_format_number(interval.low)} to {_format_number(interval.high)}")
    return "; ".join(parts)


def _interval_figure(name: str, value: float, variable, intervals) -> go.Figure:
    fig = go.Figure()
    low = float(variable.low)
    high = float(variable.high)
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[0, 0],
            mode="lines",
            line={"color": "#d1d5db", "width": 7},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for interval in intervals:
        fig.add_trace(
            go.Scatter(
                x=[interval.low, interval.high],
                y=[0, 0],
                mode="lines",
                line={"color": "#16a34a", "width": 8},
                hovertemplate="recommended %{x:.3g}<extra></extra>",
                showlegend=False,
            )
        )
        for boundary in (interval.low, interval.high):
            fig.add_vline(x=boundary, line_color="#b91c1c", line_width=2)
    fig.add_trace(
        go.Scatter(
            x=[value],
            y=[0],
            mode="markers",
            marker={"color": "#111827", "size": 10},
            hovertemplate="current %{x:.3g}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_xaxes(range=[low, high], title_text=name, fixedrange=True)
    fig.update_yaxes(visible=False, range=[-1, 1], fixedrange=True)
    fig.update_layout(
        height=86,
        margin={"l": 8, "r": 8, "t": 8, "b": 26},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _status_label(predicted_phi: float, conservative_phi: float) -> str:
    if conservative_phi <= 0.0:
        return "Conservative-surrogate feasible"
    if predicted_phi <= 0.0:
        return "Predicted feasible, not conservative"
    return "Surrogate infeasible"


def _plot_inventory_trajectory(trajectory: pd.DataFrame, target_total: float | None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trajectory["timeline"],
            y=trajectory["total_pilots"],
            name="Total pilots",
            line={"color": "#2563eb", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trajectory["timeline"],
            y=trajectory["line_pilots"],
            name="Line pilots",
            line={"color": "#0f766e", "width": 3},
        )
    )
    if target_total is not None:
        fig.add_hline(
            y=float(target_total),
            line_dash="dash",
            line_color="#b91c1c",
            annotation_text="target total pilots",
        )
    fig.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title="Pilots",
        hovermode="x unified",
    )
    return fig


def _plot_rap_trajectory(trajectory: pd.DataFrame):
    fig = go.Figure()
    for column, label, color in [
        ("wg_rap_margin", "WG RAP margin", "#2563eb"),
        ("fl_rap_margin", "FL RAP margin", "#7c3aed"),
        ("ip_rap_margin", "IP RAP margin", "#059669"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=trajectory["timeline"],
                y=trajectory[column],
                name=label,
                line={"color": color, "width": 3},
            )
        )
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#b91c1c",
        annotation_text="RAP target",
    )
    fig.update_layout(
        height=330,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title="target - observed rate",
        hovermode="x unified",
    )
    return fig


def _plot_experience_trajectory(trajectory: pd.DataFrame, minimum: float | None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trajectory["timeline"],
            y=trajectory["experience_ratio"],
            name="Experience ratio",
            line={"color": "#ea580c", "width": 3},
        )
    )
    if minimum is not None:
        fig.add_hline(
            y=float(minimum),
            line_dash="dash",
            line_color="#b91c1c",
            annotation_text="minimum experience ratio",
        )
    fig.update_layout(
        height=280,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    return fig


def _format_optional_metric(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, int)):
        return f"{float(value):.3g}"
    return str(value)


def _display_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _render_dynamic_dashboard():
    defaults = default_dynamic_artifact_paths()
    with st.sidebar.expander("Dynamic artifacts", expanded=False) as artifact_box:
        config_path = _path_input("Config", defaults.config, container=artifact_box)
        evaluations_path = _path_input("Dynamic evaluations", defaults.evaluations, container=artifact_box)
        summary_path = _path_input("Dynamic search summary", defaults.summary, container=artifact_box)
        sensitivity_default = (
            defaults.sensitivity
            if defaults.sensitivity is not None and defaults.sensitivity.exists()
            else None
        )
        report_default = (
            defaults.report
            if defaults.report is not None and defaults.report.exists()
            else None
        )
        relaxation_default = (
            defaults.relaxation_dir
            if defaults.relaxation_dir is not None and defaults.relaxation_dir.exists()
            else None
        )
        bound_default = (
            defaults.bound_relaxation_dir
            if defaults.bound_relaxation_dir is not None and defaults.bound_relaxation_dir.exists()
            else None
        )
        ipug_default = (
            defaults.ipug_diagnostic_dir
            if defaults.ipug_diagnostic_dir is not None and defaults.ipug_diagnostic_dir.exists()
            else None
        )
        paper_default = (
            defaults.paper_artifacts_dir
            if defaults.paper_artifacts_dir is not None and defaults.paper_artifacts_dir.exists()
            else None
        )
        sensitivity_path = _path_input("Local sensitivity", sensitivity_default, container=artifact_box)
        report_path = _path_input("Dynamic report", report_default, container=artifact_box)
        relaxation_dir = _path_input("Relaxation study directory", relaxation_default, container=artifact_box)
        bound_relaxation_dir = _path_input("Bound relaxation directory", bound_default, container=artifact_box)
        ipug_diagnostic_dir = _path_input("IPUG diagnostic directory", ipug_default, container=artifact_box)
        paper_artifacts_dir = _path_input("Paper artifacts directory", paper_default, container=artifact_box)

    try:
        artifacts = _load_dynamic_artifacts(
            config_path,
            evaluations_path,
            summary_path,
            sensitivity_path,
            report_path,
            relaxation_dir,
            bound_relaxation_dir,
            ipug_diagnostic_dir,
            paper_artifacts_dir,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    config = artifacts.config
    summary = artifacts.summary
    evaluations = artifacts.evaluations
    ok = evaluations[evaluations["status"] == "ok"].copy()
    feasible_count = int(summary.get("feasible_count", ok["feasible"].astype(bool).sum()))

    st.title("Viability Results Dashboard")
    st.caption(
        "Dynamic policy / finite-horizon control search. "
        "Rows shown here are direct long-horizon evaluations; reruns below are "
        f"{direct_verification_label(config)}."
    )

    metric_cols = st.columns(6)
    metric_cols[0].metric("Backend", str(summary.get("phase_backend") or config.model.phase_backend))
    metric_cols[1].metric("Evaluations", _format_optional_metric(summary.get("evaluated_count", len(evaluations))))
    metric_cols[2].metric("Feasible", str(feasible_count))
    metric_cols[3].metric("Best phi", _format_optional_metric(summary.get("best_phi")))
    metric_cols[4].metric("Active constraint", str(summary.get("best_active_constraint", "n/a")))
    best_linf = (
        None
        if artifacts.relaxation_summary is None
        else artifacts.relaxation_summary.get("best_linf_relaxation")
    )
    metric_cols[5].metric("Min relaxation", _format_optional_metric(best_linf))

    if feasible_count > 0:
        st.success("At least one direct-verified dynamic policy was found.")
    else:
        st.warning(
            "No direct-feasible dynamic policy has been found in this result bundle. "
            "Use the nearest miss and relaxation table to decide what to search or relax next."
        )

    control_col, main_col = st.columns([1, 2], gap="large")

    with control_col:
        st.subheader("Selected Schedule")
        selection_options = ["best_phi", "schedule_id"]
        if feasible_count > 0:
            selection_options.insert(1, "best_feasible")
        selection_mode = st.radio(
            "Start from",
            selection_options,
            format_func=lambda value: {
                "best_phi": "Nearest miss / best phi",
                "best_feasible": "Best feasible",
                "schedule_id": "Specific schedule",
            }[value],
        )
        schedule_id = None
        if selection_mode == "schedule_id":
            ranked = nearest_dynamic_misses(evaluations, top_n=min(100, len(evaluations)))
            schedule_id = st.selectbox("Schedule", ranked["schedule_id"].astype(str).tolist())

        selected_row = select_dynamic_schedule(
            evaluations,
            mode=selection_mode,
            schedule_id=schedule_id,
        )
        st.metric("Selected phi", f"{float(selected_row['phi']):.3g}")
        st.metric("Selected status", "Feasible" if bool(selected_row["feasible"]) else "Infeasible")
        st.caption(
            "Schedule source: "
            f"{selected_row.get('schedule_source', 'unknown')}"
            + (
                f" / {selected_row.get('template_name')}"
                if "template_name" in selected_row and pd.notna(selected_row.get("template_name"))
                else ""
            )
        )
        st.dataframe(
            dynamic_epoch_table(selected_row, config, epoch_count=artifacts.epoch_count),
            width="stretch",
            hide_index=True,
        )
        selected_key = str(selected_row["schedule_id"])
        if st.button("Re-run direct trajectory", type="primary", width="stretch"):
            with st.spinner(f"Running {direct_verification_label(config)}..."):
                st.session_state.viability_dynamic_direct_key = selected_key
                st.session_state.viability_dynamic_direct_result = run_direct_dynamic_schedule(
                    selected_row,
                    config,
                    epoch_count=artifacts.epoch_count,
                )
        st.caption(direct_verification_caveat(config))

    with main_col:
        st.subheader("Nearest Miss")
        relaxations = constraint_relaxation_table(selected_row)
        if relaxations.empty:
            st.success("The selected schedule has no positive constraint violations.")
        else:
            st.write("Smallest direct requirement relaxations needed at the selected point:")
            st.dataframe(relaxations, width="stretch", hide_index=True)

        active_counts = summary.get("active_constraint_counts")
        if not active_counts:
            active_counts = ok["active_constraint"].value_counts(dropna=False).to_dict()
        active_frame = pd.DataFrame(
            [
                {"active_constraint": str(name), "count": int(count)}
                for name, count in active_counts.items()
            ]
        ).sort_values("count", ascending=False)
        st.write("Active constraint distribution")
        st.dataframe(active_frame, width="stretch", hide_index=True)

        display = nearest_dynamic_misses(evaluations, top_n=15)
        display_columns = _display_columns(
            display,
            [
                "schedule_id",
                "template_name",
                "schedule_source",
                "phi",
                "feasible",
                "active_constraint",
                "positive_constraint_sum",
                "constraint_total_pilots_window",
                "constraint_wg_rap",
                "constraint_fl_rap",
                "constraint_ip_rap",
            ],
        )
        st.dataframe(display[display_columns], width="stretch", hide_index=True)

    trajectory_tab, relaxation_tab, authority_tab, candidate_tab, artifact_tab = st.tabs(
        [
            "Trajectories",
            "Pareto / Relaxation",
            "Control Authority",
            "Candidates",
            "Raw Artifacts",
        ]
    )

    direct_result = st.session_state.get("viability_dynamic_direct_result")
    direct_key = st.session_state.get("viability_dynamic_direct_key")
    with trajectory_tab:
        if artifacts.paper_figure_paths:
            st.subheader("Report Figures")
            figure_cols = st.columns(2)
            for index, (name, path) in enumerate(artifacts.paper_figure_paths.items()):
                figure_cols[index % 2].image(str(path), caption=name.replace("_", " ").title())
            st.divider()
        if direct_result is None or direct_key != selected_key:
            st.write("Re-run the selected schedule to populate trajectory plots.")
        elif direct_result.evaluation.status != "ok":
            st.error(f"Direct rerun failed: {direct_result.evaluation.error}")
        else:
            direct_cols = st.columns(4)
            direct_cols[0].metric("Direct phi", f"{direct_result.evaluation.phi:.3g}")
            direct_cols[1].metric(
                "Direct feasible",
                "Yes" if direct_result.evaluation.feasible else "No",
            )
            direct_cols[2].metric("Backend", str(direct_result.evaluation.phase_backend))
            direct_cols[3].metric(
                "Active constraint",
                str(direct_result.evaluation.active_constraint),
            )
            st.plotly_chart(
                _plot_inventory_trajectory(
                    direct_result.trajectory,
                    config.requirements.target_total_pilots,
                ),
                width="stretch",
            )
            st.plotly_chart(_plot_rap_trajectory(direct_result.trajectory), width="stretch")
            st.plotly_chart(
                _plot_experience_trajectory(
                    direct_result.trajectory,
                    config.requirements.min_experience_ratio,
                ),
                width="stretch",
            )

    with relaxation_tab:
        if artifacts.relaxation_summary is None:
            st.write("No requirement-relaxation study artifact was supplied.")
        else:
            st.json(artifacts.relaxation_summary)
            if artifacts.relaxation_nearest is not None:
                st.subheader("Nearest Under Relaxation")
                st.dataframe(artifacts.relaxation_nearest, width="stretch", hide_index=True)
            if artifacts.relaxation_sets is not None:
                st.subheader("Constraint Set Minima")
                st.dataframe(artifacts.relaxation_sets, width="stretch", hide_index=True)
            if artifacts.relaxation_pareto is not None:
                st.subheader("Pareto Frontier")
                st.dataframe(artifacts.relaxation_pareto, width="stretch", hide_index=True)
            if artifacts.relaxation_report:
                st.subheader("Study Report")
                st.markdown(artifacts.relaxation_report)
        if artifacts.bound_relaxation_summary is not None:
            st.subheader("Input-Bound Relaxation")
            st.json(artifacts.bound_relaxation_summary)
            if artifacts.bound_relaxation_best_by_experiment is not None:
                display = artifacts.bound_relaxation_best_by_experiment
                columns = _display_columns(
                    display,
                    [
                        "experiment_id",
                        "relaxed_variable",
                        "relaxed_high",
                        "sweep_value",
                        "phi",
                        "feasible",
                        "active_constraint",
                        "constraint_total_pilots_window",
                        "constraint_wg_rap",
                        "constraint_fl_rap",
                        "constraint_ip_rap",
                    ],
                )
                st.dataframe(display[columns].sort_values("phi"), width="stretch", hide_index=True)
        if artifacts.ipug_summary is not None:
            st.subheader("IPUG Counterfactual")
            st.json(artifacts.ipug_summary)
            if artifacts.ipug_evaluations is not None:
                columns = _display_columns(
                    artifacts.ipug_evaluations,
                    [
                        "sweep_value",
                        "phi",
                        "feasible",
                        "active_constraint",
                        "constraint_total_pilots_window",
                        "constraint_wg_rap",
                        "constraint_fl_rap",
                        "constraint_ip_rap",
                    ],
                )
                st.dataframe(
                    artifacts.ipug_evaluations[columns].sort_values("sweep_value"),
                    width="stretch",
                    hide_index=True,
                )

    with authority_tab:
        if artifacts.sensitivity is None:
            st.write("No local finite-difference diagnostic artifact was supplied.")
        else:
            response = st.selectbox(
                "Response",
                sorted(artifacts.sensitivity["response"].astype(str).unique().tolist()),
            )
            subset = artifacts.sensitivity[
                artifacts.sensitivity["response"].astype(str) == response
            ].sort_values("abs_sensitivity", ascending=False)
            st.dataframe(subset.head(20), width="stretch", hide_index=True)

    with candidate_tab:
        candidate_display = nearest_dynamic_misses(evaluations, top_n=min(250, len(evaluations)))
        st.dataframe(candidate_display, width="stretch", hide_index=True)

    with artifact_tab:
        if artifacts.paths.report and artifacts.paths.report.exists():
            st.subheader("Dynamic Control Report")
            st.markdown(artifacts.paths.report.read_text(encoding="utf-8"))
        st.write("Dynamic search summary")
        st.json(summary)
        st.write("Artifact paths")
        st.json({key: str(value) for key, value in artifacts.paths.__dict__.items()})


st.sidebar.header("Dashboard Mode")
dashboard_mode = st.sidebar.radio(
    "Mode",
    ["dynamic", "static_legacy"],
    format_func=lambda value: {
        "dynamic": "Dynamic search results",
        "static_legacy": "Legacy static sliders",
    }[value],
)

if dashboard_mode == "dynamic":
    _render_dynamic_dashboard()
    st.stop()

defaults = default_artifact_paths()
st.warning(
    "Legacy static sliders use the old constant-policy signed surrogate. "
    "They are screening guidance only and are not the default workflow for this branch."
)
with st.sidebar.expander("Legacy static artifacts", expanded=False) as static_artifact_box:
    config_path = _path_input("Config", defaults.config, container=static_artifact_box)
    surrogate_path = _path_input("Signed surrogate", defaults.surrogate, container=static_artifact_box)
    evaluations_path = _path_input("Direct evaluations", defaults.evaluations, container=static_artifact_box)
    verified_path = _path_input("Verified candidates", defaults.verified_candidates, container=static_artifact_box)
    search_summary_path = _path_input("Search summary", defaults.search_summary, container=static_artifact_box)
    verification_summary_path = _path_input(
        "Verification summary",
        defaults.verification_summary,
        container=static_artifact_box,
    )
    envelope_summary_path = _path_input(
        "Envelope summary",
        defaults.envelope_summary,
        container=static_artifact_box,
    )
    report_path = _path_input("Report", defaults.report, container=static_artifact_box)
sweep_points = st.sidebar.slider("Legacy slider sweep points", 25, 201, 121, 8)

try:
    artifacts = _load_artifacts(
        config_path,
        surrogate_path,
        evaluations_path,
        verified_path,
        search_summary_path,
        verification_summary_path,
        envelope_summary_path,
        report_path,
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

config = artifacts.config
conservative_sigma = (
    config.envelope.conservative_sigma if config.envelope is not None else 1.0
)

st.title("Interactive Viability Dashboard")
direct_label = direct_verification_label(config)
st.caption(
    "Live sliders use the signed-RAP surrogate for guidance. "
    f"{direct_label} runs only when requested."
)

control_col, main_col = st.columns([1, 2], gap="large")

with control_col:
    st.subheader("Policy Controls")
    selection_mode = st.radio(
        "Start from",
        ["near_boundary_feasible", "best_margin_feasible", "candidate_id"],
        format_func=lambda value: {
            "near_boundary_feasible": "Near-boundary feasible",
            "best_margin_feasible": "Best-margin feasible",
            "candidate_id": "Specific verified candidate",
        }[value],
    )
    candidate_id = None
    if selection_mode == "candidate_id":
        candidate_ids = artifacts.verified_candidates["candidate_id"].astype(str).tolist()
        candidate_id = st.selectbox("Candidate", candidate_ids)

    selected_row = select_dashboard_candidate(
        artifacts.verified_candidates,
        mode=selection_mode,
        candidate_id=candidate_id,
    )
    selected_values = policy_values_from_row(selected_row, config)
    selected_key = _current_policy_key(selected_row)

    if "viability_policy_values" not in st.session_state:
        st.session_state.viability_policy_values = selected_values
        st.session_state.viability_policy_source = selected_key

    if st.button("Load selected verified candidate", width="stretch"):
        st.session_state.viability_policy_values = selected_values
        st.session_state.viability_policy_source = selected_key
        st.session_state.pop("viability_direct_result", None)

    current_values = {}
    for name, variable in config.policy.variables.items():
        stored_value = st.session_state.viability_policy_values.get(
            name,
            selected_values[name],
        )
        slider_value = _policy_slider(name, variable, stored_value)
        current_values[name] = slider_value
    st.session_state.viability_policy_values = current_values

with st.spinner("Scoring current sliders with signed surrogate..."):
    current_score = score_policy_values(
        artifacts.surrogate,
        config,
        current_values,
        conservative_sigma=conservative_sigma,
    )
    sweeps = [
        local_feasible_sweep(
            artifacts.surrogate,
            config,
            current_values,
            lever,
            conservative_sigma=conservative_sigma,
            max_points=sweep_points,
        )
        for lever in config.policy.variables
    ]

with main_col:
    predicted_phi = float(current_score["predicted_phi"])
    conservative_phi = float(current_score["conservative_phi"])
    status_label = _status_label(predicted_phi, conservative_phi)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted phi", f"{predicted_phi:.3g}")
    metric_cols[1].metric("Conservative phi", f"{conservative_phi:.3g}")
    metric_cols[2].metric("Surrogate status", status_label)
    metric_cols[3].metric(
        "Predicted active constraint",
        str(current_score["predicted_active_constraint"]),
    )
    st.info(
        "Recommended slider ranges are local one-lever sweeps with all other "
        "sliders fixed. They are guidance, not direct verification."
    )

    st.caption(direct_verification_caveat(config))

    if st.button(f"Run {direct_label} for current sliders", type="primary"):
        with st.spinner(f"Running {direct_label}..."):
            st.session_state.viability_direct_result = run_direct_policy(
                current_values,
                config,
            )

    direct_result = st.session_state.get("viability_direct_result")
    if direct_result is None:
        st.warning("Run direct verification to populate trajectory plots for this point.")
    elif direct_result.evaluation.status != "ok":
        st.error(f"Direct verification failed: {direct_result.evaluation.error}")
    else:
        direct_cols = st.columns(4)
        direct_cols[0].metric("Direct phi", f"{direct_result.evaluation.phi:.3g}")
        direct_cols[1].metric(
            "Direct feasible",
            "Yes" if direct_result.evaluation.feasible else "No",
        )
        direct_cols[2].metric(
            "Backend",
            str(direct_result.evaluation.phase_backend),
        )
        direct_cols[3].metric(
            "Direct active constraint",
            str(direct_result.evaluation.active_constraint),
        )
        st.caption(
            "Active value: "
            f"{direct_result.evaluation.active_constraint_value:.3g}"
        )
        st.plotly_chart(
            _plot_inventory_trajectory(
                direct_result.trajectory,
                config.requirements.target_total_pilots,
            ),
            width="stretch",
        )
        st.plotly_chart(
            _plot_rap_trajectory(direct_result.trajectory),
            width="stretch",
        )
        st.plotly_chart(
            _plot_experience_trajectory(
                direct_result.trajectory,
                config.requirements.min_experience_ratio,
            ),
            width="stretch",
        )

with control_col:
    st.subheader("Local Feasible Ranges")
    for sweep in sweeps:
        variable = config.policy.variables[sweep.lever]
        current_value = float(current_values[sweep.lever])
        st.caption(POLICY_LABELS.get(sweep.lever, sweep.lever))
        st.plotly_chart(
            _interval_figure(sweep.lever, current_value, variable, sweep.intervals),
            width="stretch",
        )
        st.write(_interval_text(sweep.intervals))

candidate_tab, envelope_tab, direct_tab, artifact_tab = st.tabs(
    ["Verified Candidates", "Envelope Plots", "Direct Details", "Artifacts"]
)

with candidate_tab:
    display_columns = [
        "candidate_id",
        "design_id",
        "phi",
        "feasible",
        "active_constraint",
        *config.policy.variables,
    ]
    st.dataframe(
        artifacts.verified_candidates[display_columns].sort_values(["phi", "candidate_id"]),
        width="stretch",
        hide_index=True,
    )

with envelope_tab:
    for label, fixed_path, projected_path in envelope_plot_paths(artifacts.envelope_summary):
        st.subheader(label)
        cols = st.columns(2)
        cols[0].image(str(fixed_path), caption="Fixed slice")
        cols[1].image(str(projected_path), caption="Projected envelope")

with direct_tab:
    if direct_result is None:
        st.write("No direct verification has been run in this dashboard session.")
    elif direct_result.evaluation.status != "ok":
        st.error(direct_result.evaluation.error)
    else:
        st.write("Backend")
        st.code(str(direct_result.evaluation.phase_backend))
        st.write("Raw metrics")
        st.json(direct_result.evaluation.raw_metrics)
        st.write("Constraints")
        st.json(direct_result.evaluation.constraints)
        st.write("Trajectory table")
        st.dataframe(direct_result.trajectory, width="stretch", hide_index=True)

with artifact_tab:
    st.write("Search summary")
    st.json(artifacts.search_summary)
    st.write("Verification summary")
    st.json(artifacts.verification_summary)
    if artifacts.paths.report and artifacts.paths.report.exists():
        with st.expander("Report Markdown", expanded=False):
            st.markdown(artifacts.paths.report.read_text(encoding="utf-8"))
