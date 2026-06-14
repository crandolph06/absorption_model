from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.viability.dashboard import (
    POLICY_LABELS,
    DashboardArtifactPaths,
    default_artifact_paths,
    direct_verification_caveat,
    direct_verification_label,
    envelope_plot_paths,
    load_dashboard_artifacts,
    local_feasible_sweep,
    policy_values_from_row,
    run_direct_policy,
    score_policy_values,
    select_dashboard_candidate,
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


def _path_input(label: str, path: Path) -> str:
    return st.sidebar.text_input(label, value=str(path))


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


defaults = default_artifact_paths()
st.sidebar.header("Artifact Paths")
config_path = _path_input("Config", defaults.config)
surrogate_path = _path_input("Signed surrogate", defaults.surrogate)
evaluations_path = _path_input("Direct evaluations", defaults.evaluations)
verified_path = _path_input("Verified candidates", defaults.verified_candidates)
search_summary_path = _path_input("Search summary", defaults.search_summary)
verification_summary_path = _path_input(
    "Verification summary",
    defaults.verification_summary,
)
envelope_summary_path = _path_input("Envelope summary", defaults.envelope_summary)
report_path = _path_input("Report", defaults.report or Path(""))
sweep_points = st.sidebar.slider("Slider sweep points", 25, 201, 121, 8)

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
