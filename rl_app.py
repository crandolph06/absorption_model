import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from stable_baselines3 import PPO

from evaluate_manning_agent import (
    _joblib_load_brain,
    _reraise_numpy_pickle_hint,
)
from src.manning_config import SQUADRON_DATA
from src.manning_engine import CAFSimulation
from src.manning_gym import ManningEnv, SingleActionManningEnv
from src.models import PriorityMode, Qual, monthly_sortie_rap_target

_RUN_MODE_HELP = {
    "ideal": (
        "RL agent can adjust B-Course intake, FLUG/IPUG quotas, max manning %, UTE (1–30 per "
        "squadron), retention (10–100%), and PAA (1–48 per squadron)."
    ),
    "optimistic": (
        "Same levers as ideal. UTE capped at 20.0 per squadron, retention at 65%, PAA at 30 "
        "per squadron."
    ),
    "pragmatic": (
        "Agent adjusts intake, FLUG/IPUG quotas, max manning %, UTE (1–15.0 per squadron), and "
        "retention (10–50%). Squadron PAA stays at seeded values (not a policy lever)."
    ),
    "current": (
        "Fixed UTE (10.0 per squadron), retention (40%), and squadron PAA from initial data. "
        "Agent only adjusts B-Course intake, FLUG/IPUG quotas, and max manning %."
    ),
}
_REWARD_MODE_HELP = {
    "quantity_first": (
        f"Grow toward {ManningEnv.TARGET_TOTAL_PILOTS:,} total pilots (line + staff), then put "
        "more weight on line RAP shortfalls as strength increases."
    ),
    "readiness_first": (
        f"Penalize WG/FL/IP RAP shortfalls first while still rewarding progress toward "
        f"{ManningEnv.TARGET_TOTAL_PILOTS:,} total pilots."
    ),
    "key_staff_first": (
        f"Prioritize key staff: target staff count is {int(ManningEnv.KEY_STAFF_RATIO * 100)}% of "
        f"remaining slots to reach {ManningEnv.TARGET_TOTAL_PILOTS:,} line + staff; then blend "
        "into readiness-first logic."
    ),
}

BRAIN_PATH = "brains/hpc_sortie_brain_multi_output_mlp.pkl"
_PAA_BY_SQUADRON = {sq_id: paa for sq_id, paa, _, _ in SQUADRON_DATA}
_PREDICT_FEATURES = CAFSimulation._PREDICT_FEATURE_COLS
_SYLLABI_NEGLIGIBLE = 0.10
_REMAINING_TOTAL = [
    "remaining_mqt_syllabi_mean",
    "remaining_flug_syllabi_mean",
    "remaining_ipug_syllabi_mean",
]
_REMAINING_SORTIES = [
    "remaining_mqt_syllabi_sorties_only_mean",
    "remaining_flug_syllabi_sorties_only_mean",
    "remaining_ipug_syllabi_sorties_only_mean",
]

_RAP_COLOR_MAP = {
    0: "#22c55e",
    1: "#fef08a",
    2: "#fde047",
    3: "#fdba74",
    4: "#eab308",
    5: "#f97316",
    6: "#ea580c",
    7: "#ef4444",
}
# Centered moving average for upgrade-track chart (phases, not calendar months).
_UPGRADE_SMOOTH_WINDOW_PHASES = 5

_RAP_STATE_LABELS = {
    0: "All Make RAP",
    1: "WG Shortfall",
    2: "FL Shortfall",
    3: "WG + FL Shortfall",
    4: "IP Shortfall",
    5: "WG + IP Shortfall",
    6: "FL + IP Shortfall",
    7: "WG + FL + IP Shortfall",
}


def _phase_moving_average(series: pd.Series, window: int = _UPGRADE_SMOOTH_WINDOW_PHASES) -> pd.Series:
    """Centered moving average along the CAF phase timeline."""
    if series.empty:
        return series
    w = min(window, len(series))
    if w <= 1:
        return series
    return series.rolling(window=w, center=True, min_periods=1).mean()


def _clean_syllabus_preds(raw: np.ndarray) -> np.ndarray:
    vals = np.maximum(raw, 0.0)
    return np.where(vals < _SYLLABI_NEGLIGIBLE, 0.0, vals)


def _brain_features_from_history(df: pd.DataFrame, ute: float) -> pd.DataFrame:
    out = df.copy()
    for col in ("mqt_carry", "flug_carry", "ipug_carry"):
        if col not in out.columns:
            out[col] = 0.0

    out["paa"] = out["squadron_id"].map(_PAA_BY_SQUADRON)
    out["ute"] = float(ute)
    out["exp_ratio"] = out["exp_rat"]
    line_pilots = out["line_pilots"]
    mqt_qty = out["mqt_qty"] + out["mqt_carry"]
    flug_qty = out["flug_qty"] + out["flug_carry"]
    ipug_qty = out["ipug_qty"] + out["ipug_carry"]

    fls = out["fl_qty"].replace(0, 1.0)
    wgs = out["wg_qty"].replace(0, 1.0)
    out["fl_congestion"] = (ipug_qty + flug_qty) / fls
    out["wg_crowding"] = (mqt_qty + flug_qty + ipug_qty) / wgs
    out["sorties_avail"] = out["paa"] * out["ute"]
    out["pilot_to_sortie"] = np.where(
        out["sorties_avail"] != 0, line_pilots / out["sorties_avail"], 0.0
    )
    total_students = mqt_qty + flug_qty + ipug_qty
    denom_tp = line_pilots.replace(0, 1)
    out["ip_ratio"] = out["ip_qty"] / denom_tp
    denom_stud = total_students.replace(0, 0.1)
    out["ip_to_stud_ratio"] = out["ip_qty"] / denom_stud
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def _syllabus_preds_by_timeline(df: pd.DataFrame, brain, ute: float) -> pd.DataFrame:
    feat = _brain_features_from_history(df, ute)
    preds = brain.predict(feat[_PREDICT_FEATURES].fillna(0))
    for i, col in enumerate(_REMAINING_TOTAL):
        feat[col] = _clean_syllabus_preds(preds[:, 6 + i])
    for i, col in enumerate(_REMAINING_SORTIES):
        feat[col] = _clean_syllabus_preds(preds[:, 9 + i])
    feat["timeline"] = feat["year"].astype(str) + " P" + feat["phase"].astype(str)
    return (
        feat.groupby(["year", "phase", "timeline"], as_index=False)[
            _REMAINING_TOTAL + _REMAINING_SORTIES
        ]
        .mean()
        .sort_values(["year", "phase"])
    )


def _get_rap_code(row, use_blue: bool) -> int:
    suffix = "_blue" if use_blue else "_mo"
    wg_rate = row.get(f"wg_rate{suffix}", 0)
    fl_rate = row.get(f"fl_rate{suffix}", 0)
    ip_rate = row.get(f"ip_rate{suffix}", 0)

    code = 0
    if wg_rate < monthly_sortie_rap_target(Qual.WG):
        code += 1
    if fl_rate < monthly_sortie_rap_target(Qual.FL):
        code += 2
    if ip_rate < monthly_sortie_rap_target(Qual.IP):
        code += 4
    return code


def build_df_display(df: pd.DataFrame) -> pd.DataFrame:
    """CAF-wide aggregation per phase (same as manning_app)."""
    work = df.copy()
    work["timeline"] = work["year"].astype(str) + " P" + work["phase"].astype(str)

    for col in ("mqt_qty", "flug_qty", "ipug_qty", "wg_qty", "fl_qty", "ip_qty"):
        if col not in work.columns:
            work[col] = 0
    work["wg_line"] = (work["wg_qty"] - work["mqt_qty"] - work["flug_qty"]).clip(lower=0)
    work["fl_line"] = (work["fl_qty"] - work["ipug_qty"]).clip(lower=0)

    agg = {
        "wg_qty": "sum",
        "fl_qty": "sum",
        "ip_qty": "sum",
        "staff_ips": "sum",
        "staff_fls": "sum",
        "total_pilots": "sum",
        "line_pilots": "sum",
        "exp_rat": "mean",
        "percent_manned": "mean",
        "separated": "sum",
        "retained": "sum",
        "wg_rate_mo": "mean",
        "fl_rate_mo": "mean",
        "ip_rate_mo": "mean",
        "mqt_qty": "mean",
        "flug_qty": "mean",
        "ipug_qty": "mean",
        "wg_line": "mean",
        "fl_line": "mean",
    }
    for col in (
        "wg_rate_blue", "fl_rate_blue", "ip_rate_blue",
        "wg_rate_sim", "fl_rate_sim", "ip_rate_sim",
    ):
        if col in work.columns:
            agg[col] = "mean"

    out = work.groupby(["year", "phase", "timeline"], as_index=False).agg(agg).reset_index()

    # ip_qty in ``out`` is CAF-wide sum (for the population chart); line-mix needs per-sq avg.
    ip_avg = work.groupby(["year", "phase", "timeline"], as_index=False).agg(
        ip_qty_avg=("ip_qty", "mean")
    )
    out = out.merge(ip_avg, on=["year", "phase", "timeline"], how="left")

    for col in ("wg_rate_blue", "fl_rate_blue", "ip_rate_blue"):
        if col not in out.columns:
            out[col] = 0.0
    for col in ("wg_rate_sim", "fl_rate_sim", "ip_rate_sim"):
        if col not in out.columns:
            out[col] = 0.0
    for col in ("mqt_qty", "flug_qty", "ipug_qty", "wg_line", "fl_line", "ip_qty_avg"):
        if col not in out.columns:
            out[col] = 0.0
    return out


_LINE_MIX_STACK = [
    ("mqt_qty", "MQT"),
    ("wg_line", "WG (not MQT/FLUG)"),
    ("flug_qty", "FLUG"),
    ("fl_line", "FL (not IPUG)"),
    ("ipug_qty", "IPUG"),
    ("ip_qty_avg", "IP"),
]
_LINE_MIX_COLORS = ["#f59e0b", "#93c5fd", "#ec4899", "#fda4af", "#6366f1", "#00CC96"]

_ACTION_MODES = {
    "multi_action": {
        "label": "Multiple actions",
        "subdir": "multi_action",
        "help": "All policy levers may change each phase (parallelized / multi-lever training).",
    },
    "single_action": {
        "label": "Single action",
        "subdir": "single_action",
        "help": "At most one lever changes each phase (single-action training).",
    },
}
_GATE_TYPES = {
    "Book Gates": {
        "gate_subdir": "book_gates",
        "flug_window_start": 250,
        "ipug_window_start": 400,
        "help": (
            'Reflects "by-the-book" FLUG and IPUG entry requirements of 250 sorties '
            "and 400 hours, respectively."
        ),
    },
    "Real Gates": {
        "gate_subdir": "real_gates",
        "flug_window_start": 150,
        "ipug_window_start": 300,
        "help": (
            "Reflects typical FLUG and IPUG entry requirements of 150 sorties "
            "and 300 hours, respectively."
        ),
    },
}


def _action_names_for_run_mode(run_mode: str) -> list[str]:
    names = ["Intake", "FLUG", "IPUG", "Max Manning"]
    if run_mode in ("pragmatic", "optimistic", "ideal"):
        names.extend(["UTE", "Retention"])
    if run_mode in ("ideal", "optimistic"):
        names.append("PAA")
    return names


def models_dir_for(action_mode: str, gate_type: str) -> str:
    """Checkpoint directory: saved_models/{single|multi}_action/{book|real}_gates/."""
    return os.path.join(
        "saved_models",
        _ACTION_MODES[action_mode]["subdir"],
        _GATE_TYPES[gate_type]["gate_subdir"],
    )


def model_path_for_gate(
    action_mode: str, gate_type: str, reward_mode: str, run_mode: str
) -> str:
    """Path prefix for PPO checkpoint (Stable-Baselines appends ``.zip``)."""
    return os.path.join(
        models_dir_for(action_mode, gate_type),
        f"ppo_manning_agent_{reward_mode}_{run_mode}",
    )


def _display_actions_from_step(
    action: np.ndarray, action_mode: str, action_names: list[str]
) -> dict[str, int]:
    """Map policy output to per-lever -1/0/1 values for charts and CSV export."""
    if action_mode == "single_action":
        lever_idx = int(np.clip(action[0], 0, len(action_names) - 1))
        direction = int(action[1])
        values = {name: 0 for name in action_names}
        if direction != 1:
            values[action_names[lever_idx]] = direction - 1
        return values
    return {
        name: int(action[i]) - 1
        for i, name in enumerate(action_names)
    }


def run_rl_evaluation(
    run_mode: str,
    reward_mode: str,
    gate_type: str,
    action_mode: str,
    brain,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the RL policy and return per-step controls plus squadron-phase history."""
    gate = _GATE_TYPES[gate_type]
    sim_engine = CAFSimulation(
        annual_intake=200,
        retention_rate=0.40,
        brain=brain,
        flug_window_start=gate["flug_window_start"],
        ipug_window_start=gate["ipug_window_start"],
        max_manning_pct=125,
        staff_priority_mode=PriorityMode.RANDOM,
        use_upgrade_quotas=True,
        round_robin=False,
    )
    env_cls = SingleActionManningEnv if action_mode == "single_action" else ManningEnv
    env = env_cls(sim_engine, run_mode=run_mode, reward_mode=reward_mode)

    model_path = model_path_for_gate(action_mode, gate_type, reward_mode, run_mode)
    models_dir = models_dir_for(action_mode, gate_type)
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(
            f"No trained model at `{model_path}.zip`. "
            f"Place the checkpoint in `{models_dir}/`."
        )
    try:
        model = PPO.load(model_path)
    except Exception as e:
        _reraise_numpy_pickle_hint("PPO.load", e)

    action_names = _action_names_for_run_mode(run_mode)

    obs, _info = env.reset()
    terminated = False
    truncated = False
    step_rows = []

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        n_sq = max(len(sim_engine.squadrons), 1)
        avg_ute = (
            sum(sq.ute for sq in sim_engine.squadrons) / n_sq
            if sim_engine.squadrons
            else 0.0
        )
        total_paa = sum(sq.paa for sq in sim_engine.squadrons) if sim_engine.squadrons else 0

        record = {
            "Year": info["simulated_year"],
            "Phase": info["simulated_phase"],
            "Reward": reward,
            "Total Active Pilots": sim_engine.total_active_pilot_count,
            "Line Pilots": sim_engine.total_line_pilot_count,
            "Total Staff Pilots": sim_engine.total_staff_pilot_count,
            "WG Shortfall": sim_engine.current_wg_shortfall,
            "FL Shortfall": sim_engine.current_fl_shortfall,
            "IP Shortfall": sim_engine.current_ip_shortfall,
            "Intake Target": sim_engine.annual_intake,
            "FLUG Intake": sim_engine.sq_phase_flug_intake,
            "IPUG Intake": sim_engine.sq_phase_ipug_intake,
            "Retention Rate": sim_engine.retention_rate,
            "Max Manning": sim_engine.max_manning,
            "Avg UTE": avg_ute,
            "Total PAA": total_paa,
            "Experience Ratio": sim_engine.experience_ratio,
            "Number of Squadrons": len(sim_engine.squadrons),
        }
        display_actions = _display_actions_from_step(action, action_mode, action_names)
        for name in action_names:
            record[f"Action: {name}"] = display_actions[name]
        if action_mode == "single_action":
            record["Single Action Lever"] = action_names[int(np.clip(action[0], 0, len(action_names) - 1))]
            record["Single Action Direction"] = int(action[1]) - 1
        step_rows.append(record)

    step_df = pd.DataFrame(step_rows)
    hist_df = pd.DataFrame(sim_engine.history)
    return step_df, hist_df


def render_manning_charts(df_display: pd.DataFrame, df_hist: pd.DataFrame, brain, ute_val: float) -> None:
    """Plotly charts aligned with manning_app (population, exp ratio, ops health, syllabi)."""
    st.subheader("Pilot Population by Qualification")
    fig_pop = px.area(
        df_display,
        x="timeline",
        y=["wg_qty", "fl_qty", "ip_qty", "staff_fls", "staff_ips"],
        title="CAF Qualification Mix",
        labels={"value": "Count", "timeline": "Year/Phase"},
        color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#DC8F7E", "#78CAB4"],
    )
    st.plotly_chart(fig_pop, width="stretch")

    st.subheader("Line Pilot Mix: Qualification & Upgrade (Per Squadron Avg)")
    st.caption(
        "Stacked average line pilots per squadron. Upgrade tracks (MQT, FLUG, IPUG) are "
        "split out; remaining wingmen and flight leads are shown without those upgrades."
    )
    mix_cols = [c for c, _ in _LINE_MIX_STACK]
    mix_plot = df_display[["timeline"] + mix_cols].rename(
        columns={c: label for c, label in _LINE_MIX_STACK}
    )
    fig_line_mix = px.area(
        mix_plot,
        x="timeline",
        y=[label for _, label in _LINE_MIX_STACK],
        title="Line pilot composition (avg per squadron)",
        labels={"value": "Pilots", "timeline": "Year/Phase", "variable": "Category"},
        color_discrete_sequence=_LINE_MIX_COLORS,
    )
    fig_line_mix.update_traces(
        hovertemplate="%{fullData.name}: %{y:.2f}<extra></extra>",
    )
    fig_line_mix.update_layout(
        yaxis_title="Pilots per squadron (avg)",
        hovermode="x unified",
        legend=dict(title="Category"),
    )
    st.plotly_chart(fig_line_mix, width="stretch")

    st.divider()
    st.subheader("CAF Experience Ratio")
    use_blue_rap = st.toggle("Blue RAP Only", value=False, key="rl_chart_blue_rap")

    df_exp = df_display.copy()
    df_exp["rap_code"] = df_exp.apply(
        lambda row: _get_rap_code(row, use_blue_rap), axis=1
    )

    fig_exp = go.Figure()
    x_data = df_exp["timeline"].tolist()
    y_data = df_exp["exp_rat"].tolist()
    codes = df_exp["rap_code"].tolist()

    if len(x_data) > 0:
        curr_x = [x_data[0]]
        curr_y = [y_data[0]]
        curr_code = codes[0]

        for i in range(1, len(x_data)):
            if codes[i] != curr_code:
                curr_x.append(x_data[i])
                curr_y.append(y_data[i])
                fig_exp.add_trace(
                    go.Scatter(
                        x=curr_x,
                        y=curr_y,
                        mode="lines",
                        line=dict(color=_RAP_COLOR_MAP.get(curr_code, "grey"), width=3),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                curr_x = [x_data[i]]
                curr_y = [y_data[i]]
                curr_code = codes[i]
            else:
                curr_x.append(x_data[i])
                curr_y.append(y_data[i])

        fig_exp.add_trace(
            go.Scatter(
                x=curr_x,
                y=curr_y,
                mode="lines",
                line=dict(color=_RAP_COLOR_MAP.get(curr_code, "grey"), width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for code, color in _RAP_COLOR_MAP.items():
        fig_exp.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, symbol="square", color=color),
                showlegend=True,
                name=_RAP_STATE_LABELS.get(code, "Unknown"),
            )
        )

    fig_exp.add_trace(
        go.Scatter(
            x=df_exp["timeline"],
            y=df_exp["exp_rat"],
            mode="markers",
            marker=dict(size=0, opacity=0),
            hovertemplate="<b>%{text}</b><br>Exp Ratio: %{y:.1%}<extra></extra>",
            text=[_RAP_STATE_LABELS.get(c, "Unknown") for c in codes],
            showlegend=False,
        )
    )

    fig_exp.update_layout(
        title="Experience Ratio (%)",
        xaxis_title="Year/Phase",
        yaxis_title="Exp Ratio",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=500,
        legend=dict(title="RAP Status", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    fig_exp.add_hline(y=0.60, line_dash="dot", line_color="green", annotation_text="Healthy")
    fig_exp.add_hline(y=0.45, line_dash="dash", line_color="orange", annotation_text="Sortie Inequity")
    fig_exp.add_hline(y=0.40, line_dash="dot", line_color="red", annotation_text="Broken")
    st.plotly_chart(fig_exp, width="stretch")

    st.divider()
    st.subheader("Detailed Operational Health: Sortie Rates vs. Manning")
    ops_view = st.radio(
        "Operational health view",
        options=("Sorties only", "Sorties + Sims"),
        index=0,
        horizontal=True,
        key="rl_ops_health_view",
        help=(
            "Sorties only: solid = all sorties, dotted = blue sorties. "
            "Sorties + Sims: solid = sorties + sims, dotted = blue sorties + sims."
        ),
    )
    include_sims = ops_view == "Sorties + Sims"

    fig_ops = go.Figure()
    for sortie_col, blue_col, sim_col, qual_label, color in (
        ("wg_rate_mo", "wg_rate_blue", "wg_rate_sim", "WG", "#636EFA"),
        ("fl_rate_mo", "fl_rate_blue", "fl_rate_sim", "FL", "#EF553B"),
        ("ip_rate_mo", "ip_rate_blue", "ip_rate_sim", "IP", "#00CC96"),
    ):
        sorties = df_display[sortie_col]
        blue = df_display[blue_col]
        sims = df_display[sim_col] if sim_col in df_display.columns else 0.0
        if include_sims:
            solid_y = sorties + sims
            dotted_y = blue + sims
            solid_name = f"{qual_label} Sorties + Sims"
            dotted_name = f"{qual_label} Blue Sorties + Sims"
        else:
            solid_y = sorties
            dotted_y = blue
            solid_name = f"{qual_label} Rate"
            dotted_name = f"{qual_label} Blue Sorties"

        fig_ops.add_trace(
            go.Scatter(
                x=df_display["timeline"],
                y=solid_y,
                name=solid_name,
                line=dict(color=color),
                hovertemplate="%{y:.2f}<extra></extra>",
            )
        )
        fig_ops.add_trace(
            go.Scatter(
                x=df_display["timeline"],
                y=dotted_y,
                name=dotted_name,
                line=dict(color=color, dash="dot"),
                hovertemplate="%{y:.2f}<extra></extra>",
            )
        )
    fig_ops.add_hline(y=9.0, line_dash="dot", line_color="red", annotation_text="Inexp.")
    fig_ops.add_hline(y=8.0, line_dash="dot", line_color="orange", annotation_text="Exp.")

    fig_ops.add_trace(
        go.Scatter(
            x=df_display["timeline"],
            y=df_display["percent_manned"],
            name="Manning %",
            line=dict(color="white", width=3, dash="dash"),
            yaxis="y2",
            showlegend=False,
        )
    )
    fig_ops.add_trace(
        go.Scatter(
            x=df_display["timeline"],
            y=df_display["exp_rat"],
            name="Exp Ratio",
            line=dict(color="yellow", width=3, dash="dash"),
            yaxis="y2",
            showlegend=False,
        )
    )

    y_left_title = "Monthly Sorties + Sims" if include_sims else "Monthly Sorties"
    fig_ops.update_layout(
        title="Operational Health: Sortie Rates vs. Manning",
        xaxis_title="Year/Phase",
        yaxis=dict(title=y_left_title, side="left", showgrid=True),
        yaxis2=dict(
            title="Percentage",
            overlaying="y",
            side="right",
            range=[0, 2.0],
            tickformat=".0%",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0.0),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=150),
    )
    fig_ops.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=-0.25,
        xanchor="right",
        yanchor="top",
        text=(
            "<b>Right Axis Legend:</b><br>"
            "<span style='color: white; font-weight: bold; font-size: 14px'>- - -</span> Manning %<br>"
            "<span style='color: yellow; font-weight: bold; font-size: 14px'>- - -</span> Exp Ratio"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.3)",
        borderwidth=1,
        borderpad=10,
    )
    st.plotly_chart(fig_ops, width="stretch")

    st.divider()
    st.subheader("Students in Upgrade Tracks")
    use_smooth = st.toggle(
        "Smoothed",
        value=True,
        key="rl_upgrade_smooth",
        help=(
            "On: centered moving average over "
            f"{_UPGRADE_SMOOTH_WINDOW_PHASES} phases (~{_UPGRADE_SMOOTH_WINDOW_PHASES / 3:.1f} years). "
            "Off: raw end-of-phase average per squadron."
        ),
    )
    st.caption(
        "Average line pilots in MQT, FLUG, or IPUG upgrade at end of each phase "
        "(per squadron, then averaged across the CAF)."
        + (
            f" Smoothed view uses a {_UPGRADE_SMOOTH_WINDOW_PHASES}-phase centered moving average."
            if use_smooth
            else ""
        )
    )
    colors_upgrade = {"MQT": "#f59e0b", "FLUG": "#ec4899", "IPUG": "#6366f1"}
    fig_upgrades = go.Figure()
    for col, label in (
        ("mqt_qty", "MQT"),
        ("flug_qty", "FLUG"),
        ("ipug_qty", "IPUG"),
    ):
        raw_y = df_display[col]
        plot_y = _phase_moving_average(raw_y) if use_smooth else raw_y
        fig_upgrades.add_trace(
            go.Scatter(
                x=df_display["timeline"],
                y=plot_y,
                name=label,
                line=dict(color=colors_upgrade[label], width=3),
                mode="lines",
                customdata=raw_y,
                hovertemplate=(
                    f"{label}: %{{y:.2f}} pilots/sq (avg)"
                    + (
                        "<br>Raw this phase: %{customdata:.2f}<extra></extra>"
                        if use_smooth
                        else "<extra></extra>"
                    )
                ),
            )
        )
    y_title = "Pilots in upgrade (per squadron, avg)"
    if use_smooth:
        y_title += " — smoothed"
    fig_upgrades.update_layout(
        xaxis_title="Year/Phase",
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_upgrades.update_yaxes(autorange=True, rangemode="tozero")
    st.plotly_chart(fig_upgrades, width="stretch")

    if df_hist is not None and not df_hist.empty:
        st.divider()
        st.subheader("Incomplete Syllabi (Brain Prediction)")
        st.caption(
            "Y-axis is syllabus-normalized incomplete work per squadron (avg), each phase. "
            "1.0 ≈ one full syllabus incomplete at a typical squadron; values from brain outputs 6–11."
        )
        sorties_only_syll = st.toggle(
            "Sorties only",
            value=True,
            key="rl_chart4_sorties_only",
            help="On: sorties-only remainder (default). Off: sorties + sims (total syllabus).",
        )
        df_syll = _syllabus_preds_by_timeline(df_hist, brain, ute_val)
        if sorties_only_syll:
            syll_series = [
                ("remaining_mqt_syllabi_sorties_only_mean", "MQT"),
                ("remaining_flug_syllabi_sorties_only_mean", "FLUG"),
                ("remaining_ipug_syllabi_sorties_only_mean", "IPUG"),
            ]
            syll_mode_label = "Sorties only"
        else:
            syll_series = [
                ("remaining_mqt_syllabi_mean", "MQT"),
                ("remaining_flug_syllabi_mean", "FLUG"),
                ("remaining_ipug_syllabi_mean", "IPUG"),
            ]
            syll_mode_label = "Total syllabus (sorties + sims)"

        syll_colors = {"MQT": "#f59e0b", "FLUG": "#ec4899", "IPUG": "#6366f1"}
        fig_syll = go.Figure()
        for col, label in syll_series:
            fig_syll.add_trace(
                go.Scatter(
                    x=df_syll["timeline"],
                    y=df_syll[col],
                    name=label,
                    line=dict(color=syll_colors[label], width=3),
                    mode="lines",
                    hovertemplate=(
                        f"{label}: %{{y:.2f}} syllabi/sq (avg)<extra></extra>"
                    ),
                )
            )
        fig_syll.update_layout(
            xaxis_title="Year/Phase",
            yaxis_title=f"Incomplete syllabi per squadron (avg)",
            yaxis_tickformat=".2f",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_syll.update_yaxes(autorange=True, rangemode="tozero")
        st.plotly_chart(fig_syll, width="stretch")


def render_rl_charts(
    df: pd.DataFrame, active_run_mode: str, year_phase_cd: np.ndarray
) -> None:
    """RL-specific metrics and charts (population/shortfalls, levers, action heatmap)."""
    st.markdown("### RL Agent Metrics")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Final Line Pilots", int(df["Line Pilots"].iloc[-1]))
    m2.metric("Final Staff Pilots", int(df["Total Staff Pilots"].iloc[-1]))
    m3.metric("Total Active Pilots", int(df["Total Active Pilots"].iloc[-1]))
    m4.metric("Final Experience Ratio", f"{df['Experience Ratio'].iloc[-1] * 100:.0f}%")
    n_sq = df["Number of Squadrons"].iloc[-1]
    m5.metric("Final Avg WG Shortfall", round(df["WG Shortfall"].iloc[-1] / n_sq, 2))
    m6.metric("Final Avg FL Shortfall", round(df["FL Shortfall"].iloc[-1] / n_sq, 2))
    m7.metric("Final Avg IP Shortfall", round(df["IP Shortfall"].iloc[-1] / n_sq, 2))

    st.markdown("---")
    st.subheader("System Health: Population vs Shortfalls")
    fig_health = go.Figure()
    health_series = [
        ("Line Pilots", "Line Pilots", "blue"),
        ("Total Staff Pilots", "Staff Pilots", "green"),
        ("WG Shortfall", "WG Shortfall", "red"),
        ("FL Shortfall", "FL Shortfall", "orange"),
        ("IP Shortfall", "IP Shortfall", "yellow"),
    ]
    for i, (col, name, color) in enumerate(health_series):
        if i == 0:
            hovertemplate = (
                "Year: %{customdata[0]:.0f}<br>"
                "Phase: %{customdata[1]:.0f}<br>"
                "%{fullData.name}: %{y:.2f}<extra></extra>"
            )
        else:
            hovertemplate = "%{fullData.name}: %{y:.2f}<extra></extra>"
        fig_health.add_trace(
            go.Scatter(
                x=df["Time"],
                y=df[col],
                name=name,
                line=dict(color=color, width=3),
                customdata=year_phase_cd,
                hovertemplate=hovertemplate,
            )
        )
    fig_health.update_layout(
        xaxis_title="Year", yaxis_title="Count", hovermode="x unified"
    )
    st.plotly_chart(fig_health, width="stretch")

    st.subheader("Agent Strategy: Environmental Controls")
    control_metrics = [
        ("Intake Target", "Annual B-Course Intake", "green"),
        ("FLUG Intake", "Phase FLUG Quota", "blue"),
        ("IPUG Intake", "Phase IPUG Quota", "orange"),
        ("Max Manning", "Max Manning Target", "red"),
        ("Avg UTE", "Squadron Average UTE", "purple"),
        ("Retention Rate", "System Retention Rate", "teal"),
    ]
    if active_run_mode in ("ideal", "optimistic"):
        control_metrics.append(("Total PAA", "Total Fleet PAA", "brown"))

    for i in range(0, len(control_metrics), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(control_metrics):
                metric, title, color = control_metrics[i + j]
                fig = px.line(
                    df,
                    x="Time",
                    y=metric,
                    title=title,
                    custom_data=["Year", "Phase"],
                )
                fig.update_traces(
                    line_color=color,
                    hovertemplate=(
                        "Year: %{customdata[0]:.0f}<br>"
                        "Phase: %{customdata[1]:.0f}<br>"
                        f"{title}: %{{y:.2f}}<extra></extra>"
                    ),
                )
                col.plotly_chart(fig, width="stretch")

    st.subheader("Raw Action Matrix")
    st.markdown("*Blue = Decrease (-1) | White = Hold (0) | Red = Increase (+1)*")

    action_value_col = {
        "Intake": "Intake Target",
        "FLUG": "FLUG Intake",
        "IPUG": "IPUG Intake",
        "Max Manning": "Max Manning",
        "UTE": "Avg UTE",
        "Retention": "Retention Rate",
        "PAA": "Total PAA",
    }
    action_label = {-1: "Decrease", 0: "Hold", 1: "Increase"}

    action_cols = [c for c in df.columns if c.startswith("Action: ")]
    action_labels = [c.replace("Action: ", "") for c in action_cols]
    times = df["Time"].tolist()

    z_matrix = []
    customdata = []
    for col in action_cols:
        label = col.replace("Action: ", "")
        value_col = action_value_col[label]
        z_matrix.append(df[col].tolist())
        customdata.append(
            list(
                zip(
                    df["Year"],
                    df["Phase"],
                    df[value_col],
                    df[col].map(action_label),
                )
            )
        )

    fig_actions = go.Figure(
        data=go.Heatmap(
            x=times,
            y=action_labels,
            z=z_matrix,
            customdata=customdata,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            hovertemplate=(
                "Year: %{customdata[0]:.0f}<br>"
                "Phase: %{customdata[1]:.0f}<br>"
                "Current Value: %{customdata[2]:.2f}<br>"
                "Action: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig_actions.update_layout(
        yaxis_title="Action Space",
        xaxis_title="Year",
        xaxis=dict(tickmode="linear"),
    )
    st.plotly_chart(fig_actions, width="stretch")


@st.cache_resource
def load_sortie_brain(brain_path: str, brain_mtime: float):
    return _joblib_load_brain(brain_path)


st.set_page_config(page_title="RL Agent Evaluator", layout="wide")

st.title("Manning RL Agent: 20-Year Policy Evaluation")

if not os.path.exists(BRAIN_PATH):
    st.error(
        f"🚨 '{BRAIN_PATH}' not found! Copy the HPC artifact or run "
        "`hpc_train_brain_multi_output.py`."
    )
    st.stop()

_brain_mtime = os.path.getmtime(BRAIN_PATH)
cached_brain = load_sortie_brain(BRAIN_PATH, _brain_mtime)

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col1:
    run_mode = st.selectbox(
        "Run Mode", ["pragmatic", "optimistic", "current", "ideal"]
    )
    st.caption(_RUN_MODE_HELP[run_mode])
with col2:
    reward_mode = st.selectbox(
        "Reward Mode", ["readiness_first", "quantity_first", "key_staff_first"]
    )
    st.caption(_REWARD_MODE_HELP[reward_mode])
with col3:
    action_mode = st.selectbox(
        "Action Mode",
        list(_ACTION_MODES.keys()),
        index=0,
        format_func=lambda key: _ACTION_MODES[key]["label"],
    )
    st.caption(_ACTION_MODES[action_mode]["help"])
with col4:
    gate_type = st.selectbox("Gate Type", list(_GATE_TYPES.keys()), index=0)
    st.caption(_GATE_TYPES[gate_type]["help"])
with col5:
    st.write("")
    st.write("")
    run_button = st.button("🚀 Run 20-Year Simulation")

_expected_model = model_path_for_gate(action_mode, gate_type, reward_mode, run_mode)
if not os.path.exists(f"{_expected_model}.zip"):
    st.warning(
        f"No model file at `{_expected_model}.zip`. "
        f"Copy the trained checkpoint into `{models_dir_for(action_mode, gate_type)}/`."
    )

if run_button:
    with st.spinner(
        f"Evaluating {_ACTION_MODES[action_mode]['label']} / {gate_type} / "
        f"{run_mode} / {reward_mode}..."
    ):
        try:
            step_df, hist_df = run_rl_evaluation(
                run_mode=run_mode,
                reward_mode=reward_mode,
                gate_type=gate_type,
                action_mode=action_mode,
                brain=cached_brain,
            )
            st.session_state["eval_df"] = step_df
            st.session_state["hist_df"] = hist_df
            st.session_state["run_mode"] = run_mode
            st.session_state["reward_mode"] = reward_mode
            st.session_state["gate_type"] = gate_type
            st.session_state["action_mode"] = action_mode
        except Exception as e:
            st.error(f"Error loading model or running simulation: {e}")

if "eval_df" in st.session_state:
    df = st.session_state["eval_df"]
    hist_df = st.session_state.get("hist_df")
    active_run_mode = st.session_state.get("run_mode", run_mode)

    df["Time"] = df["Year"] + (df["Phase"] - 1) / 3
    year_phase_cd = df[["Year", "Phase"]].to_numpy()
    ute_val = float(df["Avg UTE"].iloc[-1]) if "Avg UTE" in df.columns else 10.0

    render_rl_charts(df, active_run_mode, year_phase_cd)

    if hist_df is not None and not hist_df.empty:
        df_display = build_df_display(hist_df)
        end_year = int(df_display["year"].iloc[-1])

        st.markdown("---")
        # st.markdown("### CAF Status (from simulation history)")
        # c1, c2, c3, c4, c5 = st.columns(5)
        # c1.metric("Final Total Pilots", int(df_display["total_pilots"].iloc[-1]))
        # c2.metric("Final Total Line Pilots", int(df_display["line_pilots"].iloc[-1]))
        # c3.metric(
        #     "Final Total Non-Line Pilots",
        #     int(df_display["staff_ips"].iloc[-1] + df_display["staff_fls"].iloc[-1]),
        # )
        # c4.metric("Final Line Exp Ratio", f"{df_display['exp_rat'].iloc[-1] * 100:.1f}%")
        # c5.metric("Total Separations", int(df_display["separated"].sum()))

        # st.markdown(f"### CAF Dashboard at Year {end_year}")
        render_manning_charts(df_display, hist_df, cached_brain, ute_val)

        st.download_button(
            label="Download Squadron-Phase History (CSV)",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="rl_simulation_history.csv",
            mime="text/csv",
        )
