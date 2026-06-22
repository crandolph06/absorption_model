import contextlib
import io
import itertools
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.engine import create_pilots, phase_upgrade_metrics, run_phase_simulation
from src.models import (
    NUM_FLS_UTC_1,
    NUM_FLS_UTC_2,
    NUM_FLS_UTC_3,
    NUM_WG_UTC_1,
    NUM_WG_UTC_2,
    NUM_WG_UTC_3,
    AssignedUTCRank,
    Assignment,
    Qual,
    SquadronConfig,
    Upgrade,
    monthly_sortie_rap_target,
)
from src.rap_state import rap_assess, rap_state_code
from src.simulation_config import DEFAULT_PHASE_LENGTH_DAYS, SimulationConfig

_UTC_CHART_LABELS = ("UTC 1", "UTC 2", "UTC 3", "Unassigned")
_SUMMARY_STATUS_SCOPE_OPTIONS = ("Overall",) + _UTC_CHART_LABELS
_UTC_STATUS_LABELS = ("UTC 1", "UTC 2", "UTC 3")
_UTC_RANK_BY_LABEL = {
    "UTC 1": AssignedUTCRank.UTC_1,
    "UTC 2": AssignedUTCRank.UTC_2,
    "UTC 3": AssignedUTCRank.UTC_3,
    "Unassigned": AssignedUTCRank.UNASSIGNED,
}
_UTC_SLOT_REQUIREMENTS: dict[AssignedUTCRank, tuple[int, int]] = {
    AssignedUTCRank.UTC_1: (NUM_FLS_UTC_1, NUM_WG_UTC_1),
    AssignedUTCRank.UTC_2: (NUM_FLS_UTC_2, NUM_WG_UTC_2),
    AssignedUTCRank.UTC_3: (NUM_FLS_UTC_3, NUM_WG_UTC_3),
}


def _sim_config(utc_wise: bool) -> SimulationConfig:
    return SimulationConfig(
        phase_length_days=DEFAULT_PHASE_LENGTH_DAYS,
        utc_wise_allocation=utc_wise,
        # Dashboard RAP charts assume syllabus first, then CT on leftover iron.
        upgrade_sortie_fraction=None,
    )


def _utc_filter_cache_key(utc_filter: AssignedUTCRank | None) -> int | None:
    return None if utc_filter is None else int(utc_filter)


def _filter_pilots_by_utc(
    pilots: list,
    utc_filter: AssignedUTCRank | None,
) -> list:
    if utc_filter is None:
        return pilots
    return [p for p in pilots if p.assigned_utc == utc_filter]


def _utc_chart_filter(key: str, utc_wise: bool) -> AssignedUTCRank | None:
    if not utc_wise:
        return None
    label = st.selectbox("UTC", _UTC_CHART_LABELS, key=key)
    return _UTC_RANK_BY_LABEL[label]


def _utc_label(rank: AssignedUTCRank | None) -> str | None:
    if rank is None:
        return None
    for label, utc_rank in _UTC_RANK_BY_LABEL.items():
        if utc_rank == rank:
            return label
    return None


def _utc_chart_layout_title(utc_filter: AssignedUTCRank | None) -> dict:
    """Plotly renders ``title=None`` as the literal string ``undefined``."""
    label = _utc_label(utc_filter)
    return {"title": label} if label else {}
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
_PREDICT_FEATURES = [
    "paa", "ute", "exp_ratio", "ip_ratio", "fl_congestion",
    "wg_crowding", "sorties_avail", "pilot_to_sortie", "ip_to_stud_ratio",
]
_SELF_TERM_HEAT_CODE = 99
_SELF_TERM_GREY = "#9ca3af"
_RATE_COLS = [
    "wg_monthly", "fl_monthly", "ip_monthly",
    "wg_blue_monthly", "fl_blue_monthly", "ip_blue_monthly",
    "wg_red_monthly", "fl_red_monthly", "ip_red_monthly",
    "wg_red_pct", "fl_red_pct", "ip_red_pct",
]

# ==============================================================================
# 1. PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(page_title="Pilot Supply Chain Analytics", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. ALLOCATION ENGINE
# ==============================================================================

@st.cache_resource
def load_brain():
    model_path = "brains/hpc_sortie_brain_multi_output_mlp.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    st.error(f"⚠️ Brain file not found at {model_path}")
    st.stop()


def is_valid_config(total, exp, ip_q, mqt, flug, ipug):
    experienced = int(total * exp)
    wg_count = total - experienced
    fl_count = experienced - ip_q
    if ip_q > experienced:
        return False
    if experienced > total:
        return False
    if (mqt + flug + ipug + ip_q) > total:
        return False
    if (mqt + flug) > wg_count:
        return False
    if ipug > fl_count:
        return False
    return True


def _prepare_input_frame(df_inputs: pd.DataFrame) -> pd.DataFrame:
    df = df_inputs.copy()
    base_features = [
        "paa", "ute", "exp_ratio", "total_pilots", "mqt_qty", "flug_qty",
        "ipug_qty", "wg_qty", "fl_qty", "ip_qty",
    ]
    for col in base_features:
        if col not in df.columns:
            df[col] = 0
    experienced = (df["total_pilots"] * df["exp_ratio"]).astype(int)
    df["wg_qty"] = df["total_pilots"] - experienced
    df["fl_qty"] = experienced - df["ip_qty"]
    ips = df["ip_qty"].replace(0, 1.0)
    df["mqt_load"] = df["mqt_qty"] / ips
    df["flug_load"] = df["flug_qty"] / ips
    df["ipug_load"] = df["ipug_qty"] / ips
    df["fl_congestion"] = (df["ipug_qty"] + df["flug_qty"]) / df["fl_qty"].replace(0, 1.0)
    df["wg_crowding"] = (df["mqt_qty"] + df["flug_qty"] + df["ipug_qty"]) / df["wg_qty"].replace(0, 1.0)
    df["sorties_avail"] = df["paa"] * df["ute"]
    df["pilot_to_sortie"] = df["total_pilots"] / df["sorties_avail"].replace(0, 1.0)
    df["total_students"] = df["mqt_qty"] + df["flug_qty"] + df["ipug_qty"]
    df["ip_ratio"] = df["ip_qty"] / df["total_pilots"].replace(0, 1)
    df["ip_to_stud_ratio"] = df["ip_qty"] / df["total_students"].replace(0, 0.1)
    return df.replace([np.inf, -np.inf], 0).fillna(0)


def _clean_syllabus_preds(raw: np.ndarray) -> np.ndarray:
    vals = np.maximum(raw, 0.0)
    return np.where(vals < _SYLLABI_NEGLIGIBLE, 0.0, vals)


def _attach_syllabus_from_upgrade_metrics(df: pd.DataFrame, metrics: dict) -> None:
    df["remaining_mqt_syllabi_mean"] = metrics["remaining_mqt_syllabi"]
    df["remaining_flug_syllabi_mean"] = metrics["remaining_flug_syllabi"]
    df["remaining_ipug_syllabi_mean"] = metrics["remaining_ipug_syllabi"]
    df["remaining_mqt_syllabi_sorties_only_mean"] = metrics["remaining_mqt_syllabi_sorties_only"]
    df["remaining_flug_syllabi_sorties_only_mean"] = metrics["remaining_flug_syllabi_sorties_only"]
    df["remaining_ipug_syllabi_sorties_only_mean"] = metrics["remaining_ipug_syllabi_sorties_only"]


def _empty_metrics_row() -> dict:
    row = {col: np.nan for col in _RATE_COLS}
    for col in _REMAINING_TOTAL + _REMAINING_SORTIES:
        row[col] = np.nan
    row["self_terminating_phase"] = False
    row["deferral_due_to_ip"] = False
    return row


def _self_term_label(row) -> str | None:
    if not row.get("self_terminating_phase"):
        return None
    if row.get("deferral_due_to_ip"):
        return "Self-Terminated (Insufficient IPs Available)"
    return "Self-Terminated (Insufficient FLs Available)"


def _self_term_mask(df: pd.DataFrame) -> pd.Series:
    if "self_terminating_phase" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["self_terminating_phase"].fillna(False).astype(bool)


def _utc_line_fl_count(pilots: list, utc: AssignedUTCRank) -> int:
    return sum(
        1
        for p in pilots
        if p.assigned_utc == utc
        and p.active
        and p.current_assignment == Assignment.LINE
        and p.qual in (Qual.FL, Qual.IP)
    )


def _utc_line_wg_count(pilots: list, utc: AssignedUTCRank) -> int:
    return sum(
        1
        for p in pilots
        if p.assigned_utc == utc
        and p.active
        and p.current_assignment == Assignment.LINE
        and p.qual == Qual.WG
        and p.upgrade != Upgrade.MQT
    )


def _utc_pilots_meet_rap(pilots: list, utc: AssignedUTCRank) -> bool:
    for p in pilots:
        if not p.active or p.assigned_utc != utc:
            continue
        if p.target_sorties <= 0:
            continue
        if p.sortie_rap_monthly < p.target_sorties - 1e-9:
            return False
    return True


def _assess_utc_mission_ready(
    pilots: list,
    utc: AssignedUTCRank,
) -> tuple[bool, str]:
    fl_req, wg_req = _UTC_SLOT_REQUIREMENTS[utc]
    fl_count = _utc_line_fl_count(pilots, utc)
    wg_count = _utc_line_wg_count(pilots, utc)
    if fl_count < fl_req:
        if wg_count < wg_req:
            if not _utc_pilots_meet_rap(pilots, utc):
                return False, "UTC not Mission Ready (WG + FL Count and RAP)"
            return False, "UTC not Mission Ready (WG + FL Count)"
        else:
            if not _utc_pilots_meet_rap(pilots, utc):
                return False, "UTC not Mission Ready (FL Count and RAP)"
            return False, "UTC not Mission Ready (FL Count)"
    elif wg_count < wg_req:
        if not _utc_pilots_meet_rap(pilots, utc):
            return False, "UTC not Mission Ready (WG Count and RAP)"
        return False, "UTC not Mission Ready (WG Count)"
    if not _utc_pilots_meet_rap(pilots, utc):
        return False, "UTC not Mission Ready (RAP)"
    return True, "UTC Mission Ready"


def _format_utc_status_label(label: str) -> str:
    prefix = "UTC not Mission Ready ("
    if label.startswith(prefix) and label.endswith(")"):
        reason = label[len(prefix):-1]
        return f"UTC not Mission Ready<br>({reason})"
    return label


def _rap_cohort_group_pilots(cohort: list) -> dict[str, list]:
    """RAP cohorts for a pilot list (same rules as ``rap_assess``)."""
    return {
        "WG": [p for p in cohort if p.qual == Qual.WG and p.upgrade != Upgrade.MQT],
        "FL": [p for p in cohort if p.qual == Qual.FL],
        "IP": [p for p in cohort if p.qual == Qual.IP],
    }


def _rap_metrics_from_cohort(cohort: list) -> dict:
    groups = _rap_cohort_group_pilots(cohort)
    rap, blue_rap, red = rap_assess(cohort)

    def _metric(group: str, value: float) -> float:
        return value if groups[group] else np.nan

    wg_monthly = _metric("WG", rap["WG"][1])
    fl_monthly = _metric("FL", rap["FL"][1])
    ip_monthly = _metric("IP", rap["IP"][1])
    wg_blue_monthly = _metric("WG", blue_rap["WG"][1])
    fl_blue_monthly = _metric("FL", blue_rap["FL"][1])
    ip_blue_monthly = _metric("IP", blue_rap["IP"][1])
    return {
        "wg_monthly": wg_monthly,
        "fl_monthly": fl_monthly,
        "ip_monthly": ip_monthly,
        "wg_blue_monthly": wg_blue_monthly,
        "fl_blue_monthly": fl_blue_monthly,
        "ip_blue_monthly": ip_blue_monthly,
        "wg_red_monthly": wg_monthly - wg_blue_monthly if groups["WG"] else np.nan,
        "fl_red_monthly": fl_monthly - fl_blue_monthly if groups["FL"] else np.nan,
        "ip_red_monthly": ip_monthly - ip_blue_monthly if groups["IP"] else np.nan,
        "wg_red_pct": _metric("WG", red["WG"][0]),
        "fl_red_pct": _metric("FL", red["FL"][0]),
        "ip_red_pct": _metric("IP", red["IP"][0]),
        "rap_state_code": rap_state_code(rap),
        "blue_rap_state_code": rap_state_code(blue_rap),
    }


def _rap_metrics_row_from_pilots(
    pilots: list,
    utc_filter: AssignedUTCRank | None,
    *,
    base: dict,
) -> dict:
    cohort = _filter_pilots_by_utc(pilots, utc_filter)
    return {**base, **_rap_metrics_from_cohort(cohort)}


def _simulate_one_row(
    row: dict,
    sim_config: SimulationConfig,
    utc_filter: AssignedUTCRank | None = None,
) -> tuple[dict, list | None]:
    total = int(row["total_pilots"])
    exp = float(row["exp_ratio"])
    ip_q = int(row["ip_qty"])
    mqt = int(row["mqt_qty"])
    flug = int(row["flug_qty"])
    ipug = int(row["ipug_qty"])
    if not is_valid_config(total, exp, ip_q, mqt, flug, ipug):
        return {**row, **_empty_metrics_row()}, None
    cfg = SquadronConfig(
        paa=int(row["paa"]),
        ute=float(row["ute"]),
        experience_ratio=exp,
        ip_qty=ip_q,
        mqt_students=mqt,
        flug_students=flug,
        ipug_students=ipug,
        total_pilots=total,
        id=99,
    )
    try:
        pilots = create_pilots(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            final_pilots = run_phase_simulation(
                cfg, pilots, sim_config=sim_config, auto_graduate=False,
            )
        cohort = _filter_pilots_by_utc(final_pilots, utc_filter)
        upgrade = phase_upgrade_metrics(final_pilots)
    except ValueError:
        return {**row, **_empty_metrics_row()}, None

    out = {**row, **_rap_metrics_from_cohort(cohort)}
    out["self_terminating_phase"] = bool(cfg.self_terminating_phase)
    out["deferral_due_to_ip"] = bool(cfg.deferral_due_to_ip)
    _attach_syllabus_from_upgrade_metrics(out, upgrade)
    return out, final_pilots


@st.cache_data(show_spinner=False)
def _single_point_summary_physics(
    paa: int,
    ute: float,
    total_pilots: int,
    exp_ratio: float,
    ip_qty: int,
    mqt_qty: int,
    flug_qty: int,
    ipug_qty: int,
    utc_wise: bool,
) -> tuple[dict, dict[int, tuple[bool, str]], list | None]:
    row = {
        "paa": paa,
        "ute": ute,
        "total_pilots": total_pilots,
        "exp_ratio": exp_ratio,
        "ip_qty": ip_qty,
        "mqt_qty": mqt_qty,
        "flug_qty": flug_qty,
        "ipug_qty": ipug_qty,
    }
    out, pilots = _simulate_one_row(row, _sim_config(utc_wise))
    if pilots is None:
        invalid = (False, "Invalid / N/A")
        utc_map = {int(utc): invalid for utc in _UTC_SLOT_REQUIREMENTS}
    else:
        utc_map = {
            int(utc): _assess_utc_mission_ready(pilots, utc)
            for utc in _UTC_SLOT_REQUIREMENTS
        }
    return out, utc_map, pilots


@st.cache_data(show_spinner=False)
def _run_physics_metrics(
    df_inputs: pd.DataFrame,
    utc_wise: bool = False,
    utc_filter_value: int | None = None,
) -> pd.DataFrame:
    utc_filter = (
        AssignedUTCRank(utc_filter_value) if utc_filter_value is not None else None
    )
    sim_config = _sim_config(utc_wise)
    rows = []
    for _, raw in df_inputs.iterrows():
        out, _ = _simulate_one_row(raw.to_dict(), sim_config, utc_filter)
        rows.append(out)
    return pd.DataFrame(rows)


def predict_metrics_ml(df_inputs, brain):
    df = _prepare_input_frame(df_inputs)
    targets = [
        "wg_monthly", "fl_monthly", "ip_monthly",
        "wg_blue_monthly", "fl_blue_monthly", "ip_blue_monthly",
    ]
    X = df[_PREDICT_FEATURES].fillna(0)
    preds = brain.predict(X)
    for i, t in enumerate(targets):
        df[t] = preds[:, i]
    df["wg_red_monthly"] = df["wg_monthly"] - df["wg_blue_monthly"]
    df["fl_red_monthly"] = df["fl_monthly"] - df["fl_blue_monthly"]
    df["ip_red_monthly"] = df["ip_monthly"] - df["ip_blue_monthly"]
    df["wg_red_pct"] = df["wg_red_monthly"] / df["wg_monthly"].replace(0, 1)
    df["fl_red_pct"] = df["fl_red_monthly"] / df["fl_monthly"].replace(0, 1)
    df["ip_red_pct"] = df["ip_red_monthly"] / df["ip_monthly"].replace(0, 1)
    for i, col in enumerate(_REMAINING_TOTAL):
        df[col] = _clean_syllabus_preds(preds[:, 6 + i])
    for i, col in enumerate(_REMAINING_SORTIES):
        df[col] = _clean_syllabus_preds(preds[:, 9 + i])
    df["self_terminating_phase"] = False
    df["deferral_due_to_ip"] = False
    return df


def compute_metrics(
    df_inputs,
    mode: str,
    brain=None,
    *,
    utc_wise: bool = False,
    utc_filter: AssignedUTCRank | None = None,
) -> pd.DataFrame:
    if mode == "Physics":
        return _run_physics_metrics(
            df_inputs,
            utc_wise=utc_wise,
            utc_filter_value=_utc_filter_cache_key(utc_filter),
        )
    if brain is None:
        raise ValueError("ML mode requires a loaded brain model")
    return predict_metrics_ml(df_inputs, brain)

def calculate_rap_code(row, is_blue=False):
    """RAP status mask (WG=1, FL=2, IP=4). Skips empty cohorts."""
    code_col = "blue_rap_state_code" if is_blue else "rap_state_code"
    if code_col in row.index and pd.notna(row.get(code_col)):
        return int(row[code_col])

    suffix = "_blue_monthly" if is_blue else "_monthly"
    wg = row.get(f"wg{suffix}", 0)
    fl = row.get(f"fl{suffix}", 0)
    ip = row.get(f"ip{suffix}", 0)
    if pd.isna(wg) or pd.isna(fl) or pd.isna(ip):
        return np.nan

    wg_n = max(0, int(row.get("wg_qty", 0)) - int(row.get("mqt_qty", 0)))
    fl_n = int(row.get("fl_qty", 0))
    ip_n = int(row.get("ip_qty", 0))

    code = 0
    if wg_n > 0 and wg < monthly_sortie_rap_target(Qual.WG):
        code += 1
    if fl_n > 0 and fl < monthly_sortie_rap_target(Qual.FL):
        code += 2
    if ip_n > 0 and ip < monthly_sortie_rap_target(Qual.IP):
        code += 4
    return code

state_labels_dict = {
    0: "All Make RAP", 1: "WG Shortfall", 2: "FL Shortfall", 3: "WG+FL Shortfall",
    4: "IP Shortfall", 5: "WG+IP Shortfall", 6: "FL+IP Shortfall", 7: "WG+FL+IP Shortfall"
}

# ==============================================================================
# 3. SIDEBAR & CONSTANTS
# ==============================================================================
with st.sidebar:
    st.header("⚙️ System Inputs")

    # allocation_mode = st.radio(
    #     "Allocation engine",
    #     options=["Physics", "ML"],
    #     index=0,
    #     help="Physics runs ``run_phase_simulation`` (default). ML uses the trained sortie brain.",
    # )
    # use_physics = allocation_mode == "Physics"

    allocation_mode = "Physics"
    use_physics = True
    brain = None if use_physics else load_brain()

    inputs = {}
    inputs['paa'] = st.slider("PAA (Aircraft)", 18, 24, 21, 1)
    inputs['ute'] = st.slider("UTE Rate", 6.0, 21.0, 10.0, 0.5)
    inputs['total_pilots'] = st.slider("Total Pilots", 25, 50, 30, 1)
    inputs['exp_ratio'] = st.slider("Experience Ratio", 0.0, 1.0, 0.45, 0.01)
    inputs['ip_qty'] = st.slider("Active IPs", 3, 10, 5, 1)
    
    st.divider()
    st.subheader("Student Load")
    inputs['mqt_qty'] = st.number_input("MQT Students", 0, 15, 4)
    inputs['flug_qty'] = st.number_input("FLUG Students", 0, 15, 4)
    inputs['ipug_qty'] = st.number_input("IPUG Students", 0, 15, 2)

    st.divider()
    utc_wise_mode = st.radio(
        "UTC-wise allocation",
        options=["Off", "On"],
        index=0,
        help=(
            "When on, roster slots are ranked into UTCs and sortie allocation "
            "prioritizes UTC 1 → 2 → 3 → unassigned."
        ),
    )
    utc_wise = utc_wise_mode == "On"

# Ranges for 1D Sweeps
sweep_ranges = {
    'ute': np.arange(6.0, 24.1, 0.5),
    'paa': np.arange(12, 31, 1),
    'total_pilots': np.arange(20, 81, 1),
    'exp_ratio': np.arange(0.20, 0.81, 0.02),
}

# Defaults based on System Inputs slider bounds
input_axis_bounds = {
    'ute': (6.0, 21.0),
    'paa': (18, 24),
    'total_pilots': (25, 50),
    'exp_ratio': (0.0, 1.0),
}

# ==============================================================================
# 4. DATA GENERATION HELPERS
# ==============================================================================
def generate_1d_sweep(
    x_var,
    *,
    utc_filter: AssignedUTCRank | None = None,
):
    """Creates a synthetic dataframe varying only the x_var."""
    x_vals = sweep_ranges.get(x_var, np.arange(0, 10))
    df_sweep = pd.DataFrame([inputs] * len(x_vals))
    df_sweep[x_var] = x_vals
    if use_physics:
        with st.spinner(f"Running {x_var} sweep…"):
            return compute_metrics(
                df_sweep,
                allocation_mode,
                brain,
                utc_wise=utc_wise,
                utc_filter=utc_filter,
            )
    return compute_metrics(df_sweep, allocation_mode, brain)

# ==============================================================================
# 5. MAIN UI & CHARTS
# ==============================================================================
st.title("✈️ Pilot Supply Chain Analytics")
engine_label = "physics allocator" if use_physics else "ML brain"
utc_caption = " · UTC-wise allocation on" if utc_wise else ""
st.caption(
    f"Interactive Dashboard for RAP Equity and Sortie Composition — "
    f"120 Day Training Phase Snapshot ({engine_label}{utc_caption})"
)
if use_physics:
    utc_hint = (
        " Enable **UTC-wise allocation** in the sidebar to filter equity, composition, "
        "and heatmap charts by UTC ."
        if not utc_wise
        else " UTC filters on charts show RAP rates for that UTC only."
    )
    st.info(
        "Charts refresh from **run_phase_simulation** in real time. "
        f"Large sweeps are cached after the first run.{utc_hint}",
        icon="⚙️",
    )

col_main, col_summary = st.columns([3, 1])

with col_main:
    # --- CHART 1: EQUITY (1D Sweep) ---
    st.subheader("📊 Sortie Equity (Total Monthly)")
    equity_utc = _utc_chart_filter("equity_utc", utc_wise)
    x_options = list(sweep_ranges.keys())
    x_var_equity = st.selectbox("X-Axis Variable", x_options, index=0, key="equity_x")

    x_vals = sweep_ranges[x_var_equity]
    sweep_min = float(np.min(x_vals))
    sweep_max = float(np.max(x_vals))
    default_min, default_max = input_axis_bounds.get(x_var_equity, (sweep_min, sweep_max))
    default_min = max(sweep_min, float(default_min))
    default_max = min(sweep_max, float(default_max))

    if default_min > default_max:
        default_min, default_max = sweep_min, sweep_max

    if np.issubdtype(x_vals.dtype, np.integer):
        x_display_min, x_display_max = st.slider(
            "Displayed X-Axis Range",
            min_value=int(sweep_min),
            max_value=int(sweep_max),
            value=(int(default_min), int(default_max)),
            step=1,
            key=f"equity_x_range_{x_var_equity}",
        )
    else:
        step = 0.5 if x_var_equity == "ute" else 0.02
        x_display_min, x_display_max = st.slider(
            "Displayed X-Axis Range",
            min_value=sweep_min,
            max_value=sweep_max,
            value=(default_min, default_max),
            step=step,
            key=f"equity_x_range_{x_var_equity}",
        )

    df_equity = generate_1d_sweep(x_var_equity, utc_filter=equity_utc)
    df_equity = df_equity[
        (df_equity[x_var_equity] >= x_display_min) &
        (df_equity[x_var_equity] <= x_display_max)
    ]
    df_equity = df_equity[~_self_term_mask(df_equity)]

    fig_equity = go.Figure()
    colors_total = {'wg_monthly': '#3b82f6', 'fl_monthly': '#8b5cf6', 'ip_monthly': '#10b981'}
    names = {'wg_monthly': 'Wingman', 'fl_monthly': 'Flight Lead', 'ip_monthly': 'Instructor'}

    for col in ['wg_monthly', 'fl_monthly', 'ip_monthly']:
        fig_equity.add_trace(go.Scatter(
            x=df_equity[x_var_equity], y=df_equity[col], name=names[col],
            line=dict(color=colors_total[col], width=3), mode='lines',
            hovertemplate='<b>%{x}</b><br>Sorties: %{y:.1f}<extra></extra>',
        ))
        
    fig_equity.add_hline(y=9.0, line_dash="dot", line_color="#b91c1c", annotation_text="9.0 Inexp.")
    fig_equity.add_hline(y=8.0, line_dash="dot", line_color="#fca5a5", annotation_text="8.0 Exp.")
    fig_equity.update_layout(
        **_utc_chart_layout_title(equity_utc),
        xaxis_title=x_var_equity.upper(),
        yaxis_title='Monthly Sorties',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
    )
    
    fig_equity.update_xaxes(range=[x_display_min, x_display_max], autorange=False)
    
    st.plotly_chart(fig_equity, width='stretch')

    # --- CHART 2: COMPOSITION (1D Sweep) ---
    st.write("---")
    st.subheader("🧱 Sortie Composition")
    comp_utc = _utc_chart_filter("comp_utc", utc_wise)
    col_comp_1, col_comp_2 = st.columns([2, 1])
    with col_comp_1:
        x_var_comp = st.selectbox("X-Axis Variable", x_options, index=3, key="comp_x") # Default exp_ratio
    with col_comp_2:
        st.write("") 
        show_trends = st.toggle("Show Total Trendlines", value=False)

    x_vals_comp = sweep_ranges[x_var_comp]
    comp_sweep_min = float(np.min(x_vals_comp))
    comp_sweep_max = float(np.max(x_vals_comp))
    comp_default_min, comp_default_max = input_axis_bounds.get(x_var_comp, (comp_sweep_min, comp_sweep_max))
    comp_default_min = max(comp_sweep_min, float(comp_default_min))
    comp_default_max = min(comp_sweep_max, float(comp_default_max))

    if comp_default_min > comp_default_max:
        comp_default_min, comp_default_max = comp_sweep_min, comp_sweep_max

    if np.issubdtype(x_vals_comp.dtype, np.integer):
        x_comp_min, x_comp_max = st.slider(
            "Displayed Composition X-Axis Range",
            min_value=int(comp_sweep_min),
            max_value=int(comp_sweep_max),
            value=(int(comp_default_min), int(comp_default_max)),
            step=1,
            key=f"comp_x_range_{x_var_comp}",
        )
    else:
        comp_step = 0.5 if x_var_comp == "ute" else 0.02
        x_comp_min, x_comp_max = st.slider(
            "Displayed Composition X-Axis Range",
            min_value=comp_sweep_min,
            max_value=comp_sweep_max,
            value=(comp_default_min, comp_default_max),
            step=comp_step,
            key=f"comp_x_range_{x_var_comp}",
        )

    df_comp = generate_1d_sweep(x_var_comp, utc_filter=comp_utc)
    df_comp = df_comp[
        (df_comp[x_var_comp] >= x_comp_min) &
        (df_comp[x_var_comp] <= x_comp_max)
    ]
    df_comp = df_comp[~_self_term_mask(df_comp)]

    fig_comp = go.Figure()
    colors = {'wg': ('#3b82f6', '#93c5fd'), 'fl': ('#8b5cf6', '#c4b5fd'), 'ip': ('#10b981', '#6ee7b7')}
    
    for role in ['wg', 'fl', 'ip']:
        fig_comp.add_trace(go.Bar(x=df_comp[x_var_comp], y=df_comp[f'{role}_blue_monthly'], name=f"{role.upper()} Blue", marker_color=colors[role][0], offsetgroup=role, hovertemplate='<b>%{x}</b><br>Sorties: %{y:.1f}<extra></extra>'))
        fig_comp.add_trace(go.Bar(x=df_comp[x_var_comp], y=df_comp[f'{role}_red_monthly'], name=f"{role.upper()} Red", marker_color=colors[role][1], offsetgroup=role, base=df_comp[f'{role}_blue_monthly'], hovertemplate='<b>%{x}</b><br>Sorties: %{y:.1f}<extra></extra>'))
        if show_trends:
            fig_comp.add_trace(go.Scatter(x=df_comp[x_var_comp], y=df_comp[f'{role}_monthly'], name=f"{role.upper()} Total Trend", line=dict(color=colors[role][0], width=2), mode='lines', hovertemplate='<b>%{x}</b><br>Sorties: %{y:.1f}<extra></extra>'))
            
    fig_comp.add_hline(y=9.0, line_dash="dot", line_color="#b91c1c", annotation_text="9.0 Inexp.")
    fig_comp.add_hline(y=8.0, line_dash="dot", line_color="#fca5a5", annotation_text="8.0 Exp.")
    fig_comp.update_layout(
        **_utc_chart_layout_title(comp_utc),
        xaxis_title=x_var_comp.upper(),
        yaxis_title='Monthly Sorties',
        barmode='group',
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_comp.update_xaxes(range=[x_comp_min, x_comp_max], autorange=False)
    st.plotly_chart(fig_comp, width='stretch')

    # --- CHART 3: HEATMAP (2D Sweep) ---
    st.write("---")
    st.subheader("🗺️ RAP State Heatmap")
    heat_utc = _utc_chart_filter("heat_utc", utc_wise)
    is_blue = st.toggle("Show Only Blue RAP Counters", value=False)

    # Generate 2D Grid
    ute_vals = sweep_ranges['ute']
    exp_vals = sweep_ranges['exp_ratio']
    grid = list(itertools.product(ute_vals, exp_vals))
    df_heat_base = pd.DataFrame(grid, columns=['ute', 'exp_ratio'])
    
    # Fill remaining constants
    for k, v in inputs.items():
        if k not in ['ute', 'exp_ratio']:
            df_heat_base[k] = v
            
    # Predict Grid
    if use_physics:
        with st.spinner("Running heatmap sweep…"):
            df_heat_preds = compute_metrics(
                df_heat_base,
                allocation_mode,
                brain,
                utc_wise=utc_wise,
                utc_filter=heat_utc,
            )
    else:
        df_heat_preds = compute_metrics(df_heat_base, allocation_mode, brain)
    
    def _heat_status(row):
        term = _self_term_label(row)
        if term:
            return term
        code = calculate_rap_code(row, is_blue)
        return state_labels_dict.get(int(code) if pd.notna(code) else -1, "Unknown")

    df_heat_preds["rap_label"] = df_heat_preds.apply(_heat_status, axis=1)
    df_heat_preds["rap_code"] = df_heat_preds.apply(
        lambda r: _SELF_TERM_HEAT_CODE if r.get("self_terminating_phase") else calculate_rap_code(r, is_blue),
        axis=1,
    )

    # Pivot for Plotly
    heat_z = df_heat_preds.pivot(index='ute', columns='exp_ratio', values='rap_code').sort_index(ascending=False)
    heat_labels = df_heat_preds.pivot(index='ute', columns='exp_ratio', values='rap_label').sort_index(ascending=False)
    
    color_map = {0: "#22c55e", 1: "#fef08a", 2: "#fde047", 3: "#fdba74", 4: "#eab308", 5: "#f97316", 6: "#ea580c", 7: "#ef4444"}
    if use_physics and df_heat_preds["self_terminating_phase"].fillna(False).any():
        color_map[_SELF_TERM_HEAT_CODE] = _SELF_TERM_GREY

    max_val = max(color_map)
    discrete_colorscale = []
    for val, hex_color in sorted(color_map.items()):
        loc = val / max_val
        discrete_colorscale.append([loc, hex_color])
        next_val_list = [v for v in color_map.keys() if v > val]
        if next_val_list:
            next_loc = (min(next_val_list) - 0.01) / max_val
            discrete_colorscale.append([next_loc, hex_color])
        else:
            discrete_colorscale.append([1.0, hex_color])

    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_z.values, x=heat_z.columns, y=heat_z.index, customdata=heat_labels.values,
        colorscale=discrete_colorscale, showscale=False, zmin=0, zmax=max_val, xgap=1, ygap=1,
        hovertemplate="<b>Status: %{customdata}</b><br>Exp Ratio: %{x:.0%}<br>UTE: %{y}<extra></extra>"
    ))
    
    for code, color in color_map.items():
        name = "Failed Sim" if code == _SELF_TERM_HEAT_CODE else state_labels_dict.get(code)
        fig_heat.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, symbol="square", color=color),
            showlegend=True, name=name,
        ))

    heat_label = _utc_label(heat_utc)
    heat_title = "Experience Ratio vs UTE"
    if heat_label:
        heat_title = f"{heat_title} — {heat_label}"
    fig_heat.update_layout(
        title=heat_title,
        xaxis_title="Experience Ratio",
        yaxis_title="UTE",
        height=500,
    )
    st.plotly_chart(fig_heat, width='stretch')

    # --- CHART 4: INCOMPLETE SYLLABI ---
    st.write("---")
    st.subheader("📉 Incomplete Syllabi (Phase Snapshot)")
    syll_source = "physics allocator" if use_physics else "ML brain"
    st.caption(
        f"Y-axis is syllabus-normalized count (not %), from the {syll_source}. "
        "1.0 ≈ one full syllabus incomplete; 0.33 ≈ one-third of a syllabus; "
        "5.0 ≈ five students' worth of incomplete syllabus (aggregate across the cohort)."
    )
    col_syll_1, col_syll_2 = st.columns([2, 1])
    with col_syll_1:
        x_var_syll = st.selectbox("X-Axis Variable", x_options, index=0, key="chart4_x")
    with col_syll_2:
        st.write("")
        sorties_only_syll = st.toggle(
            "Sorties only",
            value=True,
            key="chart4_sorties_only",
            help="On: sorties-only remainder (default). Off: sorties + sims (total syllabus).",
        )

    x_vals_syll = sweep_ranges[x_var_syll]
    syll_sweep_min = float(np.min(x_vals_syll))
    syll_sweep_max = float(np.max(x_vals_syll))
    syll_default_min, syll_default_max = input_axis_bounds.get(
        x_var_syll, (syll_sweep_min, syll_sweep_max)
    )
    syll_default_min = max(syll_sweep_min, float(syll_default_min))
    syll_default_max = min(syll_sweep_max, float(syll_default_max))

    if syll_default_min > syll_default_max:
        syll_default_min, syll_default_max = syll_sweep_min, syll_sweep_max

    if np.issubdtype(x_vals_syll.dtype, np.integer):
        x_syll_min, x_syll_max = st.slider(
            "Displayed X-Axis Range",
            min_value=int(syll_sweep_min),
            max_value=int(syll_sweep_max),
            value=(int(syll_default_min), int(syll_default_max)),
            step=1,
            key=f"chart4_x_range_{x_var_syll}",
        )
    else:
        syll_step = 0.5 if x_var_syll == "ute" else 0.02
        x_syll_min, x_syll_max = st.slider(
            "Displayed X-Axis Range",
            min_value=syll_sweep_min,
            max_value=syll_sweep_max,
            value=(syll_default_min, syll_default_max),
            step=syll_step,
            key=f"chart4_x_range_{x_var_syll}",
        )

    df_syll = generate_1d_sweep(x_var_syll)
    df_syll = df_syll[
        (df_syll[x_var_syll] >= x_syll_min) & (df_syll[x_var_syll] <= x_syll_max)
    ]

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

    fig_syll = go.Figure()
    colors_upgrade = {
        "MQT": "#f59e0b",
        "FLUG": "#ec4899",
        "IPUG": "#6366f1",
    }

    for col, label in syll_series:
        fig_syll.add_trace(
            go.Scatter(
                x=df_syll[x_var_syll],
                y=df_syll[col],
                name=label,
                line=dict(color=colors_upgrade[label], width=3),
                mode="lines",
                hovertemplate=(
                    f"<b>%{{x}}</b><br>{label}: %{{y:.2f}} syllabi<extra></extra>"
                ),
            )
        )

    fig_syll.update_layout(
        xaxis_title=x_var_syll.upper(),
        yaxis_title=f"Incomplete syllabi ({syll_mode_label})",
        yaxis_tickformat=".2f",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_syll.update_xaxes(range=[x_syll_min, x_syll_max], autorange=False)
    fig_syll.update_yaxes(autorange=True, rangemode="tozero")

    st.plotly_chart(fig_syll, width="stretch")

# ==============================================================================
# 6. SUMMARY SIDEBAR (Single Point Prediction)
# ==============================================================================
with col_summary:
    st.subheader("Status Overview")

    utc_status_map = None
    pilots_for_summary = None
    base_row_dict = None

    if use_physics and utc_wise:
        base_row_dict, utc_status_map, pilots_for_summary = _single_point_summary_physics(
            int(inputs["paa"]),
            float(inputs["ute"]),
            int(inputs["total_pilots"]),
            float(inputs["exp_ratio"]),
            int(inputs["ip_qty"]),
            int(inputs["mqt_qty"]),
            int(inputs["flug_qty"]),
            int(inputs["ipug_qty"]),
            utc_wise,
        )
    else:
        df_single = pd.DataFrame([inputs])
        base_row_dict = compute_metrics(
            df_single, allocation_mode, brain, utc_wise=utc_wise,
        ).iloc[0].to_dict()

    if utc_wise:
        summary_scope = st.selectbox(
            "Status scope",
            _SUMMARY_STATUS_SCOPE_OPTIONS,
            key="summary_status_scope",
        )
    else:
        summary_scope = "Overall"

    if (
        utc_wise
        and summary_scope != "Overall"
        and pilots_for_summary is not None
    ):
        utc_filter = _UTC_RANK_BY_LABEL[summary_scope]
        row_dict = _rap_metrics_row_from_pilots(
            pilots_for_summary, utc_filter, base=base_row_dict,
        )
    else:
        row_dict = base_row_dict

    row = pd.Series(row_dict)

    self_term = _self_term_label(base_row_dict)
    if self_term:
        label = self_term
        status_heading = "PIPELINE STATUS"
        card_bg = "#991b1b"
        title_color = "#fecaca"
    else:
        current_code = calculate_rap_code(row, is_blue=False)
        if pd.isna(current_code):
            current_code = -1
        label = state_labels_dict.get(current_code, "Invalid / N/A")
        status_heading = "SIMULATED STATUS" if use_physics else "PREDICTED STATUS"
        if utc_wise and summary_scope != "Overall":
            status_heading = f"{status_heading} — {summary_scope.upper()}"
        card_bg = "#0f172a"
        title_color = "#f8fafc"

    def _fmt(val):
        return f"{val:.1f}" if pd.notna(val) else "N/A"

    wg_t, fl_t, ip_t = _fmt(row["wg_monthly"]), _fmt(row["fl_monthly"]), _fmt(row["ip_monthly"])
    wg_b, fl_b, ip_b = _fmt(row["wg_blue_monthly"]), _fmt(row["fl_blue_monthly"]), _fmt(row["ip_blue_monthly"])
    wg_r, fl_r, ip_r = _fmt(row["wg_red_monthly"]), _fmt(row["fl_red_monthly"]), _fmt(row["ip_red_monthly"])

    st.markdown(f"""
<div style="background-color:{card_bg}; padding:20px; border-radius:15px; color:white; margin-bottom:20px;">
<p style="font-size:0.7rem; color:#94a3b8; margin-bottom:2px; letter-spacing: 0.05em;">{status_heading}</p>
<h2 style="margin:0; font-size:1.15rem; color: {title_color};">{label}</h2>
<hr style="border-color:#1e293b; margin:15px 0;">
<div style="display:flex; justify-content:space-between; text-align:center; margin-bottom:10px;">
<div style="flex:1;"></div>
<div style="flex:1;"><small style="color:#94a3b8; font-weight:bold;">WG</small></div>
<div style="flex:1;"><small style="color:#94a3b8; font-weight:bold;">FL</small></div>
<div style="flex:1;"><small style="color:#94a3b8; font-weight:bold;">IP</small></div>
</div>
<div style="display:flex; justify-content:space-between; text-align:center; margin-bottom:12px;">
<div style="flex:1; text-align:left;"><small style="color:#94a3b8;">Total</small></div>
<div style="flex:1;"><b style="font-size:1.1rem;">{wg_t}</b></div>
<div style="flex:1;"><b style="font-size:1.1rem;">{fl_t}</b></div>
<div style="flex:1;"><b style="font-size:1.1rem;">{ip_t}</b></div>
</div>
<div style="display:flex; justify-content:space-between; text-align:center; margin-bottom:8px; background: rgba(59, 130, 246, 0.1); border-radius: 4px; padding: 4px 0;">
<div style="flex:1; text-align:left; padding-left:5px;"><small style="color:#60a5fa;">Blue</small></div>
<div style="flex:1; color:#60a5fa;">{wg_b}</div>
<div style="flex:1; color:#60a5fa;">{fl_b}</div>
<div style="flex:1; color:#60a5fa;">{ip_b}</div>
</div>
<div style="display:flex; justify-content:space-between; text-align:center; background: rgba(244, 63, 94, 0.1); border-radius: 4px; padding: 4px 0;">
<div style="flex:1; text-align:left; padding-left:5px;"><small style="color:#fb7185;">Red</small></div>
<div style="flex:1; color:#fb7185;">{wg_r}</div>
<div style="flex:1; color:#fb7185;">{fl_r}</div>
<div style="flex:1; color:#fb7185;">{ip_r}</div>
</div>
</div>
""", unsafe_allow_html=True)

    if utc_wise and utc_status_map is not None:
        for utc_status_label in _UTC_STATUS_LABELS:
            utc_status_rank = _UTC_RANK_BY_LABEL[utc_status_label]
            utc_ready, utc_label = utc_status_map[int(utc_status_rank)]
            if utc_ready:
                utc_card_bg = "#166534"
                utc_title_color = "#bbf7d0"
            else:
                utc_card_bg = "#991b1b"
                utc_title_color = "#fecaca"
            st.markdown(f"""
<div style="background-color:{utc_card_bg}; padding:14px 16px; border-radius:12px; color:white; margin-bottom:10px;">
<p style="font-size:0.65rem; color:#94a3b8; margin-bottom:2px; letter-spacing: 0.05em;">{utc_status_label.upper()} STATUS </p>
<h2 style="margin:0; font-size:0.95rem; color: {utc_title_color};">{_format_utc_status_label(utc_label)}</h2>
</div>
""", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Red Air Exposure")
    burden_df = pd.DataFrame({
        'Role': ['WG', 'FL', 'IP'],
        'Red Pct': [row['wg_red_pct'], row['fl_red_pct'], row['ip_red_pct']]
    })
    fig_burden = px.bar(burden_df, y='Role', x='Red Pct', orientation='h', color_discrete_sequence=['#f43f5e'])
    fig_burden.update_layout(xaxis_tickformat='.0%', height=250, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Red Air %")
    st.plotly_chart(fig_burden, width='stretch')