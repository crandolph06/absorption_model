from datetime import datetime

import joblib
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.manning_config import SQUADRON_DATA
from src.manning_engine import CAFSimulation
from src.manning_main import setup_simulation
from src.models import PriorityMode, Qual, monthly_sortie_rap_target

BRAIN_PATH = "brains/hpc_sortie_brain_multi_output_mlp.pkl"
_PAA_BY_SQUADRON = {sq_id: paa for sq_id, paa, _, _, _ in SQUADRON_DATA}
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


def _clean_syllabus_preds(raw: np.ndarray) -> np.ndarray:
    vals = np.maximum(raw, 0.0)
    return np.where(vals < _SYLLABI_NEGLIGIBLE, 0.0, vals)


def _brain_features_from_history(df: pd.DataFrame, ute: float) -> pd.DataFrame:
    """Same feature math as CAFSimulation.predict_rates_fast, one row per history row."""
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


def _syllabus_preds_by_timeline(
    df: pd.DataFrame, brain, ute: float
) -> pd.DataFrame:
    """Brain outputs 6–11 per squadron-phase, summed CAF-wide per timeline."""
    feat = _brain_features_from_history(df, ute)
    preds = brain.predict(feat[_PREDICT_FEATURES].fillna(0))
    for i, col in enumerate(_REMAINING_TOTAL):
        feat[col] = _clean_syllabus_preds(preds[:, 6 + i])
    for i, col in enumerate(_REMAINING_SORTIES):
        feat[col] = _clean_syllabus_preds(preds[:, 9 + i])
    # 16-output brain: outputs 10–15 per squadron-phase
    # for i, col in enumerate(_REMAINING_TOTAL):
    #     feat[col] = _clean_syllabus_preds(preds[:, 10 + i])
    # for i, col in enumerate(_REMAINING_SORTIES):
    #     feat[col] = _clean_syllabus_preds(preds[:, 13 + i])
    feat["timeline"] = feat["year"].astype(str) + " P" + feat["phase"].astype(str)
    return (
        feat.groupby(["year", "phase", "timeline"], as_index=False)[
            _REMAINING_TOTAL + _REMAINING_SORTIES
        ]
        .sum()
        .sort_values(["year", "phase"])
    )


def _pipeline_n_features_in(brain):
    steps = getattr(brain, "named_steps", None)
    if steps and "scaler" in steps:
        return getattr(steps["scaler"], "n_features_in_", None)
    return getattr(brain, "n_features_in_", None)

st.set_page_config(page_title="CAF Absorption Simulator", layout="wide")

st.title("🛩️ Fighter Pilot Long-Term Manning Visualizer")
st.markdown("""
This dashboard simulates the **"Absorption Death Spiral"**. It models how adding too many students 
overwhelms the instructional capacity of a fighter squadron, causing training rates to collapse.
""")

# --- Staff Priority Backend ---
priority_options = {
    "Flight Leads First": PriorityMode.FL_FIRST,
    "Instructors First": PriorityMode.IP_FIRST,
    "Random Shuffle": PriorityMode.RANDOM
}

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")

with st.sidebar.form("sim_params"):
    years = st.slider("Years to Run", 5, 20, 20)
    intake = st.slider("Annual B-Course Intake", 10, 350, 200)
    retention = st.slider("Retention Rate (0.0 - 1.0)", 0.0, 1.0, 0.4)
    ute_val = st.slider("UTE", 6, 20, 10)
    flug_start = st.slider("FLUG Entry -- Sorties", 50, 300, 250)
    ipug_start = st.slider("IPUG Entry -- Hours", 150, 500, 400)
    max_manning_pct = st.slider("Maximum Squadron Manning (%)", 50, 150, 100)
    selected_label = st.radio(
    "Non-Line Assignment Priority",
    options=priority_options.keys(),
    horizontal=True, # <--- This makes it look like a toggle bar
    help="Determines who has the priority of going to non-line assigmnets once unit hits max capacity.",
    index=2
)

    st.markdown("---") # Visual separator
    
    round_robin = st.checkbox(
        "Round Robin Assignment", 
        value=False,
        help="If Checked: Graduates are assigned equally (1, 2, 3...). If Unchecked: Healthiest squadrons get students first."
    )

    
    run_sensitivity = st.checkbox(
        "Run Detailed Intake Analysis", 
        value=False,
        help="If checked, runs sensitivity analysis over all intake rates."
    )

    staff_priority_mode = priority_options[selected_label]
    
    # The Submit Button
    submitted = st.form_submit_button("🚀 Run Simulation")

@st.cache_resource
def load_ai_brain(brain_path: str, brain_mtime: float):
    """
    Loads the AI brain. `brain_mtime` is part of the cache key so replacing the
    file on disk (e.g. after scp from HPC) forces a reload without clearing cache.
    """
    return joblib.load(brain_path)


if not os.path.exists(BRAIN_PATH):
    st.error(
        f"🚨 '{BRAIN_PATH}' not found! Copy the HPC artifact from "
        "`outputs/single_phase/brains/` or run `hpc_train_brain_multi_output.py`."
    )
    st.stop()

_brain_mtime = os.path.getmtime(BRAIN_PATH)
cached_brain = load_ai_brain(BRAIN_PATH, _brain_mtime)

_n_fit = _pipeline_n_features_in(cached_brain)
_n_engine = len(CAFSimulation._PREDICT_FEATURE_COLS)
if _n_fit is not None and _n_fit != _n_engine:
    st.error(
        f"The loaded brain expects {_n_fit} input features but the simulator builds {_n_engine} "
        f"(including **paa** and **ute**). Retrain with the current `hpc_train_brain_multi_output.py` "
        f"(same feature list as `CAFSimulation.predict_rates_fast`) and replace `{BRAIN_PATH}`."
    )
    st.stop()

st.sidebar.caption(
    f"Brain: `{BRAIN_PATH}` · file mtime {_brain_mtime:.0f} "
    f"({datetime.fromtimestamp(_brain_mtime).strftime('%Y-%m-%d %H:%M')})"
)

# --- Run Simulation ---
if submitted:
    with st.spinner("Running Simulation..."):
        # 1. Setup & Run
        sim, squadrons = setup_simulation(round_robin=round_robin, ai_brain=cached_brain,
                                          flug_window_start=flug_start, ipug_window_start=ipug_start, annual_intake=intake,
                                          max_manning_pct=max_manning_pct, staff_priority_mode=staff_priority_mode,
                                          retention_rate=retention)
        df = sim.run_simulation(
            years_to_run=years, squadron_configs=squadrons, ute=ute_val
            ) 

        st.write("### 🔍 Debugging Tools")
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Full Simulation History (CSV)",
            data=csv,
            file_name="simulation_debug_dump.csv",
            mime="text/csv",
        )

    if df is not None and not df.empty:
        df['timeline'] = df['year'].astype(str) + " P" + df['phase'].astype(str)

        # 3. IMMEDIATE AGGREGATION (CAF Wide)
        df_display = df.groupby(['year', 'phase', 'timeline']).agg({
            'wg_qty': 'sum',
            'fl_qty': 'sum',
            'ip_qty': 'sum',
            'staff_ips': 'sum',
            'staff_fls': 'sum',
            'total_pilots': 'sum',
            'line_pilots': 'sum',
            'exp_rat': 'mean',
            'percent_manned': 'mean',
            'separated': 'sum',
            'retained': 'sum',
            'wg_rate_mo': 'mean',
            'fl_rate_mo': 'mean',
            'ip_rate_mo': 'mean',
            'wg_rate_blue': 'mean' if 'wg_rate_blue' in df.columns else lambda x: 0,
            'fl_rate_blue': 'mean' if 'fl_rate_blue' in df.columns else lambda x: 0,
            'ip_rate_blue': 'mean' if 'ip_rate_blue' in df.columns else lambda x: 0
        }).reset_index()

        # --- Top Level Metrics (Optional) ---
        st.markdown(f"### CAF Status at Year {years}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Final Total Pilots", int(df_display['total_pilots'].iloc[-1]))
        m2.metric("Final Total Line Pilots", int(df_display['line_pilots'].iloc[-1]))
        m3.metric("Final Total Non-Line Pilots", int(df_display['staff_ips'].iloc[-1] + int(df_display['staff_fls'].iloc[-1])))
        m4.metric("Final Line Exp Ratio", f"{df_display['exp_rat'].iloc[-1]*100:.1f}%")
        m5.metric("Total Separations", int(df_display['separated'].sum()))

        # --- Charts ---
        st.subheader("Pilot Population by Qualification")
        fig_pop = px.area(
            df_display, 
            x='timeline',
            y=['wg_qty', 'fl_qty', 'ip_qty', 'staff_fls', 'staff_ips'],
            title="CAF Qualification Mix",
            labels={'value': 'Count', 'timeline': 'Year/Phase'},
            color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', "#DC8F7E", "#78CAB4"]
        )
        st.plotly_chart(fig_pop, width='stretch')

        st.divider()
        st.subheader("CAF Experience Ratio")

        use_blue_rap = st.toggle("Blue RAP Only", value=False)
        color_map = {
            0: "#22c55e",          
            1: "#fef08a",          
            2: "#fde047",          
            3: "#fdba74",          
            4: "#eab308",     
            5: "#f97316",     
            6: "#ea580c",     
            7: "#ef4444" 
        }

        state_labels_dict = {
            0: "All Make RAP",          
            1: "WG Shortfall",          
            2: "FL Shortfall",          
            3: "WG + FL Shortfall",          
            4: "IP Shortfall",     
            5: "WG + IP Shortfall",     
            6: "FL + IP Shortfall",     
            7: "WG + FL + IP Shortfall"
        }

        def get_rap_code(row, use_blue):
            suffix = "_blue" if use_blue else "_mo"
            wg_rate = row.get(f'wg_rate{suffix}', 0)
            fl_rate = row.get(f'fl_rate{suffix}', 0)
            ip_rate = row.get(f'ip_rate{suffix}', 0)

            code = 0
            if wg_rate < monthly_sortie_rap_target(Qual.WG):
                code += 1
            if fl_rate < monthly_sortie_rap_target(Qual.FL):
                code += 2
            if ip_rate < monthly_sortie_rap_target(Qual.IP):
                code += 4

            return code
        
        df_display['rap_code'] = df_display.apply(lambda row: get_rap_code(row, use_blue_rap), axis=1)

        fig_exp = go.Figure()

        # --- PART A: The Multi-Colored Line ---
        x_data = df_display['timeline'].tolist()
        y_data = df_display['exp_rat'].tolist()
        codes = df_display['rap_code'].tolist()

        if len(x_data) > 0:
            curr_x = [x_data[0]]
            curr_y = [y_data[0]]
            curr_code = codes[0]
            
            for i in range(1, len(x_data)):
                if codes[i] != curr_code:
                    # Connect the segments
                    curr_x.append(x_data[i])
                    curr_y.append(y_data[i])
                    
                    fig_exp.add_trace(go.Scatter(
                        x=curr_x, y=curr_y,
                        mode='lines',
                        line=dict(color=color_map.get(curr_code, "grey"), width=3),
                        hoverinfo='skip',
                        showlegend=False # Don't show these segments in legend
                    ))
                    
                    curr_x = [x_data[i]]
                    curr_y = [y_data[i]]
                    curr_code = codes[i]
                else:
                    curr_x.append(x_data[i])
                    curr_y.append(y_data[i])
            
            # Final segment
            fig_exp.add_trace(go.Scatter(
                x=curr_x, y=curr_y,
                mode='lines',
                line=dict(color=color_map.get(curr_code, "grey"), width=3),
                hoverinfo='skip',
                showlegend=False
            ))

        # --- PART B: The "Fake" Legend Traces ---
        # This loop creates the legend items on the right side
        for code, color in color_map.items():
            fig_exp.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=12, symbol='square', color=color),
                showlegend=True,
                name=state_labels_dict.get(code, "Unknown")
            ))

        # --- PART C: Invisible Hover Data ---
        fig_exp.add_trace(go.Scatter(
            x=df_display['timeline'],
            y=df_display['exp_rat'],
            mode='markers',
            marker=dict(size=0, opacity=0),
            hovertemplate="<b>%{text}</b><br>Exp Ratio: %{y:.1%}<extra></extra>",
            text=[state_labels_dict.get(c, "Unknown") for c in codes],
            showlegend=False
        ))

        # 5. Layout
        fig_exp.update_layout(
            title="Experience Ratio (%)",
            xaxis_title="Year/Phase",
            yaxis_title="Exp Ratio",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            height=500,
            legend=dict(
                title="RAP Status",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02 # Puts the legend just outside the graph to the right
            )
        )

        # Reference Lines
        fig_exp.add_hline(y=0.60, line_dash="dot", line_color="green", annotation_text="Healthy")
        fig_exp.add_hline(y=0.45, line_dash="dash", line_color="orange", annotation_text="Sortie Inequity")
        fig_exp.add_hline(y=0.40, line_dash="dot", line_color="red", annotation_text="Broken")

        st.plotly_chart(fig_exp, width='stretch')

    st.divider()
    st.subheader("Detailed Operational Health: Sortie Rates vs. Manning")
            
    # Create a Dual-Axis Chart using Graph Objects
    fig_health = go.Figure()

    # --- Left Axis: Aging Rates (Sorties/Sims per Month) ---
    # Live Flying Rates (Solid Lines)
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['wg_rate_mo'], name='WG Rate', line=dict(color='#636EFA'), hovertemplate='%{y:.1f}'))
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['fl_rate_mo'], name='FL Rate', line=dict(color='#EF553B'), hovertemplate='%{y:.1f}'))
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['ip_rate_mo'], name='IP Rate', line=dict(color='#00CC96'), hovertemplate='%{y:.1f}'))
    
    # Blue/Sim Rates (Dotted Lines)
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['wg_rate_blue'], name='WG Blue Rate', line=dict(color='#636EFA', dash='dot'), hovertemplate='%{y:.1f}'))
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['fl_rate_blue'], name='FL Blue Rate', line=dict(color='#EF553B', dash='dot'), hovertemplate='%{y:.1f}'))
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['ip_rate_blue'], name='IP Blue Rate', line=dict(color='#00CC96', dash='dot'), hovertemplate='%{y:.1f}'))

    fig_health.add_hline(y=9.0, line_dash="dot", line_color="red", annotation_text="Inexp.")
    fig_health.add_hline(y=8.0, line_dash="dot", line_color="orange", annotation_text="Exp.")

    # --- Right Axis: Percentages (0-100%+) ---
    # Manning % (Thick White Dash)
    fig_health.add_trace(go.Scatter(
        x=df_display['timeline'], 
        y=df_display['percent_manned'], 
        name='Manning %', 
        line=dict(color='white', width=3, dash='dash'),
        yaxis='y2',
        showlegend=False
    ))
    
    # Exp Ratio (Thick Yellow Dash)
    fig_health.add_trace(go.Scatter(
        x=df_display['timeline'], 
        y=df_display['exp_rat'], 
        name='Exp Ratio', 
        line=dict(color='yellow', width=3, dash='dash'),
        yaxis='y2',
        showlegend=False
    ))

    # Layout for Dual Axis
    fig_health.update_layout(
        title="Operational Health: Sortie Rates vs. Manning",
        xaxis_title="Year/Phase",
        # Left Axis Settings
        yaxis=dict(
            title="Monthly Events (Sorties/Sims)",
            side='left',
            showgrid=True 
        ),
        # Right Axis Settings
        yaxis2=dict(
            title="Percentage",
            overlaying='y',
            side='right',
            range=[0, 2.0],
            tickformat='.0%',
            showgrid=False
        ),
        legend=dict(
                orientation="h", 
                yanchor="top", y=-0.25, 
                xanchor="left", x=0.0
            ),
            
            hovermode="x unified",
            margin=dict(l=50, r=50, t=50, b=150) # Increased bottom margin for legends
        )

    fig_health.add_annotation(
            xref="paper", yref="paper",
            x=1, y=-0.25,  # Bottom Right Position
            xanchor="right", yanchor="top",
            text=(
                "<b>Right Axis Legend:</b><br>"
                "<span style='color: white; font-weight: bold; font-size: 14px'>- - -</span> Manning %<br>"
                "<span style='color: yellow; font-weight: bold; font-size: 14px'>- - -</span> Exp Ratio"
            ),
            showarrow=False,
            align="left",
            bgcolor="rgba(0,0,0,0)", # Transparent background
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1,
            borderpad=10
        )

    st.plotly_chart(fig_health, width='stretch')

    if df is not None and not df.empty:
        # --- Chart 4: Incomplete syllabi from brain (CAF aggregate over time) ---
        st.divider()
        st.subheader("Incomplete Syllabi (Brain Prediction)")
        # 16-output brain: values from brain outputs 10–15
        st.caption(
            "Y-axis is syllabus-normalized count (not %), summed across squadrons each phase. "
            "1.0 ≈ one full syllabus incomplete; values from brain outputs 6–11 (same as single-phase dashboard)."
        )
        sorties_only_syll = st.toggle(
            "Sorties only",
            value=True,
            key="manning_chart4_sorties_only",
            help="On: sorties-only remainder (default). Off: sorties + sims (total syllabus).",
        )
        df_syll = _syllabus_preds_by_timeline(df, cached_brain, ute_val)
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

        colors_upgrade = {"MQT": "#f59e0b", "FLUG": "#ec4899", "IPUG": "#6366f1"}
        fig_syll = go.Figure()
        for col, label in syll_series:
            fig_syll.add_trace(
                go.Scatter(
                    x=df_syll["timeline"],
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
            xaxis_title="Year/Phase",
            yaxis_title=f"Incomplete syllabi ({syll_mode_label})",
            yaxis_tickformat=".2f",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_syll.update_yaxes(autorange=True, rangemode="tozero")
        st.plotly_chart(fig_syll, width="stretch")

    # --- Stability Frontier Section ---
    if run_sensitivity:
        st.divider()
        st.header("📉 Absorption Capacity")
        st.write("Calculates the 'health' of the CAF across different intake levels.")

        # Create a progress bar
        sensitivity_progress = st.progress(0, text="Initializing Analysis...")

        # Define range to test
        test_range = list(range(100, 351, 25)) 
        stability_data = []

        master_sim, _ = setup_simulation(round_robin=round_robin,
                                         ai_brain=cached_brain, flug_window_start=flug_start, annual_intake=intake,
                                         ipug_window_start= ipug_start, max_manning_pct=max_manning_pct,
                                         staff_priority_mode=staff_priority_mode, retention_rate=retention)

        # Loop with enumeration to update the bar
        for i, val in enumerate(test_range):

            pct_complete = (i + 1) / len(test_range)
            sensitivity_progress.progress(pct_complete, text=f"Simulating Intake: {val} pilots/yr...")

            t_sim, t_sqs = setup_simulation(round_robin=round_robin,
                                            ai_brain=cached_brain, existing_sim=master_sim, annual_intake=val,
                                            flug_window_start=flug_start, ipug_window_start=ipug_start,
                                            max_manning_pct=max_manning_pct, staff_priority_mode=staff_priority_mode,
                                            retention_rate=retention)

            t_df = t_sim.run_simulation(
                years_to_run=20, squadron_configs=t_sqs, ute=ute_val 
            )

            start_year = t_df['year'].min()
            horizons = {"5-Year": 4, "10-Year": 9, "15-Year": 14, "20-Year": 19}

            for label, year_offset in horizons.items():
                target_year = start_year + year_offset
                snapshot = t_df[t_df['year'] == target_year]
                
                if not snapshot.empty:
                    ratio = snapshot['exp_rat'].mean()
                    
                    stability_data.append({
                        "Annual Intake": val, 
                        "Exp Ratio": ratio, 
                        "Horizon": label
                    })

        # Clear bar when done
        sensitivity_progress.empty()

        analysis_df = pd.DataFrame(stability_data)

        fig_frontier = px.line(
            analysis_df, 
            x="Annual Intake", 
            y="Exp Ratio",
            color="Horizon",
            title="System Health Decay",
            labels={"Exp Ratio": "Experience Ratio", "Annual Intake": "Annual Intake"},
            color_discrete_sequence=px.colors.sequential.Reds_r 
        )
        fig_frontier.add_hline(y=0.60, line_dash="dot", line_color="green", annotation_text="Healthy (> 60%)")
        fig_frontier.add_hline(y=0.45, line_dash="dash", line_color="yellow", annotation_text="Sortie Inequity (< 45%)")
        fig_frontier.add_hline(y=0.40, line_dash="dot", line_color="red", annotation_text="Broken (< 40%)")
        st.plotly_chart(fig_frontier, width='stretch')
else:
    st.info("Set parameters and click 'Run Simulation'.")

# --- Footer / Contact Section ---
st.divider()
st.subheader("🐛 Report a Bug / Feature Request")
st.markdown("""
This simulation is a work in progress. If you notice any calculation errors, crashes, 
or have ideas for new features, please let me know!

**📧 Contact:** [Send me an email](mailto:claire.randolph@us.af.mil)
""")