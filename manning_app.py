import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import os

from src.manning_main import setup_simulation
from src.manning_engine import CAFSimulation
from src.predictor import CAFModel

# PATH = 'outputs/simulation_results.parquet'
# priority_vars = ['exp_ratio', 'ip_qty', 'total_pilots']

st.set_page_config(page_title="CAF Absorption Simulator", layout="wide")

st.title("🛩️ Fighter Pilot Long-Term Manning Visualizer")
st.markdown("""
This dashboard simulates the **"Absorption Death Spiral"**. It models how adding too many students 
overwhelms the instructional capacity of a fighter squadron, causing training rates to collapse.
""")

@st.cache_resource
def load_base_engine():
    """
    Initializes the CAFSimulation object and loads the AI Brain ONCE.
    This object will be passed to setup_simulation() to be reused.
    """
    path = 'outputs/simulation_results.parquet' # TODO Why do we need this?
    
    # Check for brain
    if not os.path.exists("sortie_brain.pkl"):
        st.error("🚨 'sortie_brain.pkl' not found! Please run 'train_brain_lite.py'.")
        st.stop()
        
    return CAFSimulation(path, sim_upgrades=True)

@st.cache_resource
def load_sandbox_models():
    """Loads the brain directly for the Sandbox tool at the bottom."""
    return joblib.load("sortie_brain.pkl")

cached_sim = load_base_engine()

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
with st.sidebar.form("sim_params"):
    years = st.sidebar.slider("Years to Run", 5, 20, 10)
    intake = st.sidebar.slider("Annual B-Course Intake", 10, 350, 150)
    retention = st.sidebar.slider("Retention Rate (0.0 - 1.0)", 0.0, 1.0, 0.4)
    ute_val = st.sidebar.slider("UTE", 6, 20, 10)

    include_upgrades = st.sidebar.checkbox(
        "Realistic Upgrade Bottlenecks", 
        value=False,
        help="If checked, student counts (MQT/FLUG/IPUG) will drastically reduce flying rates."
    )
    run_sensitivity = st.sidebar.checkbox(
        "Run Detailed Intake Analysis", value=False,
        help="If checked, runs sensitivity analysis over all intake rates.")
    
    submitted = st.form_submit_button("🚀 Run Simulation")

# --- Run Simulation ---
if submitted:
    with st.spinner("Running Simulation..."):
        # 1. Setup & Run
        sim, squadrons = setup_simulation(sim_upgrades=include_upgrades, existing_sim=cached_sim)
        df = sim.run_simulation(
            years_to_run=years, annual_intake=intake, 
            retention_rate=retention, squadron_configs=squadrons, ute=ute_val) # Took out multiple run_simulation requirements

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
            'wg_count': 'sum',
            'fl_count': 'sum',
            'ip_count': 'sum',
            'staff_ips': 'sum',
            'staff_fls': 'sum',
            'total_pilots': 'sum',
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

        # Recalculate Exp Ratio based on the summed counts
        # df_display['exp_rat'] = (df_display['fl_count'] + df_display['ip_count']) / df_display['total_pilots']

        # --- Top Level Metrics (Optional) ---
        st.markdown(f"### CAF Status at Year {years}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Total Line Pilots", int(df_display['total_pilots'].iloc[-1]))
        m2.metric("Final Total Staff Officers", int(df_display['staff_ips'].iloc[-1] + int(df_display['staff_fls'].iloc[-1])))
        m3.metric("Final Exp Ratio", f"{df_display['exp_rat'].iloc[-1]*100:.1f}%")
        m4.metric("Total Separations", int(df_display['separated'].sum()))

        # --- Charts ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Pilot Population by Qualification")
            fig_pop = px.area(
                df_display, 
                x='timeline', 
                y=['wg_count', 'fl_count', 'ip_count', 'staff_fls', 'staff_ips'],
                title="CAF Qualification Mix",
                labels={'value': 'Count', 'timeline': 'Year/Phase'},
                color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', "#DC8F7E", "#78CAB4"]
            )
            st.plotly_chart(fig_pop, width='stretch')

        with col2:
            st.subheader("CAF Experience Ratio")
            fig_exp = px.line(
                df_display, 
                x='timeline', 
                y='exp_rat', 
                title="Experience Ratio (%)",
                labels={'exp_rat': 'Exp Ratio', 'timeline': 'Year/Phase'}
            )
            
            # Reference Lines
            fig_exp.add_hline(y=0.60, line_dash="dot", line_color="green", annotation_text="Healthy (> 60%)")
            fig_exp.add_hline(y=0.45, line_dash="dash", line_color="yellow", annotation_text="Sortie Inequity (< 45%)")
            fig_exp.add_hline(y=0.40, line_dash="dot", line_color="red", annotation_text="Broken (< 40%)")
            
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
            margin=dict(l=50, r=50, t=50, b=100) # Increased bottom margin for legends
        )

    fig_health.add_annotation(
            xref="paper", yref="paper",
            x=1, y=-0.15,  # Bottom Right Position
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

        base_sim, base_squadrons = setup_simulation(sim_upgrades=include_upgrades)

        # Loop with enumeration to update the bar
        for i, val in enumerate(test_range):

            pct_complete = (i + 1) / len(test_range)
            sensitivity_progress.progress(pct_complete, text=f"Simulating Intake: {val} pilots/yr...")

            t_sim, t_sqs = setup_simulation(sim_upgrades=include_upgrades)

            t_df = t_sim.run_simulation(
                years_to_run=20, 
                annual_intake=val, 
                retention_rate=retention, 
                squadron_configs=t_sqs, 
                ute=ute_val # TODO fix UTE issue 
            )

            start_year = t_df['year'].min()
            horizons = {"5-Year": 4, "10-Year": 9, "15-Year": 14, "20-Year": 19}

            for label, year_offset in horizons.items():
                target_year = start_year + year_offset
                snapshot = t_df[t_df['year'] == target_year]
                
                if not snapshot.empty:
                    total_pilots = snapshot['total_pilots'].sum()
                    exp_pilots = snapshot['fl_count'].sum() + snapshot['ip_count'].sum()
                    ratio = exp_pilots / total_pilots if total_pilots > 0 else 0
                    
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
        fig_frontier.add_hline(y=0.45, line_dash="dot", line_color="yellow", annotation_text="Runaway Inequity")
        st.plotly_chart(fig_frontier, width='stretch')
else:
    st.info("Set parameters and click 'Run Simulation'.")

# ==============================================================================
# 5. AI SANDBOX (With Ground Truth Overlay)
# ==============================================================================
st.divider()
st.header("🧠 AI Sortie Predictor (Sandbox)")
st.markdown("Adjust sliders to test specific constraints. **Toggle 'Show Ground Truth' to see actual data points.**")

try:
    brain_models = load_sandbox_models()
    
    # --- SLIDERS ---
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        sb_paa = c1.slider("PAA", 12, 24, 18)
        sb_ute = c2.slider("UTE", 6.0, 20.0, 10.0, step=0.1)
        sb_pilots = c3.slider("Line Pilots", 15, 80, 40)
        sb_ips = c4.slider("Active IPs", 2, 25, 6)
        sb_ratio = c5.slider("Experience Ratio", 0.1, 0.8, 0.45, step=0.01)
        
    col_v, col_m, col_c = st.columns([1, 1, 3])
    upgrade_type = col_v.radio("Vary:", ["MQT", "FLUG", "IPUG"])
    view_mode = col_m.radio("View:", ["Total Rates", "Blue Air"])
    
    show_truth = col_c.checkbox("Show Ground Truth (Actual Data)", value=True)
    
    # --- 1. GENERATE AI PREDICTIONS (Lines) ---
    feature_names = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty']
    plot_data = []
    
    for x in range(16):
        m = x if upgrade_type == "MQT" else 0
        f = x if upgrade_type == "FLUG" else 0
        i = x if upgrade_type == "IPUG" else 0
        
        input_vec = pd.DataFrame([[
            sb_paa, sb_ute, sb_ratio, sb_pilots, m, f, i, sb_ips
        ]], columns=feature_names)
        
        wg = brain_models['wg_monthly'].predict(input_vec)[0]
        fl = brain_models['fl_monthly'].predict(input_vec)[0]
        ip = brain_models['ip_monthly'].predict(input_vec)[0]
        
        wg_b = brain_models['wg_blue_monthly'].predict(input_vec)[0]
        fl_b = brain_models['fl_blue_monthly'].predict(input_vec)[0]
        ip_b = brain_models['ip_blue_monthly'].predict(input_vec)[0]
        
        if view_mode == "Total Rates":
            plot_data.append({"Count": x, "Rate": wg, "Role": "WG"})
            plot_data.append({"Count": x, "Rate": fl, "Role": "FL"})
            plot_data.append({"Count": x, "Rate": ip, "Role": "IP"})
        else:
            plot_data.append({"Count": x, "Rate": wg_b, "Role": "WG (Blue)"})
            plot_data.append({"Count": x, "Rate": fl_b, "Role": "FL (Blue)"})
            plot_data.append({"Count": x, "Rate": ip_b, "Role": "IP (Blue)"})
            
    df_plot = pd.DataFrame(plot_data)

    # --- 2. FETCH GROUND TRUTH (Dots) ---
    truth_data = []
    if show_truth:
        # Access the raw dataframe directly from the engine
        df_raw = cached_sim.df
        
        # Map the slider selection to the actual column name in the parquet file
        var_col_map = {"MQT": "mqt_qty", "FLUG": "flug_qty", "IPUG": "ipug_qty"}
        target_col = var_col_map[upgrade_type]

        # Filter the dataframe (Approximate matches for continuous vars)
        mask = (
            (df_raw['paa'] == sb_paa) & 
            (df_raw['ute'] == sb_ute) & 
            (df_raw['total_pilots'].between(sb_pilots - 2, sb_pilots + 2)) &
            (df_raw['ip_qty'].between(sb_ips - 1, sb_ips + 1)) &
            (df_raw['exp_ratio'].between(sb_ratio - 0.05, sb_ratio + 0.05))
        )
        filtered = df_raw[mask]

        if upgrade_type == "MQT":
            mask &= (df_raw['flug_qty'] == 0) & (df_raw['ipug_qty'] == 0)
        elif upgrade_type == "FLUG":
            mask &= (df_raw['mqt_qty'] == 0) & (df_raw['ipug_qty'] == 0)
        elif upgrade_type == "IPUG":
            mask &= (df_raw['mqt_qty'] == 0) & (df_raw['flug_qty'] == 0)

        filtered = df_raw[mask]

        if not filtered.empty:
            # Average the rates for each step of the x-axis variable
            grouped = filtered.groupby(target_col)[
                ['wg_monthly', 'fl_monthly', 'ip_monthly', 'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly']
            ].mean().reset_index()
            
            for _, row in grouped.iterrows():
                x_val = row[target_col]
                if x_val > 15: continue 

                if view_mode == "Total Rates":
                    truth_data.append({"Count": x_val, "Rate": row['wg_monthly'], "Role": "WG"})
                    truth_data.append({"Count": x_val, "Rate": row['fl_monthly'], "Role": "FL"})
                    truth_data.append({"Count": x_val, "Rate": row['ip_monthly'], "Role": "IP"})
                else:
                    truth_data.append({"Count": x_val, "Rate": row['wg_blue_monthly'], "Role": "WG (Blue)"})
                    truth_data.append({"Count": x_val, "Rate": row['fl_blue_monthly'], "Role": "FL (Blue)"})
                    truth_data.append({"Count": x_val, "Rate": row['ip_blue_monthly'], "Role": "IP (Blue)"})
    
    # --- 3. BUILD CHART (Using Graph Objects for Mix of Lines/Dots) ---
    fig_sb = go.Figure()
    
    # Define Colors
    colors = {"WG": "#636EFA", "FL": "#EF553B", "IP": "#00CC96"}
    if view_mode == "Blue Air":
        colors = {"WG": "#1E90FF", "FL": "#87CEFA", "IP": "#4682B4"} # Just fallback
    
    # Plot Lines (AI Prediction)
    for role in df_plot['Role'].unique():
        subset = df_plot[df_plot['Role'] == role]
        # Determine color based on role name
        c = "#888888"
        if "WG" in role: c = colors["WG"] if "Blue" not in role else "#1E90FF"
        if "FL" in role: c = colors["FL"] if "Blue" not in role else "#87CEFA"
        if "IP" in role: c = colors["IP"] if "Blue" not in role else "#4682B4"

        fig_sb.add_trace(go.Scatter(
            x=subset['Count'], y=subset['Rate'], 
            name=role, mode='lines', 
            line=dict(color=c, width=3)
        ))

    # Plot Dots (Ground Truth)
    if truth_data:
        df_truth = pd.DataFrame(truth_data)
        for role in df_truth['Role'].unique():
            subset = df_truth[df_truth['Role'] == role]
            
            # Match color to the line
            c = "#888888"
            if "WG" in role: c = colors["WG"] if "Blue" not in role else "#1E90FF"
            if "FL" in role: c = colors["FL"] if "Blue" not in role else "#87CEFA"
            if "IP" in role: c = colors["IP"] if "Blue" not in role else "#4682B4"
            
            fig_sb.add_trace(go.Scatter(
                x=subset['Count'], y=subset['Rate'], 
                name=f"{role} (Actual)", mode='markers',
                marker=dict(color=c, size=8, symbol="circle-open", line=dict(width=2))
            ))

    # Add Reference Lines
    if view_mode == "Total Rates":
        fig_sb.add_hline(y=9.0, line_dash="dot", line_color="red", annotation_text="Inexp.")
        fig_sb.add_hline(y=8.0, line_dash="dot", line_color="orange", annotation_text="Exp.")
        
    fig_sb.update_layout(
        title=f"Predicted vs. Actual ({upgrade_type})",
        xaxis_title=f"{upgrade_type} Count",
        yaxis_title="Sorties/Month",
        hovermode="x unified"
    )

    st.plotly_chart(fig_sb, width='stretch')

except Exception as e:
    st.warning(f"Sandbox Error: {e}")
