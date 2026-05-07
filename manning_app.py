import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from src.manning_main import setup_simulation
from src.models import PriorityMode

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
    years = st.slider("Years to Run", 5, 20, 10)
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
        value=True,
        help="If Checked: Graduates are assigned equally (1, 2, 3...). If Unchecked: Healthiest squadrons get students first."
    )

    include_upgrades = st.checkbox(
        "Realistic Upgrade Bottlenecks", 
        value=True,
        help="If checked, student counts will drastically reduce flying rates (using AI Brain)."
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
def load_ai_brain():
    """
    Loads the heavy AI Brain ONCE.
    This object is passed to the simulation engine to prevent reloading.
    """
    # Check for brain
    if not os.path.exists("brains/hpc_sortie_brain_lite.pkl"):
        st.error("🚨 'hpc_sortie_brain_lite.pkl' not found! Please run 'hpc_train_brain_lite.py' on HPC.")
    # if not os.path.exists("brains/sortie_brain.pkl"):
        # st.error("🚨 'sortie_brain.pkl' not found! Please run 'train_brain_lite.py'.")
        st.stop()

    return joblib.load("brains/hpc_sortie_brain_lite.pkl")        
    # return joblib.load("brains/sortie_brain.pkl")        

cached_brain = load_ai_brain()

# --- Run Simulation ---
if submitted:
    with st.spinner("Running Simulation..."):
        # 1. Setup & Run
        sim, squadrons = setup_simulation(round_robin=round_robin, sim_upgrades=include_upgrades, ai_brain=cached_brain,
                                          flug_window_start=flug_start, ipug_window_start=ipug_start,
                                          max_manning_pct=max_manning_pct, staff_priority_mode=staff_priority_mode)
        df = sim.run_simulation(
            years_to_run=years, annual_intake=intake, 
            retention_rate=retention, squadron_configs=squadrons, ute=ute_val) 

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
            if wg_rate < 9: code += 1
            if fl_rate < 8: code += 2
            if ip_rate < 8: code += 4

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

        master_sim, _ = setup_simulation(round_robin=round_robin, sim_upgrades=include_upgrades,
                                         ai_brain=cached_brain, flug_window_start=flug_start, 
                                         ipug_window_start= ipug_start, max_manning_pct=max_manning_pct,
                                         staff_priority_mode=staff_priority_mode)

        # Loop with enumeration to update the bar
        for i, val in enumerate(test_range):

            pct_complete = (i + 1) / len(test_range)
            sensitivity_progress.progress(pct_complete, text=f"Simulating Intake: {val} pilots/yr...")

            t_sim, t_sqs = setup_simulation(round_robin=round_robin, sim_upgrades=include_upgrades,
                                            ai_brain=cached_brain, existing_sim=master_sim,
                                            flug_window_start=flug_start, ipug_window_start=ipug_start,
                                            max_manning_pct=max_manning_pct, staff_priority_mode=staff_priority_mode)

            t_df = t_sim.run_simulation(
                years_to_run=20, 
                annual_intake=val, 
                retention_rate=retention, 
                squadron_configs=t_sqs, 
                ute=ute_val 
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

# # ==============================================================================
# # 5. AI SANDBOX (With Ground Truth Overlay)
# # ==============================================================================
# st.divider()
# st.header("🧠 AI Sortie Predictor (Sandbox)")
# st.markdown("Adjust sliders to test specific constraints. **Toggle 'Show Ground Truth' to see actual data points.**")

# try:
#     brain_models = load_sandbox_models()
    
#     # --- SLIDERS ---
#     with st.container():
#         c1, c2, c3, c4, c5 = st.columns(5)
#         sb_paa = c1.slider("PAA", 12, 24, 18)
#         sb_ute = c2.slider("UTE", 6.0, 20.0, 10.0, step=0.1)
#         sb_pilots = c3.slider("Line Pilots", 15, 80, 40)
#         sb_ips = c4.slider("Active IPs", 2, 25, 6)
#         sb_ratio = c5.slider("Experience Ratio", 0.1, 0.8, 0.45, step=0.01)
        
#     col_v, col_m, col_c = st.columns([1, 1, 3])
#     upgrade_type = col_v.radio("Vary:", ["MQT", "FLUG", "IPUG"])
#     view_mode = col_m.radio("View:", ["Total Rates", "Blue Air"])
    
#     show_truth = col_c.checkbox("Show Ground Truth (Actual Data)", value=True)
    
#     # --- 1. GENERATE AI PREDICTIONS (Lines) ---
#     feature_names = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty']
#     plot_data = []
    
#     for x in range(16):
#         m = x if upgrade_type == "MQT" else 0
#         f = x if upgrade_type == "FLUG" else 0
#         i = x if upgrade_type == "IPUG" else 0
        
#         input_vec = pd.DataFrame([[
#             sb_paa, sb_ute, sb_ratio, sb_pilots, m, f, i, sb_ips
#         ]], columns=feature_names)
        
#         wg = brain_models['wg_monthly'].predict(input_vec)[0]
#         fl = brain_models['fl_monthly'].predict(input_vec)[0]
#         ip = brain_models['ip_monthly'].predict(input_vec)[0]
        
#         wg_b = brain_models['wg_blue_monthly'].predict(input_vec)[0]
#         fl_b = brain_models['fl_blue_monthly'].predict(input_vec)[0]
#         ip_b = brain_models['ip_blue_monthly'].predict(input_vec)[0]
        
#         if view_mode == "Total Rates":
#             plot_data.append({"Count": x, "Rate": wg, "Role": "WG"})
#             plot_data.append({"Count": x, "Rate": fl, "Role": "FL"})
#             plot_data.append({"Count": x, "Rate": ip, "Role": "IP"})
#         else:
#             plot_data.append({"Count": x, "Rate": wg_b, "Role": "WG (Blue)"})
#             plot_data.append({"Count": x, "Rate": fl_b, "Role": "FL (Blue)"})
#             plot_data.append({"Count": x, "Rate": ip_b, "Role": "IP (Blue)"})
            
#     df_plot = pd.DataFrame(plot_data)

#     # --- 2. FETCH GROUND TRUTH (Dots) ---
#     truth_data = []
#     if show_truth:
#         # Access the raw dataframe directly from the engine
#         df_raw = cached_sim.df # TODO bring back parquet file for this widget
        
#         # Map the slider selection to the actual column name in the parquet file
#         var_col_map = {"MQT": "mqt_qty", "FLUG": "flug_qty", "IPUG": "ipug_qty"}
#         target_col = var_col_map[upgrade_type]

#         # Filter the dataframe (Approximate matches for continuous vars)
#         mask = (
#             (df_raw['paa'] == sb_paa) & 
#             (df_raw['ute'] == sb_ute) & 
#             (df_raw['total_pilots'].between(sb_pilots - 2, sb_pilots + 2)) &
#             (df_raw['ip_qty'].between(sb_ips - 1, sb_ips + 1)) &
#             (df_raw['exp_ratio'].between(sb_ratio - 0.05, sb_ratio + 0.05))
#         )
#         filtered = df_raw[mask]

#         if upgrade_type == "MQT":
#             mask &= (df_raw['flug_qty'] == 0) & (df_raw['ipug_qty'] == 0)
#         elif upgrade_type == "FLUG":
#             mask &= (df_raw['mqt_qty'] == 0) & (df_raw['ipug_qty'] == 0)
#         elif upgrade_type == "IPUG":
#             mask &= (df_raw['mqt_qty'] == 0) & (df_raw['flug_qty'] == 0)

#         filtered = df_raw[mask]

#         if not filtered.empty:
#             # Average the rates for each step of the x-axis variable
#             grouped = filtered.groupby(target_col)[
#                 ['wg_monthly', 'fl_monthly', 'ip_monthly', 'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly']
#             ].mean().reset_index()
            
#             for _, row in grouped.iterrows():
#                 x_val = row[target_col]
#                 if x_val > 15: continue 

#                 if view_mode == "Total Rates":
#                     truth_data.append({"Count": x_val, "Rate": row['wg_monthly'], "Role": "WG"})
#                     truth_data.append({"Count": x_val, "Rate": row['fl_monthly'], "Role": "FL"})
#                     truth_data.append({"Count": x_val, "Rate": row['ip_monthly'], "Role": "IP"})
#                 else:
#                     truth_data.append({"Count": x_val, "Rate": row['wg_blue_monthly'], "Role": "WG (Blue)"})
#                     truth_data.append({"Count": x_val, "Rate": row['fl_blue_monthly'], "Role": "FL (Blue)"})
#                     truth_data.append({"Count": x_val, "Rate": row['ip_blue_monthly'], "Role": "IP (Blue)"})
    
#     # --- 3. BUILD CHART (Using Graph Objects for Mix of Lines/Dots) ---
#     fig_sb = go.Figure()
    
#     # Define Colors
#     colors = {"WG": "#636EFA", "FL": "#EF553B", "IP": "#00CC96"}
#     if view_mode == "Blue Air":
#         colors = {"WG": "#1E90FF", "FL": "#87CEFA", "IP": "#4682B4"} # Just fallback
    
#     # Plot Lines (AI Prediction)
#     for role in df_plot['Role'].unique():
#         subset = df_plot[df_plot['Role'] == role]
#         # Determine color based on role name
#         c = "#888888"
#         if "WG" in role: c = colors["WG"] if "Blue" not in role else "#1E90FF"
#         if "FL" in role: c = colors["FL"] if "Blue" not in role else "#87CEFA"
#         if "IP" in role: c = colors["IP"] if "Blue" not in role else "#4682B4"

#         fig_sb.add_trace(go.Scatter(
#             x=subset['Count'], y=subset['Rate'], 
#             name=role, mode='lines', 
#             line=dict(color=c, width=3)
#         ))

#     # Plot Dots (Ground Truth)
#     if truth_data:
#         df_truth = pd.DataFrame(truth_data)
#         for role in df_truth['Role'].unique():
#             subset = df_truth[df_truth['Role'] == role]
            
#             # Match color to the line
#             c = "#888888"
#             if "WG" in role: c = colors["WG"] if "Blue" not in role else "#1E90FF"
#             if "FL" in role: c = colors["FL"] if "Blue" not in role else "#87CEFA"
#             if "IP" in role: c = colors["IP"] if "Blue" not in role else "#4682B4"
            
#             fig_sb.add_trace(go.Scatter(
#                 x=subset['Count'], y=subset['Rate'], 
#                 name=f"{role} (Actual)", mode='markers',
#                 marker=dict(color=c, size=8, symbol="circle-open", line=dict(width=2))
#             ))

#     # Add Reference Lines
#     if view_mode == "Total Rates":
#         fig_sb.add_hline(y=9.0, line_dash="dot", line_color="red", annotation_text="Inexp.")
#         fig_sb.add_hline(y=8.0, line_dash="dot", line_color="orange", annotation_text="Exp.")
        
#     fig_sb.update_layout(
#         title=f"Predicted vs. Actual ({upgrade_type})",
#         xaxis_title=f"{upgrade_type} Count",
#         yaxis_title="Sorties/Month",
#         hovermode="x unified"
#     )

#     st.plotly_chart(fig_sb, width='stretch')

# except Exception as e:
#     st.warning(f"Sandbox Error: {e}")


# --- Footer / Contact Section ---
st.divider()
st.subheader("🐛 Report a Bug / Feature Request")
st.markdown("""
This simulation is a work in progress. If you notice any calculation errors, crashes, 
or have ideas for new features, please let me know!

**📧 Contact:** [Send me an email](mailto:claire.randolph@us.af.mil)
""")