import streamlit as st
import pandas as pd
import plotly.express as px
from src.manning_main import setup_simulation
import plotly.graph_objects as go

st.set_page_config(page_title="CAF Absorption Simulator", layout="wide")

st.title("🛩️ Fighter Pilot Long-Term Manning Visualizer")
st.markdown("""
This dashboard simulates pilot career progression over 10-20 years. 
It visualizes the "Absorption Death Spiral" where adding too many students collapses the instructional capacity of the force.

""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
years = st.sidebar.slider("Years to Run", 5, 20, 10)
intake = st.sidebar.slider("Annual B-Course Intake", 10, 350, 150)
retention = st.sidebar.slider("Retention Rate (0.0 - 1.0)", 0.0, 1.0, 0.4)
ute_val = st.sidebar.slider("UTE", 6, 20, 10)

include_upgrades = st.sidebar.checkbox(
    "Realistic Upgrade Bottlenecks", 
    value=False,
    help="If checked, student counts (MQT/FLUG/IPUG) will drastically reduce flying rates."
)

st.sidebar.header("Advanced Analysis")
run_sensitivity = st.sidebar.checkbox("Run Detailed Intake Analysis")

# --- Run Simulation ---
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running Simulation..."):
        # 1. Setup & Run
        sim, squadrons = setup_simulation(sim_upgrades=include_upgrades)
        df = sim.run_simulation(years, intake, retention, squadrons, ute_val)
        
        # 2. Add Timeline Column
        df['timeline'] = df['year'].astype(str) + " P" + df['phase'].astype(str)

        # 3. IMMEDIATE AGGREGATION (CAF Wide)
        df_display = df.groupby(['year', 'phase', 'timeline']).agg({
            'wg_count': 'sum',
            'fl_count': 'sum',
            'ip_count': 'sum',
            'staff_ips': 'sum',
            'staff_fls': 'sum',
            'total_pilots': 'sum',
            'percent_manned': 'mean',
            'separated': 'sum',
            'retained': 'sum',
            'wg_rate_mo': 'mean',
            'fl_rate_mo': 'mean',
            'ip_rate_mo': 'mean',
            'wg_rate_blue': 'mean',
            'fl_rate_blue': 'mean',
            'ip_rate_blue': 'mean'
        }).reset_index()

        # Recalculate Exp Ratio based on the summed counts
        df_display['exp_rat'] = (df_display['fl_count'] + df_display['ip_count']) / df_display['total_pilots']

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
            st.plotly_chart(fig_pop, use_container_width=True)

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
            
            st.plotly_chart(fig_exp, use_container_width=True)
        
    st.divider()
    st.subheader("Detailed Operational Health: Sortie Rates vs. Manning")
            
    # Create a Dual-Axis Chart using Graph Objects
    fig_health = go.Figure()

    # --- Left Axis: Aging Rates (Sorties/Sims per Month) ---
    # Live Flying Rates (Solid Lines)
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['wg_rate_mo'], name='WG Rate', line=dict(color='#636EFA')))
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['fl_rate_mo'], name='FL Rate', line=dict(color='#EF553B')))
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['ip_rate_mo'], name='IP Rate', line=dict(color='#00CC96')))
    
    # Blue/Sim Rates (Dotted Lines)
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['wg_rate_blue'], name='WG Blue Rate', line=dict(color='#636EFA', dash='dot')))
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['fl_rate_blue'], name='FL Blue Rate', line=dict(color='#EF553B', dash='dot')))
    fig_health.add_trace(go.Scatter(x=df_display['timeline'], y=df_display['ip_rate_blue'], name='IP Blue Rate', line=dict(color='#00CC96', dash='dot')))

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
            showgrid=False # Cleaner look
        ),
        # Right Axis Settings
        yaxis2=dict(
            title="Percentage",
            overlaying='y',
            side='right',
            autorange=True,
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
            x=1, y=-0.14,  # Bottom Right Position
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

    st.plotly_chart(fig_health, use_container_width=True)

    # --- Stability Frontier Section ---
    if run_sensitivity:
        st.divider()
        st.header("📉 Absorption Capacity")
        st.write("Calculates the 'health' of the CAF across different intake levels.")

        with st.spinner("Calculating stability across intake range... (Approx. 30 Seconds)"):
            # Define range to test
            test_range = list(range(100, 351, 25)) # Larger step for speed
            stability_data = []

            for val in test_range:
                t_sim, t_sqs = setup_simulation(sim_upgrades=include_upgrades)
                t_df = t_sim.run_simulation(years_to_run=20, annual_intake=val, retention_rate=retention, squadron_configs=t_sqs, ute=ute_val)
                
                start_year = t_df['year'].min()
                horizons = {
                    "5-Year": 4,
                    "10-Year": 9,
                    "20-Year": 19
                }

                for label, year_offset in horizons.items():
                    target_year = start_year + year_offset
                    # Filter for the specific year
                    snapshot = t_df[t_df['year'] == target_year]
                    
                    if not snapshot.empty:
                        # Aggregate fleet wide for that year
                        total_pilots = snapshot['total_pilots'].sum()
                        exp_pilots = snapshot['fl_count'].sum() + snapshot['ip_count'].sum()
                        
                        ratio = exp_pilots / total_pilots if total_pilots > 0 else 0
                        
                        stability_data.append({
                            "Annual Intake": val, 
                            "Exp Ratio": ratio, 
                            "Horizon": label
                        })

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
            st.plotly_chart(fig_frontier, use_container_width=True)
else:
    st.info("Set parameters and click 'Run Simulation'.")