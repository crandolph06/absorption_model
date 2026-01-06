import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.manning_main import setup_debug_simulation

st.set_page_config(page_title="CAF Absorption Simulator", layout="wide")

st.title("🛩️ Fighter Pilot Long-Term Manning Visualizer")
st.markdown("""
This dashboard simulates pilot career progression over 10-20 years. This is a simulation that wishes away all upgrade bottlenecks; pilots automatically attain the next qual at 250 sorties for FLs and 400 hours for IPs. 
            In reality, bottlenecks are much more restrictive than indicated on this dashboard. This is intended to visualize concepts and highlight the most optimistic scenario for decision-makers.
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
years = st.sidebar.slider("Years to Run", 5, 20, 10)
intake = st.sidebar.slider("Annual B-Course Intake", 10, 350, 150)
retention = st.sidebar.slider("Retention Rate (0.0 - 1.0)", 0.0, 1.0, 0.4)

st.sidebar.header("Sortie Generation")
use_custom_ute = st.sidebar.checkbox("Override Baseline UTE (10.0)?", value=False)
if use_custom_ute:
    ute_val = st.sidebar.slider("Custom UTE Rate", 6.0, 20.0, 10.0)
else:
    ute_val = 10.0


st.sidebar.header("Advanced Analysis")
run_sensitivity = st.sidebar.checkbox("Run Stability Frontier Analysis")

# --- Run Simulation ---
if st.sidebar.button("Run Simulation"):
    sim, squadrons = setup_debug_simulation()
    st.session_state['sim_df'] = sim.run_simulation(years, intake, retention, squadrons, ute_val)
    
if 'sim_df' in st.session_state:
    df = st.session_state['sim_df']
    df['timeline'] = df['year'].astype(str) + " P" + df['phase'].astype(str)

    # --- Top Level Metrics ---
    st.markdown(f"### Overall Stats")
    m1, m2, m3 = st.columns(3)
    m1.metric("Final Total Pilots", df['total_pilots'].iloc[-1])
    m2.metric("Final Exp Ratio", f"{df['exp_rat'].iloc[-1]*100:.1f}%")
    m3.metric("Total Separations", df['separated'].sum())

    # --- Charts ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pilot Population by Qualification")

        # squadron_ids = [sq.id for sq in sim.squadrons]
        squadron_ids = sorted(df['squadron_id'].unique().tolist())
        selected_sq = st.selectbox("Select View", options=["All Squadrons"] + squadron_ids)

        # 3. Filter the Data
        if selected_sq == "All Squadrons":
            df_display = df.groupby(['year', 'phase', 'timeline']).agg({
                'wg_count': 'sum',
                'fl_count': 'sum',
                'ip_count': 'sum',
                'total_pilots': 'sum',
                'percent_manned': 'mean',
                'staff_ips': 'sum',
                'staff_fls': 'sum',
                'separated': 'sum',
                'wg_rate_mo': 'mean',
                'fl_rate_mo': 'mean',
                'ip_rate_mo': 'mean'
            }).reset_index()
            df_display['exp_rat'] = (df_display['fl_count'] + df_display['ip_count']) / df_display['total_pilots']

        else:
            df_display = df[df['squadron_id'] == selected_sq].copy()

        # --- Population Mix Chart ---
        fig_pop = px.area(df_display, x='timeline', y=['wg_count', 'fl_count', 'ip_count', 'staff_fls', 'staff_ips'],
                          title="Qualification Mix Over Time",
                          labels={'value': 'Count', 'timeline': 'Year/Phase',
                                  'wg_rate_mo': 'WG Monthly Rate',
                                  'fl_rate_mo': 'FL Monthly Rate',
                                  'ip_rate_mo': 'IP Monthly Rate'},
                          color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', "#DC8F7E", "#78CAB4"],
                          hover_data={'wg_rate_mo': ':.1f',
                                      'fl_rate_mo': ':.1f',
                                      'ip_rate_mo': ':.1f'})
        st.plotly_chart(fig_pop, use_container_width=True)

    with col2:
        st.subheader("'Bathtub or Cliff?' (Experience Ratio)")
        fig_exp = px.line(df_display, x='timeline', y='exp_rat', 
                          title="Experience Ratio (%)",
                          labels={'exp_rat': 'Exp Ratio', 'timeline': 'Year/Phase'})
        # Add a "Red Line" for critical health (e.g., 25%)
        fig_exp.add_hline(y=0.45, line_dash="dash", line_color="yellow", annotation_text="Runaway Sortie Inequity Line")
        fig_exp.add_hline(
            y=0.60, 
            line_dash="dot", 
            line_color="green", 
            annotation_text="Healthy (> 60% Exp)",
            annotation_position="bottom right"
        )

        fig_exp.add_hline(
            y=0.40, 
            line_dash="dot", 
            line_color="red", 
            annotation_text="Broken (< 40% Exp)",
            annotation_position="bottom right"
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    # --- NEW: Stability Frontier Section ---
    if run_sensitivity:
        st.divider()
        st.header("📉 System Stability Frontier")
        st.write("This chart calculates the final 'health' of the fleet across different intake levels to identify the point of system collapse.")

        with st.spinner("Calculating stability across intake range..."):
            # 1. Define range to test (e.g., 100 to 350)
            test_range = list(range(100, 351, 10))
            stability_data = []

            for val in test_range:
                # We must re-initialize the sim/squadrons for every loop to prevent data carry-over
                test_sim, test_sqs = setup_debug_simulation()
                test_df = test_sim.run_simulation(years_to_run=20, annual_intake=intake, ute=val, retention_rate=retention, squadron_configs=test_sqs)
                
                start_year = test_df['year'].min()
                
                horizons = {
                    "5-Year Mark": 4,
                    "10-Year Mark": 9,
                    "15-Year Mark": 14,
                    "20-Year Mark": 19
                }

                for label, year_offset in horizons.items():
                    target_year = start_year + year_offset
                    # Capture the ratio from the final phase of that year
                    snapshot = test_df[test_df['year'] == target_year]
                    if not snapshot.empty:
                        ratio = snapshot['exp_rat'].iloc[-1]
                        stability_data.append({
                            "Annual Intake": val, 
                            "Exp Ratio": ratio, 
                            "Horizon": label
                        })

            analysis_df = pd.DataFrame(stability_data)

            # Create the Frontier Chart
            fig_frontier = px.line(
                analysis_df, 
                x="Annual Intake", 
                y="Exp Ratio",
                color="Horizon",
                title="System Health Decay: Multi-Decade Stability",
                labels={"Exp Ratio": "Experience Ratio (%)", "Annual Intake": "Annual B-Course Intake"},
                # Sequential color palette to show progression of time
                color_discrete_sequence=px.colors.sequential.Reds_r 
            )

            fig_frontier.add_hline(
                y=0.60, 
                line_dash="dot", 
                line_color="green", 
                annotation_text="Healthy (> 60% Exp)",
                annotation_position="bottom right"
            )

            fig_frontier.add_hline(
                y=0.45, 
                line_dash="dot", 
                line_color="yellow", 
                annotation_text="Runaway Sortie Inequity (< 45%)",
                annotation_position="top right"
            )

            fig_frontier.add_hline(
                y=0.40, 
                line_dash="dot", 
                line_color="red", 
                annotation_text="Broken (< 40% Exp)",
                annotation_position="bottom right"
            )

            fig_frontier.update_layout(
                legend_title_text='Time Horizon',
                hovermode="x unified",
                yaxis_tickformat='.0%'
            )

            st.plotly_chart(fig_frontier, use_container_width=True)
            
    # --- Retention vs Separation Chart ---
    st.subheader("Retention vs Separations")
    fig_sep = go.Figure()
    fig_sep.add_trace(go.Bar(x=df['timeline'], y=df['retained'], name='Retained', marker_color='green'))
    fig_sep.add_trace(go.Bar(x=df['timeline'], y=df['separated'], name='Separated', marker_color='red'))
    fig_sep.update_layout(barmode='stack', title="Phase-by-Phase Retention")
    st.plotly_chart(fig_sep, use_container_width=True)

    # Raw Data
    with st.expander("View Raw Simulation Data"):
        st.dataframe(df)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see the results.")

    # TODO Sensitivity analysis for UTE, Retention
    # Chart depicting percent manning