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
retention = st.sidebar.slider("Retention Rate (0.0 - 1.0)", 0.2, 0.6, 0.4)

# --- Run Simulation ---
if st.sidebar.button("Run Simulation"):
    sim, squadrons = setup_debug_simulation()
    
    # Run the engine
    df = sim.run_simulation(years, intake, retention, squadrons)
    
    # Create a Date column for better X-axis plotting
    df['timeline'] = df['year'].astype(str) + " P" + df['phase'].astype(str)

    # --- Top Level Metrics ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Final Total Pilots", df['total_pilots'].iloc[-1])
    m2.metric("Final Exp Ratio", f"{df['exp_rat'].iloc[-1]*100:.1f}%")
    m3.metric("Total Separations", df['separated'].sum())

    # --- Charts ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pilot Population by Qualification")
        fig_pop = px.area(df, x='timeline', y=['wg_count', 'fl_count', 'ip_count'],
                          title="Roster Composition Over Time",
                          labels={'value': 'Count', 'timeline': 'Year/Phase'},
                          color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'])
        st.plotly_chart(fig_pop, use_container_width=True)

    with col2:
        st.subheader("The 'Bathtub' (Experience Ratio)")
        fig_exp = px.line(df, x='timeline', y='exp_rat', 
                          title="Experience Ratio (%)",
                          labels={'exp_rat': 'Exp Ratio', 'timeline': 'Year/Phase'})
        # Add a "Red Line" for critical health (e.g., 25%)
        fig_exp.add_hline(y=0.45, line_dash="dash", line_color="yellow", annotation_text="Runaway Sortie Inequity Line")
        st.plotly_chart(fig_exp, use_container_width=True)

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