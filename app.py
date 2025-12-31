import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide", page_title="Fighter Squadron Absorption Research")

st.title("✈️ Fighter Squadron Absorption & Health Dashboard")
st.markdown("Analyzing the 'Supply Chain' of pilot experience and the impact of upgrade demand.")

@st.cache_data(ttl=60) # Refresh cache every minute to catch new sweep data
def load_data():
    if not os.path.exists("outputs/research_data.csv"):
        return pd.DataFrame()
    # Read CSV with low_memory=False to handle the large dataset
    return pd.read_csv("outputs/research_data.csv", low_memory=False)

try:
    df = load_data()

    if df.empty:
        st.info("Sweep in progress... Waiting for first rows to be written to CSV.")
        st.stop()

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Global Simulation Constants")
    
    # Check if total_pilots was successfully repaired/added
    if 'total_pilots' not in df.columns:
        st.sidebar.error("Column 'total_pilots' missing. Run repair script!")
        st.stop()

    paa_sel = st.sidebar.selectbox("PAA (Primary Aircraft Available)", sorted(df['paa'].unique()))
    total_pilots_sel = st.sidebar.selectbox("Total Pilot Inventory", sorted(df['total_pilots'].unique()))
    
    # Filter the dataframe based on sidebar
    f_df = df[(df['paa'] == paa_sel) & (df['total_pilots'] == total_pilots_sel)]

    # --- ROW 1: THE SAFETY ZONE (RQ1) ---
    st.header("RQ1: The 'Healthy Squadron' Boundary")
    st.write("Heatmap of RAP State (0=Healthy, 7=Critical) across Experience Ratios and IP Staffing.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        # MATCHED TO YOUR CSV: using 'mqt_qty' and 'ipug_qty'
        mqt_load = st.selectbox("MQT Student Load", sorted(f_df['mqt_qty'].unique()))
        ipug_load = st.selectbox("IPUG Student Load", sorted(f_df['ipug_qty'].unique()))
    
    with col2:
        heatmap_df = f_df[(f_df['mqt_qty'] == mqt_load) & (f_df['ipug_qty'] == ipug_load)]
        
        if not heatmap_df.empty:
            # We use mean here to handle multiple iterations per config
            pivot_df = heatmap_df.pivot_table(
                index='ip_qty', 
                columns='exp_ratio', 
                values='rap_state_code', 
                aggfunc='mean'
            )
            
            fig_heat = px.imshow(
                pivot_df,
                labels=dict(x="Experience Ratio", y="IP Quantity", color="Avg State"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="RdYlGn_r",
                aspect="auto",
                title=f"Squadron Health: {mqt_load} MQTs, {ipug_load} IPUGs"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("No data for this specific student load combination yet.")

    # --- ROW 2: THE RED AIR TAX & AGING (RQ3) ---
    st.divider()
    st.header("RQ3: The 'Red Air Tax' & Aging Rates")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("IP Red Air Burden")
        # Scatter plot showing how IPs get taxed as MQT load increases
        fig_red = px.scatter(
            f_df, 
            x="mqt_qty", 
            y="ip_red_pct", 
            color="rap_state_label",
            size="ute",
            hover_data=['ip_qty', 'exp_ratio'],
            title="IP Red Air % vs. MQT Load"
        )
        st.plotly_chart(fig_red, use_container_width=True)

    with col4:
        st.subheader("Blue Aging Rate (Wingmen)")
        # Box plot showing the actual Blue Air experience gain
        fig_aging = px.box(
            f_df, 
            x="exp_ratio", 
            y="wg_blue_monthly", 
            color="paa",
            title="Wingman Blue Aging Rate (Sorties/Mo) by Exp Ratio"
        )
        st.plotly_chart(fig_aging, use_container_width=True)

    # --- DATA EXPLORER ---
    st.divider()
    with st.expander("View Raw Research Data"):
        st.write(f"Showing first 500 of {len(df)} rows.")
        st.dataframe(df.head(500))

except Exception as e:
    st.error(f"Dashboard Error: {e}")
    st.info("The sweep is likely still running. Try refreshing in a minute.")