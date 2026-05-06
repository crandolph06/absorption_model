import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import itertools
import os
import joblib

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
# 2. ML ENGINE SETUP
# ==============================================================================
@st.cache_resource
def load_brain():
    model_path = "brains/hpc_sortie_brain_lite.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    st.error(f"⚠️ Brain file not found at {model_path}")
    st.stop()

brain = load_brain()

def predict_metrics(df_inputs):
    """Takes a dataframe of inputs, calculates features, and returns predictions."""
    df = df_inputs.copy()
    
    # Feature Engineering (Matching Training Data)
    df['total_students'] = df['mqt_qty'] + df['flug_qty'] + df['ipug_qty']
    df['ip_ratio'] = df['ip_qty'] / df['total_pilots'].replace(0, 1)
    df['ip_to_stud_ratio'] = df['ip_qty'] / df['total_students'].replace(0, 0.1)
    
    features = [
        'paa', 'ute', 'exp_ratio', 'total_pilots', 
        'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty', 
        'ip_ratio', 'ip_to_stud_ratio'
    ]
    
    targets = [
        'wg_monthly', 'fl_monthly', 'ip_monthly', 
        'wg_blue_monthly', 'fl_blue_monthly', 'ip_blue_monthly'
    ]
    
    # Predict
    for t in targets:
        if t in brain:
            df[t] = brain[t].predict(df[features])
            
    # Calculate Red Air
    df['wg_red_monthly'] = df['wg_monthly'] - df['wg_blue_monthly']
    df['fl_red_monthly'] = df['fl_monthly'] - df['fl_blue_monthly']
    df['ip_red_monthly'] = df['ip_monthly'] - df['ip_blue_monthly']
    
    # Red Air Percentages (Safe Division)
    df['wg_red_pct'] = df['wg_red_monthly'] / df['wg_monthly'].replace(0, 1)
    df['fl_red_pct'] = df['fl_red_monthly'] / df['fl_monthly'].replace(0, 1)
    df['ip_red_pct'] = df['ip_red_monthly'] / df['ip_monthly'].replace(0, 1)
    
    return df

def calculate_rap_code(row, is_blue=False):
    """Calculates RAP status mask. 1=WG, 2=FL, 4=IP"""
    suffix = "_blue_monthly" if is_blue else "_monthly"
    wg = row.get(f'wg{suffix}', 0)
    fl = row.get(f'fl{suffix}', 0)
    ip = row.get(f'ip{suffix}', 0)
    
    code = 0
    if wg < 9: code += 1
    if fl < 8: code += 2
    if ip < 8: code += 4
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
    st.caption("Adjust variables to predict system outcomes.")
    
    inputs = {}
    inputs['paa'] = st.slider("PAA (Aircraft)", 12, 30, 18, 1)
    inputs['ute'] = st.slider("UTE Rate", 6.0, 24.0, 10.0, 0.5)
    inputs['total_pilots'] = st.slider("Total Pilots", 20, 80, 40, 1)
    inputs['exp_ratio'] = st.slider("Experience Ratio", 0.20, 0.80, 0.45, 0.01)
    inputs['ip_qty'] = st.slider("Active IPs", 2, 20, 6, 1)
    
    st.divider()
    st.subheader("Student Load")
    inputs['mqt_qty'] = st.number_input("MQT Students", 0, 20, 2)
    inputs['flug_qty'] = st.number_input("FLUG Students", 0, 20, 2)
    inputs['ipug_qty'] = st.number_input("IPUG Students", 0, 20, 1)

# Ranges for 1D Sweeps
sweep_ranges = {
    'ute': np.arange(6.0, 24.1, 0.5),
    'paa': np.arange(12, 31, 1),
    'total_pilots': np.arange(20, 81, 1),
    'exp_ratio': np.arange(0.20, 0.81, 0.02),
    'ip_qty': np.arange(2, 21, 1),
    'mqt_qty': np.arange(0, 21, 1)
}

# ==============================================================================
# 4. DATA GENERATION HELPERS
# ==============================================================================
def generate_1d_sweep(x_var):
    """Creates a synthetic dataframe varying only the x_var."""
    x_vals = sweep_ranges.get(x_var, np.arange(0, 10))
    df_sweep = pd.DataFrame([inputs] * len(x_vals))
    df_sweep[x_var] = x_vals
    return predict_metrics(df_sweep)

# ==============================================================================
# 5. MAIN UI & CHARTS
# ==============================================================================
st.title("✈️ Predictive Supply Chain Analytics")
st.caption("Interactive Dashboard powered by HPC ML Brain -- Interpolating the Simulation Surface")

col_main, col_summary = st.columns([3, 1])

with col_main:
    # --- CHART 1: EQUITY (1D Sweep) ---
    st.subheader("📊 Sortie Equity (Total Monthly)")
    x_options = list(sweep_ranges.keys())
    x_var_equity = st.selectbox("X-Axis Variable", x_options, index=0, key="equity_x")
    
    df_equity = generate_1d_sweep(x_var_equity)
    
    fig_equity = go.Figure()
    colors_total = {'wg_monthly': '#3b82f6', 'fl_monthly': '#8b5cf6', 'ip_monthly': '#10b981'}
    names = {'wg_monthly': 'Wingman', 'fl_monthly': 'Flight Lead', 'ip_monthly': 'Instructor'}
    
    for col in ['wg_monthly', 'fl_monthly', 'ip_monthly']:
        fig_equity.add_trace(go.Scatter(
            x=df_equity[x_var_equity], y=df_equity[col], name=names[col], 
            line=dict(color=colors_total[col], width=3), mode='lines'
        ))
        
    fig_equity.add_hline(y=9.0, line_dash="dot", line_color="#b91c1c", annotation_text="9.0 Inexp.")
    fig_equity.add_hline(y=8.0, line_dash="dot", line_color="#fca5a5", annotation_text="8.0 Exp.")
    fig_equity.update_layout(xaxis_title=x_var_equity.upper(), yaxis_title='Monthly Sorties', hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20), height=350)
    st.plotly_chart(fig_equity, width='stretch')

    # --- CHART 2: COMPOSITION (1D Sweep) ---
    st.write("---")
    st.subheader("🧱 Sortie Composition")
    col_comp_1, col_comp_2 = st.columns([2, 1])
    with col_comp_1:
        x_var_comp = st.selectbox("X-Axis Variable", x_options, index=3, key="comp_x") # Default exp_ratio
    with col_comp_2:
        st.write("") 
        show_trends = st.toggle("Show Total Trendlines", value=False)
    
    df_comp = generate_1d_sweep(x_var_comp)
    
    fig_comp = go.Figure()
    colors = {'wg': ('#3b82f6', '#93c5fd'), 'fl': ('#8b5cf6', '#c4b5fd'), 'ip': ('#10b981', '#6ee7b7')}
    
    for role in ['wg', 'fl', 'ip']:
        fig_comp.add_trace(go.Bar(x=df_comp[x_var_comp], y=df_comp[f'{role}_blue_monthly'], name=f"{role.upper()} Blue", marker_color=colors[role][0], offsetgroup=role))
        fig_comp.add_trace(go.Bar(x=df_comp[x_var_comp], y=df_comp[f'{role}_red_monthly'], name=f"{role.upper()} Red", marker_color=colors[role][1], offsetgroup=role, base=df_comp[f'{role}_blue_monthly']))
        if show_trends:
            fig_comp.add_trace(go.Scatter(x=df_comp[x_var_comp], y=df_comp[f'{role}_monthly'], name=f"{role.upper()} Total Trend", line=dict(color=colors[role][0], width=2), mode='lines'))
            
    fig_comp.add_hline(y=9.0, line_dash="dot", line_color="#b91c1c", annotation_text="9.0 Inexp.")
    fig_comp.add_hline(y=8.0, line_dash="dot", line_color="#fca5a5", annotation_text="8.0 Exp.")
    fig_comp.update_layout(xaxis_title=x_var_comp.upper(), yaxis_title='Monthly Sorties', barmode='group', height=450, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_comp, width='stretch')

    # --- CHART 3: HEATMAP (2D Sweep) ---
    st.write("---")
    st.subheader("🗺️ RAP State Heatmap")
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
    df_heat_preds = predict_metrics(df_heat_base)
    
    # Calculate Status
    df_heat_preds['rap_code'] = df_heat_preds.apply(lambda r: calculate_rap_code(r, is_blue), axis=1)
    df_heat_preds['rap_label'] = df_heat_preds['rap_code'].map(state_labels_dict)

    # Pivot for Plotly
    heat_z = df_heat_preds.pivot(index='ute', columns='exp_ratio', values='rap_code').sort_index(ascending=False)
    heat_labels = df_heat_preds.pivot(index='ute', columns='exp_ratio', values='rap_label').sort_index(ascending=False)
    
    color_map = {0: "#22c55e", 1: "#fef08a", 2: "#fde047", 3: "#fdba74", 4: "#eab308", 5: "#f97316", 6: "#ea580c", 7: "#ef4444"}
    
    # Custom discrete color mapping
    max_val = 7
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
        fig_heat.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='square', color=color), showlegend=True, name=state_labels_dict.get(code)))

    fig_heat.update_layout(xaxis_title="Experience Ratio", yaxis_title="UTE", height=500)
    st.plotly_chart(fig_heat, width='stretch')

# ==============================================================================
# 6. SUMMARY SIDEBAR (Single Point Prediction)
# ==============================================================================
with col_summary:
    st.subheader("Status Overview")
    
    # Predict exactly the point defined in the sidebar
    df_single = pd.DataFrame([inputs])
    row = predict_metrics(df_single).iloc[0]
    
    current_code = calculate_rap_code(row, is_blue=False)
    label = state_labels_dict.get(current_code, "Unknown")
    
    wg_t, fl_t, ip_t = f"{row['wg_monthly']:.1f}", f"{row['fl_monthly']:.1f}", f"{row['ip_monthly']:.1f}"
    wg_b, fl_b, ip_b = f"{row['wg_blue_monthly']:.1f}", f"{row['fl_blue_monthly']:.1f}", f"{row['ip_blue_monthly']:.1f}"
    wg_r, fl_r, ip_r = f"{row['wg_red_monthly']:.1f}", f"{row['fl_red_monthly']:.1f}", f"{row['ip_red_monthly']:.1f}"

    st.markdown(f"""
<div style="background-color:#0f172a; padding:20px; border-radius:15px; color:white; margin-bottom:20px;">
<p style="font-size:0.7rem; color:#94a3b8; margin-bottom:2px; letter-spacing: 0.05em;">PREDICTED STATUS</p>
<h2 style="margin:0; font-size:1.3rem; color: #f8fafc;">{label}</h2>
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

    st.write("---")
    st.subheader("Red Air Exposure")
    burden_df = pd.DataFrame({
        'Role': ['WG', 'FL', 'IP'],
        'Red Pct': [row['wg_red_pct'], row['fl_red_pct'], row['ip_red_pct']]
    })
    fig_burden = px.bar(burden_df, y='Role', x='Red Pct', orientation='h', color_discrete_sequence=['#f43f5e'])
    fig_burden.update_layout(xaxis_tickformat='.0%', height=250, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Red Air %")
    st.plotly_chart(fig_burden, width='stretch')