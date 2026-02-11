import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

# 1. PAGE CONFIG MUST BE FIRST
st.set_page_config(page_title="Pilot Supply Chain Analytics", layout="wide")

# 2. MODERN STYLING
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# 3. SINGLE DATA LOADING FUNCTION
# DEFAULT_DATA_PATH = "outputs/research_data.csv"
DEFAULT_DATA_PATH = "outputs/simulation_results.parquet"

@st.cache_data
def load_data(uploaded_file):
    # Priority 1: User Upload
    if uploaded_file is not None:
        if uploaded_file.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        return pd.read_csv(uploaded_file)
    
    # Priority 2: Automated Path in Repo
    if os.path.exists(DEFAULT_DATA_PATH):
        if DEFAULT_DATA_PATH.endswith('.parquet'):
            return pd.read_parquet(DEFAULT_DATA_PATH)
        return pd.read_csv(DEFAULT_DATA_PATH)
    
    # Priority 3: Mock Data Fallback (Only if file is missing)
    st.warning("Default file not found. Loading mock data.")
    data = [[18, 10, 180, 0.3, 3, 30, 2, 2, 2, 1, "WG Shortfall", 1, "WG Shortfall", 4.0, 4.62, 14.0, 14.0, 3.78, 11.31, 13.08, 0.84, 2.69, 0.92, 0.18, 0.19, 0.06]]
    cols = ["paa", "ute", "total_capacity", "exp_ratio", "ip_qty", "total_pilots", "mqt_qty", "flug_qty", "ipug_qty", "rap_state_code", "rap_state_label", "blue_rap_state_code", "blue_rap_state_label", "mqt_monthly", "wg_monthly", "fl_monthly", "ip_monthly", "wg_blue_monthly", "fl_blue_monthly", "ip_blue_monthly", "wg_red_monthly", "fl_red_monthly", "ip_red_monthly", "wg_red_pct", "fl_red_pct", "ip_red_pct"]
    return pd.DataFrame(data, columns=cols)

# 4. SIDEBAR - ONLY ONE FILE UPLOADER
with st.sidebar:
    st.header("📊 Data Settings")
    uploaded_file = st.file_uploader("Upload data", type=["csv", "parquet"])
    
    # Load the data once
    with st.spinner('Loading data...'):
        df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"Loaded {len(df):,} rows")

    # 5. SIDEBAR FILTERS
    st.header("Scenario Filters")
    inputs = {}
    filter_cols = ['paa', 'ute', 'total_pilots', 'exp_ratio', 'ip_qty', 'mqt_qty', 'flug_qty', 'ipug_qty']
    
    # Pre-processing columns to avoid UI errors
    if 'exp_ratio' in df.columns:
        df['exp_ratio'] = df['exp_ratio'].round(2)
    if 'ute' in df.columns:
        df['ute'] = df['ute'].round(1)

    for col in filter_cols:
        if col in df.columns:
            options = sorted(df[col].unique())
            inputs[col] = st.selectbox(f"Select {col.replace('_', ' ').upper()}", options, index=0)

# --- UI HEADER ---
st.title("✈️ Pilot Supply Chain Analytics")
st.caption("Interactive Dashboard for RAP Equity and Sortie Composition -- 120 Day Training Phase Snapshot")

# --- DATA PROCESSING LOGIC ---
def get_filtered_data(target_x):
    mask = pd.Series([True] * len(df))
    for col, val in inputs.items():
        if col != target_x:
            mask &= (df[col] == val)
    
    filtered = df[mask].copy()
    if filtered.empty: return filtered

    agg = filtered.groupby(target_x).mean(numeric_only=True).reset_index()
    if 'wg_red_monthly' not in agg.columns:
        agg['wg_red_monthly'] = agg['wg_monthly'] - agg['wg_blue_monthly']
        agg['fl_red_monthly'] = agg['fl_monthly'] - agg['fl_blue_monthly']
        agg['ip_red_monthly'] = agg['ip_monthly'] - agg['ip_blue_monthly']
    return agg

# --- MAIN LAYOUT ---
col_main, col_summary = st.columns([3, 1])

with col_main:
    # CHART 1: EQUITY # TODO update X axis title based on selectbox
    st.subheader("📊 Sortie Equity (Total Monthly)")
    x_options = [c for c in ['ute', 'paa', 'total_pilots', 'exp_ratio'] if c in df.columns]
    ix_equity = x_options.index('ute') if 'ute' in x_options else 0
    x_var_equity = st.selectbox("X-Axis Variable", x_options, index=ix_equity, key="equity_x")
    equity_data = get_filtered_data(x_var_equity)

    if not equity_data.empty:
        fig_equity = go.Figure()
        colors_total = {'wg_monthly': '#3b82f6', 'fl_monthly': '#8b5cf6', 'ip_monthly': '#10b981'}
        names = {'wg_monthly': 'Wingman', 'fl_monthly': 'Flight Lead', 'ip_monthly': 'Instructor'}
        for col in ['wg_monthly', 'fl_monthly', 'ip_monthly']:
            fig_equity.add_trace(go.Scatter(x=equity_data[x_var_equity], y=equity_data[col], name=names[col], 
                                          line=dict(color=colors_total[col], width=3), mode='lines+markers'))
        fig_equity.add_hline(y=9.0, line_dash="dot", line_color="#b91c1c", annotation_text="9.0 Inexp.")
        fig_equity.add_hline(y=8.0, line_dash="dot", line_color="#fca5a5", annotation_text="8.0 Exp.")
        fig_equity.update_layout(xaxis_title='UTE', yaxis_title='Monthly Sorties', hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20), height=350)
        st.plotly_chart(fig_equity, width='stretch')

    # CHART 2: COMPOSITION # TODO update X axis title based on selectbox
    st.write("---")
    st.subheader("🧱 Sortie Composition")
    col_comp_1, col_comp_2 = st.columns([2, 1])
    with col_comp_1:
        ix_comp = x_options.index('exp_ratio') if 'exp_ratio' in x_options else 0
        x_var_comp = st.selectbox("X-Axis Variable", x_options, index=ix_comp, key="comp_x")
    with col_comp_2:
        st.write("") # Spacer
        show_trends = st.toggle("Show Total Trendlines", value=False)
    
    comp_data = get_filtered_data(x_var_comp)
    if not comp_data.empty:
        fig_comp = go.Figure()
        colors = {'wg': ('#3b82f6', '#93c5fd'), 'fl': ('#8b5cf6', '#c4b5fd'), 'ip': ('#10b981', '#6ee7b7')}
        for role in ['wg', 'fl', 'ip']:
            fig_comp.add_trace(go.Bar(x=comp_data[x_var_comp], y=comp_data[f'{role}_blue_monthly'], name=f"{role.upper()} Blue", marker_color=colors[role][0], offsetgroup=role))
            fig_comp.add_trace(go.Bar(x=comp_data[x_var_comp], y=comp_data[f'{role}_red_monthly'], name=f"{role.upper()} Red", marker_color=colors[role][1], offsetgroup=role, base=comp_data[f'{role}_blue_monthly']))
            if show_trends:
                fig_comp.add_trace(go.Scatter(x=comp_data[x_var_comp], y=comp_data[f'{role}_monthly'], name=f"{role.upper()} Total Trend", line=dict(color=colors[role][0], width=2), mode='lines+markers'))
        fig_comp.add_hline(y=9.0, line_dash="dot", line_color="#b91c1c", annotation_text="9.0 Inexp.")
        fig_comp.add_hline(y=8.0, line_dash="dot", line_color="#fca5a5", annotation_text="8.0 Exp.")
        fig_comp.update_layout(xaxis_title='Experience Ratio', yaxis_title='Monthly Sorties', barmode='group', height=450, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_comp, width='stretch')

    # CHART 3: HEATMAP
    st.write("---")
    st.subheader("🗺️ RAP State Heatmap")
    is_blue = st.toggle("Show Only Blue RAP Counters", value=False)
    code_col = "blue_rap_state_code" if is_blue else "rap_state_code"
    label_col = "blue_rap_state_label" if is_blue else "rap_state_label"

    
    heat_mask = pd.Series([True] * len(df))
    for col, val in inputs.items():
        # We do NOT filter by UTE or EXP_RATIO here because they are the X and Y axes of the map
        if col not in ['ute', 'exp_ratio']:
            heat_mask &= (df[col] == val)

    df_heat_filtered = df[heat_mask].copy()

    if not df_heat_filtered.empty:
        # 3. Pivot using the FILTERED data
        heat_df = df_heat_filtered.pivot_table(index='ute', columns='exp_ratio', values=code_col, aggfunc='first').sort_index(ascending=False)
        label_df = df_heat_filtered.pivot_table(index='ute', columns='exp_ratio', values=label_col, aggfunc='first').sort_index(ascending=False)

    else:
        heat_df = df.pivot_table(index='ute', columns='exp_ratio', values=code_col, aggfunc='first').sort_index(ascending=False)
        label_df = df.pivot_table(index='ute', columns='exp_ratio', values=label_col, aggfunc='first').sort_index(ascending=False)

    
    color_map = {0: "#22c55e", 1: "#fef08a", 2: "#fde047", 3: "#fdba74", 4: "#eab308", 5: "#f97316", 6: "#ea580c", 7: "#ef4444"}
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
        z=heat_df.values, x=heat_df.columns, y=heat_df.index, customdata=label_df.values,
        colorscale=discrete_colorscale, showscale=False, zmin=0, zmax=max_val, xgap=1, ygap=1,
        hovertemplate="<b>Status: %{customdata}</b><br>Exp Ratio: %{x:.0%}<br>UTE: %{y}<extra></extra>"
    ))
    
    state_labels_dict = {0: "All Make RAP", 1: "WG Shortfall", 2: "FL Shortfall", 3: "WG+FL Shortfall", 4: "IP Shortfall", 5: "WG+IP Shortfall", 6: "FL+IP Shortfall", 7: "WG+FL+IP Shortfall"}
    for code, color in color_map.items():
        fig_heat.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='square', color=color), showlegend=True, name=state_labels_dict.get(code)))

    fig_heat.update_layout(xaxis_title="Experience Ratio", yaxis_title="UTE", height=500)
    # fig_heat.update_layout(xaxis_title="Experience Ratio", yaxis_title="UTE", height=500, xaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig_heat, width='stretch')

# --- SUMMARY SIDEBAR ---
with col_summary:
    st.subheader("Status Overview")
    
    mask = pd.Series([True] * len(df))
    for col, val in inputs.items():
        mask &= (df[col] == val)
    current_match = df[mask]

    if not current_match.empty:
        row = current_match.mean(numeric_only=True)
        label = current_match.iloc[0]['rap_state_label']
        
        # Pre-formatting variables makes the HTML string cleaner/easier to debug
        wg_t, fl_t, ip_t = f"{row['wg_monthly']:.1f}", f"{row['fl_monthly']:.1f}", f"{row['ip_monthly']:.1f}"
        wg_b, fl_b, ip_b = f"{row['wg_blue_monthly']:.1f}", f"{row['fl_blue_monthly']:.1f}", f"{row['ip_blue_monthly']:.1f}"
        wg_r, fl_r, ip_r = f"{row['wg_red_monthly']:.1f}", f"{row['fl_red_monthly']:.1f}", f"{row['ip_red_monthly']:.1f}"

        # NOTE: Indentation removed inside the string to prevent Markdown Code Block triggering
        st.markdown(f"""
<div style="background-color:#0f172a; padding:20px; border-radius:15px; color:white; margin-bottom:20px;">
<p style="font-size:0.7rem; color:#94a3b8; margin-bottom:2px; letter-spacing: 0.05em;">OVERALL STATUS</p>
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
    else:
        st.info("No exact match for these filters.")

# ==============================================================================
# 🕵️‍♂️ TRUTH INSPECTOR (Raw Data vs. ML Brain)
# ==============================================================================
st.set_page_config(layout="wide")
st.title("⚖️ The Moment of Truth: Physics vs. AI")
st.markdown("""
**Compare & Contrast:** * **Dots** ⚫ match your inputs in the **Raw Simulation Data** (Historical).
* **Lines** ➖ are the **ML Brain's Predictions** (theoretical).
* *If the dots are far from the lines, the model might be over-smoothing the chaos.*
""")

# --- 1. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    # 1. Load Brain
    brain_path = "brains/hpc_sortie_brain_lite.pkl"
    model = None
    if os.path.exists(brain_path):
        model = joblib.load(brain_path)
    
    # 2. Load Raw Data (Parquet)
    # Update this path to where your aggregated parquet lives
    data_path = "outputs/simulation_results.parquet" 
    df = None
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        
    return model, df

brain, df_raw = load_resources()

if brain is None or df_raw is None:
    st.error("⚠️ Missing Files! Ensure `hpc_sortie_brain_lite.pkl` and `outputs/simulation_results.parquet` are in the folder.")
    st.stop()

# --- 2. CONTROLS ---
with st.container():
    st.subheader("1. Scenario Settings")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # PAA (Discrete in Raw Data)
        valid_paas = sorted(df_raw['paa'].unique())
        sb_paa = st.selectbox("PAA", valid_paas, index=0)
        
    with col2:
        # UTE (Discrete in Raw Data)
        valid_utes = sorted(df_raw['ute'].unique())
        default_ute = 10.0 if 10.0 in valid_utes else valid_utes[0]
        sb_ute = st.selectbox("UTE Rate", valid_utes, index=valid_utes.index(default_ute))
        
    with col3:
        sb_pilots = st.slider("Line Pilots", 20, 80, 40)
        
    with col4:
        sb_ips = st.slider("Active IPs", 2, 20, 6)

    with col5:
        sb_exp = st.slider("Exp Ratio", 0.2, 0.8, 0.45)

    st.subheader("2. Comparison Controls")
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        # The variable to sweep on X-axis
        upgrade_type = st.radio("Vary Student Load:", ["MQT", "FLUG", "IPUG"], horizontal=True)
        col_map = {"MQT": "mqt_qty", "FLUG": "flug_qty", "IPUG": "ipug_qty"}
        target_col = col_map[upgrade_type]
        
    with c_col2:
        # Tolerance for Raw Data Search
        st.caption("🔍 Raw Data Search Window")
        tol_pilots = st.slider("Pilot Tolerance (+/-)", 0, 5, 2, help="Widen to find more raw data points.")
        tol_ips = st.slider("IP Tolerance (+/-)", 0, 3, 1)

# --- 3. ENGINE: GET DATA ---

# A. RAW DATA LOOKUP
def get_raw_data():
    # Exact match on PAA/UTE (Physics constraints)
    mask = (df_raw['paa'] == sb_paa) & (df_raw['ute'] == sb_ute)
    
    # Fuzzy match on Pilots/IPs/Exp (Because exact matches are rare)
    mask &= df_raw['total_pilots'].between(sb_pilots - tol_pilots, sb_pilots + tol_pilots)
    mask &= df_raw['ip_qty'].between(sb_ips - tol_ips, sb_ips + tol_ips)
    
    # We don't filter Exp Ratio strictly, but we color/size by it maybe? 
    # Or just filter loosely to keep the data relevant
    mask &= df_raw['exp_ratio'].between(sb_exp - 0.1, sb_exp + 0.1)
    
    return df_raw[mask].copy()

# B. BRAIN PREDICTION
def get_brain_prediction(x_range):
    # Create scenario frame
    scen = pd.DataFrame({
        'paa': sb_paa, 'ute': sb_ute, 'total_pilots': sb_pilots,
        'ip_qty': sb_ips, 'exp_ratio': sb_exp,
        'mqt_qty': 0, 'flug_qty': 0, 'ipug_qty': 0
    }, index=range(len(x_range)))
    
    # Apply sweep
    scen[target_col] = x_range
    
    # Calc Features
    scen['total_students'] = scen['mqt_qty'] + scen['flug_qty'] + scen['ipug_qty']
    scen['ip_ratio'] = scen['ip_qty'] / scen['total_pilots'].replace(0, 1)
    scen['ip_to_stud_ratio'] = scen['ip_qty'] / scen['total_students'].replace(0, 0.1)
    
    # Predict
    features = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty', 'ip_ratio', 'ip_to_stud_ratio']
    
    preds = scen[[target_col]].copy()
    targets = ['wg_monthly', 'fl_monthly', 'ip_monthly']
    
    for t in targets:
        if t in brain:
            preds[t] = brain[t].predict(scen[features])
            
    return preds

# --- 4. DEEP DIVE INSPECTION ---
st.divider()
st.subheader("3. Deep Dive Inspector")
st.markdown("Isolate one metric and **color the raw data** to see which variables are causing the spread.")

# A. Controls for Deep Dive
dd_col1, dd_col2 = st.columns(2)
with dd_col1:
    inspect_metric = st.selectbox(
        "Metric to Inspect", 
        ["wg_monthly", "fl_monthly", "ip_monthly", "wg_blue_monthly"],
        format_func=lambda x: x.replace("_monthly", " Sorties").replace("_", " ").title()
    )

with dd_col2:
    color_by = st.selectbox(
        "Color Raw Dots By...",
        ["total_pilots", "ip_qty", "ute", "exp_ratio"],
        format_func=lambda x: x.replace("_", " ").title()
    )

# B. Get Data
df_lookup = get_raw_data()
df_pred = get_brain_prediction(range(0, 16))

# C. Build the Chart
fig = go.Figure()

# LAYER 1: RAW DATA (Colored Scatter)
if not df_lookup.empty:
    fig.add_trace(go.Scatter(
        x=df_lookup[target_col],
        y=df_lookup[inspect_metric],
        mode='markers',
        name='Raw Simulation',
        marker=dict(
            size=10,
            color=df_lookup[color_by], # Dynamic Coloring
            colorscale='Viridis',      # Blue -> Green -> Yellow
            showscale=True,
            colorbar=dict(title=color_by.replace("_", " ").title()),
            line=dict(width=1, color='DarkSlateGrey') # Border for visibility
        ),
        text=[
            f"Pilots: {p}<br>IPs: {i}<br>Exp: {e:.2f}" 
            for p, i, e in zip(df_lookup['total_pilots'], df_lookup['ip_qty'], df_lookup['exp_ratio'])
        ],
        hovertemplate="<b>Raw Data Point</b><br>Sorties: %{y:.2f}<br>%{text}<extra></extra>"
    ))
else:
    st.warning("⚠️ No Raw Data found. Widen the Tolerance sliders above.")

# LAYER 2: ML PREDICTION (The "Truth" Line)
fig.add_trace(go.Scatter(
    x=df_pred[target_col],
    y=df_pred[inspect_metric],
    mode='lines',
    name='AI Prediction',
    line=dict(color='red', width=4, dash='solid'),
    hovertemplate="<b>AI Prediction</b><br>Sorties: %{y:.2f}<extra></extra>"
))

# Layout Updates
fig.update_layout(
    title=f"Inspecting {inspect_metric}: Colored by {color_by}",
    xaxis_title=f"Number of {upgrade_type} Students",
    yaxis_title="Sorties per Month",
    height=600,
    template="plotly_dark",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

# # --- 4. EXECUTE & PLOT ---
# df_lookup = get_raw_data()
# df_pred = get_brain_prediction(range(0, 16)) # Sweep 0-15 students

# # Colors
# colors = {"wg_monthly": "#636EFA", "fl_monthly": "#EF553B", "ip_monthly": "#00CC96"}
# names = {"wg_monthly": "Wingman", "fl_monthly": "Flt Lead", "ip_monthly": "Instructor"}

# fig = go.Figure()

# # LAYER 1: RAW DATA (Dots)
# if not df_lookup.empty:
#     for metric in colors.keys():
#         fig.add_trace(go.Box(
#             x=df_lookup[target_col],
#             y=df_lookup[metric],
#             name=f"{names[metric]} (Raw)",
#             marker_color=colors[metric],
#             boxpoints='all', # Show all dots
#             jitter=0.3,      # Spread them out so they don't overlap
#             pointpos=0,      # Center them
#             fillcolor='rgba(0,0,0,0)', # Transparent box
#             line=dict(width=0), # Hide box lines, just show dots
#             showlegend=False,
#             hoverinfo='y'
#         ))
# else:
#     st.warning("⚠️ No Raw Data found for these settings. Try widening the Pilot/IP Tolerance.")

# # LAYER 2: BRAIN (Lines)
# for metric in colors.keys():
#     fig.add_trace(go.Scatter(
#         x=df_pred[target_col],
#         y=df_pred[metric],
#         mode='lines',
#         name=f"{names[metric]} (AI)",
#         line=dict(color=colors[metric], width=4),
#     ))

# # LAYOUT
# fig.update_layout(
#     title=f"Sortie Rates: Raw Data vs. AI Prediction ({upgrade_type} Sweep)",
#     xaxis_title=f"Number of {upgrade_type} Students",
#     yaxis_title="Sorties / Month",
#     height=600,
#     hovermode="x unified",
#     template="plotly_dark"
# )

# st.plotly_chart(fig, use_container_width=True)

# # --- 5. DEBUGGING TABLE ---
# with st.expander("See Underlying Data"):
#     st.write("First 5 rows of matched Raw Data:")
#     st.dataframe(df_lookup.head())
#     st.write("AI Predictions:")
#     st.dataframe(df_pred.head())