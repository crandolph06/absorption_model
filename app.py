import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

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
DEFAULT_DATA_PATH = "outputs/research_data.csv"

@st.cache_data
def load_data(uploaded_file):
    # Priority 1: User Upload
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    
    # Priority 2: Automated Path in Repo
    if os.path.exists(DEFAULT_DATA_PATH):
        return pd.read_csv(DEFAULT_DATA_PATH)
    
    # Priority 3: Mock Data Fallback (Only if file is missing)
    st.warning("Default CSV not found. Loading mock data.")
    data = [[18, 10, 180, 0.3, 3, 30, 2, 2, 2, 1, "WG Shortfall", 1, "WG Shortfall", 4.0, 4.62, 14.0, 14.0, 3.78, 11.31, 13.08, 0.84, 2.69, 0.92, 0.18, 0.19, 0.06]]
    cols = ["paa", "ute", "total_capacity", "exp_ratio", "ip_qty", "total_pilots", "mqt_qty", "flug_qty", "ipug_qty", "rap_state_code", "rap_state_label", "blue_rap_state_code", "blue_rap_state_label", "mqt_monthly", "wg_monthly", "fl_monthly", "ip_monthly", "wg_blue_monthly", "fl_blue_monthly", "ip_blue_monthly", "wg_red_monthly", "fl_red_monthly", "ip_red_monthly", "wg_red_pct", "fl_red_pct", "ip_red_pct"]
    return pd.DataFrame(data, columns=cols)

# 4. SIDEBAR - ONLY ONE FILE UPLOADER
with st.sidebar:
    st.header("📊 Data Settings")
    uploaded_file = st.file_uploader("Upload an override CSV", type="csv")
    
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
    x_options = [c for c in ['ute', 'paa', 'total_pilots'] if c in df.columns]
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
        st.plotly_chart(fig_equity, use_container_width=True)

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
        fig_comp.update_layout(xaxis_title='Experience Ratio', yaxis_title='Monthly Sorties', barmode='group', height=450, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_comp, use_container_width=True)

    # CHART 3: HEATMAP
    st.write("---")
    st.subheader("🗺️ RAP State Heatmap")
    is_blue = st.toggle("Show Only Blue RAP Counters", value=False)
    code_col = "blue_rap_state_code" if is_blue else "rap_state_code"
    label_col = "blue_rap_state_label" if is_blue else "rap_state_label"

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

    fig_heat.update_layout(xaxis_title="Experience Ratio", yaxis_title="UTE", height=500, xaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig_heat, use_container_width=True)

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
        st.plotly_chart(fig_burden, use_container_width=True)
    else:
        st.info("No exact match for these filters.")