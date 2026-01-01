import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textwrap import dedent

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pilot Supply Chain Analytics", layout="wide")

# Modern Styling
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Mock Data matching the specified columns list order
        data = [
            [18, 10, 180, 0.3, 3, 30, 2, 2, 2, 1, "WG Shortfall", 1, "WG Shortfall", 4.0, 4.62, 14.0, 14.0, 3.78, 11.31, 13.08, 0.84, 2.69, 0.92, 0.18, 0.19, 0.06],
            [18, 12, 216, 0.3, 3, 30, 2, 2, 2, 1, "WG Shortfall", 1, "WG Shortfall", 4.0, 5.5, 15.2, 15.2, 4.5, 12.5, 14.0, 1.0, 2.7, 1.2, 0.18, 0.20, 0.07],
            [18, 14, 252, 0.3, 3, 30, 2, 2, 2, 0, "All Make RAP", 0, "All Make RAP", 4.0, 12.0, 12.5, 12.5, 10.0, 11.0, 11.5, 2.0, 1.5, 1.0, 0.15, 0.15, 0.05],
            [18, 10, 180, 0.5, 3, 30, 2, 2, 2, 0, "All Make RAP", 0, "All Make RAP", 4.0, 11.0, 11.5, 11.5, 9.5, 10.0, 10.0, 1.5, 1.5, 1.5, 0.15, 0.15, 0.05],
            [18, 10, 180, 0.7, 3, 30, 2, 2, 2, 0, "All Make RAP", 0, "All Make RAP", 4.0, 13.0, 13.0, 13.0, 12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 0.08, 0.08, 0.04]
        ]
        cols = [
            "paa", "ute", "total_capacity", "exp_ratio", "ip_qty", "total_pilots", 
            "mqt_qty", "flug_qty", "ipug_qty", "rap_state_code", "rap_state_label", 
            "blue_rap_state_code", "blue_rap_state_label", "mqt_monthly", 
            "wg_monthly", "fl_monthly", "ip_monthly", "wg_blue_monthly", 
            "fl_blue_monthly", "ip_blue_monthly", "wg_red_monthly", 
            "fl_red_monthly", "ip_red_monthly", "wg_red_pct", "fl_red_pct", "ip_red_pct"
        ]
        df = pd.DataFrame(data, columns=cols)
    return df

# --- UI HEADER ---
st.title("✈️ Pilot Supply Chain Analytics")
st.caption("Interactive Dashboard for RAP Equity and Sortie Composition")

uploaded_file = st.sidebar.file_uploader("Upload research_data.csv", type="csv")
df = load_data(uploaded_file)

if 'exp_ratio' in df.columns:
    df['exp_ratio'] = df['exp_ratio'].round(2)
    
if 'ute' in df.columns:
    # Optional: Round ute if it also has decimals (e.g. 12.5)
    df['ute'] = df['ute'].round(1)

# --- SIDEBAR FILTERS ---
st.sidebar.header("Scenario Filters")
inputs = {}
filter_cols = ['paa', 'ute', 'total_pilots', 'exp_ratio', 'ip_qty', 'mqt_qty', 'flug_qty', 'ipug_qty']

for col in filter_cols:
    if col in df.columns:
        options = sorted(df[col].unique())
        inputs[col] = st.sidebar.selectbox(f"Select {col.replace('_', ' ').upper()}", options, index=0)

# --- DATA PROCESSING ---
def get_filtered_data(target_x):
    mask = pd.Series([True] * len(df))
    for col, val in inputs.items():
        print(col)
        if col != target_x:
            mask &= (df[col] == val)
    
    filtered = df[mask].copy()
    if filtered.empty:
        return filtered

    # Aggregate by X-axis variable
    agg = filtered.groupby(target_x).mean(numeric_only=True).reset_index()
    
    # Pre-calculate components if not present
    if 'wg_red_monthly' not in agg.columns:
        agg['wg_red_monthly'] = agg['wg_monthly'] - agg['wg_blue_monthly']
        agg['fl_red_monthly'] = agg['fl_monthly'] - agg['fl_blue_monthly']
        agg['ip_red_monthly'] = agg['ip_monthly'] - agg['ip_blue_monthly']
        
    return agg

# --- MAIN LAYOUT ---
col_main, col_summary = st.columns([3, 1])

with col_main:
    # CHART 1: EQUITY
    st.subheader("📊 Sortie Equity (Total Monthly)")
    x_options = [c for c in ['ute', 'paa', 'total_pilots', 'exp_ratio'] if c in df.columns]
    # Find the index of 'ute' in the list options. Defaults to 0 if not found.
    ix_equity = x_options.index('ute') if 'ute' in x_options else 0
    x_var_equity = st.selectbox("X-Axis Variable", x_options, index=ix_equity, key="equity_x")
    equity_data = get_filtered_data(x_var_equity)

    if not equity_data.empty:
        fig_equity = go.Figure()
        colors_total = {'wg_monthly': '#3b82f6', 'fl_monthly': '#8b5cf6', 'ip_monthly': '#10b981'}
        names = {'wg_monthly': 'Wingman', 'fl_monthly': 'Flight Lead', 'ip_monthly': 'Instructor'}
        
        for col in ['wg_monthly', 'fl_monthly', 'ip_monthly']:
            fig_equity.add_trace(go.Scatter(
                x=equity_data[x_var_equity], 
                y=equity_data[col], 
                name=names[col], 
                line=dict(color=colors_total[col], width=3), 
                mode='lines+markers'
            ))
        
        # Reference Lines
        fig_equity.add_hline(y=9.0, line_dash="dot", line_color="#b91c1c", annotation_text="9.0 Inexp.")
        fig_equity.add_hline(y=8.0, line_dash="dot", line_color="#fca5a5", annotation_text="8.0 Exp.")
        
        fig_equity.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20), height=350)
        st.plotly_chart(fig_equity, use_container_width=True)

    # CHART 2: COMPOSITION
    st.write("---")
    st.subheader("🧱 Sortie Composition")
    
    col_comp_1, col_comp_2 = st.columns([2, 1])
    with col_comp_1:
        # Find the index of 'exp_ratio'
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
            # Blue Bar (Standard Ops)
            fig_comp.add_trace(go.Bar(
                x=comp_data[x_var_comp], 
                y=comp_data[f'{role}_blue_monthly'], 
                name=f"{role.upper()} Blue", 
                marker_color=colors[role][0], 
                offsetgroup=role,
                hovertemplate=f"{role.upper()} Blue: %{{y:.2f}}<extra></extra>"
            ))
            # Red Bar (Red Air Burden)
            fig_comp.add_trace(go.Bar(
                x=comp_data[x_var_comp], 
                y=comp_data[f'{role}_red_monthly'], 
                name=f"{role.upper()} Red", 
                marker_color=colors[role][1], 
                offsetgroup=role, 
                base=comp_data[f'{role}_blue_monthly'],
                hovertemplate=f"{role.upper()} Red: %{{y:.2f}}<extra></extra>"
            ))
            
            # Trendlines for Total Sorties per role
            if show_trends:
                fig_comp.add_trace(go.Scatter(
                    x=comp_data[x_var_comp], 
                    y=comp_data[f'{role}_monthly'], 
                    name=f"{role.upper()} Total Trend", 
                    line=dict(color=colors[role][0], width=2), 
                    mode='lines+markers',
                    marker=dict(symbol='circle-open', size=8)
                ))

        fig_comp.add_hline(y=9.0, line_dash="dot", line_color="#b91c1c", opacity=0.5, annotation_text="9.0 Inexp.")
        fig_comp.add_hline(y=8.0, line_dash="dot", line_color="#fca5a5", annotation_text="8.0 Exp.")
        fig_comp.update_layout(
            barmode='group', 
            height=450, 
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title=x_var_comp.replace('_', ' ').upper())
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- HEATMAP SECTION ---
    st.write("---")
    st.subheader("🗺️ RAP State Heatmap")

    col_heat_1, col_heat_2 = st.columns([2, 1])
    with col_heat_2:
        is_blue = st.toggle("Show Only Blue RAP Counters", value=False)
        code_col = "blue_rap_state_code" if is_blue else "rap_state_code"
        label_col = "blue_rap_state_label" if is_blue else "rap_state_label"

    # Create pivot table
    heat_df = df.pivot_table(
        index='ute', 
        columns='exp_ratio', 
        values=code_col, 
        aggfunc='first'
    ).sort_index(ascending=False)

    label_df = df.pivot_table(
       index='ute', 
        columns='exp_ratio', 
        values=label_col, 
        aggfunc='first'
    ).sort_index(ascending=False)

    state_labels = {
        0: "All Make RAP",
        1: "WG Shortfall",
        2: "FL Shortfall",
        3: "WG + FL Shortfall",
        4: "IP Shortfall",
        5: "WG + IP Shortfall",
        6: "FL + IP Shortfall",
        7: "WG + FL + IP Shortfall"
    }

    # Define the Discrete Color Mapping
    color_map = {
        0: "#22c55e",  # Green
        1: "#fef08a",  # Yellow 1
        2: "#fde047",  # Yellow 2
        4: "#eab308",  # Yellow 3 (Darker)
        3: "#fdba74",  # Orange 1
        5: "#f97316",  # Orange 2
        6: "#ea580c",  # Orange 3 (Darker)
        7: "#ef4444"   # Red
    }

    # Create the custom discrete colorscale for the heatmap
    # We normalize the keys to a 0-1 scale for Plotly
    max_val = 9
    discrete_colorscale = []
    for val, hex_color in sorted(color_map.items()):
        loc = val / max_val
        discrete_colorscale.append([loc, hex_color])
        # Add a second point just before the next one to create solid blocks instead of gradients
        next_val_list = [v for v in color_map.keys() if v > val]
        if next_val_list:
            next_loc = (min(next_val_list) - 0.01) / max_val
            discrete_colorscale.append([next_loc, hex_color])
        else:
            discrete_colorscale.append([1.0, hex_color])

    fig_heat = go.Figure()

    # 1. Add the Heatmap (Hide the default colorbar)
    fig_heat.add_trace(go.Heatmap(
        z=heat_df.values,
        x=heat_df.columns,
        y=heat_df.index,
        # This sends the 2D array of text labels to the chart
        customdata=label_df.values, 
        colorscale=discrete_colorscale,
        showscale=False,
        zmin=0,
        zmax=7, 
        xgap=1,
        ygap=1,
        # %{customdata} pulls the text string from our label_df
        hovertemplate=(
            "<b>Status: %{customdata}</b><br>" + 
            "Exp Ratio: %{x:.0%}<br>" +
            "UTE: %{y}<br>" +
            "<extra></extra>"
        )
    ))

    for code, color in color_map.items():
        fig_heat.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, symbol='square', color=color),
            showlegend=True,
            name=state_labels.get(code, f"State {code}")
        ))

    fig_heat.update_layout(
        xaxis_title="Experience Ratio",
        yaxis_title="UTE",
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            title="RAP State Legend",
            bordercolor="Gray",
            borderwidth=1,
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

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