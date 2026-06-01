import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from evaluate_manning_agent import run_evaluation 

st.set_page_config(page_title="RL Agent Evaluator", layout="wide")

st.title("Manning RL Agent: 20-Year Policy Evaluation")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    run_mode = st.selectbox("Run Mode", ["pragmatic", "optimistic", "current", "ideal"])
with col2:
    reward_mode = st.selectbox("Reward Mode", ["readiness_first", "quantity_first", "key_staff_first"])
with col3:
    st.write("") # Spacer
    st.write("")
    run_button = st.button("🚀 Run 20-Year Simulation")

if run_button:
    with st.spinner(f"Evaluating {run_mode} / {reward_mode} agent..."):
        try:
            df = run_evaluation(run_mode=run_mode, reward_mode=reward_mode)
            st.session_state['eval_df'] = df
        except Exception as e:
            st.error(f"Error loading model or running simulation: {e}")

if 'eval_df' in st.session_state:
    df = st.session_state['eval_df']
    
    # Create a continuous time axis for clean plotting
    df['Time'] = df['Year'] + (df['Phase'] - 1) / 3
    year_phase_cd = df[["Year", "Phase"]].to_numpy()

    # KPI Headers
    st.markdown("### Final Simulation Metrics")
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    m1.metric("Final Line Pilots", int(df['Total Pilots'].iloc[-1]))
    m2.metric("Final Staff Pilots", int(df['Total Staff Pilots'].iloc[-1]))
    m3.metric("Final Experience Ratio", f"{df['Experience Ratio'].iloc[-1]*100:.0f}%")
    n_sq = df['Number of Squadrons'].iloc[-1]
    m4.metric("Final Avg WG Shortfall", round(df['WG Shortfall'].iloc[-1] / n_sq, 2))
    m5.metric("Final Avg FL Shortfall", round(df['FL Shortfall'].iloc[-1] / n_sq, 2))
    m6.metric("Final Avg IP Shortfall", round(df['IP Shortfall'].iloc[-1] / n_sq, 2))
    m7.metric("Final Retention Rate", f"{df['Retention Rate'].iloc[-1]*100:.0f}%")
    m8.metric("Cumulative Reward", round(df['Reward'].sum(), 1))
    
    st.markdown("---")
    
    # ---------------------------------------------------------
    # PLOT 1: Population & Readiness
    # ---------------------------------------------------------
    st.subheader("System Health: Population vs Shortfalls")
    fig_health = go.Figure()
    health_series = [
        ("Total Pilots", "Total Line Pilots", "blue"),
        ("Total Staff Pilots", "Total Staff Pilots", "green"),
        ("WG Shortfall", "WG Shortfall", "red"),
        ("FL Shortfall", "FL Shortfall", "orange"),
        ("IP Shortfall", "IP Shortfall", "yellow"),
    ]
    for i, (col, name, color) in enumerate(health_series):
        if i == 0:
            hovertemplate = (
                "Year: %{customdata[0]:.0f}<br>"
                "Phase: %{customdata[1]:.0f}<br>"
                "%{fullData.name}: %{y:.2f}<extra></extra>"
            )
        else:
            hovertemplate = "%{fullData.name}: %{y:.2f}<extra></extra>"
        fig_health.add_trace(
            go.Scatter(
                x=df["Time"],
                y=df[col],
                name=name,
                line=dict(color=color, width=3),
                customdata=year_phase_cd,
                hovertemplate=hovertemplate,
            )
        )

    fig_health.update_layout(xaxis_title="Year", yaxis_title="Count", hovermode="x unified")
    st.plotly_chart(fig_health, width='stretch')
    
    # ---------------------------------------------------------
    # PLOT 2: The Agent's Strategy (Levers Pulled)
    # ---------------------------------------------------------
    st.subheader("Agent Strategy: Environmental Controls")
    
    # Define all possible control variables and their styling
    control_metrics = [
        ("Intake Target", "Annual B-Course Intake", "green"),
        ("FLUG Intake", "Phase FLUG Quota", "blue"),
        ("IPUG Intake", "Phase IPUG Quota", "orange"),
        ("Max Manning", "Max Manning Target", "red"),
        ("Avg UTE", "Squadron Average UTE", "purple"),
        ("Retention Rate", "System Retention Rate", "teal")
    ]
    
    # Only add PAA if we are running a mode that modifies it
    if run_mode in ["ideal", "optimistic"]:
        control_metrics.append(("Total PAA", "Total Fleet PAA", "brown"))
        
    # Dynamically generate rows with exactly 3 columns each
    for i in range(0, len(control_metrics), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(control_metrics):
                metric, title, color = control_metrics[i + j]
                fig = px.line(
                    df,
                    x="Time",
                    y=metric,
                    title=title,
                    custom_data=["Year", "Phase"],
                )
                fig.update_traces(
                    line_color=color,
                    hovertemplate=(
                        "Year: %{customdata[0]:.0f}<br>"
                        "Phase: %{customdata[1]:.0f}<br>"
                        f"{title}: %{{y:.2f}}<extra></extra>"
                    ),
                )
                col.plotly_chart(fig, width='stretch')
        
    # ---------------------------------------------------------
    # PLOT 3: Action Heatmap
    # ---------------------------------------------------------
    st.subheader("Raw Action Matrix")
    st.markdown("*Blue = Decrease (-1) | White = Hold (0) | Red = Increase (+1)*")
    
    # Map each action lever to the control value the agent is adjusting
    action_value_col = {
        "Intake": "Intake Target",
        "FLUG": "FLUG Intake",
        "IPUG": "IPUG Intake",
        "Max Manning": "Max Manning",
        "UTE": "Avg UTE",
        "Retention": "Retention Rate",
        "PAA": "Total PAA",
    }

    action_label = {-1: "Decrease", 0: "Hold", 1: "Increase"}

    action_cols = [c for c in df.columns if c.startswith("Action: ")]
    action_labels = [c.replace("Action: ", "") for c in action_cols]
    times = df["Time"].tolist()

    z_matrix = []
    customdata = []
    for col in action_cols:
        label = col.replace("Action: ", "")
        value_col = action_value_col[label]
        z_matrix.append(df[col].tolist())
        customdata.append(
            list(
                zip(
                    df["Year"],
                    df["Phase"],
                    df[value_col],
                    df[col].map(action_label),
                )
            )
        )

    fig_actions = go.Figure(
        data=go.Heatmap(
            x=times,
            y=action_labels,
            z=z_matrix,
            customdata=customdata,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            hovertemplate=(
                "Year: %{customdata[0]:.0f}<br>"
                "Phase: %{customdata[1]:.0f}<br>"
                "Current Value: %{customdata[2]:.2f}<br>"
                "Action: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig_actions.update_layout(
        yaxis_title="Action Space",
        xaxis_title="Year",
        xaxis=dict(tickmode="linear"),
    )
    st.plotly_chart(fig_actions, width='stretch')