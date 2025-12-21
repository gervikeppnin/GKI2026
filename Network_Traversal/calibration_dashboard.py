#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for Hydraulic Model Calibration.

This dashboard visualizes the water distribution network and allows
users to adjust pipe roughness values to calibrate the model against
real sensor measurements.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Import our WNTR model builder
from wntr_model_builder import (
    build_network_from_csv,
    get_pipe_groups,
    update_roughness_by_group,
    run_simulation,
    get_network_geometry,
    get_sensor_names,
    get_measured_pressures,
    get_boundary_flow,
    get_available_strategies,
    load_csv_data
)

# Page configuration
st.set_page_config(
    page_title="Hydraulic Model Calibration Dashboard",
    page_icon="💧",
    layout="wide"
)

# Custom CSS - removed hardcoded colors to support dark mode
st.markdown("""
<style>
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stMetric {
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data_and_model(strategy="decade"):
    """Load CSV data and build the WNTR model (cached)."""
    data_dir = "Shared Materials"
    
    # Load raw data
    data = load_csv_data(data_dir)
    
    # Build network
    wn = build_network_from_csv(data_dir)
    
    # Get pipe groups using selected strategy
    pipe_groups = get_pipe_groups(wn, data['pipes'], strategy=strategy)
    
    # Get sensor names
    sensor_names = get_sensor_names(data_dir)
    
    # Get measured pressures
    measured_df = get_measured_pressures(data_dir)
    
    # Get boundary flow
    boundary_df = get_boundary_flow(data_dir)
    
    return wn, data, pipe_groups, sensor_names, measured_df, boundary_df


def run_sim_with_roughness(roughness_values: dict, pipe_groups: dict, sensor_names: list,
                           reservoir_head_offset: float = 20.0, strategy: str = "decade"):
    """Run simulation with specified roughness values and reservoir head."""
    # Rebuild network with specified reservoir head
    wn = build_network_from_csv("Shared Materials", reservoir_head_offset=reservoir_head_offset)
    
    # Rebuild pipe groups for this instance
    pipes_df = pd.read_csv("Shared Materials/pipes_csv.csv")
    pipe_groups = get_pipe_groups(wn, pipes_df, strategy=strategy)
    
    # Update roughness values
    update_roughness_by_group(wn, roughness_values, pipe_groups)
    
    # Run simulation
    results = run_simulation(wn, sensor_names)
    
    # Get updated geometry
    nodes_df, pipes_df = get_network_geometry(wn)
    
    return results, nodes_df, pipes_df


def create_network_map(nodes_df: pd.DataFrame, pipes_df: pd.DataFrame, 
                       pressures: pd.DataFrame, sensor_names: list) -> go.Figure:
    """Create interactive network map with Plotly."""
    
    fig = go.Figure()
    
    # Determine color scale for roughness
    roughness_min = pipes_df['roughness'].min()
    roughness_max = pipes_df['roughness'].max()
    
    # Add pipes as lines (color by H-W C-factor)
    for _, pipe in pipes_df.iterrows():
        # Color based on H-W C-factor (80 to 140 typical range)
        # Higher C = smoother = blue, Lower C = rougher = red
        hw_c = pipe['roughness']
        normalized = (hw_c - 80) / (140 - 80)  # 0=rough(C=80), 1=smooth(C=140)
        normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
        
        # Blue (smooth/high C) to red (rough/low C) gradient
        r = int((1 - normalized) * 255)
        b = int(normalized * 255)
        color = f'rgb({r}, 100, {b})'
        
        fig.add_trace(go.Scatter(
            x=[pipe['start_x'], pipe['end_x']],
            y=[pipe['start_y'], pipe['end_y']],
            mode='lines',
            line=dict(color=color, width=1),
            hoverinfo='text',
            hovertext=f"Pipe: {pipe['name']}<br>H-W C-factor: {hw_c:.1f}<br>Diameter: {pipe['diameter']:.1f} mm",
            showlegend=False
        ))
    
    # Add junctions as scatter points
    # Get pressures at first timestep if available
    if not pressures.empty:
        first_timestep_pressures = pressures.iloc[0]
        node_pressures = []
        for name in nodes_df['name']:
            if name in first_timestep_pressures.index:
                node_pressures.append(first_timestep_pressures[name])
            else:
                node_pressures.append(0)
        nodes_df = nodes_df.copy()
        nodes_df['pressure'] = node_pressures
    else:
        nodes_df = nodes_df.copy()
        nodes_df['pressure'] = 0
    
    # Regular junctions
    regular_nodes = nodes_df[~nodes_df['name'].str.startswith('Sensor')]
    
    fig.add_trace(go.Scatter(
        x=regular_nodes['x'],
        y=regular_nodes['y'],
        mode='markers',
        marker=dict(
            size=3,
            color=regular_nodes['pressure'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Pressure (bar)", x=1.02)
        ),
        hoverinfo='text',
        hovertext=[f"{row['name']}<br>Elev: {row['elevation']:.1f} m<br>Pressure: {row['pressure']:.2f} bar" 
                   for _, row in regular_nodes.iterrows()],
        name='Junctions'
    ))
    
    # Highlight sensors with larger markers
    sensor_nodes = nodes_df[nodes_df['name'].isin(sensor_names)]
    
    fig.add_trace(go.Scatter(
        x=sensor_nodes['x'],
        y=sensor_nodes['y'],
        mode='markers+text',
        marker=dict(
            size=12,
            color='red',
            symbol='star',
            line=dict(color='white', width=1)
        ),
        text=sensor_nodes['name'],
        textposition='top center',
        textfont=dict(size=10, color='red'),
        hoverinfo='text',
        hovertext=[f"<b>{row['name']}</b><br>Elev: {row['elevation']:.1f} m<br>Pressure: {row['pressure']:.2f} bar" 
                   for _, row in sensor_nodes.iterrows()],
        name='Sensors'
    ))
    
    fig.update_layout(
        title="Network Map (Pipes colored by roughness, nodes by pressure)",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        showlegend=True,
        legend=dict(x=0, y=1),
        hovermode='closest',
        height=600,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_sensor_comparison_chart(measured_df: pd.DataFrame, 
                                    simulated: dict,
                                    selected_sensor: str) -> go.Figure:
    """Create time-series comparison chart for a sensor."""
    
    # Get measured data for this sensor
    sensor_measured = measured_df[measured_df['sensor'] == selected_sensor].copy()
    sensor_measured = sensor_measured.sort_values('hour')
    
    # Get simulated data
    sim_pressures = simulated.get(selected_sensor, np.zeros(25))
    
    fig = go.Figure()
    
    # Measured line
    fig.add_trace(go.Scatter(
        x=sensor_measured['hour'],
        y=sensor_measured['pressure_avg'],
        mode='lines+markers',
        name='Measured Pressure',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Simulated line (hours 0-24)
    hours = list(range(len(sim_pressures)))
    fig.add_trace(go.Scatter(
        x=hours,
        y=sim_pressures,
        mode='lines+markers',
        name='Simulated Pressure',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"Pressure Comparison - {selected_sensor}",
        xaxis_title="Hour",
        yaxis_title="Pressure (bar)",
        legend=dict(x=0.02, y=0.98),
        height=350,
        hovermode='x unified',
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_boundary_flow_chart(boundary_df: pd.DataFrame) -> go.Figure:
    """Create boundary flow monitor chart."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=boundary_df['hour'],
        y=boundary_df['boundary_flow'],
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='green', width=2),
        marker=dict(size=6),
        name='Boundary Flow'
    ))
    
    fig.update_layout(
        title="Boundary Flow (24-hour Pattern)",
        xaxis_title="Hour",
        yaxis_title="Flow (L/s)",
        height=300,
        hovermode='x unified',
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def calculate_mse_per_sensor(measured_df: pd.DataFrame, simulated: dict) -> dict:
    """Calculate MSE for each sensor."""
    mse_values = {}
    
    for sensor in simulated:
        sensor_measured = measured_df[measured_df['sensor'] == sensor].sort_values('hour')
        measured_values = sensor_measured['pressure_avg'].values
        sim_values = simulated[sensor]
        
        # Match lengths (use minimum)
        min_len = min(len(measured_values), len(sim_values))
        if min_len > 0:
            mse = np.mean((measured_values[:min_len] - sim_values[:min_len]) ** 2)
            mse_values[sensor] = mse
        else:
            mse_values[sensor] = 0
    
    return mse_values


def main():
    st.title("💧 Hydraulic Model Calibration Dashboard")
    st.markdown("Adjust pipe roughness values and compare simulation results with real sensor measurements.")
    
    st.sidebar.header("⚙️ Configuration")
    
    # Strategy Selector
    strategies = get_available_strategies()
    # Check session state for strategy
    if 'grouping_strategy' not in st.session_state:
        st.session_state.grouping_strategy = "decade"
        
    selected_strategy_key = st.sidebar.selectbox(
        "Grouping Strategy",
        options=list(strategies.keys()),
        format_func=lambda x: strategies[x],
        index=list(strategies.keys()).index(st.session_state.grouping_strategy)
    )
    
    # Handle strategy change
    if selected_strategy_key != st.session_state.grouping_strategy:
        st.session_state.grouping_strategy = selected_strategy_key
        # Clear roughness values to trigger re-init for new groups
        if 'roughness_values' in st.session_state:
            del st.session_state.roughness_values
        # Clear specific slider keys
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith('slider_')]
        for k in keys_to_delete:
            del st.session_state[k]
        st.rerun()

    # Load data and model with selected strategy
    with st.spinner("Loading network model..."):
        wn, data, pipe_groups, sensor_names, measured_df, boundary_df = load_data_and_model(strategy=st.session_state.grouping_strategy)
    
    # ========== SIDEBAR ==========
    # Fixed reservoir head (competitors only adjust roughness)
    reservoir_head = 20.0  # Fixed: 20m offset above 21.6m elevation = 41.6m total head
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Roughness Controls")
    st.sidebar.markdown("Adjust Hazen-Williams C-factor by Material/Year groups")
    st.sidebar.markdown("*Lower C = rougher pipe = more head loss = lower pressure*")
    
    # Initialize session state for roughness values (H-W C-factors)
    # Start with C=100 (moderate roughness) for calibration challenge
    if 'roughness_values' not in st.session_state:
        st.session_state.roughness_values = {group: 100.0 for group in pipe_groups}
    
    # Create sliders for each group
    roughness_values = {}
    
    # Sort groups for consistent display
    sorted_groups = sorted(pipe_groups.keys())
    
    # Show only groups with significant pipe counts
    significant_groups = [g for g in sorted_groups if len(pipe_groups[g]) >= 10]
    
    st.sidebar.markdown(f"**Groups with 10+ pipes: {len(significant_groups)}**")
    
    # Use expanders for groups
    with st.sidebar.expander("📊 Roughness Sliders", expanded=True):
        for group in significant_groups[:15]:  # Limit to first 15 to avoid overwhelming
            pipe_count = len(pipe_groups[group])
            default_val = st.session_state.roughness_values.get(group, 120.0)
            value = st.slider(
                f"{group} ({pipe_count} pipes)",
                min_value=80.0,
                max_value=140.0,
                value=float(default_val),
                step=5.0,
                key=f"slider_{group}"
            )
            roughness_values[group] = value
    
    # Fill in remaining groups with default value
    for group in sorted_groups:
        if group not in roughness_values:
            roughness_values[group] = st.session_state.roughness_values.get(group, 120.0)
    
    # Store updated values
    st.session_state.roughness_values = roughness_values
    
    # Run simulation button
    run_simulation_btn = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # ========== AUTOMATED TRAINING SECTION ==========
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Automated Training")
    
    training_method = st.sidebar.selectbox(
        "Training Method",
        ["Hill Climbing", "Gradient Estimation", "Random Search"],
        key="training_method"
    )
    
    n_steps = st.sidebar.slider("Training Steps", min_value=5, max_value=50, value=15, key="n_steps")
    
    run_training_btn = st.sidebar.button("🎯 Run Training", use_container_width=True)
    
    # Show success message if we just completed training
    if st.session_state.get('just_trained', False):
        trained_mae = st.session_state.get('trained_mae', 0)
        st.sidebar.success(f"✅ Training complete! Best MAE: {trained_mae:.4f} bar")
        st.sidebar.info("Sliders and visualization updated with trained values.")
        st.session_state.just_trained = False
    
    # Training progress container
    training_log = []
    
    if run_training_btn:
        st.sidebar.markdown("**Training Progress:**")
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        # Initialize C-factors from current slider values
        current_c = {group: roughness_values[group] for group in sorted_groups}
        best_mae = float('inf')
        best_c = current_c.copy()
        
        for step in range(n_steps):
            progress_bar.progress((step + 1) / n_steps)
            
            if training_method == "Hill Climbing":
                # Try a random perturbation
                trial_c = current_c.copy()
                for group in sorted_groups:
                    delta = np.random.uniform(-3.0, 3.0)
                    trial_c[group] = np.clip(trial_c[group] + delta, 80.0, 140.0)
                
                # Evaluate
                results, _, _ = run_sim_with_roughness(trial_c, pipe_groups, sensor_names, reservoir_head, strategy=st.session_state.grouping_strategy)
                mse_per_sensor = calculate_mse_per_sensor(measured_df, results['sensor_pressures'])
                mae = np.mean([np.sqrt(v) for v in mse_per_sensor.values()])  # RMSE as proxy
                
                if mae < best_mae:
                    best_mae = mae
                    best_c = trial_c.copy()
                    current_c = trial_c.copy()
                    training_log.append(f"Step {step+1}: MAE improved to {mae:.4f}")
                    
            elif training_method == "Gradient Estimation":
                # Finite difference gradient
                epsilon = 2.0
                gradients = {}
                
                # Base evaluation
                # Base evaluation
                results, _, _ = run_sim_with_roughness(current_c, pipe_groups, sensor_names, reservoir_head, strategy=st.session_state.grouping_strategy)
                mse_per_sensor = calculate_mse_per_sensor(measured_df, results['sensor_pressures'])
                base_mae = np.mean([np.sqrt(v) for v in mse_per_sensor.values()])
                
                for group in sorted_groups:
                    # Positive perturbation
                    trial_c = current_c.copy()
                    trial_c[group] = np.clip(trial_c[group] + epsilon, 80.0, 140.0)
                    results, _, _ = run_sim_with_roughness(trial_c, pipe_groups, sensor_names, reservoir_head, strategy=st.session_state.grouping_strategy)
                    mse_plus = calculate_mse_per_sensor(measured_df, results['sensor_pressures'])
                    mae_plus = np.mean([np.sqrt(v) for v in mse_plus.values()])
                    
                    gradients[group] = (mae_plus - base_mae) / epsilon
                
                # Update in negative gradient direction
                for group in sorted_groups:
                    current_c[group] = np.clip(
                        current_c[group] - 2.0 * gradients[group], 
                        80.0, 140.0
                    )
                
                # Evaluate new position
                results, _, _ = run_sim_with_roughness(current_c, pipe_groups, sensor_names, reservoir_head)
                mse_per_sensor = calculate_mse_per_sensor(measured_df, results['sensor_pressures'])
                mae = np.mean([np.sqrt(v) for v in mse_per_sensor.values()])
                
                if mae < best_mae:
                    best_mae = mae
                    best_c = current_c.copy()
                    training_log.append(f"Step {step+1}: MAE improved to {mae:.4f}")
                    
            else:  # Random Search
                trial_c = {}
                for group in sorted_groups:
                    trial_c[group] = np.random.uniform(80.0, 140.0)
                
                results, _, _ = run_sim_with_roughness(trial_c, pipe_groups, sensor_names, reservoir_head)
                mse_per_sensor = calculate_mse_per_sensor(measured_df, results['sensor_pressures'])
                mae = np.mean([np.sqrt(v) for v in mse_per_sensor.values()])
                
                if mae < best_mae:
                    best_mae = mae
                    best_c = trial_c.copy()
                    training_log.append(f"Step {step+1}: MAE improved to {mae:.4f}")
            
            status_text.text(f"Step {step+1}/{n_steps} - Best MAE: {best_mae:.4f}")
        
        # Update roughness values (will be used on next rerun)
        st.session_state.roughness_values = best_c
        st.session_state.trained_mae = best_mae
        st.session_state.just_trained = True
        
        # Clear the individual slider keys so they pick up new values on rerun
        for group in best_c.keys():
            slider_key = f"slider_{group}"
            if slider_key in st.session_state:
                del st.session_state[slider_key]
        
        # Rerun to update sliders and visualization
        st.rerun()
    
    # ========== MAIN CONTENT ==========
    
    # Run simulation if button pressed or on first load
    if run_simulation_btn or 'sim_results' not in st.session_state:
        with st.spinner("Running hydraulic simulation..."):
            results, nodes_df, pipes_df = run_sim_with_roughness(
                roughness_values, pipe_groups, sensor_names,
                reservoir_head_offset=reservoir_head
            )
            st.session_state.sim_results = results
            st.session_state.nodes_df = nodes_df
            st.session_state.pipes_df = pipes_df
    
    results = st.session_state.sim_results
    nodes_df = st.session_state.nodes_df
    pipes_df = st.session_state.pipes_df
    
    # Calculate MAE metrics (sqrt of MSE for each sensor, then average)
    mse_per_sensor = calculate_mse_per_sensor(measured_df, results['sensor_pressures'])
    mae_per_sensor = {s: np.sqrt(mse) for s, mse in mse_per_sensor.items()}
    total_mae = np.mean(list(mae_per_sensor.values())) if mae_per_sensor else 0
    
    # ========== METRICS ROW ==========
    st.header("📊 Calibration Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # MAE with color indicator
        if total_mae < 0.5:
            mae_color = "🟢"
        elif total_mae < 2.0:
            mae_color = "🟡"
        else:
            mae_color = "🔴"
        st.metric("Total MAE", f"{mae_color} {total_mae:.2f} bar")
    
    with col2:
        st.metric("Network Nodes", f"{len(wn.junction_name_list):,}")
    
    with col3:
        st.metric("Network Pipes", f"{len(wn.pipe_name_list):,}")
    
    with col4:
        st.metric("Active Sensors", f"{len(sensor_names)}")
    
    # Per-sensor MAE breakdown
    with st.expander("📈 Per-Sensor MAE Breakdown"):
        sensor_cols = st.columns(len(sensor_names))
        for i, sensor in enumerate(sorted(sensor_names)):
            mae = mae_per_sensor.get(sensor, 0)
            with sensor_cols[i]:
                st.metric(sensor, f"{mae:.2f} bar")
    
    # ========== NETWORK MAP ==========
    st.header("🗺️ Network Map")
    
    if not results['all_pressures'].empty:
        fig_map = create_network_map(nodes_df, pipes_df, results['all_pressures'], sensor_names)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Simulation did not produce results. Check model configuration.")
    
    # ========== TIME SERIES COMPARISON ==========
    st.header("📈 Time-Series Comparison")
    
    # Sensor selector
    selected_sensor = st.selectbox("Select Sensor", sorted(sensor_names))
    
    if selected_sensor:
        fig_comparison = create_sensor_comparison_chart(
            measured_df, results['sensor_pressures'], selected_sensor
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Multi-sensor view
    with st.expander("🔍 View All Sensors"):
        cols = st.columns(2)
        for i, sensor in enumerate(sorted(sensor_names)):
            with cols[i % 2]:
                fig = create_sensor_comparison_chart(measured_df, results['sensor_pressures'], sensor)
                st.plotly_chart(fig, use_container_width=True, key=f"sensor_chart_{sensor}")
    
    # ========== BOUNDARY FLOW MONITOR ==========
    st.header("🌊 Boundary Flow Monitor")
    
    fig_boundary = create_boundary_flow_chart(boundary_df)
    st.plotly_chart(fig_boundary, use_container_width=True)
    
    # ========== FOOTER ==========
    st.markdown("---")
    st.markdown(
        "**Hydraulic Model Calibration Dashboard** | "
        "Built with Streamlit, WNTR, and Plotly | "
        "Data: Veitur Water Network"
    )


if __name__ == "__main__":
    main()
