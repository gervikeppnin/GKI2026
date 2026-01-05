"""
Visualization utilities for Network Traversal Dashboard.
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict

def create_network_map(nodes_df: pd.DataFrame, pipes_df: pd.DataFrame, 
                       pressures: pd.DataFrame, sensor_names: List[str],
                       flows: Dict[str, float] = None,
                       pumps_df: pd.DataFrame = None,
                       valves_df: pd.DataFrame = None,
                       reservoirs_df: pd.DataFrame = None) -> go.Figure:
    """Create interactive network map with Plotly (Optimized similar to visualize_network.py)."""
    fig = go.Figure()
    
    # --- Build Coordinate Lookup ---
    # We need this for Pumps/Valves which define connections by Node Name
    coords = {}
    if not nodes_df.empty:
        for _, row in nodes_df.iterrows():
            coords[str(row['name'])] = (row['x'], row['y'])
    if reservoirs_df is not None and not reservoirs_df.empty:
        for _, row in reservoirs_df.iterrows():
            coords[str(row['name'])] = (row['x'], row['y'])

    # --- 1. Pipes (Optimized Single Trace) ---
    pipe_x = []
    pipe_y = []
    arrow_x = []
    arrow_y = []
    arrow_angles = []
    
    # Iterate to build coordinate lists
    for _, pipe in pipes_df.iterrows():
        # Pipe Lines
        pipe_x.extend([pipe['start_x'], pipe['end_x'], None])
        pipe_y.extend([pipe['start_y'], pipe['end_y'], None])
        
        # Flow Arrows
        if flows:
            flow = flows.get(str(pipe['name']), 0.0)
            if abs(flow) > 0.001:
                mx = (pipe['start_x'] + pipe['end_x']) / 2
                my = (pipe['start_y'] + pipe['end_y']) / 2
                dx = pipe['end_x'] - pipe['start_x']
                dy = pipe['end_y'] - pipe['start_y']
                
                if flow < 0:
                    dx, dy = -dx, -dy
                
                angle = np.degrees(np.arctan2(dy, dx))
                arrow_x.append(mx)
                arrow_y.append(my)
                arrow_angles.append(angle)

    # Trace: Pipes (Lines) - Mid-Grey to work on Light & Dark backgrounds
    fig.add_trace(go.Scattergl(
        x=pipe_x, y=pipe_y,
        mode='lines',
        line=dict(color='#999999', width=1),
        hoverinfo='skip',
        name='Pipes'
    ))
    
    # Trace: Flow Arrows - Dark Grey
    if arrow_x:
        fig.add_trace(go.Scattergl(
            x=arrow_x, y=arrow_y,
            mode='markers',
            marker=dict(
                symbol='triangle-right',
                size=8,
                color='#444444',
                angle=arrow_angles
            ),
            hoverinfo='skip',
            name='Flow Direction'
        ))

    # --- 2. Reservoirs ---
    if reservoirs_df is not None and not reservoirs_df.empty:
        fig.add_trace(go.Scatter(
            x=reservoirs_df['x'],
            y=reservoirs_df['y'],
            mode='markers+text',
            marker=dict(size=15, color='green', symbol='square', line=dict(width=2, color='darkgreen')),
            text=reservoirs_df['name'],
            textposition="top center",
            name='Reservoirs'
        ))

    # --- 3. Pumps ---
    if pumps_df is not None and not pumps_df.empty:
        px, py = [], []
        for _, row in pumps_df.iterrows():
            s, e = str(row['start']), str(row['end'])
            if s in coords and e in coords:
                sx, sy = coords[s]
                ex, ey = coords[e]
                px.append((sx + ex) / 2)
                py.append((sy + ey) / 2)
        
        if px:
            fig.add_trace(go.Scatter(
                x=px, y=py,
                mode='markers',
                marker=dict(size=12, color='black', symbol='diamond', line=dict(width=1, color='white')),
                name='Pumps'
            ))

    # --- 4. Valves ---
    if valves_df is not None and not valves_df.empty:
        vx, vy = [], []
        for _, row in valves_df.iterrows():
            s, e = str(row['start']), str(row['end'])
            if s in coords and e in coords:
                sx, sy = coords[s]
                ex, ey = coords[e]
                vx.append((sx + ex) / 2)
                vy.append((sy + ey) / 2)
        
        if vx:
            fig.add_trace(go.Scatter(
                x=vx, y=vy,
                mode='markers',
                marker=dict(size=10, color='orange', symbol='triangle-up', line=dict(width=1, color='darkorange')),
                name='Valves'
            ))

    # --- 5. Junctions ---
    if not pressures.empty:
        current_p = pressures.iloc[0]
        nodes_df['pressure'] = nodes_df['name'].map(current_p).fillna(0)
    else:
        nodes_df['pressure'] = 0.0

    regular_nodes = nodes_df[~nodes_df['name'].isin(sensor_names)]
    node_colors = np.where(regular_nodes['pressure'] < 0, 'red', 'blue')
    
    fig.add_trace(go.Scattergl(
        x=regular_nodes['x'],
        y=regular_nodes['y'],
        mode='markers',
        marker=dict(
            size=6,
            color=node_colors,
            opacity=0.7
        ),
        text=[f"{n}<br>P: {p:.2f} bar" for n, p in zip(regular_nodes['name'], regular_nodes['pressure'])],
        hoverinfo='text',
        name='Junctions'
    ))

    # --- 6. Sensors ---
    sensors = nodes_df[nodes_df['name'].isin(sensor_names)]
    fig.add_trace(go.Scatter(
        x=sensors['x'],
        y=sensors['y'],
        mode='markers+text',
        marker=dict(size=14, color='gold', symbol='star', line=dict(width=1, color='black')),
        text=sensors['name'],
        textposition="top center",
        name='Sensors'
    ))
    
    # Theme-neutral layout
    fig.update_layout(
        title="Network Map",
        height=700,
        showlegend=True,
        # Transparent background to support Light/Dark Streamlit themes
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        dragmode='pan',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

def create_sensor_comparison_chart(measured_df: pd.DataFrame, 
                                   simulated_pressures: Dict[str, np.ndarray],
                                   selected_sensor: str) -> go.Figure:
    """Time-series comparison."""
    # Filter measured
    measure_series = measured_df[measured_df['sensor'] == selected_sensor].sort_values('hour')
    
    sim_series = simulated_pressures.get(selected_sensor, [])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=measure_series['hour'], y=measure_series['pressure_avg'],
        name='Measured', mode='lines+markers', line=dict(color='#00CC96')
    ))
    
    if len(sim_series) > 0:
        fig.add_trace(go.Scatter(
            x=list(range(len(sim_series))), y=sim_series,
            name='Simulated', mode='lines+markers', 
            line=dict(color='#EF553B', dash='dash')
        ))
        
    fig.update_layout(
        title=f"Sensor: {selected_sensor}",
        xaxis_title="Hour", yaxis_title="Pressure (bar)",
        template="plotly_dark", height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_gp_uncertainty(surrogate, theta_star, group_names: List[str]) -> go.Figure:
    """
    Plot the internal uncertainty of the GP at the best point.
    """
    fig = go.Figure()
    
    try:
        # Complex kernel: Constant * Matern + White
        k = surrogate.gp.kernel_
        # k.k1 is likely Product(Constant, Matern)
        # Checking kernel structure flexibility
        if hasattr(k, 'k1') and hasattr(k.k1, 'k2'):
             matern = k.k1.k2
        elif hasattr(k, 'k2'): # Might be direct Sum
             matern = k.k2
        else:
             # Fallback or direct Matern
             matern = k
             
        if hasattr(matern, 'length_scale'):
            length_scales = matern.length_scale
            # Relevance = 1/LengthScale
            relevance = 1.0 / (length_scales + 1e-6)
            
            fig.add_trace(go.Bar(
                x=group_names,
                y=relevance,
                name="Feature Relevance",
                marker_color='teal'
            ))
            
            fig.update_layout(
                title="Brain Activation: Parameter Relevance (1/LengthScale)",
                yaxis_title="Relevance (ARD)",
                xaxis_title="Pipe Group",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        else:
            fig.add_annotation(text="Kernel has no length_scale", showarrow=False)
            
    except Exception as e:
        fig.add_annotation(text=f"Could not extract kernel params: {e}", showarrow=False)

    return fig

def plot_gp_slice(surrogate, theta_star: np.ndarray, 
                 target_idx: int, group_name: str, 
                 bounds: tuple = (60, 150)) -> go.Figure:
    """
    Plot 1D Slice of the GP fitness landscape for a specific parameter (group).
    """
    x_grid = np.linspace(bounds[0], bounds[1], 100)
    
    # Construct input matrix
    X_probe = np.tile(theta_star, (100, 1))
    X_probe[:, target_idx] = x_grid
    
    # Predict
    mean, std = surrogate.gp.predict(X_probe, return_std=True)
    
    fig = go.Figure()
    
    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_grid, x_grid[::-1]]),
        y=np.concatenate([mean - 1.96*std, (mean + 1.96*std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='95% Confidence'
    ))
    
    # Mean
    fig.add_trace(go.Scatter(
        x=x_grid, y=mean,
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Predicted Score'
    ))
    
    # Optimal Point
    if target_idx < len(theta_star):
        opt_val = theta_star[target_idx]
        opt_score, _ = surrogate.gp.predict(theta_star.reshape(1,-1), return_std=True)
        
        fig.add_trace(go.Scatter(
            x=[opt_val], y=[opt_score[0]],
            mode='markers',
            marker=dict(color='gold', size=12, symbol='star'),
            name='Current Best'
        ))
    
    fig.update_layout(
        title=f"Brain Slice: {group_name}",
        xaxis_title="Roughness (C-Factor)",
        yaxis_title="Predicted Fitness",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_brain_map(pipes_df: pd.DataFrame, 
                   groups_map: Dict[str, List[str]], 
                   metric_values: Dict[str, float],
                   metric_name: str = "Relevance",
                   nodes_df: pd.DataFrame = None) -> go.Figure:
    """
    Plot the network map with pipes colored by a specific brain metric (e.g. Relevance).
    """
    fig = go.Figure()

    # 1. Map metric to pipes
    # Create a mapping: pipe_name -> metric_value
    pipe_metric = {}
    for g_name, pipes in groups_map.items():
        val = metric_values.get(g_name, 0.0)
        for p in pipes:
            pipe_metric[p] = val

    # 2. Prepare Coordinate Lists with Color
    # We can't use single Line trace easily for variable color unless we segment.
    # Plotly Scattergl lines can handle color array if we use 'marker' mode? No.
    # We use segments. Or safer: Use Scattergl with 'lines' and 'line.color' array?
    # No, line.color array is not fully supported for gradient lines in all versions.
    # Better approach: Segments (Start->End) with NaN separators and color list?
    # Or just use the pipe_groups to create one trace PER GROUP (if groups < 1000).
    # Since we have ~10-50 groups usually, One Trace Per Group is efficient and allows legend toggling!
    
    # 2b. One Trace per Group
    # Normalize metric for colorscale if needed, or let Plotly handle it.
    # Actually, we want a continuous colorscale. 
    # To do that efficiently with 2000 pipes, we might need a dedicated approach.
    
    # Let's try: One single Scattergl trace using line segments separated by None, 
    # but color? 
    # Plotly Scattergl 'lines' does NOT support per-segment color easily.
    # Current best practice for large networks with variable edge color:
    # Use 'Scattergl' with mode='lines' for geometry, but color is tricky.
    # Alternative: Use 'Scatter' (not gl) if size < 5000? Maybe slow.
    
    # Compromise: One Trace Per Group.
    # Limit: If we have thousands of groups (e.g. per-pipe), this fails.
    # But usually < 100 groups.
    
    # Check number of groups
    if len(groups_map) > 200:
        # Fallback for many groups: Binning or similar?
        pass

    # Get Value Range for Color Scale
    vals = list(metric_values.values())
    if not vals: vals = [0]
    min_v, max_v = min(vals), max(vals)
    
    import plotly.colors as pc
    # Use Viridis or Plasma
    colorscale = pc.sequential.Viridis
    
    def get_color(val):
        # Normalize to 0-1
        if max_v - min_v < 1e-9: n = 0.5
        else: n = (val - min_v) / (max_v - min_v)
        # Sample color from scale (hacky w/o proper interpolation func, just discrete)
        idx = int(n * (len(colorscale) - 1))
        return colorscale[idx]

    # Pre-calculate coordinates per group
    # pipe_df id lookup
    pipe_lookup = pipes_df.set_index('name')
    
    for g_name, pipes in groups_map.items():
        if g_name not in metric_values: continue
        val = metric_values[g_name]
        c = get_color(val)
        
        # Collect coords
        gx, gy = [], []
        
        # Filter pipes_df by this group (optimized)
        # Using index is faster?
        # Or just filtering:
        # group_pipes = pipes_df[pipes_df['name'].isin(pipes)]
        # This might be slow inside loop.
        
        # Iterate over pipes is safer if we built a dict
        
        # Lets build dict of coords first?
        # Already have pipe_lookup
        
        for p in pipes:
             if p in pipe_lookup.index:
                 row = pipe_lookup.loc[p]
                 gx.extend([row['start_x'], row['end_x'], None])
                 gy.extend([row['start_y'], row['end_y'], None])
        
        if gx:
            fig.add_trace(go.Scattergl(
                x=gx, y=gy,
                mode='lines',
                line=dict(color=c, width=3),
                name=f"{g_name} ({val:.3f})",
                hoverinfo='name'
            ))

    # Add ColorBar (Dummy Trace)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            cmin=min_v, cmax=max_v,
            showscale=True,
            colorbar=dict(title=metric_name)
        ),
        showlegend=False
    ))

    fig.update_layout(
        title=f"Brain Map: {metric_name}",
        height=700,
        showlegend=False, # Too many groups likely
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        dragmode='pan',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig
