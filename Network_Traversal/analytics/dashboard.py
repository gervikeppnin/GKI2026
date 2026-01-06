#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for Hydraulic Model Calibration.
Simplified to use only model selection (no pipe group controls).
"""

import streamlit as st
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from pathlib import Path

import sys
import os
import plotly.express as px

# Ensure project root is in path
# Point to GKI2026 (parent of Network_Traversal)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import Network_Traversal.core.engine as engine_module
from Network_Traversal.core.model_storage import list_models, load_model, CalibrationModel
import Network_Traversal.analytics.vis_utils as vis_utils
import Network_Traversal.core.data_utils as data_utils
import importlib
importlib.reload(data_utils)
importlib.reload(engine_module)
from Network_Traversal.core.engine import SimulationEngine, get_network_geometry
from Network_Traversal.core.data_utils import get_available_dates

logger = logging.getLogger(__name__)


def main():
    st.set_page_config(page_title="Hydraulic Calibration", page_icon="💧", layout="wide")
    st.markdown("<style>.metric-card {padding:1rem;border-radius:0.5rem;}</style>", unsafe_allow_html=True)

    st.title("💧 Hydraulic Model Calibration Dashboard")
    
    # ==========================================================================
    # Sidebar Configuration
    # ==========================================================================
    st.sidebar.header("⚙️ Configuration")
    
    # --- Data Settings ---
    include_rejected = st.sidebar.checkbox(
        "Include Rejected Data", 
        value=False,
        help="Include data with missing sensors or incomplete hours (from 'rejected' folder)."
    )
    
    # --- Date Selection ---
    available_dates = get_available_dates(include_rejected=include_rejected)
    
    if 'simulation_date' not in st.session_state:
        st.session_state.simulation_date = available_dates[0] if available_dates else None

    # Reset if current date not in new list (unless list is empty)
    if st.session_state.simulation_date not in available_dates and available_dates:
         st.session_state.simulation_date = available_dates[0]

    selected_date = st.sidebar.selectbox(
        "📅 Simulation Date",
        available_dates,
        index=available_dates.index(st.session_state.simulation_date) if st.session_state.simulation_date in available_dates else 0,
        help="Select the date for sensor measurements and boundary conditions"
    )
    
    if selected_date != st.session_state.simulation_date:
        st.session_state.simulation_date = selected_date
        if 'sim_results' in st.session_state:
            del st.session_state.sim_results
        st.rerun()

    # --- Load Engine ---
    with st.spinner(f"Loading Simulation Engine ({st.session_state.simulation_date})..."):
        engine = SimulationEngine(date=st.session_state.simulation_date, include_rejected=include_rejected)
        engine.build_network()
        sensor_names = engine.sensor_names
        measured_df = engine.data.get('sensors', pd.DataFrame())
        boundary_df = engine.data.get('boundary_flow', pd.DataFrame())

    # ==========================================================================
    # Model Selection (Simplified - No Group Controls)
    # ==========================================================================
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Model Selection")
    
    models = list_models()
    
    if not models:
        st.sidebar.warning("No trained models found in `models/` directory.")
        st.sidebar.info("Run training to generate calibration models.")
        selected_model = None
    else:
        # Create display options
        model_options = ["Default", "Baseline (Constant)"] + [
            f"{m.path.stem} (MAE: {m.mae:.4f}) - {m.name}" for m in models
        ]
        
        if 'selected_model_idx' not in st.session_state:
            st.session_state.selected_model_idx = 0
        
        selected_idx = st.sidebar.selectbox(
            "Calibration Model",
            range(len(model_options)),
            format_func=lambda i: model_options[i],
            index=st.session_state.selected_model_idx,
            help="Select a trained calibration model to apply"
        )
        
        if selected_idx != st.session_state.selected_model_idx:
            st.session_state.selected_model_idx = selected_idx
            if 'sim_results' in st.session_state:
                del st.session_state.sim_results
            st.rerun()
        
        if selected_idx == 0:
            selected_model = None
        elif selected_idx == 1:
            # Baseline
            selected_model = "BASELINE"
            baseline_c = st.sidebar.slider(
                "Baseline Roughness (C-Factor)", 
                min_value=60, max_value=150, value=110, step=5,
                help="Uniform roughness for all pipes"
            )
        else:
            selected_model = models[selected_idx - 2]
            
            # Show model info
            with st.sidebar.expander("📊 Model Details", expanded=False):
                st.write(f"**Name:** {selected_model.name}")
                st.write(f"**MAE:** {selected_model.mae:.4f} bar")
                st.write(f"**Pipes:** {selected_model.n_pipes}")
                st.write(f"**Algorithm:** {selected_model.algorithm}")
                if selected_model.training_date:
                    st.write(f"**Training Date:** {selected_model.training_date}")
                st.write(f"**Format:** v{selected_model.version}")

    # ==========================================================================
    # Run Simulation
    # ==========================================================================
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    if run_button or 'sim_results' not in st.session_state:
        with st.spinner("Simulating..."):
            # Apply model if selected
            # Apply model if selected
            if selected_model is not None:
                if selected_model == "BASELINE":
                    # Create uniform roughness map
                    pipe_roughness = {p: float(baseline_c) for p in engine.wn.pipe_name_list}
                    engine.apply_pipe_roughness(pipe_roughness)
                elif selected_model.version >= 2:
                    # New format: direct per-pipe roughness
                    engine.apply_pipe_roughness(selected_model.pipe_roughness)
                else:
                    # Legacy format: need to use groups
                    # For legacy models, pipe_roughness is actually group_roughness
                    st.warning("Using legacy model format. Results may not be optimal.")
                    pipe_groups = engine.get_pipe_groups('decade')
                    engine.update_roughness(selected_model.pipe_roughness, pipe_groups)
            
            # Run simulation
            results = engine.run_simulation()
            st.session_state.sim_results = results
            
            # Cache geometry for plotting
            nodes_df, pipes_df = get_network_geometry(engine.wn)
            st.session_state.nodes_df = nodes_df
            st.session_state.pipes_df = pipes_df
            st.session_state.engine_data = engine.data

            # Try loading Surrogate Brain
            st.session_state.selected_surrogate = None
            if selected_model and selected_model != "BASELINE" and selected_model.path:
                # Try standard pattern: {stem}_surrogate.pkl
                brain_path = selected_model.path.parent / f"{selected_model.path.stem}_surrogate.pkl"
                
                # Check for alternative pattern: model_1.json -> model_surrogate_1.pkl
                if not brain_path.exists():
                     import re
                     # Check if stem ends with _\d+
                     match = re.search(r'(.+)_(\d+)$', selected_model.path.stem)
                     if match:
                         base = match.group(1)
                         idx = match.group(2)
                         alt_path = selected_model.path.parent / f"{base}_surrogate_{idx}.pkl"
                         if alt_path.exists():
                             brain_path = alt_path

                if brain_path.exists():
                     try:
                         import joblib
                         st.session_state.selected_surrogate = joblib.load(brain_path)
                         # Need to know which group corresponds to which index in theta vector
                         # The surrogate was trained on vector of values.
                         # We need the groups mapping order.
                         # Model metadata usually has 'group_names'
                         st.session_state.surrogate_groups = selected_model.metadata.get('group_names', [])
                         # Try to get groups map
                         st.session_state.groups_map = selected_model.metadata.get('groups_map', {})
                     except Exception as e:
                         pass
                         # If model stores pipe_roughness, we need to reconstruct group vector
                         # Or just use the pipe_roughness values for one pipe per group
                         # Actually, for V2 trained models, we can extract from metadata or reconstruct 
                     except Exception as e:
                         pass

    # ==========================================================================
    # Display Results
    # ==========================================================================
    results = st.session_state.sim_results
    
    if results.success:
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Simulation MAE", f"{results.mae:.2f} bar")
        c2.metric("Pipes", len(engine.wn.pipe_name_list))
        c3.metric("Sensors", len(sensor_names))

        
        display_name = "Default"
        if selected_model == "BASELINE":
            display_name = f"Baseline (C={baseline_c})"
        elif selected_model:
            display_name = selected_model.name
        
        c4.metric("Model", display_name)
        
        # Network Map
        st.header("🗺️ Network Map")
        
        show_arrows = st.checkbox("Show Flow Arrows (Performance Cost)", value=False)
        
        flows_dict = {}
        if show_arrows:
            flows_dict = results.all_flows_lps.iloc[0].to_dict() if not results.all_flows_lps.empty else {}
        
        fig_map = vis_utils.create_network_map(
            st.session_state.nodes_df,
            st.session_state.pipes_df,
            results.all_pressures_bar,
            sensor_names,
            flows=flows_dict,
            pumps_df=st.session_state.engine_data['pumps'],
            valves_df=st.session_state.engine_data['valves'],
            reservoirs_df=st.session_state.engine_data['reservoir']
        )
        st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})
        
        # Sensor Charts
        st.header("📈 Sensor Comparison")
        
        cols = st.columns(2)
        for i, sensor_name in enumerate(sensor_names):
            with cols[i % 2]:
                fig_chart = vis_utils.create_sensor_comparison_chart(
                    measured_df, results.sensor_pressures_bar, sensor_name
                )
                st.plotly_chart(fig_chart, use_container_width=True)


        # --- Roughness Analysis ---
        st.header("🔍 Roughness Analysis")
        st.write("Inspect how the calibrated roughness correlates with pipe features.")
        
        if 'pipes' in engine.data:
            pipes_static = engine.data['pipes'].copy()
            
            # Get current roughness state
            # engine.wn is current.
            try:
                # WNTR 1.x: Iterate via pipe_name_list
                current_roughness = {}
                for name in engine.wn.pipe_name_list:
                    link = engine.wn.get_link(name)
                    current_roughness[name] = link.roughness
                
                # Add to dataframe
                pipes_static['current_roughness'] = pipes_static['name'].map(current_roughness)
                
                # Feature Selection
                # Check available columns
                available_cols = [c for c in ['year', 'diameter', 'length'] if c in pipes_static.columns]
                
                if available_cols:
                    feature = st.selectbox("Select Feature to Analyze", available_cols, format_func=lambda x: x.capitalize())
                    
                    import plotly.express as px  # Ensure availability
                    # Create Plot
                    # User requested "Always Box" (candlestick style)
                    fig_rough = px.box(
                        pipes_static, 
                        x=feature, 
                        y='current_roughness',
                        hover_data=['name'],
                        title=f"Roughness Distribution by {feature.capitalize()}"
                    )

                    fig_rough.update_layout(yaxis_title="Roughness (C-Factor)")
                    st.plotly_chart(fig_rough, use_container_width=True)
                    
                else:
                    st.warning("No suitable pipe features (year, diameter, length) found in data.")
            except Exception as e:
                st.error(f"Could not load roughness analysis: {e}")
        else:
             st.warning("Pipe metadata not available.")

        # --- Demand vs Inflow Analysis ---
        st.header("🌊 Demand & Inflow Analysis")
        
        if not boundary_df.empty:
            import plotly.graph_objects as go
            
            real_flow = boundary_df.iloc[:24, 1].values
            total_base = engine.data['junctions']['bas_demand'].sum()
            hours = list(range(24))
            
            fig_flow = go.Figure()
            fig_flow.add_trace(go.Scatter(
                x=hours, y=real_flow,
                mode='lines+markers', name='Real Inflow (Boundary)',
                line=dict(color='blue', width=3)
            ))
            fig_flow.add_trace(go.Scatter(
                x=hours, y=[total_base]*24,
                mode='lines', name='Original Base Demand (Unscaled)',
                line=dict(color='gray', dash='dash')
            ))
            fig_flow.update_layout(
                title="System Water Balance: Inflow vs Demand",
                xaxis_title="Hour",
                yaxis_title="Flow (L/s)",
                legend=dict(orientation="h", y=1.1),
                height=400
            )
            st.plotly_chart(fig_flow, use_container_width=True)
            
            avg_mult = real_flow.mean() / total_base if total_base > 0 else 0
            st.info(f"The simulation scales the Original Base Demand by an average of **{avg_mult:.2f}x** "
                    "to match the Real Inflow. This ensures mass balance.")

        # --- Multi-Date Analysis ---
        st.markdown("---")
        st.header("📅 Multi-Date Reliability Analysis")
        st.markdown("Test your current model across all available dates to ensure robustness.")
        
        if st.button("Analyze Across All Dates"):
            if not available_dates:
                st.warning("No dates found to analyze.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                multi_results = []
                sensor_errors_list = []
                
                for idx, d in enumerate(available_dates):
                    status_text.text(f"Simulating date: {d} ({idx+1}/{len(available_dates)})")
                    
                    # Setup Engine for this date
                    iter_engine = SimulationEngine(date=d, include_rejected=include_rejected)
                    iter_engine.build_network()
                    
                    # Apply model if one is selected (uncalibrated uses default roughness=1.0)
                    if selected_model is not None:
                        if selected_model == "BASELINE":
                             pipe_roughness = {p: float(baseline_c) for p in iter_engine.wn.pipe_name_list}
                             iter_engine.apply_pipe_roughness(pipe_roughness)
                        elif selected_model.version >= 2:
                            iter_engine.apply_pipe_roughness(selected_model.pipe_roughness)
                        else:
                            iter_groups = iter_engine.get_pipe_groups('decade')
                            iter_engine.update_roughness(selected_model.pipe_roughness, iter_groups)
                    
                    # Run
                    r = iter_engine.run_simulation()
                    
                    if r.success:
                        multi_results.append({'date': d, 'mae': r.mae})
                        
                        for s_name, s_avg_p in iter_engine.measured_pressures.items():
                            sim_series = r.sensor_pressures_bar.get(s_name, [])
                            if len(sim_series) > 0:
                                sim_val = float(np.mean(sim_series))
                                err = abs(sim_val - s_avg_p)
                                sensor_errors_list.append({
                                    'date': d,
                                    'sensor': s_name,
                                    'error': err,
                                    'simulated': sim_val,
                                    'measured': s_avg_p
                                })
                    
                    progress_bar.progress((idx + 1) / len(available_dates))
                
                status_text.text("Analysis Complete!")
                progress_bar.empty()
                
                if multi_results:
                    res_df = pd.DataFrame(multi_results)
                    err_df = pd.DataFrame(sensor_errors_list)
                    
                    # Summary Metrics
                    st.subheader("Global Performance Metrics")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mean MAE", f"{res_df['mae'].mean():.4f}")
                    c2.metric("MAE Variance", f"{res_df['mae'].var():.6f}")
                    c3.metric("MAE Std. Dev", f"{res_df['mae'].std():.4f}")
                    c4.metric("Worst MAE", f"{res_df['mae'].max():.4f}")
                    
                    # Distribution Plot
                    import plotly.express as px
                    
                    st.subheader("Model Stability Analysis")
                    tab1, tab2, tab3 = st.tabs(["MAE Distribution", "Error Evolution", "Detailed Sensor Variance"])
                    
                    with tab1:
                        fig_dist = px.box(res_df, y="mae", points="all",
                                          title="MAE Distribution across Dates",
                                          hover_data=['date'])
                        fig_dist.update_layout(yaxis_title="Mean Absolute Error (bar)")
                        st.plotly_chart(fig_dist, use_container_width=True)

                    with tab2:
                        # Convert date string (DDMMYY) to datetime for sorting/plotting
                        plot_df = err_df.copy() if not err_df.empty else pd.DataFrame()
                        
                        if not plot_df.empty:
                            try:
                                plot_df['datetime'] = pd.to_datetime(plot_df['date'], format='%d%m%y')
                                plot_df = plot_df.sort_values('datetime')
                            except Exception:
                                pass
                                
                            fig_evol = px.line(plot_df, x="datetime", y="error", color="sensor", markers=True,
                                              title="Error Evolution over Time (per Sensor)",
                                              labels={"error": "Absolute Error (bar)", "datetime": "Simulation Date", "sensor": "Sensor"})
                            fig_evol.update_layout(yaxis_title="Absolute Error (bar)")
                            st.plotly_chart(fig_evol, use_container_width=True)
                        else:
                            st.warning("No sensor error data available to plot.")
                    
                    with tab3:
                        if not err_df.empty:
                            sensor_stats = err_df.groupby('sensor')['error'].agg(['mean', 'std', 'max']).reset_index()
                            sensor_stats.columns = ['Sensor', 'Mean Error', 'Std Dev', 'Max Error']
                            
                            st.write("Error Statistics per Sensor:")
                            st.dataframe(sensor_stats.style.background_gradient(cmap='Reds'), use_container_width=True)
                            
                            fig_sens = px.box(err_df, x="sensor", y="error",
                                              title="Error Distribution per Sensor (All Dates)",
                                              color="sensor")
                            st.plotly_chart(fig_sens, use_container_width=True)
                else:
                    st.error("No successful simulations to analyze.")



        # --- Agent Brain Visualization ---
        if 'selected_surrogate' in st.session_state and st.session_state.selected_surrogate:
            st.markdown("---")
            st.header("🧠 Agent Brain (Gaussian Process)")
            st.markdown("Visualize what the agent learned about the fitness landscape.")
            
            surrogate = st.session_state.selected_surrogate
            group_names = st.session_state.surrogate_groups
            
            # Reconstruct Theta Star (Optimal Vector)
            # We need values in same order as group_names
            theta_star = []
            if group_names:
                # We need to find the roughness for reach group.
                # The model has pipe_roughness. We can just pick one pipe for each group.
                # But we don't have group mapping handy in dashboard easily...
                # Wait, metadata has 'original_groups' (dict) potentially?
                # saved_model stores 'pipe_groups' inverted?
                # Actually, the 'group_roughness' is not stored in calibration model struct directly if V2 expanded.
                # However, our save_model saves metadata['group_names'].
                # And usually metadata['original_groups'] if migrated, but new ones?
                # Let's rely on reconstructing from pipe_roughness using the fact that all pipes in a group have same val.
                # We need to map group -> val.
                
                # Check if we can get group values
                # We need the group mapping. 'pipe_groups' is not in CalibrationModel by default for V2 load.
                # Ideally, we should have saved 'group_roughness' in metadata or similar.
                # Let's check save_model implementation... 
                # It saves 'group_names' in metadata.
                # And `save_model` takes `group_roughness`.
                # But `CalibrationModel` struct doesn't keep it. 
                # It's likely we can't easily reconstruct without the map.
                # However, we can approximate or use metadata if we updated save_model?
                # I did not update save_model to store group_values in metadata.
                # But I did update run_ig_v2 to save...
                
                # Workaround: The surrogate is only useful if we know the inputs.
                # If we lack the mapping, we can't plot slices accurately by name.
                pass
            
            # If we simply can't map, we can still plot uncertainty bar chart by Index
            if not group_names:
                group_names = [f"Dim {i}" for i in range(surrogate.n_features)]
                
            # Try to get theta vector from the internal GP or just use dummy?
            # Surrogate doesn't store theta_star.
            # But the agent's 'best_theta' was used.
            # We can try to guess values if we can't finding them.
            # OR better: The user just wants to see the brain.
            # We can plot Feature Relevance (Length scales) which is independent of current theta.
            
            st.subheader("Feature Relevance")
            
            # --- Relevance Metrics ---
            relevance = {}
            if surrogate and hasattr(surrogate.gp, 'kernel_'):
                try:
                    k = surrogate.gp.kernel_
                    if hasattr(k, 'k1') and hasattr(k.k1, 'k2'): matern = k.k1.k2
                    elif hasattr(k, 'k2'): matern = k.k2
                    else: matern = k
                    if hasattr(matern, 'length_scale'):
                         vals = 1.0 / (matern.length_scale + 1e-6)
                         for i, name in enumerate(group_names):
                             relevance[name] = vals[i]
                except: pass

            tab_brain1, tab_brain2, tab_brain3 = st.tabs(["🗺️ Brain Map", "📊 Relevance Bars", "🔪 Slices"])
            
            with tab_brain1:
                st.write("Visualizing Feature Relevance (1/LengthScale) on the network.")
                if 'groups_map' in st.session_state and st.session_state.groups_map and relevance:
                     fig_map = vis_utils.plot_brain_map(
                         st.session_state.pipes_df,
                         st.session_state.groups_map,
                         relevance
                     )
                     st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("Map visualization requires 'groups_map' in model metadata (re-train to generate) and valid surrogate.")

            with tab_brain2:
                st.write("Which groups does the agent think are most critical?")
                fig_unc = vis_utils.plot_gp_uncertainty(surrogate, None, group_names)
                st.plotly_chart(fig_unc, use_container_width=True)
            
            with tab_brain3:
                slice_group = st.selectbox("Select Parameter to Slice", group_names)
                
                if slice_group:
                    idx = group_names.index(slice_group)
                    
                    # Calculate actual theta from model's pipe_roughness and groups_map
                    current_theta = np.ones(len(group_names)) * 100.0
                    
                    # If we have groups_map and pipe_roughness, compute group means
                    if 'groups_map' in st.session_state and st.session_state.groups_map:
                        groups_map = st.session_state.groups_map
                        pipe_roughness = selected_model.pipe_roughness
                        
                        for i, gname in enumerate(group_names):
                            if gname in groups_map:
                                pipes_in_group = groups_map[gname]
                                vals = [pipe_roughness.get(p, 100.0) for p in pipes_in_group]
                                if vals:
                                    current_theta[i] = float(np.mean(vals))
                    
                    st.caption(f"Current Best θ: {[f'{v:.1f}' for v in current_theta]}")
                    
                    fig_slice = vis_utils.plot_gp_slice(surrogate, current_theta, idx, slice_group)
                    st.plotly_chart(fig_slice, use_container_width=True)
            
    else:
        st.error(f"Simulation Failed: {results.error_message}")


if __name__ == "__main__":
    main()
