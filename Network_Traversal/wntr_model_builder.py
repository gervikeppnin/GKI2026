#!/usr/bin/env python3
"""
WNTR-based Model Builder for Veitur Water Network.

This module builds a WNTR WaterNetworkModel from CSV files and provides
functions for roughness calibration and simulation.
"""

import pandas as pd
import numpy as np
import wntr
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_csv_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV files into dataframes.
    
    Args:
        data_dir: Path to directory containing CSV files
        
    Returns:
        Dictionary of dataframes keyed by data type
    """
    data_path = Path(data_dir)
    
    data = {
        'junctions': pd.read_csv(data_path / "junctions_csv.csv"),
        'pipes': pd.read_csv(data_path / "pipes_csv.csv"),
        'pumps': pd.read_csv(data_path / "pumps_csv.csv"),
        'pump_curves': pd.read_csv(data_path / "pump_curves_csv.csv"),
        'valves': pd.read_csv(data_path / "valves_csv.csv"),
        'reservoir': pd.read_csv(data_path / "reservoir_csv.csv"),
        'boundary_flow': pd.read_csv(data_path / "boundary_flow.csv"),
        'sensors': pd.read_csv(data_path / "sensor_measurements.csv"),
    }
    
    return data


def build_network_from_csv(data_dir: str = "Shared Materials", 
                           reservoir_head_offset: float = 20.0) -> wntr.network.WaterNetworkModel:
    """Build a WNTR WaterNetworkModel from CSV files.
    
    Args:
        data_dir: Path to directory containing CSV files
        reservoir_head_offset: Head to add above reservoir elevation (meters).
                               Default 20m for realistic pressures ~4-8 bar.
        
    Returns:
        Constructed WaterNetworkModel
    """
    data = load_csv_data(data_dir)
    
    # Create empty network
    wn = wntr.network.WaterNetworkModel()
    
    # Set hydraulic options (using Hazen-Williams for WNTRSimulator compatibility)
    wn.options.hydraulic.headloss = 'H-W'  # Hazen-Williams
    wn.options.time.duration = 24 * 3600  # 24 hours in seconds
    wn.options.time.hydraulic_timestep = 3600  # 1 hour
    wn.options.time.report_timestep = 3600
    wn.options.hydraulic.demand_multiplier = 1.0
    
    # Calculate demand pattern from boundary flow
    # 1. Get total base demand from junctions (L/s)
    total_base_demand_lps = data['junctions']['bas_demand'].sum()
    
    # 2. Get boundary flow profile (L/s)
    boundary_flow = data['boundary_flow']['boundary_flow']
    
    # 3. Calculate multipliers (if total demand > 0)
    if total_base_demand_lps > 0:
        multipliers = boundary_flow / total_base_demand_lps
    else:
        multipliers = [1.0] * 24
        
    # 4. Add pattern to network
    wn.add_pattern('DynamicDemand', multipliers.tolist())
    
    # Add junctions with demand pattern
    for _, row in data['junctions'].iterrows():
        name = str(row['name'])
        elevation = float(row['z']) if pd.notna(row['z']) else 0.0
        base_demand = float(row['bas_demand']) if pd.notna(row['bas_demand']) else 0.0
        
        # Convert L/s to m³/s for WNTR (1 L/s = 0.001 m³/s)
        base_demand_m3s = base_demand / 1000.0
        
        wn.add_junction(
            name,
            base_demand=base_demand_m3s,
            demand_pattern='DynamicDemand',
            elevation=elevation,
            coordinates=(
                float(row['x']) if pd.notna(row['x']) else 0.0,
                float(row['y']) if pd.notna(row['y']) else 0.0
            )
        )
    
    # Add reservoirs
    for _, row in data['reservoir'].iterrows():
        name = str(row['name'])
        elevation = float(row['z']) if pd.notna(row['z']) else 0.0
        # Set head = elevation + offset for supply pressure
        head = elevation + reservoir_head_offset
        
        wn.add_reservoir(
            name,
            base_head=head,
            head_pattern=None,
            coordinates=(
                float(row['x']) if pd.notna(row['x']) else 0.0,
                float(row['y']) if pd.notna(row['y']) else 0.0
            )
        )
    
    # Add pump curves first (needed by pumps)
    curve_data = {}
    for _, row in data['pump_curves'].iterrows():
        curve_id = str(row['curve_id'])
        flow_lps = float(row['flow_lps'])
        head_m = float(row['head_m'])
        
        if curve_id not in curve_data:
            curve_data[curve_id] = []
        # Convert L/s to m³/s
        curve_data[curve_id].append((flow_lps / 1000.0, head_m))
    
    for curve_id, points in curve_data.items():
        wn.add_curve(curve_id, 'HEAD', points)
    
    # Add pipes
    for _, row in data['pipes'].iterrows():
        name = str(row['name'])
        start_node = str(row['start'])
        end_node = str(row['end'])
        length = float(row['length']) if pd.notna(row['length']) else 100.0
        diameter = float(row['diameter']) if pd.notna(row['diameter']) else 100.0
        roughness = float(row['roughness']) if pd.notna(row['roughness']) else 1.0
        minor_loss = float(row['minorLoss']) if pd.notna(row['minorLoss']) else 0.0
        status_str = str(row['status']) if pd.notna(row['status']) else 'OPEN'
        
        # Convert diameter from mm to m
        diameter_m = diameter / 1000.0
        # Convert D-W roughness (mm) to Hazen-Williams C-factor
        # D-W roughness 1.0 mm (new steel) -> HW C ~ 120-130
        # Lower D-W roughness = smoother = higher C-factor
        # Formula approximation: C = 130 - 20 * (dw_roughness - 0.5)
        hw_c_factor = 130 - 20 * (roughness - 0.5)
        hw_c_factor = max(80, min(140, hw_c_factor))  # Clamp to realistic range
        
        # Determine check valve status
        cv = status_str.upper() == 'CV'
        initial_status = 'CLOSED' if status_str.upper() == 'CLOSED' else 'OPEN'
        
        try:
            wn.add_pipe(
                name,
                start_node_name=start_node,
                end_node_name=end_node,
                length=length,
                diameter=diameter_m,
                roughness=hw_c_factor,
                minor_loss=minor_loss,
                initial_status=initial_status,
                check_valve=cv
            )
        except Exception as e:
            print(f"Warning: Could not add pipe {name}: {e}")
    
    # Add pumps
    for _, row in data['pumps'].iterrows():
        name = str(row['name'])
        start_node = str(row['start'])
        end_node = str(row['end'])
        curve = str(row['curve'])
        
        try:
            wn.add_pump(
                name,
                start_node_name=start_node,
                end_node_name=end_node,
                pump_type='HEAD',
                pump_parameter=curve,
                speed=1.0,
                pattern=None
            )
        except Exception as e:
            print(f"Warning: Could not add pump {name}: {e}")
    
    # Add valves (using TCV for initial testing)
    for _, row in data['valves'].iterrows():
        name = str(row['name'])
        start_node = str(row['start'])
        end_node = str(row['end'])
        diameter = float(row['diameter']) if pd.notna(row['diameter']) else 100.0
        minor_loss = float(row['minorLoss']) if pd.notna(row['minorLoss']) else 0.0
        
        # Convert diameter to m
        diameter_m = diameter / 1000.0
        
        try:
            wn.add_valve(
                name,
                start_node_name=start_node,
                end_node_name=end_node,
                diameter=diameter_m,
                valve_type='TCV',  # Throttle control valve, fully open
                minor_loss=minor_loss,
                initial_setting=0  # Fully open
            )
        except Exception as e:
            print(f"Warning: Could not add valve {name}: {e}")
    
    return wn


def get_pipe_groups(wn: wntr.network.WaterNetworkModel, 
                    pipes_df: pd.DataFrame,
                    strategy: str = "decade") -> Dict[str, List[str]]:
    """Get pipe groups using different strategies.
    
    Args:
        wn: The water network model
        pipes_df: DataFrame with pipe data
        strategy: One of:
            - "decade": Group by decade (6 groups)
            - "year": Group by year (41 groups)
            - "diameter": Group by diameter range (5 groups)
            - "decade_diameter": Group by decade + diameter (30 groups)
            - "5year": Group by 5-year period (12 groups)
        
    Returns:
        Dictionary mapping group name to list of pipe names
    """
    pipes_df = pipes_df.copy()
    
    if strategy == "decade":
        # Original: by decade (6 groups)
        pipes_df['decade'] = (pipes_df['year'] // 10) * 10
        pipes_df['group'] = 'steel_' + pipes_df['decade'].fillna(2000).astype(int).astype(str)
        
    elif strategy == "year":
        # By individual year (41 groups)
        pipes_df['group'] = 'year_' + pipes_df['year'].fillna(2000).astype(int).astype(str)
        
    elif strategy == "diameter":
        # By diameter range (5 groups)
        def diameter_group(d):
            if d < 30:
                return 'tiny_<30mm'
            elif d < 50:
                return 'small_30-50mm'
            elif d < 80:
                return 'medium_50-80mm'
            elif d < 150:
                return 'large_80-150mm'
            else:
                return 'xlarge_>150mm'
        pipes_df['group'] = pipes_df['diameter'].apply(diameter_group)
        
    elif strategy == "decade_diameter":
        # Combined: decade + diameter (up to 30 groups)
        pipes_df['decade'] = (pipes_df['year'] // 10) * 10
        def diam_size(d):
            if d < 50:
                return 'small'
            elif d < 100:
                return 'medium'
            else:
                return 'large'
        pipes_df['diam_cat'] = pipes_df['diameter'].apply(diam_size)
        pipes_df['group'] = (
            pipes_df['decade'].fillna(2000).astype(int).astype(str) + '_' +
            pipes_df['diam_cat']
        )
        
    elif strategy == "5year":
        # By 5-year period (12 groups)
        pipes_df['period'] = (pipes_df['year'] // 5) * 5
        pipes_df['group'] = 'period_' + pipes_df['period'].fillna(2000).astype(int).astype(str)
        
    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")
    
    groups = {}
    for group_name in pipes_df['group'].unique():
        group_pipes = pipes_df[pipes_df['group'] == group_name]['name'].tolist()
        # Filter to only include pipes that exist in the network
        existing_pipes = [p for p in group_pipes if p in wn.pipe_name_list]
        if existing_pipes:
            groups[group_name] = existing_pipes
    
    return groups


def get_available_strategies() -> Dict[str, str]:
    """Return available grouping strategies with descriptions."""
    return {
        "decade": "By decade (6 groups: 1970s, 1980s, ... 2020s)",
        "year": "By individual year (41 groups)",
        "diameter": "By diameter range (5 groups: tiny, small, medium, large, xlarge)",
        "decade_diameter": "By decade + diameter size (18 groups)",
        "5year": "By 5-year period (12 groups)",
    }


def update_roughness_by_group(wn: wntr.network.WaterNetworkModel,
                               group_roughness: Dict[str, float],
                               pipe_groups: Dict[str, List[str]]) -> None:
    """Update pipe roughness values by group.
    
    Args:
        wn: The water network model to modify
        group_roughness: Dictionary mapping group name to roughness value (Hazen-Williams C)
        pipe_groups: Dictionary mapping group name to list of pipe names
    """
    for group_name, hw_c_factor in group_roughness.items():
        if group_name not in pipe_groups:
            continue
        
        for pipe_name in pipe_groups[group_name]:
            try:
                pipe = wn.get_link(pipe_name)
                pipe.roughness = hw_c_factor
            except KeyError:
                pass  # Pipe not in network


def run_simulation(wn: wntr.network.WaterNetworkModel,
                   sensor_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Run hydraulic simulation and get pressure results.
    
    Args:
        wn: The water network model
        sensor_names: Optional list of sensor node names to extract pressures for
        
    Returns:
        Dictionary with:
        - 'all_pressures': DataFrame of all node pressures over time
        - 'sensor_pressures': Dict of sensor name -> pressure array (in bar)
        - 'all_flows': DataFrame of all link flows over time
    """
    # Run simulation using WNTR's built-in simulator (cross-platform compatible)
    sim = wntr.sim.WNTRSimulator(wn)
    
    try:
        results = sim.run_sim()
    except Exception as e:
        print(f"Simulation error: {e}")
        # Return empty results
        return {
            'all_pressures': pd.DataFrame(),
            'sensor_pressures': {},
            'all_flows': pd.DataFrame()
        }
    
    # Get pressure results (in meters, convert to bar: 1 bar ≈ 10.2 m)
    pressure_m = results.node['pressure']
    pressure_bar = pressure_m / 10.2
    
    # Extract sensor pressures if specified
    sensor_pressures = {}
    if sensor_names:
        for sensor in sensor_names:
            if sensor in pressure_bar.columns:
                sensor_pressures[sensor] = pressure_bar[sensor].values
            else:
                sensor_pressures[sensor] = np.zeros(len(pressure_bar))
    
    # Get flow results (in m³/s, convert to L/s)
    flow_m3s = results.link['flowrate']
    flow_lps = flow_m3s * 1000
    
    return {
        'all_pressures': pressure_bar,
        'sensor_pressures': sensor_pressures,
        'all_flows': flow_lps
    }


def get_network_geometry(wn: wntr.network.WaterNetworkModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract network geometry for visualization.
    
    Args:
        wn: The water network model
        
    Returns:
        Tuple of (nodes_df, pipes_df) with geometry information
    """
    # Extract node data
    nodes_data = []
    for node_name in wn.node_name_list:
        node = wn.get_node(node_name)
        coord = node.coordinates if hasattr(node, 'coordinates') else (0, 0)
        nodes_data.append({
            'name': node_name,
            'x': coord[0] if coord else 0,
            'y': coord[1] if coord else 0,
            'elevation': node.elevation if hasattr(node, 'elevation') else 0,
            'type': node.node_type
        })
    nodes_df = pd.DataFrame(nodes_data)
    
    # Extract pipe data
    pipes_data = []
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        start_node = wn.get_node(pipe.start_node_name)
        end_node = wn.get_node(pipe.end_node_name)
        start_coord = start_node.coordinates if hasattr(start_node, 'coordinates') else (0, 0)
        end_coord = end_node.coordinates if hasattr(end_node, 'coordinates') else (0, 0)
        
        pipes_data.append({
            'name': pipe_name,
            'start_node': pipe.start_node_name,
            'end_node': pipe.end_node_name,
            'start_x': start_coord[0] if start_coord else 0,
            'start_y': start_coord[1] if start_coord else 0,
            'end_x': end_coord[0] if end_coord else 0,
            'end_y': end_coord[1] if end_coord else 0,
            'length': pipe.length,
            'diameter': pipe.diameter * 1000,  # Convert to mm
            'roughness': pipe.roughness,  # H-W C-factor (dimensionless)
        })
    pipes_df = pd.DataFrame(pipes_data)
    
    return nodes_df, pipes_df


def calculate_mse(measured: Dict[str, float], simulated: Dict[str, float]) -> float:
    """Calculate Mean Squared Error between measured and simulated pressures.
    
    Args:
        measured: Dictionary of sensor name -> measured pressure (bar)
        simulated: Dictionary of sensor name -> simulated pressure (bar)
        
    Returns:
        MSE value
    """
    errors = []
    for sensor in measured:
        if sensor in simulated:
            error = (measured[sensor] - simulated[sensor]) ** 2
            errors.append(error)
    
    return np.mean(errors) if errors else 0.0


def get_sensor_names(data_dir: str = "Shared Materials") -> List[str]:
    """Get list of sensor names from the sensor measurements file.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of unique sensor names
    """
    sensors_df = pd.read_csv(Path(data_dir) / "sensor_measurements.csv")
    return sorted(sensors_df['sensor'].unique().tolist())


def get_measured_pressures(data_dir: str = "Shared Materials") -> pd.DataFrame:
    """Get measured pressures from sensor measurements file.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DataFrame with columns: hour, sensor, pressure_avg
    """
    return pd.read_csv(Path(data_dir) / "sensor_measurements.csv")


def get_boundary_flow(data_dir: str = "Shared Materials") -> pd.DataFrame:
    """Get boundary flow data.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DataFrame with hour and flow columns
    """
    df = pd.read_csv(Path(data_dir) / "boundary_flow.csv")
    df = df.rename(columns={df.columns[0]: 'hour'})
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Testing WNTR Model Builder")
    print("=" * 60)
    
    # Build network
    print("\nBuilding network from CSV files...")
    wn = build_network_from_csv("Shared Materials")
    
    print(f"  Junctions: {len(wn.junction_name_list)}")
    print(f"  Reservoirs: {len(wn.reservoir_name_list)}")
    print(f"  Pipes: {len(wn.pipe_name_list)}")
    print(f"  Pumps: {len(wn.pump_name_list)}")
    print(f"  Valves: {len(wn.valve_name_list)}")
    
    # Get pipe groups
    print("\nLoading pipe data for groups...")
    pipes_df = pd.read_csv("Shared Materials/pipes_csv.csv")
    groups = get_pipe_groups(wn, pipes_df)
    print(f"  Pipe groups: {len(groups)}")
    for group_name, pipes in sorted(groups.items()):
        print(f"    {group_name}: {len(pipes)} pipes")
    
    # Get sensor names
    sensor_names = get_sensor_names()
    print(f"\nSensors: {sensor_names}")
    
    # Run simulation
    print("\nRunning simulation...")
    results = run_simulation(wn, sensor_names)
    
    if not results['all_pressures'].empty:
        print("  ✓ Simulation completed successfully!")
        print(f"  Timesteps: {len(results['all_pressures'])}")
        
        # Show sensor pressures at first timestep
        print("\nSensor pressures at t=0 (bar):")
        for sensor, pressures in results['sensor_pressures'].items():
            if len(pressures) > 0:
                print(f"  {sensor}: {pressures[0]:.2f} bar")
    else:
        print("  ✗ Simulation failed")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
