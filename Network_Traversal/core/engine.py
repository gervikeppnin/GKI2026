#!/usr/bin/env python3
"""
Core Simulation Engine for Veitur Water Network.

This module builds a WNTR WaterNetworkModel from CSV files and provides
functions for roughness calibration and simulation (using EPANET 2.2).
Manages network state, demand patterns, and hydraulic simulation.
"""

import wntr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from .config import (
    DEMAND_MULTIPLIER,
    HYDRAULIC_TIMESTEP,
    DATA_DIR,
    BAR_TO_METERS,
    LPS_TO_M3S,
    MM_TO_M,
    ENABLE_PRESSURE_VALVES,
    PIPE_ROUGHNESS_BY_MATERIAL,
    REFERENCE_YEAR,
    ROUGHNESS_MIN,
    ROUGHNESS_MAX
)
from .data_utils import load_csv_data, get_sensor_names

# Configure logging
# logging.basicConfig(level=logging.INFO) # Removing side-effect
logger = logging.getLogger(__name__)

@dataclass
class SimulationResult:
    """Structured result from a simulation run."""
    all_pressures_bar: pd.DataFrame
    sensor_pressures_bar: Dict[str, np.ndarray]
    all_flows_lps: pd.DataFrame
    success: bool = True
    error_message: Optional[str] = None
    mae: float = 0.0  # Mean Absolute Error vs Measurements
    negative_pressure_count: int = 0  # Number of nodes with negative pressure at any time step

class SimulationEngine:
    """
    Manages the WNTR (Water Network Tool used for Resilience) model creation, simulation, and evaluation.
    Previously NetworkManager.
    """
    def __init__(self, data_dir: Optional[Path] = None, date: Optional[str] = None, include_rejected: bool = False):
        self.data_dir = data_dir or DATA_DIR
        self.date = date
        self.include_rejected = include_rejected
        self.data = load_csv_data(self.data_dir, self.date, include_rejected=self.include_rejected)
        self.wn: Optional[wntr.network.WaterNetworkModel] = None
        
        # Cache sensor data for evaluation
        self.sensor_names = self._get_sensor_names_from_data()
        self.measured_pressures = self._load_measured_pressures()

    def set_date(self, date: str) -> None:
        """Switch simulation context to a specific date."""
        if self.date != date:
            self.date = date
            self.data = load_csv_data(self.data_dir, self.date, include_rejected=self.include_rejected)
            # Re-cache sensor names and pressures for the new date
            self.sensor_names = self._get_sensor_names_from_data()
            self.measured_pressures = self._load_measured_pressures()
            # Clear WN model so it rebuilds with new patterns next run
            self.wn = None
            logger.info(f"Engine switched to date: {date}")

    def _get_sensor_names_from_data(self) -> List[str]:
        """Extract sensor names from currently loaded data."""
        if 'sensors' in self.data and not self.data['sensors'].empty:
            return sorted(self.data['sensors']['sensor'].unique().tolist())
        return get_sensor_names(self.data_dir)

    def _load_measured_pressures(self) -> Dict[str, float]:
        """Load average measured pressures for sensors."""
        if 'sensors' not in self.data or self.data['sensors'].empty:
            return {}
        
        df = self.data['sensors']
        avg_pressures = {}
        for sensor in self.sensor_names:
            val = df[df['sensor'] == sensor]['pressure_avg'].mean()
            avg_pressures[sensor] = float(val)
        return avg_pressures

    def build_network(self) -> wntr.network.WaterNetworkModel:
        """Builds and returns the WNTR network model from loaded data."""
        self.wn = wntr.network.WaterNetworkModel()
        
        # Hydraulic Options
        self.wn.options.hydraulic.headloss = 'H-W'  # Hazen-Williams
        self.wn.options.time.duration = 24 * 3600
        self.wn.options.time.hydraulic_timestep = HYDRAULIC_TIMESTEP
        self.wn.options.time.report_timestep = HYDRAULIC_TIMESTEP
        self.wn.options.hydraulic.demand_multiplier = DEMAND_MULTIPLIER

        self._add_pattern()
        self._add_junctions()
        self._add_reservoirs()
        self._add_curves()
        self._add_pipes()
        self._add_pumps()
        self._add_valves()
        
        return self.wn

    def _add_pattern(self):
        """Adds demand pattern based on boundary flow data.
        
        Calculates multipliers to scale Total Base Demand to match Total Inflow (Boundary Flow).
        Multiplier(t) = BoundaryFlow(t) / TotalBaseDemand
        """
        # Default fallback
        multipliers = [1.0] * 24
        
        try:
            # 1. Calculate Total Base Demand
            if 'junctions' in self.data and not self.data['junctions'].empty:
                # Sum bas_demand (L/s)
                total_base_demand = self.data['junctions']['bas_demand'].sum()
            else:
                total_base_demand = 0.0
                
            # 2. Get Boundary Flow
            if 'boundary_flow' in self.data and not self.data['boundary_flow'].empty:
                bf = self.data['boundary_flow']
                # Ensure we have data for 24 hours
                if len(bf) >= 24:
                    # Extract flow column (assuming 2nd column is flow as seen in preview)
                    # boundary_flow.csv structure: index, flow
                    flows = bf.iloc[:24, 1].values
                    
                    if total_base_demand > 0:
                        multipliers = (flows / total_base_demand).tolist()
                        logger.info(f"Generated Demand Pattern: Mean Multiplier={np.mean(multipliers):.2f}")
                    else:
                        logger.warning("Total base demand is 0, cannot scale pattern. Using 1.0.")
                        
        except Exception as e:
            logger.error(f"Error creating demand pattern: {e}. Using default 1.0.")

        self.wn.add_pattern('DynamicDemand', multipliers)

    def _add_junctions(self):
        """Adds junctions to the network."""
        for row in self.data['junctions'].itertuples():
            # Convert L/s to m³/s for WNTR
            base_demand_m3s = (row.bas_demand * LPS_TO_M3S) if pd.notna(row.bas_demand) else 0.0
            
            self.wn.add_junction(
                str(row.name),
                base_demand=base_demand_m3s,
                demand_pattern='DynamicDemand',
                elevation=float(row.z) if pd.notna(row.z) else 0.0,
                coordinates=(
                    float(row.x) if pd.notna(row.x) else 0.0,
                    float(row.y) if pd.notna(row.y) else 0.0
                )
            )

    def _add_reservoirs(self):
        """Adds reservoirs to the network.
        
        The 'z' column in reservoir_csv.csv is the hydraulic head (m),
        not ground elevation. This is per the data README specification.
        """
        for row in self.data['reservoir'].itertuples():
            # z is the hydraulic head (m) as specified in the data README
            base_head = float(row.z) if pd.notna(row.z) else 0.0
            logger.info(f"Reservoir {row.name}: head={base_head:.1f}m")
            
            self.wn.add_reservoir(
                str(row.name),
                base_head=base_head,
                coordinates=(
                    float(row.x) if pd.notna(row.x) else 0.0,
                    float(row.y) if pd.notna(row.y) else 0.0
                )
            )

    def _add_curves(self):
        """Adds pump curves."""
        curve_data = {}
        for row in self.data['pump_curves'].itertuples():
            cid = str(row.curve_id)
            if cid not in curve_data:
                curve_data[cid] = []
            curve_data[cid].append((row.flow_lps * LPS_TO_M3S, row.head_m))
        
        for cid, points in curve_data.items():
            self.wn.add_curve(cid, 'HEAD', points)

    def _add_pipes(self):
        """Adds pipes with material and age-based roughness.
        
        Hazen-Williams C is calculated based on:
        - Pipe material (from 'tags' column)
        - Pipe age (from 'year' column)
        Using standard engineering degradation models.
        """
        for row in self.data['pipes'].itertuples():
            name = str(row.name)
            
            # Calculate Hazen-Williams C based on material and age
            material = str(row.tags).lower() if pd.notna(row.tags) else 'default'
            year = int(row.year) if pd.notna(row.year) else 2000
            hw_c = self._calculate_hw_roughness(material, year)

            status = 'CLOSED' if str(row.status).upper() == 'CLOSED' else 'OPEN'
            cv = str(row.status).upper() == 'CV'

            # Use actual diameter from CSV (no artificial overrides)
            row_diameter = float(row.diameter) if pd.notna(row.diameter) else 100.0

            try:
                self.wn.add_pipe(
                    name,
                    start_node_name=str(row.start),
                    end_node_name=str(row.end),
                    length=float(row.length) if pd.notna(row.length) else 100.0,
                    diameter=(row_diameter * MM_TO_M),
                    roughness=hw_c,
                    minor_loss=float(row.minorLoss) if pd.notna(row.minorLoss) else 0.0,
                    initial_status=status,
                    check_valve=cv
                )
            except Exception as e:
                logger.warning(f"Could not add pipe {name}: {e}")
    
    def _calculate_hw_roughness(self, material: str, install_year: int) -> float:
        """Calculate Hazen-Williams C-factor based on material and age.
        
        Args:
            material: Pipe material type (e.g., 'steel', 'pvc')
            install_year: Year the pipe was installed
            
        Returns:
            Hazen-Williams C-factor (typically 80-150)
        """
        # Get material properties or use defaults
        mat_props = PIPE_ROUGHNESS_BY_MATERIAL.get(
            material, 
            PIPE_ROUGHNESS_BY_MATERIAL['default']
        )
        
        # Calculate pipe age in decades
        age_years = REFERENCE_YEAR - install_year
        age_decades = max(0, age_years) / 10.0
        
        # Apply age degradation
        base_c = mat_props['base_c']
        decay = mat_props['age_decay_per_decade'] * age_decades
        min_c = mat_props['min_c']
        
        hw_c = max(min_c, base_c - decay)
        
        # Clamp to valid range
        return max(ROUGHNESS_MIN, min(ROUGHNESS_MAX, hw_c))

    def _add_pumps(self):
        """Adds pumps."""
        for row in self.data['pumps'].itertuples():
            name = str(row.name)
            try:
                self.wn.add_pump(
                    name,
                    start_node_name=str(row.start),
                    end_node_name=str(row.end),
                    pump_type='HEAD',
                    pump_parameter=str(row.curve),
                    speed=1.0
                )
            except Exception as e:
                logger.warning(f"Could not add pump {name}: {e}")

    def _add_valves(self):
        """Adds valves with proper pressure control.
        
        Valve settings are in bar in the CSV and need conversion to meters.
        EPANET supports all valve types: PRV, PSV, PBV, FCV, TCV, GPV.
        """
        for row in self.data['valves'].itertuples():
            name = str(row.name)
            v_type = str(row.type).upper() if pd.notna(row.type) else 'TCV'
            setting = float(row.setting) if pd.notna(row.setting) else 0.0
            
            # Get start/end nodes
            start_node = str(row.start)
            end_node = str(row.end)

            if v_type in ['PRV', 'PSV', 'PBV']:
                if ENABLE_PRESSURE_VALVES:
                    # Convert pressure setting from bar to meters head
                    setting = setting * BAR_TO_METERS
                    logger.info(f"Valve {name}: {v_type} with setting {setting:.1f}m")
                else:
                    # Fallback: convert to open TCV (disables pressure control)
                    logger.info(f"Converting {v_type} valve {name} to TCV (valves disabled)")
                    v_type, setting = 'TCV', 0.0
            elif v_type == 'FCV':
                setting *= LPS_TO_M3S  # L/s to m3/s

            try:
                self.wn.add_valve(
                    name,
                    start_node_name=start_node,
                    end_node_name=end_node,
                    diameter=(float(row.diameter) * MM_TO_M) if pd.notna(row.diameter) else 0.1,
                    valve_type=v_type,
                    minor_loss=float(row.minorLoss) if pd.notna(row.minorLoss) else 0.0,
                    initial_setting=setting
                )
            except Exception as e:
                logger.warning(f"Could not add valve {name}: {e}")

    def run_simulation(self) -> SimulationResult:
        """Run hydraulic simulation using EPANET.
        
        Uses EpanetSimulator which calls the EPANET2.2 library for full
        hydraulic solver support including all valve types.
        """
        import uuid
        import os
        
        if not self.wn:
            self.build_network()
        
        # Use EPANET simulator for full hydraulic accuracy
        sim = wntr.sim.EpanetSimulator(self.wn)
        
        # Use unique file prefix for thread-safety in parallel execution
        temp_prefix = f"temp_{uuid.uuid4().hex[:8]}"
        
        try:
            results = sim.run_sim(file_prefix=temp_prefix)
            
            # Process Pressures
            pressure_m = results.node['pressure']
            pressure_bar = pressure_m / BAR_TO_METERS
            
            sensor_pressures = {}
            if self.sensor_names:
                for sensor in self.sensor_names:
                    if sensor in pressure_bar.columns:
                        sensor_pressures[sensor] = pressure_bar[sensor].values
                    else:
                        sensor_pressures[sensor] = np.zeros(len(pressure_bar))

            # Process Flows
            flow_lps = results.link['flowrate'] / LPS_TO_M3S 
            
            # Calculate MAE
            mae = self._calculate_mae(sensor_pressures)
            
            return SimulationResult(
                all_pressures_bar=pressure_bar,
                sensor_pressures_bar=sensor_pressures,
                all_flows_lps=flow_lps,
                success=True,
                mae=mae
            )

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return SimulationResult(
                all_pressures_bar=pd.DataFrame(),
                sensor_pressures_bar={},
                all_flows_lps=pd.DataFrame(),
                success=False,
                error_message=str(e),
                mae=float('inf')
            )
        finally:
            # Clean up temp files to avoid disk space issues
            for ext in ['.inp', '.bin', '.out', '.rpt', '.hyd']:
                try:
                    temp_file = f"{temp_prefix}{ext}"
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass

    def _calculate_mae(self, sensor_pressures: Dict[str, np.ndarray]) -> float:
        """Calculate Mean Absolute Error against measured data."""
        if not self.measured_pressures:
            return 0.0
        
        errors = []
        for sensor, measured in self.measured_pressures.items():
            sim_series = sensor_pressures.get(sensor, [])
            if len(sim_series) > 0:
                # Average simulated pressure over time to compare with average measured
                sim_avg = float(np.mean(sim_series))
                errors.append(abs(sim_avg - measured))
        
        return float(np.mean(errors)) if errors else 0.0

    def get_pipe_groups(self, strategy: str = "decade") -> Dict[str, List[str]]:
        """Get pipe groups using different strategies."""
        return get_pipe_groups(self.wn, self.data['pipes'], strategy)

    def update_roughness(self, group_roughness: Dict[str, float], pipe_groups: Dict[str, List[str]]) -> None:
        """Update pipe roughness values by group."""
        if not self.wn:
            self.build_network()
        update_roughness_by_group(self.wn, group_roughness, pipe_groups)

    def apply_pipe_roughness(self, pipe_roughness: Dict[str, float]) -> None:
        """Apply roughness values directly to individual pipes.
        
        This method applies per-pipe roughness values without needing
        to know the grouping strategy used during training.
        
        Args:
            pipe_roughness: Dictionary mapping pipe ID to roughness value
        """
        if not self.wn:
            self.build_network()
        
        applied = 0
        for pipe_name, roughness in pipe_roughness.items():
            try:
                self.wn.get_link(pipe_name).roughness = float(roughness)
                applied += 1
            except KeyError:
                pass  # Pipe not in network (may have been removed or renamed)
        
        logger.debug(f"Applied roughness to {applied}/{len(pipe_roughness)} pipes")


# --- Standalone Functions (Legacy/Helper) ---
# Kept for backward compatibility or direct usage if needed

def get_pipe_groups(wn: wntr.network.WaterNetworkModel, 
                    pipes_df: pd.DataFrame,
                    strategy: str = "decade") -> Dict[str, List[str]]:
    """Get pipe groups using different strategies."""
    pipes_df = pipes_df.copy()
    
    if strategy == "decade":
        pipes_df['decade'] = (pipes_df['year'] // 10) * 10
        pipes_df['group'] = 'steel_' + pipes_df['decade'].fillna(2000).astype(int).astype(str)
        
    elif strategy == "year":
        pipes_df['group'] = 'year_' + pipes_df['year'].fillna(2000).astype(int).astype(str)
        
    elif strategy == "detailed":
        # Group by Material + Decade + Diameter Class
        def diameter_class(d):
            return 'sm' if d < 150 else 'lg'
            
        pipes_df['decade'] = (pipes_df['year'] // 10) * 10
        # Clean material tag
        pipes_df['mat'] = pipes_df['tags'].fillna('unk').astype(str).str.lower()
        pipes_df['d_cls'] = pipes_df['diameter'].apply(diameter_class)
        
        pipes_df['group'] = pipes_df['mat'] + '_' + pipes_df['decade'].fillna(2000).astype(int).astype(str) + '_' + pipes_df['d_cls']
    
    elif strategy == "diameter":
        def diameter_group(d):
            if d < 30: return 'tiny_<30mm'
            if d < 50: return 'small_30-50mm'
            if d < 80: return 'medium_50-80mm'
            if d < 150: return 'large_80-150mm'
            return 'xlarge_>150mm'
        pipes_df['group'] = pipes_df['diameter'].apply(diameter_group)
        
    elif strategy == "decade_diameter":
        pipes_df['decade'] = (pipes_df['year'] // 10) * 10
        def diam_size(d):
            if d < 50: return 'small'
            if d < 100: return 'medium'
            return 'large'
        pipes_df['diam_cat'] = pipes_df['diameter'].apply(diam_size)
        pipes_df['group'] = (
            pipes_df['decade'].fillna(2000).astype(int).astype(str) + '_' + pipes_df['diam_cat']
        )
        
    elif strategy == "5year":
        pipes_df['period'] = (pipes_df['year'] // 5) * 5
        pipes_df['group'] = 'period_' + pipes_df['period'].fillna(2000).astype(int).astype(str)
    
    elif strategy == "5year_diameter":
        # 5-year period
        pipes_df['period'] = (pipes_df['year'] // 5) * 5
        pipes_df['period_str'] = pipes_df['period'].fillna(2000).astype(int).astype(str)
        
        # Diameter class
        def diam_class(d):
            if d < 50: return 'sm'
            if d < 100: return 'md'
            if d < 200: return 'lg'
            return 'xl'
        pipes_df['diam'] = pipes_df['diameter'].apply(diam_class)
        
        # Combine: e.g., "2000_md"
        pipes_df['group'] = pipes_df['period_str'] + '_' + pipes_df['diam']
    
    elif strategy == "5year_diameter_length":
        # 5-year period
        pipes_df['period'] = (pipes_df['year'] // 5) * 5
        pipes_df['period_str'] = pipes_df['period'].fillna(2000).astype(int).astype(str)
        
        # Diameter class
        def diam_class(d):
            if d < 50: return 'sm'
            if d < 100: return 'md'
            if d < 200: return 'lg'
            return 'xl'
        pipes_df['diam'] = pipes_df['diameter'].apply(diam_class)
        
        # Length class
        def length_class(l):
            if l < 50: return 'short'
            if l < 200: return 'mid'
            return 'long'
        pipes_df['len'] = pipes_df['length'].apply(length_class)
        
        # Combine: e.g., "2000_md_mid"
        pipes_df['group'] = pipes_df['period_str'] + '_' + pipes_df['diam'] + '_' + pipes_df['len']
    
    elif strategy == "smart":
        # Smart grouping: Age x Pipe Type = 6 groups
        # Age: old (<1990), mature (1990-2010), new (>2010)
        # Type: main (>=100mm), distribution (<100mm)
        ref_year = 2026
        
        def age_category(year):
            if pd.isna(year) or year < 1990:
                return 'old'      # Pre-1990: significant degradation expected
            elif year <= 2010:
                return 'mature'   # 1990-2010: moderate degradation
            else:
                return 'new'      # 2010+: minimal degradation
        
        def pipe_type(diameter):
            if diameter >= 100:
                return 'main'     # Transmission/main lines
            else:
                return 'dist'     # Distribution lines
        
        pipes_df['age'] = pipes_df['year'].apply(age_category)
        pipes_df['type'] = pipes_df['diameter'].apply(pipe_type)
        pipes_df['group'] = pipes_df['age'] + '_' + pipes_df['type']

    elif strategy == "compact":
        # Ultra-compact: just 4 groups based on age
        def age_bin(year):
            if pd.isna(year) or year < 1985:
                return 'vintage'  # Very old
            elif year < 2000:
                return 'aged'     # Older
            elif year < 2015:
                return 'mature'   # Middle-aged
            else:
                return 'recent'   # Recent
        pipes_df['group'] = pipes_df['year'].apply(age_bin)

    elif strategy == "sophisticated":
        # High-fidelity grouping: 5-year x Diameter Class x Connector status
        # This creates ~60-100 parameters for fine-grained calibration
        
        # 1. 5-Year Periods
        pipes_df['period_str'] = ((pipes_df['year'] // 5) * 5).fillna(2000).astype(int).astype(str)
        
        # 2. Detailed Diameter Classes
        def diam_detailed(d):
            if d < 30: return 'micro'    # < 30mm
            if d < 60: return 'small'    # 30-60mm
            if d < 120: return 'med'     # 60-120mm
            if d < 250: return 'large'   # 120-250mm
            return 'huge'                # > 250mm
        pipes_df['diam'] = pipes_df['diameter'].apply(diam_detailed)
        
        # 3. Connection Type (Length-based)
        # Short pipes often represent fittings/valves/connections with different loss characteristics
        def length_type(l):
            return 'conn' if l < 20.0 else 'line'
            
        pipes_df['ltype'] = pipes_df['length'].apply(length_type)
        
        # Combine: "2005_small_line"
        pipes_df['group'] = pipes_df['period_str'] + '_' + pipes_df['diam'] + '_' + pipes_df['ltype']
        
    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")
    
    groups = {}
    valid_pipes = set(wn.pipe_name_list)
    for group_name, df_group in pipes_df.groupby('group'):
        existing = [p for p in df_group['name'] if p in valid_pipes]
        if existing:
            groups[group_name] = existing
            
    return groups

def update_roughness_by_group(wn: wntr.network.WaterNetworkModel,
                               group_roughness: Dict[str, float],
                               pipe_groups: Dict[str, List[str]]) -> None:
    """Update pipe roughness."""
    for group_name, hw_c_factor in group_roughness.items():
        if group_name in pipe_groups:
             for pipe_name in pipe_groups[group_name]:
                try:
                    wn.get_link(pipe_name).roughness = hw_c_factor
                except KeyError:
                    pass

def run_simulation(wn: wntr.network.WaterNetworkModel,
                   sensor_names: Optional[List[str]] = None) -> SimulationResult:
    """Legacy/Direct: Run hydraulic simulation using WNTRSimulator."""
    sim = wntr.sim.WNTRSimulator(wn)
    try:
        results = sim.run_sim()
        pressure_m = results.node['pressure']
        pressure_bar = pressure_m / BAR_TO_METERS
        
        sensor_pressures = {}
        if sensor_names:
            for sensor in sensor_names:
                if sensor in pressure_bar.columns:
                    sensor_pressures[sensor] = pressure_bar[sensor].values
                else:
                    sensor_pressures[sensor] = np.zeros(len(pressure_bar))

        flow_lps = results.link['flowrate'] / LPS_TO_M3S 
        
        return SimulationResult(
            all_pressures_bar=pressure_bar,
            sensor_pressures_bar=sensor_pressures,
            all_flows_lps=flow_lps,
            success=True
        )
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return SimulationResult(pd.DataFrame(), {}, pd.DataFrame(), False, str(e))

# Alias for compatibility with old imports if any
NetworkManager = SimulationEngine

def get_network_geometry(wn: wntr.network.WaterNetworkModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Isolates geometry extraction logic."""
    node_list = []
    for name, node in wn.nodes():
        node_list.append({
            'name': name,
            'x': node.coordinates[0],
            'y': node.coordinates[1],
            'elevation': getattr(node, 'elevation', 0),
            'type': node.node_type
        })
    nodes_df = pd.DataFrame(node_list)

    pipe_list = []
    for name, link in wn.links():
        if link.link_type != 'Pipe': continue
        
        start_node = wn.get_node(link.start_node_name)
        end_node = wn.get_node(link.end_node_name)
        
        pipe_list.append({
            'name': name,
            'start_node': link.start_node_name,
            'end_node': link.end_node_name,
            'start_x': start_node.coordinates[0],
            'start_y': start_node.coordinates[1],
            'end_x': end_node.coordinates[0],
            'end_y': end_node.coordinates[1],
            'length': link.length,
            'diameter': link.diameter / MM_TO_M,
            'roughness': link.roughness
        })
    pipes_df = pd.DataFrame(pipe_list)
    
    return nodes_df, pipes_df

def get_available_strategies() -> Dict[str, str]:
    return {
        "decade": "By decade (6 groups)",
        "year": "By individual year (41 groups)",
        "diameter": "By diameter range (5 groups)",
        "decade_diameter": "By decade + diameter (18 groups)",
        "5year": "By 5-year period (12 groups)",
    }

if __name__ == "__main__":
    print("Testing Simulation Engine...")
    engine = SimulationEngine()
    wn = engine.build_network()
    print(f"Network built: {len(wn.pipe_name_list)} pipes.")
    
    res = engine.run_simulation()
    if res.success:
        print(f"Simulation successful. MAE: {res.mae:.4f} bar")
    else:
        print("Simulation failed.")
