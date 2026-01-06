"""
Data loading utilities for WNTR network models.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache
from .config import DATA_DIR

def load_csv_data(data_dir: Optional[Path] = None, date: Optional[str] = None, include_rejected: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files into dataframes.
    If 'date' is provided (format 'DDMMYY'), attempts to load specific daily files 
    for sensors and boundary flow ('sensors_DDMMYY.csv', 'inputflow_DDMMYY.csv').
    """
    if data_dir is None:
        data_dir = DATA_DIR

    files = {
        'junctions': "junctions_csv.csv",
        'pipes': "pipes_csv.csv",
        'pumps': "pumps_csv.csv",
        'pump_curves': "pump_curves_csv.csv",
        'valves': "valves_csv.csv",
        'reservoir': "reservoir_csv.csv",
        # Default/Fallback filenames
        'boundary_flow': "boundary_flow.csv",
        'sensors': "sensor_measurements.csv",
    }
    
    # Override for specific date if provided
    if date:
        # Check for specific files in consolidated headers, legacy, OR rejected
        # Define search roots: Main (Consolidated) and Rejected
        search_roots = [data_dir]
        
        if include_rejected:
            if data_dir.name == "consolidated":
                 search_roots.append(data_dir / "rejected")
            elif data_dir.name == "data": # Fallback
                 search_roots.append(data_dir / "rejected")

        flow_dirs = ["boundary_flow", "input_flow_20days"]
        sensor_dirs = ["pressure_sensors", "iot_measurements_20days"]
        
        files['boundary_flow'] = None
        files['sensors'] = None
        
        # Search for Flow File
        for root in search_roots:
            if files['boundary_flow'] is not None: break
            for d in flow_dirs:
                p = root / d / f"inputflow_{date}.csv"
                if p.exists():
                    files['boundary_flow'] = p
                    break
        
        # Search for Sensor File
        for root in search_roots:
            if files['sensors'] is not None: break
            for d in sensor_dirs:
                p = root / d / f"sensors_{date}.csv"
                if p.exists():
                    files['sensors'] = p
                    break
        
        if files['boundary_flow'] is None:
            # Only warn if not found in ANY location
             print(f"Warning: Date {date} requested but inputflow file not found. Using default.")
             files['boundary_flow'] = "boundary_flow.csv"

        if files['sensors'] is None:
             print(f"Warning: Date {date} requested but sensors file not found. Using default.")
             files['sensors'] = "sensor_measurements.csv"

    data = {}
    for key, filename in files.items():
        # Handle both string filenames (relative) and Path objects (absolute from override)
        if filename is None: continue 
        
        if isinstance(filename, Path):
            file_path = filename
        else:
            file_path = data_dir / filename
            
        try:
            data[key] = pd.read_csv(file_path)
            # Basic normalization for consistency
            if key == 'boundary_flow' and not data[key].empty:
                # Rename columns to standard [index/hour, flow] if needed, 
                # though engine mostly relies on position.
                pass 
        except Exception as e: # Catch all read errors (missing cols, empty, etc)
             if not date: 
                 print(f"Warning: File {filename} not found or invalid in {data_dir}. {e}")
             data[key] = pd.DataFrame()

    return data

def get_available_dates(data_dir: Optional[Path] = None, include_rejected: bool = False) -> list[str]:
    """Scan data directories to find available dates (DDMMYY)."""
    if data_dir is None:
        data_dir = DATA_DIR
        
    dates = set()
    
    search_roots = [data_dir]
    
    if include_rejected:
        if data_dir.name == "consolidated":
             search_roots.append(data_dir / "rejected")
    
    # Check for consolidated first, then legacy, then rejected
    possible_flow_dirs = ["boundary_flow", "input_flow_20days"]
    possible_sensor_dirs = ["pressure_sensors", "iot_measurements_20days"]
    
    for root in search_roots:
        if not root.exists(): continue
        
        # Scan input flow folders
        for d_name in possible_flow_dirs:
            flow_dir = root / d_name
            if flow_dir.exists():
                for f in flow_dir.glob("inputflow_*.csv"):
                    parts = f.stem.split('_')
                    if len(parts) > 1:
                        dates.add(parts[1])
                    
        # Scan sensors folder
        for d_name in possible_sensor_dirs:
            sensor_dir = root / d_name
            if sensor_dir.exists():
                for f in sensor_dir.glob("sensors_*.csv"):
                    parts = f.stem.split('_')
                    if len(parts) > 1:
                        dates.add(parts[1])
                
    return sorted(list(dates))

def get_sensor_names(data_dir: Optional[Path] = None) -> list[str]:
    """Get list of unique sensor names."""
    data = load_csv_data(data_dir) # Uses default (no date) for generic checking
    if 'sensors' in data and not data['sensors'].empty:
        return sorted(data['sensors']['sensor'].unique().tolist())
    return []

def get_measured_pressures(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Get measured pressures dataframe."""
    return load_csv_data(data_dir).get('sensors', pd.DataFrame())

def get_boundary_flow(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Get boundary flow dataframe with normalized column names."""
    df = load_csv_data(data_dir).get('boundary_flow', pd.DataFrame()).copy()
    if not df.empty:
        # Standardize 'hour' column if it's the first column
        df.rename(columns={df.columns[0]: 'hour'}, inplace=True)
    return df
