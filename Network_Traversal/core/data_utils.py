"""
Data loading utilities for WNTR network models.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache
from .config import DATA_DIR

@lru_cache(maxsize=1)
def load_csv_data(data_dir: Optional[Path] = None, date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
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
        # Check for specific files
        flow_file = data_dir / "input_flow_20days" / f"inputflow_{date}.csv"
        sensor_file = data_dir / "iot_measurements_20days" / f"sensors_{date}.csv"
        
        if flow_file.exists():
            files['boundary_flow'] = flow_file
        else:
            print(f"Warning: Date {date} requested but {flow_file.name} not found. Using default.")
            
        if sensor_file.exists():
            files['sensors'] = sensor_file
        else:
             print(f"Warning: Date {date} requested but {sensor_file.name} not found. Using default.")

    data = {}
    for key, filename in files.items():
        # Handle both string filenames (relative) and Path objects (absolute from override)
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
        except FileNotFoundError:
             if not date: # Only warn if we aren't already expecting potential missing files for specific dates
                 print(f"Warning: File {filename} not found in {data_dir}. Returning empty DataFrame.")
             data[key] = pd.DataFrame()

    return data

def get_available_dates(data_dir: Optional[Path] = None) -> list[str]:
    """Scan data directories to find available dates (DDMMYY)."""
    if data_dir is None:
        data_dir = DATA_DIR
        
    dates = set()
    
    # Scan input flow folder
    flow_dir = data_dir / "input_flow_20days"
    if flow_dir.exists():
        for f in flow_dir.glob("inputflow_*.csv"):
            # extract DDMMYY from inputflow_DDMMYY.csv
            parts = f.stem.split('_')
            if len(parts) > 1:
                dates.add(parts[1])
                
    # Scan sensors folder
    sensor_dir = data_dir / "iot_measurements_20days"
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
