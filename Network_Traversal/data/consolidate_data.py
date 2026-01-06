
import os
import shutil
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataConsolidator")

def consolidate_datasets(base_dir: Path):
    """
    Consolidates data from:
    - input_flow_20days
    - iot_measurements_20days
    - long_data/boundary_flow
    - long_data/pressure_sensors
    
    Into:
    - consolidated/boundary_flow
    - consolidated/pressure_sensors
    
    Prioritizes 'long_data' (source of truth).
    Standardizes on 0-23 hour format.
    """
    
    # 1. Setup Directories
    consolidated_dir = base_dir / "consolidated"
    cons_flow_dir = consolidated_dir / "boundary_flow"
    cons_press_dir = consolidated_dir / "pressure_sensors"
    
    if consolidated_dir.exists():
        logger.warning(f"Consolidated directory {consolidated_dir} already exists. Cleaning up...")
        shutil.rmtree(consolidated_dir)
        
    cons_flow_dir.mkdir(parents=True)
    cons_press_dir.mkdir(parents=True)
    
    # Sources
    src_long_flow = base_dir / "long_data" / "boundary_flow"
    src_long_press = base_dir / "long_data" / "pressure_sensors"
    
    src_old_flow = base_dir / "input_flow_20days"
    src_old_press = base_dir / "iot_measurements_20days"
    
    # Track processed dates (DDMMYY) to handle duplicates
    # Filename format: inputflow_DDMMYY.csv, sensors_DDMMYY.csv
    processed_dates_flow = set()
    processed_dates_press = set()
    
    # --- PHASE 1: Process Long Data (Priority) ---
    logger.info("Processing Long Data (Priority)...")
    
    # Flow
    if src_long_flow.exists():
        for f in src_long_flow.glob("inputflow_*.csv"):
            process_flow_file(f, cons_flow_dir, processed_dates_flow, source_label="LongData")
            
    # Pressure
    if src_long_press.exists():
        for f in src_long_press.glob("sensors_*.csv"):
            process_pressure_file(f, cons_press_dir, processed_dates_press, source_label="LongData")

    # --- PHASE 2: Process Old Data (Fill gaps) ---
    logger.info("Processing Old Data (Filling gaps)...")
    
    # Flow
    if src_old_flow.exists():
        for f in src_old_flow.glob("inputflow_*.csv"):
            process_flow_file(f, cons_flow_dir, processed_dates_flow, source_label="OldData", shift_hours=True)
            
    # Pressure
    if src_old_press.exists():
        for f in src_old_press.glob("sensors_*.csv"):
            process_pressure_file(f, cons_press_dir, processed_dates_press, source_label="OldData", shift_hours=True)

    logger.info(f"Consolidation Complete.")
    logger.info(f"Total Flow Files: {len(processed_dates_flow)}")
    logger.info(f"Total Pressure Files: {len(processed_dates_press)}")


def process_flow_file(file_path: Path, dest_dir: Path, processed_set: set, source_label: str, shift_hours: bool = False):
    """Reads, standardizes, and saves flow file if date not already processed."""
    # Extract date from filename: inputflow_DDMMYY.csv
    filename = file_path.name
    date_part = filename.replace("inputflow_", "").replace(".csv", "")
    
    if date_part in processed_set:
        # Skip duplicate (Long Data takes precedence)
        return
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize Columns
        # Expecting [, value] or [hour, value]
        # Check if index is unnamed
        if "value" not in df.columns and len(df.columns) >= 1:
             # assume last column is value
             df.rename(columns={df.columns[-1]: 'value'}, inplace=True)
        
        # Standardize Hour Index
        # If shift_hours=True (Old Data 1-24), subtract 1 -> 0-23
        # If already 0-23, keep.
        
        # We need to ensure we save it 'cleanly'
        # Long data has unnamed index 0..23.
        # Let's verify row count.
        if len(df) != 24 and len(df) != 26: # Allow minor variations, but warn
             logger.warning(f"File {filename} ({source_label}) has {len(df)} rows. Expected 24.")
             
        # If needing shift (1-24 -> 0-23)
        # Old data inputflow:
        # ,value
        # 1,238...
        
        # New data inputflow:
        # ,value
        # 0,238...
        
        # We assume the first column is the index/hour.
        # Just reset the index to be safe 0..23
        df = df.reset_index(drop=True)
        # If the CSV had an explicit index column that was read as a column, drop it?
        # Pandas read_csv with no index_col usually reads:
        # Unnamed: 0, value
        # 1, 238
        
        cols = [c for c in df.columns if 'value' in c.lower()]
        if not cols:
            logger.warning(f"Could not identify value column in {filename}")
            return
            
        val_col = cols[0]
        
        # Create standardized DF
        clean_df = pd.DataFrame()
        clean_df['value'] = df[val_col]
        
        # If we need to shift? 
        # Actually, since we generated a fresh 0..23 index by creating new DF, 
        # we implicitly handled 1-24 vs 0-23 row mapping IF the rows were ordered.
        
        # Validate integrity
        if shift_hours:
            # Old data was 1..24. We just mapped to 0..23 by position.
            pass
            
        # Save
        clean_df.to_csv(dest_dir / filename, index=True, index_label="")
        processed_set.add(date_part)
        
    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}")


def process_pressure_file(file_path: Path, dest_dir: Path, processed_set: set, source_label: str, shift_hours: bool = False):
    """Reads, standardizes, and saves pressure file."""
    # Filename: sensors_DDMMYY.csv
    filename = file_path.name
    date_part = filename.replace("sensors_", "").replace(".csv", "")
    
    if date_part in processed_set:
        return

    try:
        df = pd.read_csv(file_path)
        
        # Standardize Columns: [hour, pressure_avg, sensor]
        required = {'hour', 'pressure_avg', 'sensor'}
        if not required.issubset(df.columns):
             logger.warning(f"File {filename} missing columns. Found {df.columns}")
             return

        # Shift Hours if needed
        if shift_hours:
            # Shift 1-24 -> 0-23
            df['hour'] = df['hour'] - 1
            
        # Verify range
        min_h, max_h = df['hour'].min(), df['hour'].max()
        if min_h < 0 or max_h > 23:
             logger.warning(f"File {filename} ({source_label}) has hours {min_h}-{max_h}. Clamping/Checking.")
             
        # Save
        df.to_csv(dest_dir / filename, index=False)
        processed_set.add(date_part)
        
    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    base_path = Path("Network_Traversal/data")
    if not base_path.exists():
        # Try full path or relative to current wd
        base_path = Path("/Users/bjorn/Projects/AI Competition/AI Keppni/GKI2026/Network_Traversal/data")
        
    consolidate_datasets(base_path)
