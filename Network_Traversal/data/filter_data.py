
import pandas as pd
from pathlib import Path
import shutil
import os

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "Network_Traversal" / "data"
CONSOLIDATED_DIR = DATA_DIR / "consolidated"
SENSORS_DIR = CONSOLIDATED_DIR / "pressure_sensors"
FLOW_DIR = CONSOLIDATED_DIR / "boundary_flow"
REJECTED_DIR = DATA_DIR / "rejected"

def filter_data():
    if not SENSORS_DIR.exists():
        print(f"Error: {SENSORS_DIR} does not exist.")
        return

    # Ensure rejected dirs exist
    (REJECTED_DIR / "pressure_sensors").mkdir(parents=True, exist_ok=True)
    (REJECTED_DIR / "boundary_flow").mkdir(parents=True, exist_ok=True)

    sensor_files = list(SENSORS_DIR.glob("sensors_*.csv"))
    print(f"Scanning {len(sensor_files)} sensor files in {SENSORS_DIR}...")
    
    rejected_count = 0
    
    for s_file in sensor_files:
        date = s_file.stem.split('_')[1]
        try:
            df = pd.read_csv(s_file)
            
            # Check structure
            if 'sensor' not in df.columns or 'pressure_avg' not in df.columns:
                reason = "Missing required columns"
                move_to_rejected(s_file, date, reason)
                rejected_count += 1
                continue
            
            # Group by sensor and count
            counts = df.groupby('sensor').size()
            
            # Criteria 1: 24 points per sensor
            if not (counts == 24).all():
                # Get details
                bad_sensors = counts[counts != 24]
                reason = f"Incomplete data for sensors: {bad_sensors.to_dict()}"
                move_to_rejected(s_file, date, reason)
                rejected_count += 1
                continue
                
            # Criteria 2: Expecting 8 sensors
            expected_sensors = {f"Sensor-{i}" for i in range(1, 9)}
            found_sensors = set(df['sensor'].unique())
            
            if found_sensors != expected_sensors:
                missing = expected_sensors - found_sensors
                extra = found_sensors - expected_sensors
                reason = f"Missing sensors: {missing}"
                if extra:
                    reason += f", Extra: {extra}"
                move_to_rejected(s_file, date, reason)
                rejected_count += 1
                continue
            
            # If we passed, we keep it.
            
        except Exception as e:
            reason = f"Error reading file: {e}"
            move_to_rejected(s_file, date, reason)
            rejected_count += 1
            
    print(f"Finished. Rejected {rejected_count} files.")

def move_to_rejected(s_file, date, reason):
    print(f"REJECTING {date}: {reason}")
    
    # Move sensor file
    dest_s = REJECTED_DIR / "pressure_sensors" / s_file.name
    shutil.move(str(s_file), str(dest_s))
    
    # Move flow file if exists
    f_file = FLOW_DIR / f"inputflow_{date}.csv"
    if f_file.exists():
        dest_f = REJECTED_DIR / "boundary_flow" / f_file.name
        shutil.move(str(f_file), str(dest_f))
    else:
        print(f"  (No corresponding flow file found for {date})")

if __name__ == "__main__":
    filter_data()
