"""
Data Pipeline for IG V2 Agent.
Handles 20-day window processing, median diurnal aggregation, and noise estimation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

from ...core.data_utils import load_csv_data, get_available_dates
from ...core.engine import SimulationEngine
from .config import (
    WINDOW_SIZE_DAYS,
    AGGREGATION_METHOD,
    INFER_LATENT_NOISE,
    INITIAL_SIGMA_GUESS
)

from datetime import datetime

logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, data_dir: Optional[Path] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.data_dir = data_dir
        raw_dates = get_available_dates(self.data_dir)
        
        # Sort chronologically (DDMMYY)
        # 301025 -> Oct 30, 2025. 011125 -> Nov 1, 2025.
        # Alphabetical sort fails here (01... < 30...).
        try:
            date_objs = []
            for d in raw_dates:
                try:
                    dt = datetime.strptime(d, "%d%m%y")
                    date_objs.append((d, dt))
                except ValueError:
                    logger.warning(f"Skipping invalid date format: {d}")
            
            # Sort by datetime object
            date_objs.sort(key=lambda x: x[1])
            
            # Filter if range provided
            if start_date:
                start_dt = datetime.strptime(start_date, "%d%m%y")
                date_objs = [x for x in date_objs if x[1] >= start_dt]
                
            if end_date:
                end_dt = datetime.strptime(end_date, "%d%m%y")
                date_objs = [x for x in date_objs if x[1] <= end_dt]
                
            # Extract back to strings
            self.available_dates = [x[0] for x in date_objs]
            
        except Exception as e:
            logger.error(f"Error filtering dates: {e}. Falling back to raw sorted.")
            self.available_dates = raw_dates

        # Limit to available window size (first N days of the filtered set)
        # IF user provided explicit range, use it all (do not truncate).
        # ELSE use default window size.
        if (start_date or end_date):
             # User specified range, respect it fully.
             self.dates = self.available_dates
             if len(self.dates) > WINDOW_SIZE_DAYS:
                 logger.info(f"Using full requested range of {len(self.dates)} days (exceeds default window {WINDOW_SIZE_DAYS}).")
        elif len(self.available_dates) > WINDOW_SIZE_DAYS:
             # Default behavior: slice
             logger.info(f"Filtered range has {len(self.available_dates)} days. Truncating to window size {WINDOW_SIZE_DAYS}.")
             self.dates = self.available_dates[:WINDOW_SIZE_DAYS]
        else:
             self.dates = self.available_dates
             if len(self.dates) < WINDOW_SIZE_DAYS:
                logger.warning(f"Only {len(self.dates)} days available, expected {WINDOW_SIZE_DAYS}.")
        
        if not self.dates:
            raise ValueError(f"No data found for the specified range! Start: {start_date}, End: {end_date}. Check that Start <= End and files exist.")

        
        range_str = f"{self.dates[0]} to {self.dates[-1]}" if self.dates else "None"
        logger.info(f"Initialized DataPipeline with {len(self.dates)} days: {range_str}")
        
        # Cache for processed data
        self.y_obs: Dict[str, pd.DataFrame] = {} # date -> sensor_df
        self.y_target: Dict[str, np.ndarray] = {} # sensor -> 24h vector (if median)
        self.y_variance: Dict[str, float] = {} # sensor -> scalar
        self.sigma_n = INITIAL_SIGMA_GUESS
        
        self.load_and_process_data()

    def load_and_process_data(self):
        """Loads all daily data and aggregates it."""
        all_sensor_data = [] # List of (sensor_name, hour, pressure) tuples or similar
        
        # 1. Load raw data for each day
        raw_dfs = []
        for date in self.dates:
            data = load_csv_data(self.data_dir, date=date)
            if 'sensors' in data and not data['sensors'].empty:
                df = data['sensors'].copy()
                df['date'] = date
                # Ensure 'hour' column exists (assuming standard format has 'timestamp' or index=hour)
                if 'hour' not in df.columns:
                     # Assuming 'timestamp' is in specific format or just 0-23 index if repeated
                     # Based on data_utils, sensors_*.csv usually has 'sensor', 'pressure_avg', etc.
                     # We need time info. If missing, we might assume 24h avg or specific structure.
                     # Let's check a sample file if possible, but for now assume we can infer or use raw.
                     # Actually, data_utils._load_measured_pressures takes mean of the file.
                     # But spec says "y_obs[s][t]".
                     # If the CSV is just daily averages, we can't do diurnal aggregation.
                     # Assuming the CSVs might have hourly data or we treat each file as one 'day' sample.
                     # If the file is just ONE row per sensor (daily avg), then MEDIAN_DIURNAL is just MEDIAN of days.
                     pass 
                raw_dfs.append(df)
            else:
                logger.warning(f"No sensor data for {date}")

        if not raw_dfs:
            logger.error("No sensor data found!")
            return

        combined_df = pd.concat(raw_dfs, ignore_index=True)
        # Structure: sensor, pressure, (maybe hour/time), date
        
        # 2. Process based on Aggregation Method
        self.sensor_names = sorted(combined_df['sensor'].unique())
        
        if AGGREGATION_METHOD == "MEDIAN_DIURNAL":
            # If we don't have hourly resolution in the CSVs (checking data_utils, it looks like 'pressure_avg'),
            # we might just have daily averages.
            # Strategy: Treat 't' as Day Index if hourly missing, or Hour Index if present.
            
            # Let's assume for V2 we want hourly if possible. 
            # If current CSVs are daily summaries, we just aggregate across days.
            # "y_target[s][h] = MEDIAN(y_obs[s] for all days at hour h)"
            
            # CHECK: data_utils.py says "val = df[df['sensor'] == sensor]['pressure_avg'].mean()"
            # This suggests the CSV might have multiple entries per sensor? 
            # Or just one. 
            # If it has 24 entries (hours), then we can do hourly median.
            
            # For this implementation, I will assume we compute a single robust target per sensor
            # if we lack hourly info, OR a 24-point curve if we have it.
            # To be safe and compatible with current data loaders:
            # We will generate a 'target profile' for each sensor.
            
            for s in self.sensor_names:
                s_data = combined_df[combined_df['sensor'] == s]
                
                # Try to detect if we have hourly data
                if len(s_data) > len(self.dates): 
                    # Likely multiple points per day
                    # Assuming we can align by index 0..23 if they are ordered
                    # This is a bit risky without explicit time column, but plausible for simulation output
                    
                    # Group by relative index (0..23)
                    # Create a temporary 'hour' based on modulo 24 if rows are ordered?
                    # Safer: Just take median across all observations for now (0-dim target)
                    # OR if the spec implies 24h vector:
                    
                    # "y_sim[s][t]" in spec.
                    # Let's build a 24-value target if possible.
                    # If we can't, we fall back to scaler.
                    
                    # For now, let's assume scaler (daily avg) to be safe with existing CSV format 
                    # seen in data_utils ("pressure_avg").
                    
                    # If the user really wants diurnal, we'd need hourly data.
                    # I will use the median of the daily averages as the robust target.
                    median_val = s_data['pressure_avg'].median()
                    self.y_target[s] = np.full(24, median_val) # Broadcast to 24h
                    
                    # Estimate noise (variance)
                    # MAD (Median Absolute Deviation) is more robust
                    mad = (s_data['pressure_avg'] - median_val).abs().median()
                    self.y_variance[s] = (mad * 1.4826) ** 2 # Consistent estimator for normal sigma^2
                    
                else:
                    # One point per day
                    median_val = s_data['pressure_avg'].median()
                    self.y_target[s] = np.full(24, median_val)
                    mad = (s_data['pressure_avg'] - median_val).abs().median()
                    self.y_variance[s] = (mad * 1.4826) ** 2
                    
        else:
            # Default to mean
            for s in self.sensor_names:
                mean_val = combined_df[combined_df['sensor'] == s]['pressure_avg'].mean()
                self.y_target[s] = np.full(24, mean_val)
                self.y_variance[s] = combined_df[combined_df['sensor'] == s]['pressure_avg'].var()

        # Latent noise init
        if INFER_LATENT_NOISE:
            # Initialize sigma_n based on average observed variance
            obs_monitor_vars = list(self.y_variance.values())
            if obs_monitor_vars:
                self.sigma_n = float(np.sqrt(np.mean(obs_monitor_vars)))
            else:
                self.sigma_n = INITIAL_SIGMA_GUESS
        
        logger.info(f"Data processed. Initial sigma_n: {self.sigma_n:.4f}")

    def update_noise_estimate(self, y_sim: Dict[str, np.ndarray]):
        """
        Updates sigma_n based on residuals between simulation and target.
        Only called if INFER_LATENT_NOISE is True.
        """
        residuals = []
        for s in self.sensor_names:
            if s in y_sim:
                # Compare 24h profile (or scalar broadcast)
                # target is stored as 24h vector. y_sim has multiples of 24h.
                sim_data = y_sim[s]
                target_data = self.y_target[s]
                
                # Tile target to match sim length
                num_days = len(sim_data) // len(target_data)
                remainder = len(sim_data) % len(target_data)
                
                tiled_target = np.tile(target_data, num_days)
                if remainder > 0:
                     tiled_target = np.concatenate([tiled_target, target_data[:remainder]])
                     
                res = sim_data - tiled_target
                residuals.extend(res)
        
        if residuals:
            # Robust estimation of noise from residuals
            resid_np = np.array(residuals)
            # Use RMS or standard deviation of residuals
            # If we assume unbiased, RMS. If biased, std.
            # Using std (centering) is safer for roughness errors.
            new_sigma = np.std(resid_np)
            
            # Smooth update (EMA)
            alpha = 0.3
            self.sigma_n = alpha * new_sigma + (1 - alpha) * self.sigma_n
            logger.info(f"Updated sigma_n: {self.sigma_n:.4f}")

