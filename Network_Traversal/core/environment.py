#!/usr/bin/env python3
"""
Gymnasium RL Environment for WNTR Pipe Roughness Calibration.
Refactored to use consolidated SimulationEngine.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .config import DATA_DIR, ROUGHNESS_MIN, ROUGHNESS_MAX, ROUGHNESS_DEFAULT
from .engine import SimulationEngine
from .data_utils import get_available_dates

logger = logging.getLogger(__name__)

class RoughnessCalibrationEnv(gym.Env):
    """RL Environment for pipe roughness calibration using WNTR simulation."""
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data_dir: Optional[Path] = None, max_steps: int = 50, render_mode: Optional[str] = None):
        super().__init__()
        # Initialize consolidated engine
        self.engine = SimulationEngine(data_dir)
        self.wn = self.engine.build_network()
        
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Metadata from Engine
        self.pipe_groups = self.engine.get_pipe_groups(strategy='detailed')
        self.group_names = sorted(self.pipe_groups.keys())
        self.n_groups = len(self.group_names)
        self.sensor_names = self.engine.sensor_names
        self.n_sensors = len(self.sensor_names)
        
        # Available Dates for Training
        self.available_dates = get_available_dates(self.engine.data_dir)
        if not self.available_dates:
            logger.warning("No specific dates found in data folder. Using default/single-file mode.")
        else:
            logger.info(f"Found {len(self.available_dates)} days for training: {self.available_dates}")
        
        # State
        self.c_factor_min = ROUGHNESS_MIN
        self.c_factor_max = ROUGHNESS_MAX
        self.group_c_factors = np.full(self.n_groups, ROUGHNESS_DEFAULT, dtype=np.float32)
        self.current_mae = 0.0
        self.current_step = 0
        
        # Spaces
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n_groups,), dtype=np.float32)
        
        # Observation: [errors..., c_factors..., step_fraction]
        obs_dim = self.n_sensors + self.n_groups + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.group_c_factors = np.full(self.n_groups, ROUGHNESS_DEFAULT, dtype=np.float32)
        
        # Select Random Day
        current_date_str = "Default"
        if self.available_dates:
            # If seed is provided, this ensures reproducibility of the sequence of days
            if seed is not None:
                np.random.seed(seed)
            selected_date = np.random.choice(self.available_dates)
            self.engine.set_date(selected_date)
            current_date_str = str(selected_date)
        
        # Initial Run
        obs, result = self._run_and_observe()
        mae = result.mae
        
        return obs, {
            "initial_mae": mae,
            "n_groups": self.n_groups,
            "group_names": self.group_names,
            "date": current_date_str
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        
        # Update State
        self.group_c_factors = np.clip(
            self.group_c_factors + action,
            self.c_factor_min,
            self.c_factor_max
        ).astype(np.float32)
        
        # Run
        obs, result = self._run_and_observe()
        mae = result.mae
        neg_count = result.negative_pressure_count
        
        # Rewards & Termination
        if not result.success or np.isnan(mae) or np.isinf(mae):
            # Simulation failed drastically
            mae = 100.0
            neg_count = 100
            reward = -500.0
            terminated = True
        else:
            # Shaped Reward:
            # 1. Minimize Error (MAE). Scale * 10 so 0.1 bar error = -1.0 reward.
            # 2. Penalize Negative Pressure. Heavy penalty.
            reward = -(mae * 10.0) - (0.5 * neg_count)
            terminated = mae < 0.05
            
        truncated = self.current_step >= self.max_steps
        
        if terminated and reward > -100: reward += 100.0 # big bonus for solving (but not for crashing)
        
        return obs, reward, terminated, truncated, {
            "mae": mae,
            "c_factors": dict(zip(self.group_names, self.group_c_factors.tolist()))
        }

    def _run_and_observe(self) -> Tuple[np.ndarray, Any]:
        """Internal helper to run simulation via Engine and construct observation."""
        # 1. Update Engine Roughness
        roughness_map = {name: float(self.group_c_factors[i]) 
                         for i, name in enumerate(self.group_names)}
        self.engine.update_roughness(roughness_map, self.pipe_groups)
        
        # 2. Run Simulation
        result = self.engine.run_simulation()
        self.current_mae = result.mae
        
        # 3. Construct Observation
        current_errors = []
        for sensor in self.sensor_names:
            sim_val = np.mean(result.sensor_pressures_bar.get(sensor, [0]))
            meas_val = self.engine.measured_pressures.get(sensor, 0)
            current_errors.append(abs(sim_val - meas_val))
            
        norm_c = (self.group_c_factors - self.c_factor_min) / (self.c_factor_max - self.c_factor_min)
        
        obs = np.concatenate([
            current_errors,
            norm_c,
            [self.current_step / self.max_steps]
        ]).astype(np.float32)
        
        # Sanitize observation to prevent NaNs from breaking the Agent
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return obs, result

if __name__ == "__main__":
    print("Testing Centralized RL Env...")
    env = RoughnessCalibrationEnv()
    obs, info = env.reset()
    print(f"Initial MAE: {info['initial_mae']:.4f} bar")
    env.close()
