#!/usr/bin/env python3
"""
Gymnasium RL Environment for WNTR Pipe Roughness Calibration.

Uses WNTR for hydraulic simulation instead of EPANET subprocess.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from wntr_model_builder import (
    build_network_from_csv,
    get_pipe_groups,
    update_roughness_by_group,
    run_simulation,
    get_sensor_names,
    load_csv_data
)


class RoughnessCalibrationEnv(gym.Env):
    """RL Environment for pipe roughness calibration using WNTR simulation."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data_dir: str = "Shared Materials",
        max_steps: int = 50,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Load data and build initial network
        self.data = load_csv_data(str(self.data_dir))
        self.wn = build_network_from_csv(str(self.data_dir))
        
        # Get pipe groups (by material and decade)
        self.pipe_groups = get_pipe_groups(self.wn, self.data['pipes'])
        self.group_names = sorted(self.pipe_groups.keys())
        self.n_groups = len(self.group_names)
        
        # Get sensor information
        self.sensor_names = get_sensor_names(str(self.data_dir))
        self.n_sensors = len(self.sensor_names)
        
        # Load measured pressures (average across hours, in bar)
        self.sensor_df = self.data['sensors']
        self.measured_pressures = self._get_measured_pressures()
        
        # Hazen-Williams C-factor range (realistic values)
        self.c_factor_min = 80.0   # Very rough/corroded
        self.c_factor_max = 140.0  # Smooth/new pipe
        self.c_factor_default = 120.0  # Typical aged steel
        
        # Define action space: C-factor adjustments for each group
        # Actions are deltas to apply to current C-factors
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.n_groups,), dtype=np.float32
        )
        
        # Define observation space:
        # - Pressure errors at each sensor (n_sensors)
        # - Normalized C-factors for each group (n_groups)
        # - Step fraction (1)
        obs_dim = self.n_sensors + self.n_groups + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.group_c_factors = np.full(self.n_groups, self.c_factor_default, dtype=np.float32)
        self.current_errors = np.zeros(self.n_sensors, dtype=np.float32)
        
    def _get_measured_pressures(self) -> np.ndarray:
        """Get average measured pressure at each sensor (in bar)."""
        pressures = []
        for sensor in self.sensor_names:
            avg = self.sensor_df[self.sensor_df['sensor'] == sensor]['pressure_avg'].mean()
            pressures.append(avg)
        return np.array(pressures, dtype=np.float32)
    
    def _run_simulation(self) -> Dict[str, float]:
        """Run WNTR simulation and get average pressures at sensor nodes."""
        # Rebuild network for fresh state
        wn = build_network_from_csv(str(self.data_dir))
        
        # Rebuild pipe groups for this network instance
        pipe_groups = get_pipe_groups(wn, self.data['pipes'])
        
        # Create roughness dict from current C-factors
        roughness_values = {
            name: float(self.group_c_factors[i])
            for i, name in enumerate(self.group_names)
        }
        
        # Update roughness values
        update_roughness_by_group(wn, roughness_values, pipe_groups)
        
        # Run simulation
        results = run_simulation(wn, self.sensor_names)
        
        # Get average pressure at each sensor across all timesteps
        sensor_pressures = {}
        for sensor in self.sensor_names:
            if sensor in results['sensor_pressures']:
                pressures = results['sensor_pressures'][sensor]
                sensor_pressures[sensor] = float(np.mean(pressures)) if len(pressures) > 0 else 0.0
            else:
                sensor_pressures[sensor] = 0.0
        
        return sensor_pressures
    
    def _calculate_reward(self, simulated: Dict[str, float]) -> float:
        """Calculate reward based on pressure matching."""
        errors = []
        for i, sensor in enumerate(self.sensor_names):
            measured = self.measured_pressures[i]
            sim = simulated.get(sensor, 0.0)
            errors.append(abs(sim - measured))
        
        self.current_errors = np.array(errors, dtype=np.float32)
        mae = np.mean(errors)
        
        # Negative MAE as reward (higher is better)
        reward = -mae
        
        return float(reward)
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        # Normalize C-factors to [0, 1] range
        normalized_c = (self.group_c_factors - self.c_factor_min) / (self.c_factor_max - self.c_factor_min)
        
        obs = np.concatenate([
            self.current_errors,
            normalized_c,
            [self.current_step / self.max_steps]
        ])
        return obs.astype(np.float32)
    
    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        # Start with default C-factor
        self.group_c_factors = np.full(self.n_groups, self.c_factor_default, dtype=np.float32)
        
        # Run initial simulation
        simulated = self._run_simulation()
        self._calculate_reward(simulated)
        
        obs = self._get_observation()
        info = {
            "initial_mae": float(np.mean(self.current_errors)),
            "n_groups": self.n_groups,
            "group_names": self.group_names,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step by adjusting C-factors."""
        self.current_step += 1
        
        # Apply C-factor adjustments and clamp to valid range
        self.group_c_factors = np.clip(
            self.group_c_factors + action,
            self.c_factor_min,
            self.c_factor_max
        ).astype(np.float32)
        
        # Run simulation with updated C-factors
        simulated = self._run_simulation()
        reward = self._calculate_reward(simulated)
        
        mae = float(np.mean(self.current_errors))
        
        # Episode terminates if MAE < 0.05 bar
        terminated = mae < 0.05
        truncated = self.current_step >= self.max_steps
        
        # Bonus reward for solving
        if terminated:
            reward += 1.0
        
        obs = self._get_observation()
        info = {
            "mae": mae,
            "step": self.current_step,
            "c_factors": dict(zip(self.group_names, self.group_c_factors.tolist())),
            "errors": dict(zip(self.sensor_names, self.current_errors.tolist())),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            mae = float(np.mean(self.current_errors))
            print(f"Step {self.current_step}: MAE={mae:.4f} bar")
            print(f"  C-factors: {dict(zip(self.group_names, self.group_c_factors.tolist()))}")
    
    def close(self):
        """Clean up resources."""
        pass  # Nothing to clean up with WNTR


if __name__ == "__main__":
    print("Testing Roughness Calibration Environment (WNTR-based)")
    print("=" * 60)
    
    env = RoughnessCalibrationEnv()
    print(f"Sensors: {env.n_sensors}")
    print(f"Pipe Groups: {env.n_groups} - {env.group_names}")
    print(f"Measured pressures (bar): {env.measured_pressures}")
    
    obs, info = env.reset()
    print(f"\nInitial MAE: {info['initial_mae']:.4f} bar")
    print(f"Initial errors: {env.current_errors}")
    
    print("\nRunning 5 random steps...")
    for i in range(5):
        action = env.action_space.sample() * 0.5  # Smaller random adjustments
        obs, reward, term, trunc, info = env.step(action)
        print(f"  Step {i+1}: MAE={info['mae']:.4f} bar, reward={reward:.4f}")
    
    env.close()
    print("\n✓ Environment test complete!")
