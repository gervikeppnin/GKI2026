"""
Soft Actor-Critic (SAC) Training Script for Pipe Roughness Calibration.

Trains an agent to adjust pipe roughness coefficients to minimize error between
simulated and measured pressures in the water network.
"""
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from core.environment import RoughnessCalibrationEnv
from core.config import DATA_DIR, ROUGHNESS_DEFAULT

def train_sac(total_timesteps=100_000, model_name="sac_hierarchical"):
    """
    Train a Soft Actor-Critic (SAC) agent on the Roughness Calibration Environment.
    
    Args:
        total_timesteps (int): Number of steps to train.
        model_name (str): Base name for saved models.
    """
    
    # Directories
    models_dir = Path("models")
    logs_dir = Path("logs")
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    # Create Environment
    # Wrap in Monitor for logging/stats
    env = RoughnessCalibrationEnv(data_dir=DATA_DIR, max_steps=50) # 50 steps per episode
    env = Monitor(env, str(logs_dir / f"{model_name}_monitor.csv"))
    
    # Checkpoint Callback: Save every 5000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path=str(models_dir / "checkpoints"),
        name_prefix=model_name
    )
    
    # Eval Callback: Save best model based on reward
    # We use the training env for simplicity, or separate eval env
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(models_dir),
        log_path=str(logs_dir),
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    print(f"Initializing SAC Agent: {model_name}")
    print(f"Observation Space: {env.observation_space.shape}")
    print(f"Action Space: {env.action_space.shape}")
    
    # Model Configuration
    # Using default hyperparameters for SAC which are robust
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(logs_dir / "sac_tensorboard"),
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.99,
        tau=0.005,
        target_update_interval=1,
    )
    
    start_time = time.time()
    print("Starting training...")
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[checkpoint_callback, eval_callback]
    )
    
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")
    
    import json
    
    # Save Final Model
    final_path = models_dir / f"{model_name}_final"
    model.save(str(final_path))
    print(f"Saved final model weights to {final_path}.zip")

    # --- Export to JSON for Dashboard ---
    # Load the BEST model found during training (via EvalCallback)
    best_model_path = models_dir / "best_model.zip"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path} for export...")
        model = SAC.load(str(best_model_path), env=env)
    
    print("Extracting parameters for Dashboard...")
    obs, _ = env.reset()
    done = False
    
    # Run one episode to let the agent establish its preferred parameters
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
            
    # Extract info from the finished episode
    # Gym API returns info dict directly
    final_info = info 
    
    if "mae" in final_info:
        json_path = models_dir / f"{model_name}.json"
        export_data = {
            "name": "SAC Hierarchical Agent",
            "mae": float(final_info["mae"]),
            "roughness": final_info["c_factors"],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(json_path, "w") as f:
            json.dump(export_data, f, indent=4)
        print(f"Exported dashboard-ready parameters to {json_path}")
    else:
        print("Warning: Could not extract final parameters from environment info.")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC Agent for Roughness Calibration")
    parser.add_argument("--steps", type=int, default=100_000, help="Total timesteps")
    parser.add_argument("--name", type=str, default="sac_hierarchical", help="Model name")
    
    args = parser.parse_args()
    
    train_sac(args.steps, args.name)
