"""
Train and Save Calibration Models.
Runs optimization algorithms to find best roughness values and saves them to JSON.
"""

import json
import numpy as np
import logging
from pathlib import Path
from core.environment import RoughnessCalibrationEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def save_model(name: str, mae: float, c_factors: dict):
    """Save model to JSON."""
    filepath = MODELS_DIR / f"{name}.json"
    data = {
        "name": name,
        "mae": float(mae),
        "roughness": c_factors
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved model '{name}' (MAE: {mae:.4f}) to {filepath}")

def evaluate(env, c_factors):
    """Helper to evaluate specific C-factors."""
    original_c = env.group_c_factors.copy()
    env.group_c_factors = c_factors.astype(np.float32)
    _, result = env._run_and_observe()
    env.group_c_factors = original_c
    return result.mae

def train_hill_climbing(env: RoughnessCalibrationEnv, steps=50):
    logger.info(f"Training Hill Climbing Model ({steps} steps)...")
    obs, info = env.reset()
    current_mae = info['initial_mae']
    
    current_c = env.group_c_factors.copy()
    mutation_scale = 2.0
    
    for i in range(steps):
        # Mutate
        noise = np.random.uniform(-mutation_scale, mutation_scale, size=env.n_groups)
        candidate_c = np.clip(current_c + noise, env.c_factor_min, env.c_factor_max)
        
        # Evaluate
        candidate_mae = evaluate(env, candidate_c)
        
        # Accept if better
        if candidate_mae < current_mae:
            current_mae = candidate_mae
            current_c = candidate_c
            logger.info(f"  Step {i:02d}: New Best MAE {current_mae:.4f}")
            
    # Save
    c_map = dict(zip(env.group_names, current_c.tolist()))
    save_model("hill_climbing_best", current_mae, c_map)

def train_gradient_descent(env: RoughnessCalibrationEnv, steps=50):
    logger.info(f"Training Gradient Descent Model ({steps} steps)...")
    obs, info = env.reset()
    current_mae = info['initial_mae']
    
    current_c = env.group_c_factors.copy()
    epsilon = 1.0
    lr = 5.0
    
    for i in range(steps):
        gradients = np.zeros(env.n_groups)
        
        for g_idx in range(env.n_groups):
            # Calculate Gradient via finite difference
            base_val = current_c[g_idx]
            
            # Plus
            current_c[g_idx] = np.clip(base_val + epsilon, env.c_factor_min, env.c_factor_max)
            mae_plus = evaluate(env, current_c)
            
            # Minus
            current_c[g_idx] = np.clip(base_val - epsilon, env.c_factor_min, env.c_factor_max)
            mae_minus = evaluate(env, current_c)
            
            # Reset
            current_c[g_idx] = base_val
            
            gradients[g_idx] = (mae_plus - mae_minus) / (2 * epsilon)
            
        # Update
        update = -lr * gradients
        current_c = np.clip(current_c + update, env.c_factor_min, env.c_factor_max)
        
        # Evaluate new state
        current_mae = evaluate(env, current_c)
        
        if i % 5 == 0:
             logger.info(f"  Step {i:02d}: MAE {current_mae:.4f}")
             
    # Save final state
    c_map = dict(zip(env.group_names, current_c.tolist()))
    save_model("gradient_descent_best", current_mae, c_map)

def main():
    print("Initializing Environment...")
    env = RoughnessCalibrationEnv()
    
    train_hill_climbing(env, steps=50)
    train_gradient_descent(env, steps=30)
    
    env.close()
    print("\\nTraining Complete. Models saved in 'models/' directory.")

if __name__ == "__main__":
    main()
