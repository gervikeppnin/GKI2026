"""
Training algorithms for Roughness Calibration.
"""
import numpy as np
import logging
from typing import Dict, Any, Callable, Tuple, List

logger = logging.getLogger(__name__)

def evaluate_mae(env: Any, c_factors: np.ndarray) -> float:
    """Helper to evaluate MAE for a given set of C-factors."""
    # Temporarily set env state
    original_c = env.group_c_factors.copy()
    env.group_c_factors = c_factors.astype(np.float32)
    sim = env._run_and_observe()
    # _run_and_observe returns (obs, result)
    _, result = sim
    mae = result.mae
    # Restore? Or keep? usually evaluation implies trial.
    # We'll rely on the caller to manage state or the Env to persist if desired.
    # For pure evaluation, we should probably revert. But for training steps, we keep.
    # Let's assume this is a side-effect-free check? No, difficult with Env.
    # We will let the algorithms manage state.
    env.group_c_factors = original_c
    return mae

def hill_climbing_step(env: Any, current_mae: float, mutation_scale: float = 2.0) -> Tuple[float, np.ndarray, bool]:
    """Single step of Hill Climbing."""
    old_c = env.group_c_factors.copy()
    
    # Mutate
    noise = np.random.uniform(-mutation_scale, mutation_scale, size=env.n_groups)
    new_c = np.clip(old_c + noise, env.c_factor_min, env.c_factor_max)
    
    # Apply and Evaluate
    env.group_c_factors = new_c.astype(np.float32)
    _, result = env._run_and_observe()
    new_mae = result.mae
    
    if new_mae < current_mae:
        return new_mae, new_c, True
    else:
        # Revert
        env.group_c_factors = old_c
        return current_mae, old_c, False

def gradient_descent_step(env: Any, current_mae: float, epsilon: float = 1.0, lr: float = 2.0) -> Tuple[float, np.ndarray]:
    """Single step of Gradient Estimation."""
    base_c = env.group_c_factors.copy()
    gradients = np.zeros(env.n_groups, dtype=np.float32)
    
    for i in range(env.n_groups):
        # Plus
        env.group_c_factors[:] = base_c[:]
        env.group_c_factors[i] = np.clip(base_c[i] + epsilon, env.c_factor_min, env.c_factor_max)
        env.group_c_factors[i] = np.clip(base_c[i] + epsilon, env.c_factor_min, env.c_factor_max)
        _, result = env._run_and_observe()
        mae_plus = result.mae
        
        # Minus
        env.group_c_factors[:] = base_c[:]
        env.group_c_factors[i] = np.clip(base_c[i] - epsilon, env.c_factor_min, env.c_factor_max)
        env.group_c_factors[i] = np.clip(base_c[i] - epsilon, env.c_factor_min, env.c_factor_max)
        _, result = env._run_and_observe()
        mae_minus = result.mae
        
        gradients[i] = (mae_plus - mae_minus) / (2 * epsilon)
        
    # Apply Gradient Descent
    # We want to minimize MAE, so go opposite to gradient
    update = -lr * gradients
    new_c = np.clip(base_c + update, env.c_factor_min, env.c_factor_max)
    
    env.group_c_factors = new_c.astype(np.float32)
    env.group_c_factors = new_c.astype(np.float32)
    _, result = env._run_and_observe()
    final_mae = result.mae
    
    return final_mae, new_c
