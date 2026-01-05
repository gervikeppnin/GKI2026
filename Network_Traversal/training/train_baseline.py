#!/usr/bin/env python3
"""
Baseline training for Roughness Calibration Competition.
Refactored to use shared training utilities.
"""

import numpy as np
import logging
from core.environment import RoughnessCalibrationEnv
from .utils import hill_climbing_step, gradient_descent_step

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_random_baseline(env: RoughnessCalibrationEnv, n_episodes: int = 3, max_steps: int = 15):
    logger.info("=" * 60)
    logger.info("Random Baseline Agent")
    logger.info("=" * 60)
    
    best_overall_mae = float('inf')
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        logger.info(f"Episode {ep+1} - Initial MAE: {info['initial_mae']:.4f} bar")
        
        for step in range(max_steps):
            action = env.action_space.sample() * 0.5
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info['mae'] < best_overall_mae:
                best_overall_mae = info['mae']
            
            if terminated:
                logger.info(f"  Solved at step {step+1}!")
                break
                
        logger.info(f"  Final MAE: {info['mae']:.4f} bar")
        
    logger.info(f"\nBest MAE: {best_overall_mae:.4f} bar")
    return best_overall_mae

def run_hill_climbing(env: RoughnessCalibrationEnv, n_episodes: int = 2, max_steps: int = 20):
    logger.info("\n" + "=" * 60)
    logger.info("Hill Climbing Baseline")
    logger.info("=" * 60)
    
    best_overall_mae = float('inf')
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        current_mae = info['initial_mae']
        logger.info(f"Episode {ep+1} - Initial MAE: {current_mae:.4f} bar")
        
        for step in range(max_steps):
            # Use utility function logic pattern
            # Note: The utility function `hill_climbing_step` expects us to manage state or it manages it?
            # In `training_utils.py` I defined `hill_climbing_step(env, current_mae)` which modifies env in place.
            # It returns new_mae, new_c, success.
            new_mae, new_c, success = hill_climbing_step(env, current_mae)
            
            if success:
                current_mae = new_mae
                logger.info(f"  Step {step+1}: MAE improved to {current_mae:.4f} bar")
            
            if current_mae < 0.05:
                logger.info(f"  Solved at step {step+1}!")
                break
                
        if current_mae < best_overall_mae:
            best_overall_mae = current_mae
            
    logger.info(f"\nBest MAE: {best_overall_mae:.4f} bar")
    return best_overall_mae

def run_gradient_estimation(env: RoughnessCalibrationEnv, n_episodes: int = 2, max_steps: int = 20):
    logger.info("\n" + "=" * 60)
    logger.info("Gradient Estimation Baseline")
    logger.info("=" * 60)
    
    best_overall_mae = float('inf')
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        current_mae = info['initial_mae']
        logger.info(f"Episode {ep+1} - Initial MAE: {current_mae:.4f} bar")
        
        for step in range(max_steps):
            current_mae, _ = gradient_descent_step(env, current_mae)
            logger.info(f"  Step {step+1}: MAE: {current_mae:.4f} bar")
            
            if current_mae < 0.05:
                logger.info(f"  Solved at step {step+1}!")
                break

        if current_mae < best_overall_mae:
            best_overall_mae = current_mae

    logger.info(f"\nBest MAE: {best_overall_mae:.4f} bar")
    return best_overall_mae

def main():
    env = RoughnessCalibrationEnv()
    
    run_random_baseline(env)
    run_hill_climbing(env)
    run_gradient_estimation(env)
    
    env.close()

if __name__ == "__main__":
    main()
