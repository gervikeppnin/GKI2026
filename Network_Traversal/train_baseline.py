#!/usr/bin/env python3
"""
Baseline training for Roughness Calibration Competition.

Uses WNTR-based simulation for hydraulic modeling.
"""

import numpy as np
from roughness_calibration_env import RoughnessCalibrationEnv


def random_baseline(env, n_episodes=3, max_steps=15):
    """Random baseline: Take random actions."""
    print("=" * 60)
    print("Random Baseline Agent")
    print("=" * 60)
    
    best_mae = float('inf')
    best_c_factors = None
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"Episode {ep+1} - Initial MAE: {info['initial_mae']:.4f} bar")
        
        for step in range(max_steps):
            # Small random adjustments to C-factors
            action = env.action_space.sample() * 0.5
            obs, reward, term, trunc, info = env.step(action)
            
            if info['mae'] < best_mae:
                best_mae = info['mae']
                best_c_factors = info['c_factors'].copy()
            
            if term:
                print(f"  Solved at step {step+1}!")
                break
        
        print(f"  Final MAE: {info['mae']:.4f} bar")
    
    print(f"\nBest MAE: {best_mae:.4f} bar")
    print(f"Best C-factors: {best_c_factors}")
    return best_mae, best_c_factors


def hill_climbing(env, n_episodes=2, max_steps=20):
    """Hill climbing: Keep improvements only."""
    print("\n" + "=" * 60)
    print("Hill Climbing Baseline")
    print("=" * 60)
    
    best_mae = float('inf')
    best_c_factors = None
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        current_mae = info['initial_mae']
        print(f"Episode {ep+1} - Initial MAE: {current_mae:.4f} bar")
        
        # Track current C-factors
        current_c = env.group_c_factors.copy()
        
        for step in range(max_steps):
            # Save state before action
            old_c = env.group_c_factors.copy()
            old_mae = current_mae
            
            # Take a small random action
            action = np.random.uniform(-2.0, 2.0, size=env.n_groups).astype(np.float32)
            obs, reward, term, trunc, info = env.step(action)
            
            if info['mae'] < current_mae:
                # Keep improvement
                current_mae = info['mae']
                current_c = env.group_c_factors.copy()
                print(f"  Step {step+1}: MAE improved to {current_mae:.4f} bar")
            else:
                # Revert by setting C-factors back
                env.group_c_factors = old_c
            
            if term:
                print(f"  Solved at step {step+1}!")
                break
        
        if current_mae < best_mae:
            best_mae = current_mae
            best_c_factors = {name: float(c) for name, c in zip(env.group_names, current_c)}
        
        print(f"  Episode {ep+1} final MAE: {current_mae:.4f} bar")
    
    print(f"\nBest MAE: {best_mae:.4f} bar")
    print(f"Best C-factors: {best_c_factors}")
    return best_mae, best_c_factors


def gradient_estimation(env, n_episodes=2, max_steps=20):
    """Gradient estimation via finite differences."""
    print("\n" + "=" * 60)
    print("Gradient Estimation Baseline")
    print("=" * 60)
    
    best_mae = float('inf')
    best_c_factors = None
    epsilon = 1.0  # Perturbation size for gradient estimation
    learning_rate = 2.0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        current_mae = info['initial_mae']
        print(f"Episode {ep+1} - Initial MAE: {current_mae:.4f} bar")
        
        for step in range(max_steps):
            # Estimate gradient for each group
            gradients = np.zeros(env.n_groups, dtype=np.float32)
            base_c = env.group_c_factors.copy()
            
            for i in range(env.n_groups):
                # Positive perturbation
                env.group_c_factors = base_c.copy()
                env.group_c_factors[i] += epsilon
                env.group_c_factors = np.clip(env.group_c_factors, env.c_factor_min, env.c_factor_max)
                sim_plus = env._run_simulation()
                env._calculate_reward(sim_plus)
                mae_plus = float(np.mean(env.current_errors))
                
                # Negative perturbation
                env.group_c_factors = base_c.copy()
                env.group_c_factors[i] -= epsilon
                env.group_c_factors = np.clip(env.group_c_factors, env.c_factor_min, env.c_factor_max)
                sim_minus = env._run_simulation()
                env._calculate_reward(sim_minus)
                mae_minus = float(np.mean(env.current_errors))
                
                # Gradient: direction that decreases MAE
                gradients[i] = (mae_plus - mae_minus) / (2 * epsilon)
            
            # Take step in negative gradient direction (to decrease MAE)
            action = -learning_rate * gradients
            
            # Apply action
            env.group_c_factors = np.clip(
                base_c + action,
                env.c_factor_min,
                env.c_factor_max
            ).astype(np.float32)
            
            # Evaluate new position
            sim = env._run_simulation()
            env._calculate_reward(sim)
            new_mae = float(np.mean(env.current_errors))
            
            if new_mae < current_mae:
                current_mae = new_mae
                print(f"  Step {step+1}: MAE improved to {current_mae:.4f} bar")
            else:
                # Reduce learning rate if no improvement
                learning_rate *= 0.8
            
            if current_mae < 0.05:
                print(f"  Solved at step {step+1}!")
                break
        
        if current_mae < best_mae:
            best_mae = current_mae
            best_c_factors = {name: float(c) for name, c in zip(env.group_names, env.group_c_factors)}
        
        print(f"  Episode {ep+1} final MAE: {current_mae:.4f} bar")
        learning_rate = 2.0  # Reset for next episode
    
    print(f"\nBest MAE: {best_mae:.4f} bar")
    print(f"Best C-factors: {best_c_factors}")
    return best_mae, best_c_factors


def main():
    print("=" * 60)
    print("ROUGHNESS CALIBRATION BASELINES (WNTR-based)")
    print("=" * 60)
    
    env = RoughnessCalibrationEnv()
    
    print(f"\nConfiguration:")
    print(f"  Sensors: {env.n_sensors}")
    print(f"  Pipe Groups: {env.n_groups}")
    print(f"  Groups: {env.group_names}")
    print(f"  Measured pressures (bar): {env.measured_pressures}")
    print(f"  C-factor range: [{env.c_factor_min}, {env.c_factor_max}]")
    
    # Run baselines
    random_mae, _ = random_baseline(env, n_episodes=2, max_steps=10)
    hill_mae, _ = hill_climbing(env, n_episodes=2, max_steps=15)
    grad_mae, grad_c = gradient_estimation(env, n_episodes=1, max_steps=10)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Random baseline MAE:      {random_mae:.4f} bar")
    print(f"Hill climbing MAE:        {hill_mae:.4f} bar")
    print(f"Gradient estimation MAE:  {grad_mae:.4f} bar")
    
    if grad_c:
        print(f"\nBest C-factors found:")
        for name, c in grad_c.items():
            print(f"  {name}: {c:.1f}")
    
    env.close()


if __name__ == "__main__":
    main()
