#!/usr/bin/env python3
"""
Parallelized Constrained Black-Box Inverse Optimization for Pipe Roughness.
Features: 
- Parallelized Differential Evolution using multiprocessing Pool
- Suppressed EPANET terminal warnings
- Thread-safe temp files via UUID
"""

import numpy as np
import pandas as pd
import logging
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

from scipy.optimize import differential_evolution

from .config import DATA_DIR, ROUGHNESS_MIN, ROUGHNESS_MAX, ROUGHNESS_DEFAULT

# Silence EPANET warnings
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('wntr.epanet.toolkit').setLevel(logging.ERROR)
logging.getLogger('wntr.epanet.io').setLevel(logging.ERROR)

# Constants
DEFAULT_MAX_EVALUATIONS = 500
DEFAULT_POPULATION_SIZE = 15
CONVERGENCE_FAILURE_PENALTY = 1e8
NEGATIVE_PRESSURE_PENALTY = 2000.0

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OptimizationResult:
    best_solutions: List[Dict[str, float]]
    objective_values: List[float]
    pressure_fields: List[pd.DataFrame]
    n_evaluations: int
    converged: bool
    optimization_history: List[float]
    parameter_names: List[str]
    algorithm: str
    elapsed_time: float = 0.0

# =============================================================================
# Worker State (for multiprocessing Pool)
# =============================================================================

_worker_state = None

def _init_worker(data_dir, dates, parameter_names, pipe_groups):
    """Initialize worker process with lazy-loaded engines."""
    global _worker_state
    
    # Suppress logging in workers to keep progress bar clean
    import logging
    logging.getLogger('core.engine').setLevel(logging.WARNING)
    logging.getLogger('wntr').setLevel(logging.ERROR)
    
    _worker_state = {
        'data_dir': data_dir,
        'dates': dates,
        'params': parameter_names,
        'groups': pipe_groups,
        'engines': {}
    }

def _evaluate_theta(theta):
    """Evaluate objective across all dates in worker process."""
    global _worker_state
    from .engine import SimulationEngine
    
    theta_dict = {name: float(theta[i]) for i, name in enumerate(_worker_state['params'])}
    
    date_scores = []
    for date in _worker_state['dates']:
        # Lazy-load engine for this date
        if date not in _worker_state['engines']:
            eng = SimulationEngine(_worker_state['data_dir'], date)
            eng.build_network()
            _worker_state['engines'][date] = eng
        
        eng = _worker_state['engines'][date]
        eng.update_roughness(theta_dict, _worker_state['groups'])
        
        try:
            res = eng.run_simulation()
            if not res.success:
                return CONVERGENCE_FAILURE_PENALTY
            
            # Check for negative pressures
            if not res.all_pressures_bar.empty and (res.all_pressures_bar < -0.01).any().any():
                return NEGATIVE_PRESSURE_PENALTY
            
            # Calculate error
            err, count = 0.0, 0
            for sensor, measured in eng.measured_pressures.items():
                sim = res.sensor_pressures_bar.get(sensor)
                if sim is not None and len(sim) > 0:
                    err += (float(np.mean(sim)) - measured) ** 2
                    count += 1
            date_scores.append(err / count if count > 0 else 1e5)
        except Exception:
            return CONVERGENCE_FAILURE_PENALTY
    
    return float(np.mean(date_scores))

# =============================================================================
# Inverse Optimizer
# =============================================================================

class InverseOptimizer:
    """Parallel inverse optimizer for pipe roughness calibration."""
    
    def __init__(self, data_dir=None, date=None, dates=None, grouping_strategy='decade'):
        from .engine import SimulationEngine
        
        self.data_dir = data_dir or DATA_DIR
        self.dates = dates if dates else ([date] if date else ['011125'])
        self.grouping_strategy = grouping_strategy
        
        # Initialize to get parameter structure
        temp_engine = SimulationEngine(self.data_dir, self.dates[0])
        temp_engine.build_network()
        self.pipe_groups = temp_engine.get_pipe_groups(strategy=grouping_strategy)
        self.parameter_names = sorted(self.pipe_groups.keys())
        self.n_params = len(self.parameter_names)
        self.bounds = [(ROUGHNESS_MIN, ROUGHNESS_MAX)] * self.n_params
        
        logger.info(f"InverseOptimizer: {self.n_params} params, {len(self.dates)} dates")

    def optimize(self, algorithm='differential_evolution', max_evaluations=500, **kwargs) -> OptimizationResult:
        import time
        start_time = time.time()
        
        n_workers = min(multiprocessing.cpu_count(), 10)
        logger.info(f"Starting optimization with {n_workers} parallel workers...")
        
        # Create pool with initialized workers
        pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(self.data_dir, self.dates, self.parameter_names, self.pipe_groups)
        )
        
        try:
            if algorithm == 'differential_evolution':
                from tqdm import tqdm
                
                max_iter = max_evaluations // DEFAULT_POPULATION_SIZE
                pbar = tqdm(total=max_iter, desc="Optimizing", unit="gen",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Best: {postfix}')
                pbar.set_postfix_str("--")
                
                best_so_far = [float('inf')]  # Use list for mutability in callback
                
                def callback(xk, convergence):
                    # Update progress bar
                    pbar.update(1)
                    return False  # Don't stop
                
                result = differential_evolution(
                    _evaluate_theta,
                    bounds=self.bounds,
                    maxiter=max_iter,
                    popsize=DEFAULT_POPULATION_SIZE,
                    workers=pool.map,  # Use pool.map for parallel evaluation
                    updating='deferred',
                    polish=False,  # Disable - polish uses main process where _worker_state is None
                    seed=kwargs.get('seed'),
                    callback=callback
                )
                pbar.close()
                
                best_theta, best_val = result.x, result.fun
                print(f"\n✓ Best J(θ) = {best_val:.6f}")
            else:
                raise ValueError(f"Algorithm {algorithm} not supported.")
        finally:
            pool.close()
            pool.join()
        
        elapsed = time.time() - start_time
        theta_dict = {name: float(best_theta[i]) for i, name in enumerate(self.parameter_names)}
        
        logger.info(f"Optimization complete in {elapsed:.1f}s. Best J(θ) = {best_val:.6f}")
        
        return OptimizationResult(
            best_solutions=[theta_dict],
            objective_values=[best_val],
            pressure_fields=[],
            n_evaluations=max_evaluations,
            converged=best_val < CONVERGENCE_FAILURE_PENALTY,
            optimization_history=[],
            parameter_names=self.parameter_names,
            algorithm=algorithm,
            elapsed_time=elapsed
        )