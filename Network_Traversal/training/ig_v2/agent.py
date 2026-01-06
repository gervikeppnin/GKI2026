"""
Pipe Roughness Calibration IG V2 Agent.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import copy
from dataclasses import dataclass
from joblib import Parallel, delayed
from tqdm import tqdm


from sklearn.cluster import KMeans

from ...core.engine import SimulationEngine
from .config import (
    TOTAL_BUDGET_UNITS,
    COST_1_DAY_SIM,
    COST_FULL_WINDOW_SIM,
    PENALTY_LAMBDA,
    EARLY_EXIT_THRESHOLD,
    MIN_SNR_THRESHOLD,
    ROUGHNESS_MIN,
    ROUGHNESS_MAX,
    INFER_LATENT_NOISE,
    WINDOW_SIZE_DAYS,
    MAX_GROUPS,
    MAX_GP_SAMPLES
)
from .data import DataPipeline
from .surrogate import SurrogateModel

logger = logging.getLogger(__name__)

@dataclass
class Action:
    type: str # 'KEEP', 'SPLIT', 'MERGE', 'PERTURB'
    target_group: Optional[str] = None
    target_groups: Optional[List[str]] = None
    new_theta: Optional[Dict[str, float]] = None # If Perturb
    split_feature: str = 'diameter' # 'diameter' or 'slope'
    cost: float = COST_1_DAY_SIM
    
    
    def __repr__(self):
        return f"Action({self.type}, T={self.target_group or self.target_groups}, F={self.split_feature}, C={self.cost})"

import logging
import warnings

def run_single_date(date: str, pipe_roughness: Dict[str, float], data_dir: Optional[str] = None):
    """
    Helper function for parallel execution.
    Instantiates a fresh SimulationEngine for the given date and runs the simulation.
    """
    # Suppress logs in worker process
    logging.getLogger("wntr").setLevel(logging.ERROR)
    logging.getLogger("Network_Traversal").setLevel(logging.ERROR)
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Instantiate engine for this specific date
    # Note: This loads data each time. If overhead is too high, we might need shared memory or optimized loading.
    # However, for EPANET simulation time >> loading time typically.
    sim = SimulationEngine(data_dir=data_dir, date=date)
    sim.apply_pipe_roughness(pipe_roughness)
    res = sim.run_simulation()
    
    sensor_pressures = {}
    if res.success:
        # We only need the sensor pressures for the noise model update
        # We process them here to avoid sending full huge objects back
        for s, p_array in res.sensor_pressures_bar.items():
            # Truncate to 24h if needed
            if len(p_array) > 24:
                p_array = p_array[:24]
            sensor_pressures[s] = p_array
            
    return (date, res.success, res.mae, sensor_pressures)


class PipeRoughnessCalibrationIGV2:
    def __init__(
        self, 
        data_pipeline: DataPipeline, 
        sim_engine: SimulationEngine,
        initial_pipe_roughness: Optional[Dict[str, float]] = None,
        budget: float = TOTAL_BUDGET_UNITS,
        cost_full_sim: float = COST_FULL_WINDOW_SIM,
        window_size: int = WINDOW_SIZE_DAYS,
        max_gp_samples: int = MAX_GP_SAMPLES
    ):
        self.data = data_pipeline
        self.engine = sim_engine
        self.budget = budget
        self.cost_full_sim = cost_full_sim
        self.window_size = window_size
        self.max_gp_samples = max_gp_samples
        
        # State
        self.budget_used = 0.0
        self.iteration = 0
        self.history: List[Dict] = []
        
        # Structure variables
        # G: pipe_id -> group_id
        # theta: group_id -> roughness
        self.group_mapping: Dict[str, str] = {} # pipe -> group
        self.theta: Dict[str, float] = {}       # group -> value
        
        if not self.engine.wn:
            self.engine.build_network()
            
        # Initialize G0 using 'smart' heuristic
        self._initialize_structure(initial_pipe_roughness)

        # Track BEST state
        self.best_theta = copy.deepcopy(self.theta)
        self.best_group_mapping = copy.deepcopy(self.group_mapping)
        self.best_surrogate = None # Track best brain state
        
        # Adaptive Exploration State
        self.steps_since_improvement = 0
        self.current_sigma = 2.0 # Start with Fine Tuning
        self.max_stagnation = 15 # Steps before switching to Explore mode
        self.explore_sigma = 10.0
        self.fine_tune_sigma = 2.0
        
        # Initialize Surrogate
        self._rebuild_surrogate()
        
        # Metrics
        self.best_mae = float('inf')

    def _initialize_structure(self, initial_pipe_roughness: Optional[Dict[str, float]] = None):
        """Initializes G_0 and theta_0."""
        
        if initial_pipe_roughness:
            # Reconstruct groups from input values
            # This preserves the structure of the warm-start model
            value_groups = {}
            for p, r in initial_pipe_roughness.items():
                # Round to avoid float precision issues creating duplicate groups
                r_val = round(float(r), 4) 
                if r_val not in value_groups:
                    value_groups[r_val] = []
                value_groups[r_val].append(p)
            
            self.theta = {}
            self.group_mapping = {}
            
            # Create groups
            for r_val, pipes in value_groups.items():
                # Name group by its value, e.g., G_139.6003
                g_name = f"G_{r_val}"
                self.theta[g_name] = r_val
                for p in pipes:
                    self.group_mapping[p] = g_name
            
            # Ensure we cover all pipes if possible, or fallback to default for missing
            # If pipes are missing from initial_pipe_roughness but exist in self.sim.params?
            # We used pipes from the input dict.
            # We should probably ensure all pipes in self.sim are covered.
            # But specific pipes list comes from smart grouping usually.
            
            logger.info(f"Initialized with {len(self.theta)} groups (reconstructed from input values).")
            
        else:
            # Use engine's smart grouping (Default)
            groups = self.engine.get_pipe_groups(strategy="smart")
            
            self.group_mapping = {}
            self.theta = {}
            
            for g_name, pipes in groups.items():
                self.theta[g_name] = 100.0 # Default C-factor center
                for p in pipes:
                    self.group_mapping[p] = g_name
                    
            logger.info(f"Initialized with {len(self.theta)} groups using Smart strategy.")

        # Initialize Constraint Data
        # Load pipe ages for monotonicity constraint
        # Expecting sim.data['pipes'] to have 'name' and 'year'
        self.pipe_5yr_buckets = {}
        if 'pipes' in self.engine.data:
            self.pipes_df = self.engine.data['pipes'].set_index('name') 
            logger.info(f"Loaded {len(self.pipes_df)} pipes for structural operations.")
            
            pdf = self.engine.data['pipes']
            for row in pdf.itertuples():
                try:
                    p_name = str(row.name)
                    year = float(row.year) if pd.notna(row.year) else 2000
                    bucket = int((year // 5) * 5)
                    self.pipe_5yr_buckets[p_name] = bucket
                except Exception:
                    pass
        self.sorted_buckets = sorted(list(set(self.pipe_5yr_buckets.values())))
        logger.info(f"Loaded {len(self.pipe_5yr_buckets)} pipes into {len(self.sorted_buckets)} age buckets for constraints.")

        # Calculate Pipe Slopes
        # Join pipes with junctions to get elevations
        if 'junctions' in self.engine.data and 'pipes' in self.engine.data:
            j_df = self.engine.data['junctions'].set_index('name')
            elev_map = j_df['z'].to_dict()
            
            slopes = []
            for idx, row in self.pipes_df.iterrows():
                try:
                    z_start = float(elev_map.get(row['start'], 0.0))
                    z_end = float(elev_map.get(row['end'], 0.0))
                    length = float(row['length'])
                    if length < 0.1: length = 0.1 # Avoid div by zero
                    
                    slope = abs(z_start - z_end) / length
                    slopes.append(slope)
                except Exception:
                    slopes.append(0.0)
            
            self.pipes_df['slope'] = slopes
            logger.info("Calculated pipe slopes for structural splitting.")


    def _check_monotonicity_penalty(self, theta: Dict[str, float]) -> float:
        """
        Calculates penalty if age monotonicity is violated.
        Constraint: Older pipes (lower C) <= Younger pipes (higher C).
        buckets are sorted by year (1970, 1975, ...)
        Therefore: mean(bucket[i]) <= mean(bucket[i+1])
        """
        if not self.pipe_5yr_buckets:
            return 0.0
            
        # 1. Map theta to individual pipes
        # This allows the constraint to work regardless of current grouping strategy
        pipe_roughness = {}
        for pipe, group in self.group_mapping.items():
            pipe_roughness[pipe] = theta.get(group, 100.0)
            
        # 2. Compute mean per bucket
        bucket_sums = {b: 0.0 for b in self.sorted_buckets}
        bucket_counts = {b: 0 for b in self.sorted_buckets}
        
        for pipe, r in pipe_roughness.items():
            b = self.pipe_5yr_buckets.get(pipe)
            if b is not None and b in bucket_sums:
                bucket_sums[b] += r
                bucket_counts[b] += 1
                
        # 3. Check Monotonicity
        penalty = 0.0
        prev_mean = -float('inf')
        
        for b in self.sorted_buckets:
            if bucket_counts[b] == 0:
                continue
                
            curr_mean = bucket_sums[b] / bucket_counts[b]
            
            # Constraint: prev_mean (Older) <= curr_mean (Younger)
            # Violation: curr_mean < prev_mean
            if curr_mean < prev_mean:
                diff = prev_mean - curr_mean
                penalty += diff * 5.0 
            
            prev_mean = curr_mean
            
        return penalty

    def _rebuild_surrogate(self, transfer_data: Optional[Tuple] = None):
        """
        Re-initializes surrogate model for current dimensionality k.
        
        Args:
            transfer_data: Tuple (old_X, old_y, split_idx, old_length_scales) to preserve history.
        """
        k = len(self.theta)
        
        initial_length_scales = None
        
        # Knowledge Transfer: Restore history if provided
        if transfer_data:
            # Unpack with support for different modes
            # Mode 1: Split (Default if len=3) -> (old_X, old_y, split_idx)
            # Mode 2: Explicit -> (old_X, old_y, mode, *args)
            
            old_X, old_y = transfer_data[0], transfer_data[1]
            old_ls = None
            
            mode = 'SPLIT'
            args = []
            
            if isinstance(transfer_data[2], str) and transfer_data[2] == 'MERGE':
                mode = 'MERGE'
                if len(transfer_data) >= 6:
                    args = [transfer_data[3], transfer_data[4]]
                    old_ls = transfer_data[5]
            else:
                mode = 'SPLIT'
                if len(transfer_data) >= 3:
                     # Check if 4th element exists for length scales
                     # (old_X, old_y, split_idx, old_ls)
                     if len(transfer_data) >= 4:
                         old_ls = transfer_data[3]
                     args = [transfer_data[2]]
            
            logger.info(f"Rebuild Surrogate Mode: {mode}. Old X len: {len(old_X)}. First shape: {old_X[0].shape if old_X else 'N/A'}")

            new_X = []
            valid_y = []
            
            # Transfer Length Scales
            if old_ls is not None:
                try:
                    if mode == 'SPLIT':
                        idx = args[0]
                        val = old_ls[idx]
                        ls_prime = np.delete(old_ls, idx)
                        # SPLIT Logic: remove target, append A, append B.
                        ls_new = np.concatenate([ls_prime, [val, val]])
                        initial_length_scales = ls_new
                        logger.info(f"Transferred length scale for SPLIT logic (idx {idx}). New shape: {len(ls_new)}")
                        
                    elif mode == 'MERGE':
                        # MERGE Logic: remove drop_idx.
                        # (keep_idx, drop_idx)
                        _, drop_idx = args
                        ls_new = np.delete(old_ls, drop_idx)
                        initial_length_scales = ls_new
                        logger.info(f"Transferred length scales for MERGE logic. Dropped {drop_idx}. New shape: {len(ls_new)}")
                except Exception as e:
                    logger.warning(f"Failed to transfer length scales: {e}")

            for i, x in enumerate(old_X):
                try:
                    if mode == 'SPLIT':
                        idx = args[0]
                        val = x[idx]
                        x_prime = np.delete(x, idx)
                        x_new = np.concatenate([x_prime, [val, val]])
                        new_X.append(x_new)
                    elif mode == 'MERGE':
                        keep_idx, drop_idx = args
                        x_new = np.delete(x, drop_idx)
                        new_X.append(x_new)
                        
                    valid_y.append(old_y[i])
                except Exception as e:
                    logger.warning(f"Failed to transfer data point: {e}")
            
            self.surrogate = SurrogateModel(n_features=k, 
                                            bounds=(ROUGHNESS_MIN, ROUGHNESS_MAX),
                                            initial_length_scale=initial_length_scales,
                                            max_samples=self.max_gp_samples)
            
            if new_X:
                self.surrogate.X_train = new_X
                self.surrogate.y_train = valid_y
                # Fit immediately so we don't return random predictions
                try:
                    self.surrogate.gp.fit(np.array(new_X), np.array(valid_y))
                    self.surrogate.is_fitted = True
                    logger.info(f"Transferred {len(new_X)} history points to new brain structure. New shape: {new_X[0].shape if new_X else 'N/A'}")
                except Exception as e:
                    logger.error(f"Failed to retrain transferred surrogate: {e}")
        else:
            self.surrogate = SurrogateModel(n_features=k, 
                                            bounds=(ROUGHNESS_MIN, ROUGHNESS_MAX),
                                            max_samples=self.max_gp_samples)

    def step(self) -> bool:
        """
        Executes one iteration of the optimization loop.
        Returns False if budget exhausted or early exit.
        """
        if self.budget_used >= self.budget:
            logger.info("Budget exhausted.")
            return False

        self.iteration += 1
        logger.info(f"--- Iteration {self.iteration} (Budget: {self.budget_used:.1f}/{self.budget}) ---")
        
        # 1. PROPOSE actions
        actions = self._propose_actions()
        
        # 2. SCORE actions
        best_action = None
        best_score = -float('inf')
        
        current_k = len(self.theta)
        
        for action in actions:
            # Estimate Delta k
            delta_k = 0
            if action.type == 'SPLIT': delta_k = 1
            if action.type == 'MERGE': delta_k = -1
            
            # Predict IG
            # For structural changes, IG is hard to predict with fixed-dim surrogate over theta.
            # We use a heuristic:
            # - Perturb: Use Surrogate Variance
            # - Split: High heuristic score if variance in group residuals is high (not implemented in surrogate, need latent)
            # - Merge: Score if correlations high
            
            # Simplified V2 Implementation:
            # If Perturb:
            cand_theta_array = self._get_candidate_theta(action)
            ig = self.surrogate.calculate_ig(cand_theta_array, 0.0)
            
            # Reconstruct dict for constraint check
            cand_dict = {k: v for k, v in zip(self.theta.keys(), cand_theta_array)}
            
            # Calculate Constraint Penalty
            constraint_pen = self._check_monotonicity_penalty(cand_dict)
            
            # Penalize complexity and constraints
            # score = (ig / action.cost) - (PENALTY_LAMBDA * delta_k) - constraint_pen
            # Removing cost division to make IG comparable to Penalty (IG ~ 0.1-1.0, Lambda ~ 0.05)
            # If we divide by 20, IG becomes 0.005, which is < Lambda, suppressing all splits.
            score = ig - (PENALTY_LAMBDA * delta_k) - constraint_pen
            
            if score > best_score:
                best_score = score
                best_action = action
                
        if not best_action:
            logger.info("No viable actions found.")
            return False

        # Check for early exit based on low score
        # Relaxed threshold to allow recovery from constraint violations (which cause negative scores)
        # We allow negative scores if we are still exploring constraints
        # But if it is extremely low, we might want to stop. For now, set very low to allow recovery.
        if best_score < -1000.0: 
            logger.info(f"Early exit: Score {best_score:.4f} is too low.")
            return False

        # 5. SELECT FIDELITY (Duration) based on uncertainty?
        # Simulating logic: if IG is high, maybe verify with full window?
        # Specification says: "IF Surrogate_Uncertainty is High ... Duration = 1_DAY" -> fast exploration
        # "ELSE ... Duration = FULL_WINDOW" -> confirm
        # Actually it's usually opposite? High uncertainty -> Need more data (Full)?
        # Or: High uncertainty -> Quick check to see if we are in right ballpark?
        # The spec says:
        # "IF Surrogate_Uncertainty is High: Duration = 1_DAY" (Cheap exploration)
        # "ELSE: Duration = FULL_WINDOW" (High fidelity confirmation when we are confident)
        # We will follow spec.
        
        # Retrieve uncertainty of the action
        _, std = self.surrogate.predict(self._get_candidate_theta(best_action).reshape(1, -1))
        
        if std > 0.1: # Threshold for "High"
            duration = 1 # 1 day exploration
            cost = float(COST_1_DAY_SIM)
        else:
            duration = self.window_size
            cost = self.cost_full_sim
            
        best_action.cost = cost # Update cost handling
        
        # Check budget
        if self.budget_used + cost > self.budget:
            logger.info("Not enough budget for selected action.")
            return False

        # 6. RUN Simulator
        logger.info(f"Executing {best_action} with duration {duration} days.")
        self._execute_action(best_action, duration)
        
        self.budget_used += cost
        
        return True

    def _propose_actions(self) -> List[Action]:
        """Proposes a set of candidate actions."""
        actions = []
        
        # 1. Perturb (Exploration/Optimization)
        # Propose random perturbation of current best theta
        # or use acquisition function gradient.
        # Simple: Random sampling around current mean
        current_theta_vec = np.array(list(self.theta.values()))
        
        # Sample 5 PERTURB actions
        mean_theta = current_theta_vec
        
        # Log current mode occasionally
        if np.random.random() < 0.05:
             mode = "EXPLORE" if self.current_sigma > self.fine_tune_sigma + 1e-6 else "FINE TUNE"
             logger.info(f"Proposing actions in {mode} mode (Sigma={self.current_sigma})")

        for _ in range(5):
            noise = np.random.normal(0, self.current_sigma, size=mean_theta.shape) 
            cand_vec = np.clip(mean_theta + noise, ROUGHNESS_MIN, ROUGHNESS_MAX)
            
            # Reconstruct theta dict
            cand_theta = {k: v for k, v in zip(self.theta.keys(), cand_vec)}
            
            actions.append(Action(type='PERTURB', new_theta=cand_theta, cost=COST_1_DAY_SIM))
            
        # 2. Structural (Split/Merge)
        # Randomly select a group to split
        if len(self.theta) < MAX_GROUPS:
             # Find groups with > 1 pipe
             candidate_groups = []
             inverted_mapping = {}
             for p, g in self.group_mapping.items():
                 if g not in inverted_mapping: inverted_mapping[g] = []
                 inverted_mapping[g].append(p)
             
             for g, pipes in inverted_mapping.items():
                 if len(pipes) > 1:
                     candidate_groups.append(g)
                     
             if candidate_groups:
                 target_group = np.random.choice(candidate_groups)
                 # Randomly choose feature: Diameter or Slope
                 feat = np.random.choice(['diameter', 'slope'])
                 actions.append(Action(type='SPLIT', target_group=target_group, split_feature=feat, cost=COST_1_DAY_SIM))
                 
        # 3. Structural (Merge)
        # Attempt to merge similar groups (Regularization)
        # Only merge if we have enough complexity and groups are very similar
        if len(self.theta) > 5 and np.random.random() < 0.3:
            # Find closest pair
            keys = list(self.theta.keys())
            curr_vals = list(self.theta.values())
            
            best_pair = None
            min_diff = float('inf')
            
            # Sample random pairs to avoid O(N^2)
            for _ in range(10):
                i, j = np.random.choice(len(keys), 2, replace=False)
                diff = abs(curr_vals[i] - curr_vals[j])
                if diff < min_diff:
                    min_diff = diff
                    best_pair = (keys[i], keys[j])
            
            # Strict threshold: Only merge if they are effectively identical (< 1.0 C-factor)
            if best_pair and min_diff < 1.0:
                actions.append(Action(type='MERGE', target_groups=list(best_pair), cost=COST_1_DAY_SIM))
        
        return actions

    def _get_candidate_theta(self, action: Action) -> np.ndarray:
        """Extracts vector representation of theta for an action."""
        if action.new_theta:
            return np.array(list(action.new_theta.values()))
        else:
            # For structural actions, we might need mapping.
            # Fallback to current
            return np.array(list(self.theta.values()))

    def _execute_action(self, action: Action, duration: int):
        """Runs the simulation and updates models."""
        
        # Apply theta
        if action.type == 'PERTURB' and action.new_theta:
            self.theta = action.new_theta
        elif action.type == 'SPLIT':
            # Split target_group into two based on Diameter
            # 1. Identify pipes in group
            # 2. Get their diameters
            # 3. Split by median
            
            target_g = action.target_group
            pipes_in_group = [p for p, g in self.group_mapping.items() if g == target_g]
            
            if len(pipes_in_group) < 2:
                logger.warning(f"Cannot split group {target_g}: too few pipes.")
                return 

            # Get feature values (Diameter or Slope)
            feature_vals = []
            split_feat = action.split_feature or 'diameter' # default
            
            for p in pipes_in_group:
                val = 0.0
                try:
                     val = float(self.pipes_df.loc[p, split_feat])
                except:
                     # Fallback defaults
                     if split_feat == 'diameter': val = 100.0
                     else: val = 0.0
                feature_vals.append(val)
            
            median_val = np.median(feature_vals)
            
            # Create new groups
            g_a = f"{target_g}_A"
            g_b = f"{target_g}_B"
            
            # Update Mapping
            count_a = 0
            count_b = 0
            
            for i, p in enumerate(pipes_in_group):
                val = feature_vals[i]
                if val <= median_val and count_a <= len(pipes_in_group)/2: # balancing check
                    self.group_mapping[p] = g_a
                    count_a += 1
                else:
                    self.group_mapping[p] = g_b
                    count_b += 1
            
            # Initialize Theta (inherit parent value)
            parent_val = self.theta[target_g]
            
            # Capture History for Knowledge Transfer
            old_keys = list(self.theta.keys())
            split_idx = old_keys.index(target_g)
            old_X = copy.deepcopy(self.surrogate.X_train)
            old_y = copy.deepcopy(self.surrogate.y_train)
            old_ls = self.surrogate.get_length_scales()
            
            del self.theta[target_g]
            self.theta[g_a] = parent_val
            self.theta[g_b] = parent_val
            
            # Rebuild Surrogate for new dimension with history transfer
            self._rebuild_surrogate(transfer_data=(old_X, old_y, split_idx, old_ls))
            
            logger.info(f"Split {target_g} by {split_feat} -> {g_a} ({count_a}), {g_b} ({count_b})")
            
        elif action.type == 'MERGE' and action.target_groups:
            # Merge two groups into one
            g1, g2 = action.target_groups
            
            # Keep g1, remove g2 (merge g2 into g1)
            # Update Mapping
            count_moved = 0
            for p, g in self.group_mapping.items():
                if g == g2:
                    self.group_mapping[p] = g1
                    count_moved += 1
            
            # Capture History BEFORE modifying theta
            old_keys = list(self.theta.keys())
            try:
                idx1 = old_keys.index(g1)
                idx2 = old_keys.index(g2)
                
                old_X = copy.deepcopy(self.surrogate.X_train)
                old_y = copy.deepcopy(self.surrogate.y_train)
                old_ls = self.surrogate.get_length_scales()
                
                # We will keep idx1, drop idx2.
                # Update Theta
                # New value? Average? Or just g1? 
                # Since we selected them based on similarity, keeping g1 is fine.
                # Or average:
                # new_val = (self.theta[g1] + self.theta[g2]) / 2
                # self.theta[g1] = new_val
                
                del self.theta[g2]
                
                # Rebuild Surrogate
                # We need to tell it to drop idx2
                self._rebuild_surrogate(transfer_data=(old_X, old_y, 'MERGE', idx1, idx2, old_ls))
                
                logger.info(f"Merged {g2} into {g1} (Moved {count_moved} pipes).")
            except ValueError:
                logger.warning(f"Could not merge {g1}, {g2} - keys not found.")
            
        # Build vector for simulation
        # Map group_theta to per_pipe_roughness
        pipe_roughness = {}
        for pipe, group in self.group_mapping.items():
            pipe_roughness[pipe] = self.theta.get(group, 100.0)
            
        # Run Simulation
        # We need to run for 'duration' days.
        # The SimulationEngine is single-day.
        # So we run 'duration' times (or samples)
        # Spec says: "Duration = 1-Day or Full-Window"
        
        # We interpret 'duration' as how many different dates we simulate.
        dates_to_run = self.data.dates[:duration]
        
        # Parallel Execution
        # Use simple joblib parallelization
        results = Parallel(n_jobs=-1)(
            delayed(run_single_date)(date, pipe_roughness, self.engine.data_dir) 
            for date in dates_to_run
        )
        
        aggregated_score = 0.0
        all_y_sim = {} # needed for noise update
        
        for date, success, mae, sensor_pressures in results:
            if success:
                aggregated_score -= mae
                
                # Collect outputs for noise update
                for s, p_array in sensor_pressures.items():
                    if s not in all_y_sim: all_y_sim[s] = []
                    all_y_sim[s].extend(p_array)
            else:
                aggregated_score -= 5.0 # Reduced penalty (was 100) to keep MAE signal visible
                
        # Average score
        final_score = aggregated_score / len(dates_to_run)

        
        # 7. UPDATE PosteriorModel (Surrogate)
        current_theta_vec = np.array(list(self.theta.values()))
        try:
            self.surrogate.update(current_theta_vec, final_score)
        except ValueError as e:
            logger.error(f"Surrogate Update Failed: {e}")
            shapes = [x.shape for x in self.surrogate.X_train]
            logger.error(f"X_train shapes: {shapes}")
            raise e
        
        # 8. UPDATE Noise
        if INFER_LATENT_NOISE:
            # Need properly shaped y_sim.
            # data_pipeline expects a dict of aggregated arrays?
            # Actually update_noise_estimate wants residuals.
            # We collected all_y_sim. Let's form the dict.
            # If duration < full window, we only update partial info?
            # Yes.
            
            # Reformat for update_noise_estimate
            # It expects y_sim corresponding to the data it has (20 days).
            # If we only ran 1 day, we can't compare to 20 day targets easily unless we compare day-by-day.
            # Simplified: skip noise update on partial runs, or implement daily comparison.
            # We will skip for 1-day to avoid bias.
            if duration == WINDOW_SIZE_DAYS:
                # Convert lists to arrays
                y_sim_formatted = {k: np.array(v) for k, v in all_y_sim.items()}
                self.data.update_noise_estimate(y_sim_formatted)
                
        # Update current best
        current_mae = -final_score
        if current_mae < self.best_mae:
            logger.info(f"New Best Model Found! MAE: {current_mae:.4f}")
            self.best_mae = current_mae
            self.best_theta = copy.deepcopy(self.theta)
            self.best_group_mapping = copy.deepcopy(self.group_mapping)
            self.best_surrogate = copy.deepcopy(self.surrogate)
            
            # Reset Adaptive State
            self.steps_since_improvement = 0
            self.current_sigma = self.fine_tune_sigma
        else:
            # Stagnation Logic
            self.steps_since_improvement += 1
            if self.steps_since_improvement >= self.max_stagnation:
                self.current_sigma = self.explore_sigma
                if self.steps_since_improvement == self.max_stagnation:
                     logger.info("Stagnation detected. Switching to EXPLORE mode (Sigma=10.0).")
            else:
                self.current_sigma = self.fine_tune_sigma
            
        logger.info(f"Action Complete. Score: {final_score:.4f}, Sigma_n: {self.data.sigma_n:.4f}, best_mae: {self.best_mae:.4f}")

    def run(self):
        """Runs the budget loop."""
        logger.info("Starting Optimization Loop...")
        
        # Initial Evaluation of Warm Start
        if self.iteration == 0 and self.theta:
             logger.info("Performing initial evaluation of warm-start model...")
             # Run 1-day check to establish baseline
             # We use a dummy KEEP action which falls through to simulation
             # but keeps current theta.
             dummy_action = Action(type='KEEP', cost=COST_1_DAY_SIM)
             # We force duration=20 (1 day) to get a quick score
             self._execute_action(dummy_action, duration=20) 
             logger.info(f"Initial Baseline MAE: {self.best_mae:.4f}")

        # Progress Bar
        pbar = tqdm(total=self.budget, desc="Budget Used", unit="units")
        
        while self.step():
            # Update progress bar
            # We need to calculate how much budget was just used.
            # step() doesn't return cost, but self.budget_used is cumulative.
            # So we can just set pbar.n to current budget_used
            pbar.n = min(self.budget_used, self.budget)
            pbar.set_postfix({
                "Best MAE": f"{self.best_mae:.4f}",
                "Groups": len(self.theta)
            })
            pbar.refresh()
            
        pbar.close()
        logger.info("Optimization Finished.")
        
        return self.best_theta, self.best_group_mapping
