"""
Surrogate Model for IG V2 Agent.
Uses Gaussian Process Regression to estimate pipe roughness effects and calculate Information Gain.
"""
import numpy as np
import logging
from typing import Tuple, List, Optional

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import differential_entropy, norm

from .config import LIKELIHOOD_TYPE

logger = logging.getLogger(__name__)

class SurrogateModel:
    def __init__(self, n_features: int, bounds: Tuple[float, float], initial_length_scale: Optional[np.ndarray] = None, max_samples: int = 100):
        self.n_features = n_features
        self.bounds = bounds
        self.max_samples = max_samples
        
        # Kernel: Constant * Matern(nu=2.5) + Noise
        # nu=2.5 corresponds to Matérn 5/2 (twice differentiable)
        # Bounds: (0.1, 200.0) allow for both local and global trends
        
        if initial_length_scale is not None and len(initial_length_scale) == n_features:
            ls = initial_length_scale
        else:
            ls = 10.0 * np.ones(n_features) # Default smooth physics
            
        kernel = ConstantKernel(1.0) * Matern(length_scale=ls, 
                                              length_scale_bounds=(0.1, 200.0), 
                                              nu=2.5) + \
                 WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1.0))
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=42
        )
        
        self.X_train: List[np.ndarray] = []
        self.y_train: List[float] = [] # Fitness/Likelihood scores
        self.is_fitted = False

    def update(self, theta: np.ndarray, score: float):
        """Add new observation and retrain GP."""
        self.X_train.append(theta)
        self.y_train.append(score)
        
        # Prune if necessary
        if len(self.X_train) > self.max_samples:
            self._prune_data()
        
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        
        try:
            self.gp.fit(X, y)
            self.is_fitted = True
        except Exception as e:
            logger.error(f"GP Fitting failed: {e}")

    def _prune_data(self):
        """Prune data to keep model efficient.
        Strategy: Keep top N best scores + most recent K.
        default: Top 20 best, rest recent.
        """
        if len(self.X_train) <= self.max_samples:
            return

        N_BEST = 20
        N_RECENT = self.max_samples - N_BEST
        
        # Indices
        all_indices = np.arange(len(self.y_train))
        
        # Sort by score (descending, assuming higher score = better)
        # y_train contains 'score' which in agent is -MAE (so higher is closer to 0, better)
        sorted_indices = np.argsort(self.y_train)[::-1]
        best_indices = sorted_indices[:N_BEST]
        
        # Get recent indices that are NOT in best_indices
        recent_indices = []
        # Traverse backwards
        for idx in range(len(self.y_train) - 1, -1, -1):
            if len(recent_indices) >= N_RECENT:
                break
            if idx not in best_indices:
                recent_indices.append(idx)
        
        # Combine
        keep_indices = np.concatenate([best_indices, recent_indices])
        keep_indices = np.sort(keep_indices) # optional, but kept chronological
        
        # Filter
        self.X_train = [self.X_train[i] for i in keep_indices]
        self.y_train = [self.y_train[i] for i in keep_indices]
        
        logger.info(f"Pruned GP memory. Kept {len(self.X_train)} samples "
                    f"(Best {N_BEST}, Recent {len(recent_indices)}).")

    def predict(self, theta: np.ndarray) -> Tuple[float, float]:
        """Predict mean and std for a candidate theta."""
        if not self.is_fitted:
            return 0.0, 1.0 # High uncertainty
            
        theta = theta.reshape(1, -1)
        mean, std = self.gp.predict(theta, return_std=True)
        return float(mean[0]), float(std[0])

    def calculate_ig(self, candidate_theta: np.ndarray, current_best_posterior_entropy: float) -> float:
        """
        Estimate Information Gain for a candidate action.
        IG(a) = H(Theta | D) - E_y [ H(Theta | D u {a, y}) ]
        
        Approximation:
        We use the variance of the surrogate prediction as a proxy for IG in active learning context (Uncertainty Sampling).
        Higher predictive variance -> Higher Information Gain potential.
        
        Ref: Budgeted Active Learning.
        """
        if not self.is_fitted:
            return 1.0 # High value for initial exploration
            
        _, std = self.predict(candidate_theta)
        
        # IG ~ Variance for Gaussian likelihoods
        # Scale to avoid tiny numbers
        formatted_ig = std 
        
        return formatted_ig

    def get_posterior_entropy(self) -> float:
        """Estimate current posterior entropy of the parameter space."""
        # This is a rough proxy using limited samples, in a real Bayesian methods we'd integrate.
        # Here we just return the average predictive standard deviation at known points??
        # Or better: Entropy of the GP at the optimum?
        # For simplicity in this budget loop, we might track the variance of the 'best' candidate.
        return 0.0 # Placeholder if not strictly needed for the simplified IG loop

    def get_length_scales(self) -> np.ndarray:
        """Returns the current learned length scales (inverse importance)."""
        if not self.is_fitted:
            # Return initial kernel's length scales
            k = self.gp.kernel
            # Drill down to Matern
            # Kernel structure: Product(Constant, Matern) + WhiteKernel
            # Depending on sklearn version and ops, structure varies.
            # Initial kernel passed to constructor is usually preserved in .kernel if not fitted?
            # self.gp.kernel is the initial one if not fitted.
            try:
                # Based on init: Constant * Matern + White
                # k.k1 = Constant * Matern
                # k.k1.k2 = Matern
                if hasattr(k, 'k1') and hasattr(k.k1, 'k2'):
                    return k.k1.k2.length_scale
                # Fallback
                return 10.0 * np.ones(self.n_features)
            except:
                return 10.0 * np.ones(self.n_features)
        
        # If fitted, self.gp.kernel_ is the fitted kernel
        try:
            k = self.gp.kernel_
            # Same structure: (Constant * Matern) + White
            # k.k1 = Constant * Matern
            # k.k1.k2 = Matern
            if hasattr(k, 'k1') and hasattr(k.k1, 'k2'):
                return k.k1.k2.length_scale
            # Sometimes optimization might simplify structure?
            return 10.0 * np.ones(self.n_features)
        except:
             return 10.0 * np.ones(self.n_features)
