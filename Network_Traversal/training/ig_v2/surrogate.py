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
    def __init__(self, n_features: int, bounds: Tuple[float, float], initial_length_scale: Optional[np.ndarray] = None):
        self.n_features = n_features
        self.bounds = bounds
        
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
        
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        
        try:
            self.gp.fit(X, y)
            self.is_fitted = True
        except Exception as e:
            logger.error(f"GP Fitting failed: {e}")

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
