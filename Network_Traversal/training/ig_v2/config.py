"""
Configuration parameters for the IG V2 Pipe Roughness Calibration Agent.
"""
from ...core.config import (
    HYDRAULIC_TIMESTEP,
    DATA_DIR,
    ROUGHNESS_MIN,
    ROUGHNESS_MAX
)

# Temporal & Data Handling
WINDOW_SIZE_DAYS        = 20                # Total data history available
SAMPLING_FREQ_MIN       = 15                # Sensor reporting interval (minutes)
AGGREGATION_METHOD      = "MEDIAN_DIURNAL"  # Options: [MEDIAN_DIURNAL, MEAN_FULL, PEAK_ONLY]

# Noise & Inference
INFER_LATENT_NOISE      = True              # Enable if sensor noise profile is unknown
LIKELIHOOD_TYPE         = "STUDENT_T"       # Options: [GAUSSIAN, LAPLACE, STUDENT_T]
MIN_SNR_THRESHOLD       = 2.0               # Min signal-to-noise to accept update
INITIAL_SIGMA_GUESS     = 0.5               # Initial guess for sigma_n (bar, approx)

# Optimization & Budget
TOTAL_BUDGET_UNITS      = 1000.0            # Total computational credits
COST_1_DAY_SIM          = 1.0               # Credit cost for partial check (1 unit per day)
COST_FULL_WINDOW_SIM    = 1.0               # Base cost (per day unit)
PENALTY_LAMBDA          = 0.05              # Complexity penalty per group added (k)
EARLY_EXIT_THRESHOLD    = 1e-6              # Stop if (IG/Cost) < threshold
CONVERGENCE_LIMIT_ENTROPY = 0.1             # Stop if posterior entropy drops below this

# Structural
MAX_GROUPS              = 15               # Cap on number of groups to prevent explosion
MAX_GP_SAMPLES          = 100              # Max number of samples to keep in GP (pruning limit)
