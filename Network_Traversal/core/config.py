"""
Configuration settings for the WNTR network model.
This file serves as the single source of truth for physical constants and parameters.
"""
from pathlib import Path
from typing import Final, Dict

# Paths
BASE_DIR: Final[Path] = Path(__file__).parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data" / "consolidated"

# Hydraulic Parameters
DEMAND_MULTIPLIER: Final[float] = 1.0
HYDRAULIC_TIMESTEP: Final[int] = 3600  # 1 hour in seconds

# Reservoir Configuration
# The reservoir elevation in the CSV is ground level (21.6m).
# This offset represents the water surface height above ground.
# Note: Pump-1 adds substantial head (~91m at shutoff), so offset should be minimal.
# Testing shows 0m offset gives more realistic pressures.
RESERVOIR_HEAD_OFFSET: Final[float] = 0.0  # meters above ground elevation

# Valve Configuration
# Set to True to enable PRV/PSV valves with their CSV settings
# Set to False to convert them to open TCVs (old behavior)
ENABLE_PRESSURE_VALVES: Final[bool] = True  # Enable for proper pressure zone regulation

# Roughness Calibration - Default values when not using material-based
ROUGHNESS_MIN: Final[float] = 60.0
ROUGHNESS_MAX: Final[float] = 140.0
ROUGHNESS_DEFAULT: Final[float] = 100.0

# Material-based Hazen-Williams C coefficients
# Based on standard engineering tables for steel pipes
# Reference: Lamont (1981), Walski (1984)
PIPE_ROUGHNESS_BY_MATERIAL: Final[Dict[str, Dict]] = {
    'steel': {
        'base_c': 140,           # New steel pipe C-factor
        'age_decay_per_decade': 5,  # C reduction per decade of age
        'min_c':60,             # Minimum C-factor for very old pipes
    },
    'ductile_iron': {
        'base_c': 140,
        'age_decay_per_decade': 3,
        'min_c': 100,
    },
    'pvc': {
        'base_c': 150,
        'age_decay_per_decade': 1,
        'min_c': 140,
    },
    'concrete': {
        'base_c': 120,
        'age_decay_per_decade': 2,
        'min_c': 90,
    },
    'default': {
        'base_c': 110,
        'age_decay_per_decade': 5,
        'min_c': 80,
    },
}

# Current reference year for age calculations
REFERENCE_YEAR: Final[int] = 2026

# Unit Conversions
BAR_TO_METERS: Final[float] = 10.197  # 1 bar approx 10.197 meters head
LPS_TO_M3S: Final[float] = 0.001
MM_TO_M: Final[float] = 0.001

# File names
PIPES_CSV: Final[str] = "pipes_csv.csv"

