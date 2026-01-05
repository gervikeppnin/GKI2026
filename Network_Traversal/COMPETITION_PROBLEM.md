# Veitur Pipe Roughness Calibration Competition

## Problem Overview

Water distribution networks are complex systems of pipes, pumps, valves, and reservoirs that deliver water to consumers. Accurate hydraulic simulation of these networks is essential for operations, planning, and leak detection. However, simulations require **pipe roughness coefficients** that are often unknown or change over time due to aging, corrosion, and deposits.

**Your Challenge**: Develop an RL agent that learns to calibrate pipe roughness coefficients by matching simulated pressures to real sensor measurements.

## The Problem

You are given:
1. **Network Topology**: A water distribution network with ~2,350 junctions and ~2,400 pipes
2. **Pipe Properties**: Material (steel), installation year, diameter, length
3. **Sensor Measurements**: Hourly pressure readings from 8 IoT sensors
4. **Boundary Conditions**: Inflow rates at the reservoir (used to scale network demand)

You must:
1. Adjust pipe roughness values (Hazen-Williams C-factors) to minimize the difference between simulated and measured pressures
2. Learn a generalizable relationship between pipe characteristics (age, material) and roughness
3. Do this efficiently within a limited number of simulation steps

## Dynamic Demand
The simulation uses a **time varying demand pattern** derived from the boundary flow data (`boundary_flow.csv`). The base demand of the network is scaled hourly to match the total system inflow, creating realistic pressure fluctuations throughout the day.

## Why This Matters

- **Real-world impact**: Accurate calibration improves leak detection, pressure management, and energy efficiency
- **Transferable knowledge**: Models that learn pipe aging patterns can be applied to other networks
- **Industry need**: Water utilities spend significant resources on manual calibration

## Technical Details

### Environment

```python
import gymnasium as gym
from roughness_calibration_env import RoughnessCalibrationEnv

env = RoughnessCalibrationEnv()
obs, info = env.reset()
```

### State Space (15 dimensions)
- **Pressure errors** (8 values): Current difference between simulated and measured pressure at each sensor (bar)
- **Normalized C-factors** (6 values): Current Hazen-Williams C-factor for each pipe group, normalized to [0,1]
- **Step fraction** (1 value): Progress through episode (0 to 1)

### Action Space (6 dimensions, continuous)
- Adjustment to Hazen-Williams C-factor for each pipe group
- Range: [-5.0, +5.0] per step
- C-factors clamped to [80, 140]

### Hazen-Williams C-Factor
The C-factor represents pipe roughness:
- **C = 140**: Very smooth (new plastic or lined pipe)
- **C = 120**: Typical aged steel (default starting value)
- **C = 100**: Moderately corroded
- **C = 80**: Very rough/corroded

Higher C = smoother pipe = less head loss

### Pipe Groups
Pipes are grouped by installation decade:
- `steel_1970`: Pipes installed 1970-1979 (159 pipes)
- `steel_1980`: Pipes installed 1980-1989 (233 pipes)
- `steel_1990`: Pipes installed 1990-1999 (494 pipes)
- `steel_2000`: Pipes installed 2000-2009 (1,111 pipes)
- `steel_2010`: Pipes installed 2010-2019 (141 pipes)
- `steel_2020`: Pipes installed 2020-2029 (242 pipes)

### Reward Function
```python
reward = -mean_absolute_error(simulated_pressure, measured_pressure)
```

Lower MAE = higher reward. Target: MAE < 0.05 bar

### Episode
- **Maximum steps**: 50 actions per episode
- **Termination**: Early termination if MAE < 0.05 bar (with +1.0 bonus reward)
- **Truncation**: Episode ends after 50 steps

## Evaluation Criteria

Submissions will be evaluated on:

1. **Final MAE** (50%): Mean Absolute Error between simulated and measured pressures on held-out test days
2. **Efficiency** (25%): Number of simulation steps required to achieve good calibration
3. **Generalization** (25%): Performance on networks/days not seen during training

## Provided Resources

| File | Description |
|------|-------------|
| `roughness_calibration_env.py` | Gymnasium RL environment |
| `epanet_engine.py` | EPANET-based network simulation engine |
| `calibration_dashboard.py` | Interactive Streamlit dashboard for visualization |
| `train_baseline.py` | Baseline training script |
| `data/*.csv` | Network data and sensor measurements |

## Getting Started

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install gymnasium numpy pandas wntr streamlit plotly

# Test the environment
python roughness_calibration_env.py

# Run baselines
python train_baseline.py

# Launch interactive dashboard
streamlit run calibration_dashboard.py
```

## Submission Format

Submit a Python script that:
1. Implements a function `train_agent(env)` that returns a trained policy
2. Implements a function `evaluate_agent(env, policy)` that returns (final_mae, steps_taken)

```python
def train_agent(env):
    """Train your agent and return a policy function."""
    # Your training code here
    return policy

def evaluate_agent(env, policy):
    """Evaluate the policy and return metrics."""
    obs, info = env.reset()
    steps = 0
    done = False
    
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
    
    return info['mae'], steps
```

## Baseline Performance

| Method | Best MAE (bar) | Steps |
|--------|---------------|-------|
| Random | Variable | 50 |
| Hill Climbing | Variable | 50 |
| Gradient Estimation | Variable | 50 |

Can you beat the baselines with a smarter approach?

## Hints for Competitors

1. **Grouping Strategy**: Consider alternative ways to group pipes (by diameter, location, flow rate)
2. **Policy Architecture**: Neural networks can learn complex roughness patterns
3. **Reward Shaping**: Add intermediate rewards for improvement
4. **Multi-objective**: Balance accuracy with minimal roughness changes
5. **Physics-Informed**: Incorporate pipe aging models into your approach
6. **Exploration**: The pressure-roughness relationship is non-linear; explore the action space wisely

## Technical Notes

### EPANET Simulation
This competition uses [EPANET 2.2](https://www.epa.gov/water-research/epanet) for hydraulic simulation via WNTR Python bindings. EPANET provides:
- Industry-standard hydraulic solver
- Full valve support (PRV, PSV, PBV, FCV, TCV)
- Accurate pressure and flow calculations

### Hazen-Williams vs Darcy-Weisbach
The model uses Hazen-Williams headloss formula. C-factors are calculated based on pipe material and age using standard engineering tables.

## Prize Categories

- **Best Overall**: Lowest MAE on test data
- **Most Efficient**: Fewest simulation steps to achieve calibration
- **Most Innovative**: Novel approach to roughness learning
- **Best Documentation**: Clearest explanation of methodology

---

**Good luck!**

*This competition is organized by Veitur in collaboration with the ML/RL community.*
