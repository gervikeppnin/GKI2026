from wntr_model_builder import build_network_from_csv
import wntr
import pandas as pd
import numpy as np

print("Building network with 20m offset...")
# Setup with default 20m offset
wn = build_network_from_csv('Shared Materials', reservoir_head_offset=20.0)

# Check reservoir head
res = wn.get_node(wn.reservoir_name_list[0])
print(f"Reservoir '{res.name}' Head Total: {res.head_total_pattern.base_value} m")

print("\nRunning simulation...")
sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()

# Analyze negative pressures
pressures = results.node['pressure']
min_pressure = pressures.min().min()
print(f"Global Minimum Pressure: {min_pressure:.2f} m")

# Find worst nodes
worst_time = pressures.min(axis=1).idxmin()
print(f"\nWorst Time Step: {worst_time}s ({worst_time/3600:.1f}h)")

idx = worst_time
p_at_worst = pressures.loc[idx]
neg_nodes = p_at_worst[p_at_worst < 0].sort_values()

print(f"Number of nodes with negative pressure at worst time: {len(neg_nodes)}")
print("\nTop 10 worst nodes:")
for node_name, p_val in neg_nodes.head(10).items():
    node = wn.get_node(node_name)
    print(f"  {node_name:<15} Elev: {node.elevation:<6.1f} Pressure: {p_val:<6.2f}m")

# Check boundary flow / demand at this time
boundary_demand = results.node['demand'].loc[idx].sum() * 1000
print(f"\nTotal System Demand at worst time: {boundary_demand:.2f} L/s")
