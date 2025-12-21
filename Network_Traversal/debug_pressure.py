from wntr_model_builder import build_network_from_csv
import wntr
import pandas as pd
import numpy as np

print("Building network with 20m offset...")
# Setup with default 20m offset
wn = build_network_from_csv('data', reservoir_head_offset=20.0)

# Check reservoir head
res = wn.get_node(wn.reservoir_name_list[0])
print(f"Reservoir '{res.name}' Base Head: {res.base_head:.2f} m")

print("\nRunning simulation...")
sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()

# Analyze negative pressures
# WNTR pressure is in METERS
pressures = results.node['pressure']
min_pressure = pressures.min().min()
print(f"Global Minimum Pressure: {min_pressure:.2f} m ({(min_pressure*0.098):.2f} bar)")

# Find worst nodes
worst_time = pressures.min(axis=1).idxmin()
print(f"\nWorst Time Step: {worst_time}s ({worst_time/3600:.1f}h)")

idx = worst_time
p_at_worst = pressures.loc[idx]
neg_nodes = p_at_worst[p_at_worst < 0].sort_values()

print(f"Number of nodes with negative pressure at worst time: {len(neg_nodes)}")
print("\nTop 10 worst nodes:")
print(f"{'Node':<15} {'Elev(m)':<10} {'Head(m)':<10} {'Pressure(m)':<15}")
for node_name, p_val in neg_nodes.head(10).items():
    node = wn.get_node(node_name)
    head_val = results.node['head'].loc[idx, node_name]
    print(f"{node_name:<15} {node.elevation:<10.2f} {head_val:<10.2f} {p_val:<15.2f}")

# Check boundary flow / demand at this time
boundary_demand = results.node['demand'].loc[idx].sum() * 1000
print(f"\nTotal System Demand at worst time: {boundary_demand:.2f} L/s")

print("\n=== SENSOR CHECK ===")
print(f"{'Sensor':<15} {'Elev(m)':<10} {'Sim(bar)':<10} {'Meas(bar)':<10}")
sensors_df = pd.read_csv("data/sensor_measurements.csv")
sensor_names = sensors_df['sensor'].unique()
for s in sensor_names:
    if s in wn.node_name_list:
        node = wn.get_node(s)
        sim_p = results.node['pressure'].loc[idx, s] * 0.0980665 # bar
        meas_p = sensors_df[sensors_df['sensor']==s]['pressure_avg'].mean()
        print(f"{s:<15} {node.elevation:<10.2f} {sim_p:<10.2f} {meas_p:<10.2f}")

print("\n=== PUMP CHECK ===")
# WNTR uses link results for pumps
flow = results.link['flowrate'].loc[idx] * 1000 # L/s
status = results.link['status'].loc[idx]
headloss = results.link['headloss'].loc[idx]
# For pumps, headloss is negative of head added (usually) or separate attribute?
# WNTR Link results include flowrate, velocity, headloss, status, setting.
# For pumps, head gain is often -headloss.

for pump_name in wn.pump_name_list:
    f = flow[pump_name]
    s = status[pump_name]
    h = headloss[pump_name]
    print(f"{pump_name:<10} Flow: {f:<8.2f} L/s  Status: {s:<5}  Headloss: {h:<8.2f}m")
