import json
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
model_a_path = Path("Network_Traversal/models/ig_v2_calibrated.json")
model_b_path = Path("Network_Traversal/models/ig_v2_calibrated_1.json")

def load_roughness(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['pipe_roughness']

try:
    r_a = load_roughness(model_a_path)
    r_b = load_roughness(model_b_path)
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    exit(1)

# Find common pipes
pipes_a = set(r_a.keys())
pipes_b = set(r_b.keys())
common_pipes = list(pipes_a.intersection(pipes_b))

print(f"Model A (Unnumbered): {len(r_a)} pipes")
print(f"Model B (Numbered _1): {len(r_b)} pipes")
print(f"Common Pipes: {len(common_pipes)}")

# Analysis
diffs = []
vals_a = []
vals_b = []

for p in common_pipes:
    va = r_a[p]
    vb = r_b[p]
    vals_a.append(va)
    vals_b.append(vb)
    diffs.append(va - vb)

diffs = np.array(diffs)
vals_a = np.array(vals_a)
vals_b = np.array(vals_b)

print("\n--- Statistics ---")
print(f"Mean Difference (A - B): {np.mean(diffs):.4f}")
print(f"Mean Absolute Diff:      {np.mean(np.abs(diffs)):.4f}")
print(f"Max Difference:          {np.max(np.abs(diffs)):.4f}")

print("\n--- Diversity (Unique Values) ---")
print(f"Unique Values in A: {len(np.unique(vals_a))}")
print(f"Unique Values in B: {len(np.unique(vals_b))}")
print(f"Unique Values in A (rounded to 2 decimals): {len(np.unique(np.round(vals_a, 2)))}")

# Check widely different pipes
df = pd.DataFrame({'pipe': common_pipes, 'A': vals_a, 'B': vals_b, 'diff': diffs})
df['abs_diff'] = df['diff'].abs()
df = df.sort_values('abs_diff', ascending=False)

print("\n--- Top 5 Biggest Differences ---")
print(df.head(5).to_string(index=False))
