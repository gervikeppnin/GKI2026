#!/usr/bin/env python3
"""
Network Connection Analysis Script
Analyzes connections at each junction and creates a connectivity matrix/dataframe.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "data"

def load_data():
    """Load all network component data."""
    junctions = pd.read_csv(DATA_DIR / "junctions_csv.csv")
    pipes = pd.read_csv(DATA_DIR / "pipes_csv.csv")
    pumps = pd.read_csv(DATA_DIR / "pumps_csv.csv")
    valves = pd.read_csv(DATA_DIR / "valves_csv.csv")
    reservoirs = pd.read_csv(DATA_DIR / "reservoir_csv.csv")
    return junctions, pipes, pumps, valves, reservoirs

def analyze_connections():
    """Analyze all connections at each junction."""
    junctions, pipes, pumps, valves, reservoirs = load_data()
    
    # Initialize connection counts for each junction
    connection_data = defaultdict(lambda: {
        'pipes_in': 0,      # Pipes ending here
        'pipes_out': 0,     # Pipes starting here
        'pumps_in': 0,      # Pumps ending here  
        'pumps_out': 0,     # Pumps starting here
        'valves_in': 0,     # Valves ending here
        'valves_out': 0,    # Valves starting here
        'total_connections': 0
    })
    
    # Count pipe connections
    for _, pipe in pipes.iterrows():
        connection_data[pipe['start']]['pipes_out'] += 1
        connection_data[pipe['end']]['pipes_in'] += 1
    
    # Count pump connections
    for _, pump in pumps.iterrows():
        connection_data[pump['start']]['pumps_out'] += 1
        connection_data[pump['end']]['pumps_in'] += 1
    
    # Count valve connections
    for _, valve in valves.iterrows():
        connection_data[valve['start']]['valves_out'] += 1
        connection_data[valve['end']]['valves_in'] += 1
    
    # Calculate totals
    for node in connection_data:
        data = connection_data[node]
        data['total_pipes'] = data['pipes_in'] + data['pipes_out']
        data['total_pumps'] = data['pumps_in'] + data['pumps_out']
        data['total_valves'] = data['valves_in'] + data['valves_out']
        data['total_connections'] = (data['total_pipes'] + 
                                      data['total_pumps'] + 
                                      data['total_valves'])
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(connection_data, orient='index')
    df.index.name = 'node'
    df = df.reset_index()
    
    # Sort by total connections (descending)
    df = df.sort_values('total_connections', ascending=False)
    
    return df

def main():
    print("=" * 70)
    print("NETWORK CONNECTION ANALYSIS")
    print("=" * 70)
    
    df = analyze_connections()
    
    # Save full analysis to CSV
    output_path = DATA_DIR / "connection_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"\nFull analysis saved to: {output_path}")
    
    # Print summary statistics
    print(f"\n{'─' * 70}")
    print("SUMMARY STATISTICS")
    print(f"{'─' * 70}")
    print(f"Total nodes analyzed: {len(df)}")
    print(f"Average connections per node: {df['total_connections'].mean():.2f}")
    print(f"Max connections: {df['total_connections'].max()}")
    print(f"Min connections: {df['total_connections'].min()}")
    
    # Top 20 most connected nodes
    print(f"\n{'─' * 70}")
    print("TOP 20 MOST CONNECTED NODES")
    print(f"{'─' * 70}")
    top_20 = df.head(20)[['node', 'total_pipes', 'total_pumps', 'total_valves', 'total_connections']]
    print(top_20.to_string(index=False))
    
    # Nodes with only 1 connection (dead ends)
    dead_ends = df[df['total_connections'] == 1]
    print(f"\n{'─' * 70}")
    print(f"DEAD END NODES (1 connection): {len(dead_ends)}")
    print(f"{'─' * 70}")
    if len(dead_ends) <= 10:
        print(dead_ends[['node', 'total_pipes', 'total_pumps', 'total_valves']].to_string(index=False))
    else:
        print(dead_ends.head(10)[['node', 'total_pipes', 'total_pumps', 'total_valves']].to_string(index=False))
        print(f"... and {len(dead_ends) - 10} more")
    
    # Specific analysis for Junction-455840
    print(f"\n{'─' * 70}")
    print("ANALYSIS FOR Junction-455840")
    print(f"{'─' * 70}")
    target = df[df['node'] == 'Junction-455840']
    if not target.empty:
        row = target.iloc[0]
        print(f"  Pipes (in/out/total): {row['pipes_in']}/{row['pipes_out']}/{row['total_pipes']}")
        print(f"  Pumps (in/out/total): {row['pumps_in']}/{row['pumps_out']}/{row['total_pumps']}")
        print(f"  Valves (in/out/total): {row['valves_in']}/{row['valves_out']}/{row['total_valves']}")
        print(f"  TOTAL CONNECTIONS: {row['total_connections']}")
    else:
        print("  Junction-455840 not found in connection data!")
    
    print(f"\n{'=' * 70}")
    
    return df

if __name__ == "__main__":
    df = main()
