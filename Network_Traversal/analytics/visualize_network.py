#!/usr/bin/env python3
"""
Interactive Network Visualization Script
Visualizes the water distribution network with:
- Scroll-wheel zooming & Drag panning
- Dynamic symbol sizing
- Hydraulic Pressure Visualization (Red=Negative, Blue=Positive)
- Flow Direction Arrows (appear on zoom)
- Click-to-identify with pressure/flow data

Usage:
    python visualize_network.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path to import epanet_engine
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

try:
    from core.engine import SimulationEngine
except ImportError:
    print("Error: Could not import 'core.engine'. Make sure you are running this from the root directory or correct environment.")
    sys.exit(1)

# Data directory
DATA_DIR = ROOT_DIR / "data"

# Target junction to highlight
TARGET_JUNCTION = "Junction-455840"


def load_data():
    """Load all network component data."""
    print(f"Loading data from: {DATA_DIR}")
    try:
        junctions = pd.read_csv(DATA_DIR / "junctions_csv.csv")
        pipes = pd.read_csv(DATA_DIR / "pipes_csv.csv")
        pumps = pd.read_csv(DATA_DIR / "pumps_csv.csv")
        valves = pd.read_csv(DATA_DIR / "valves_csv.csv")
        reservoirs = pd.read_csv(DATA_DIR / "reservoir_csv.csv")
        return junctions, pipes, pumps, valves, reservoirs
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        exit(1)

def run_simulation_full():
    """Run WNTR simulation to get pressures and flows."""
    print("Running hydraulic simulation (this may take a moment)...")
    engine = SimulationEngine(data_dir=DATA_DIR)
    sim_result = engine.run_simulation()
    
    if not sim_result.success:
        print(f"Simulation failed: {sim_result.error_message}")
        return {}, {}
        
    # Get pressure at t=0
    pressures = sim_result.all_pressures_bar.iloc[0].to_dict()
    
    # Get flows at t=0
    # all_flows_lps columns are link names
    flows = sim_result.all_flows_lps.iloc[0].to_dict()
    
    return pressures, flows


def count_connections(pipes, pumps, valves, junction_name):
    """Count all connections to a junction."""
    pipe_count = len(pipes[pipes['start'] == junction_name]) + len(pipes[pipes['end'] == junction_name])
    pump_count = len(pumps[pumps['start'] == junction_name]) + len(pumps[pumps['end'] == junction_name])
    valve_count = len(valves[valves['start'] == junction_name]) + len(valves[valves['end'] == junction_name])
    return pipe_count, pump_count, valve_count


class NetworkVisualizer:
    def __init__(self):
        self.junctions, self.pipes, self.pumps, self.valves, self.reservoirs = load_data()
        self.pressures, self.flows = run_simulation_full()
        self.junction_coords = self._build_coord_lookup()
        self.labels = []
        self.markers = {}
        self.show_labels = False
        self.quiver = None 
        
    def _build_coord_lookup(self):
        """Build coordinate lookup dictionary."""
        coords = {}
        for _, row in self.junctions.iterrows():
            coords[row['name']] = (row['x'], row['y'])
        for _, row in self.reservoirs.iterrows():
            coords[row['name']] = (row['x'], row['y'])
        return coords
    
    def create_plot(self):
        """Create the interactive network plot."""
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        plt.subplots_adjust(bottom=0.1)
        
        # --- Plotting Data ---
        
        # 1. Pipes & Flow Arrows
        print("Plotting pipes and calculating flows...")
        pipe_segments = []
        
        # Arrays for Quiver (Flow Arrows)
        arrow_x, arrow_y = [], []
        arrow_u, arrow_v = [], []
        
        for _, pipe in self.pipes.iterrows():
            start, end = pipe['start'], pipe['end']
            if start in self.junction_coords and end in self.junction_coords:
                x1, y1 = self.junction_coords[start]
                x2, y2 = self.junction_coords[end]
                pipe_segments.append([(x1, y1), (x2, y2)])
                
                # Flow Arrow Logic
                flow = self.flows.get(str(pipe['name']), 0.0)
                if abs(flow) > 0.001: # Only plot meaningful flow
                    # Calculate midpoint
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # Direction vector
                    dx, dy = x2 - x1, y2 - y1
                    
                    # If flow is negative, flip direction (flow is End -> Start)
                    if flow < 0:
                        dx, dy = -dx, -dy
                        
                    # Normalize vector
                    length = np.hypot(dx, dy)
                    if length > 0:
                        dx, dy = dx/length, dy/length
                        
                    arrow_x.append(mx)
                    arrow_y.append(my)
                    arrow_u.append(dx)
                    arrow_v.append(dy)

        # Plot Pipes
        from matplotlib.collections import LineCollection
        lc = LineCollection(pipe_segments, colors='#CCCCCC', linewidths=0.5, alpha=0.6, zorder=1)
        self.ax.add_collection(lc)
        
        # Plot Arrows (Quiver)
        print(f"  Plotting {len(arrow_x)} flow arrows...")
        self.quiver = self.ax.quiver(arrow_x, arrow_y, arrow_u, arrow_v, 
                                   color='#555555', scale=30, width=0.002, 
                                   headwidth=4, headlength=5, pivot='mid', zorder=1.5,
                                   alpha=0.6)
        
        # 2. Junctions (Colored by Pressure)
        print("Plotting junctions...")
        
        # Prepare pressure colors
        junction_colors = []
        neg_count = 0
        for name in self.junctions['name']:
            p = self.pressures.get(str(name), 0.0)
            if p < 0:
                junction_colors.append('red')
                neg_count += 1
            else:
                junction_colors.append('blue')
        
        print(f"  Found {neg_count} junctions with negative pressure.")
        
        self.markers['junctions'] = self.ax.scatter(
            self.junctions['x'], self.junctions['y'], 
            c=junction_colors, s=5, alpha=0.7, edgecolors='none', zorder=2, label='Junctions (Blue=Pos, Red=Neg)'
        )
        
        # 3. Reservoirs
        print("Plotting reservoirs...")
        res_x, res_y = self.reservoirs['x'], self.reservoirs['y']
        self.markers['reservoirs'] = self.ax.scatter(
            res_x, res_y, c='green', s=100, marker='s', edgecolors='darkgreen', linewidth=2, zorder=5, label='Reservoirs'
        )
        for _, res in self.reservoirs.iterrows():
            lbl = self.ax.annotate(res['name'], (res['x'], res['y']), xytext=(5, 5), textcoords="offset points", 
                                   fontsize=9, color='darkgreen', fontweight='bold', visible=False)
            self.labels.append(lbl)

        # 4. Pumps
        print("Plotting pumps...")
        pump_x, pump_y = [], []
        for _, pump in self.pumps.iterrows():
            if pump['start'] in self.junction_coords and pump['end'] in self.junction_coords:
                x1, y1 = self.junction_coords[pump['start']]
                x2, y2 = self.junction_coords[pump['end']]
                pump_x.append((x1+x2)/2)
                pump_y.append((y1+y2)/2)
                self.ax.plot([x1, x2], [y1, y2], 'black', linewidth=1.5, alpha=0.8, zorder=4)

        self.markers['pumps'] = self.ax.scatter(
            pump_x, pump_y, c='black', s=60, marker='D', edgecolors='white', linewidth=1.0, zorder=6, label='Pumps'
        )
        
        # 5. Valves
        print("Plotting valves...")
        valve_x, valve_y = [], []
        for _, valve in self.valves.iterrows():
            if valve['start'] in self.junction_coords and valve['end'] in self.junction_coords:
                x1, y1 = self.junction_coords[valve['start']]
                x2, y2 = self.junction_coords[valve['end']]
                valve_x.append((x1+x2)/2)
                valve_y.append((y1+y2)/2)
                self.ax.plot([x1, x2], [y1, y2], 'orange', linewidth=1.5, alpha=0.8, zorder=4)

        self.markers['valves'] = self.ax.scatter(
            valve_x, valve_y, c='orange', s=60, marker='^', edgecolors='darkorange', linewidth=1.5, zorder=6, label='Valves'
        )

        # 6. Highlight Target
        if TARGET_JUNCTION in self.junction_coords:
            tx, ty = self.junction_coords[TARGET_JUNCTION]
            tp = self.pressures.get(TARGET_JUNCTION, 0.0)
            self.ax.scatter(tx, ty, c='yellow', s=400, marker='*', edgecolors='black', linewidth=1.5, zorder=10, label=TARGET_JUNCTION)
            
            pc, pmc, vc = count_connections(self.pipes, self.pumps, self.valves, TARGET_JUNCTION)
            
            self.ax.annotate(f"{TARGET_JUNCTION}\n({pc} pipes, {pmc} pumps)\nP: {tp:.2f} bar", 
                             (tx, ty), xytext=(10, 10), textcoords="offset points",
                             fontsize=10, fontweight='bold', color='black',
                             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.8), zorder=11)

        # --- Interactive Features ---
        self.ax.autoscale()
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.2)
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('Easting')
        self.ax.set_ylabel('Northing')
        self.ax.set_title("Water Distribution Network Panel (Red=Neg P, Zoom for Flow Arrows)")

        # Events
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.ax.callbacks.connect('xlim_changed', self.update_marker_sizes)
        self.ax.callbacks.connect('ylim_changed', self.update_marker_sizes)
        
        # Click Annotation (Reusable)
        self.click_annotation = self.ax.annotate("", xy=(0,0), xytext=(10, 10), textcoords="offset points",
                                               bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
                                               fontsize=10, fontweight='bold', zorder=20)
        self.click_annotation.set_visible(False)

        # Toggle Button
        self.btn_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.btn = Button(self.btn_ax, 'Toggle Labels')
        self.btn.on_clicked(self.toggle_labels)
        
        # Setup arrays for identification
        self.setup_lookup_arrays()

        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
    
    def setup_lookup_arrays(self):
        # Arrays for fast click lookup
        self.all_nodes_x = self.junctions['x'].values
        self.all_nodes_y = self.junctions['y'].values
        self.all_nodes_names = self.junctions['name'].values

    def identify_node(self, x, y):
        """ Find closest node to (x,y) and show label with pressure """
        dists = np.sqrt((self.all_nodes_x - x)**2 + (self.all_nodes_y - y)**2)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        xlim = self.ax.get_xlim()
        view_width = xlim[1] - xlim[0]
        threshold = view_width * 0.05 
        
        if min_dist < threshold:
            name = self.all_nodes_names[min_idx]
            nx, ny = self.all_nodes_x[min_idx], self.all_nodes_y[min_idx]
            pressure = self.pressures.get(str(name), 0.0)
            
            label_text = f"{name}\nP: {pressure:.2f} bar"
            
            self.click_annotation.xy = (nx, ny)
            self.click_annotation.set_text(label_text)
            self.click_annotation.set_visible(True)
            
            if pressure < 0:
                self.click_annotation.get_bbox_patch().set_edgecolor('red')
                self.click_annotation.get_bbox_patch().set_linewidth(2)
            else:
                self.click_annotation.get_bbox_patch().set_edgecolor('blue')
                self.click_annotation.get_bbox_patch().set_linewidth(1)
                
        else:
            self.click_annotation.set_visible(False)

    def update_marker_sizes(self, event_ax):
        """ Dynamically update marker sizes based on visual span of the plot """
        if event_ax != self.ax: return
        
        xlim = self.ax.get_xlim()
        view_width = xlim[1] - xlim[0]
        
        # Optimization: Panning doesn't change zoom level, so skip updates if width is stable
        if hasattr(self, 'last_view_width') and abs(self.last_view_width - view_width) < 1.0:
            return
            
        self.last_view_width = view_width
        
        scale_factor = 2000 / max(view_width, 100) 
        scale_factor = np.clip(scale_factor, 0.5, 12.0)
        
        sizes = {
            'junctions': 6 * scale_factor,
            'reservoirs': 80 * scale_factor,
            'pumps': 60 * scale_factor,
            'valves': 60 * scale_factor
        }
        
        for kind, collection in self.markers.items():
            if collection:
                collection.set_sizes([sizes[kind]])
        
        # Update Arrow Scale and Visibility
        if self.quiver:
            if view_width > 5000: # Zoomed out heavily
                 self.quiver.set_alpha(0.0) # Hide
            else:
                 self.quiver.set_alpha(0.6) # Show
                 # Dynamic scale: smaller value = larger arrows in quiver
                 # We want them to look consistent or slightly growing.
                 # Inverse relationship with zoom level essentially.
                 q_scale = max(view_width / 20.0, 10.0)
                 self.quiver.scale = q_scale
                
    def on_scroll(self, event):
        """ Zoom functionality """
        if event.inaxes != self.ax: return
        scale = 1.1 if event.button == 'up' else 1/1.1
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_width = (xlim[1] - xlim[0]) * scale
        new_height = (ylim[1] - ylim[0]) * scale
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self.press = event.xdata, event.ydata
        self.cur_xlim = self.ax.get_xlim()
        self.cur_ylim = self.ax.get_ylim()
        self.is_dragging = False

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax: return
        dx = event.xdata - self.press[0]
        dy = event.ydata - self.press[1]
        
        if abs(dx) > 0 or abs(dy) > 0:
            self.is_dragging = True
            
        self.cur_xlim -= dx
        self.cur_ylim -= dy
        self.ax.set_xlim(self.cur_xlim)
        self.ax.set_ylim(self.cur_ylim)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.press is None: return
        if not getattr(self, 'is_dragging', False) and event.inaxes == self.ax:
            self.identify_node(event.xdata, event.ydata)
        self.press = None
        self.is_dragging = False
        self.fig.canvas.draw_idle()

    def toggle_labels(self, event):
        self.show_labels = not self.show_labels
        for lbl in self.labels:
            lbl.set_visible(self.show_labels)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

def main():
    print("Initializing Visualizer...")
    viz = NetworkVisualizer()
    print("Creating plot...")
    viz.create_plot()
    print("Displaying...")
    # Trigger initial update for markers
    viz.update_marker_sizes(viz.ax)
    viz.show()

if __name__ == "__main__":
    main()
