"""
Custom Plotting Script for AerisLab Simulation Logs

This script allows you to load simulation data from a CSV log file and generate
custom plots with full control over matplotlib figure properties.

Usage:
1. Modify the CONFIGURATION section below to point to your log file and output directory.
2. Modify the PLOT CONFIGURATION section to define what to plot and how it looks.
3. Run the script: python scripts/custom_plot.py

Dependencies:
- pandas
- matplotlib
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to the simulation CSV log file
# Example: "output/knacke_demo_20260209_091450/logs/simulation.csv"
# You should update this path to point to your specific run's log file.
LOG_FILE = "output/knacke_demo_20260209_091450/logs/simulation.csv"

# Directory where the plot will be saved
OUTPUT_DIR = "output/custom_plots"

# Name of the output file
OUTPUT_FILENAME = "custom_plot.png"

# Time range to plot (start_time, end_time). Set to None to plot the entire simulation.
# Example: TIME_RANGE = (0.0, 10.0)
TIME_RANGE = None

# ==============================================================================
# PLOT CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# Figure Properties
# These properties are passed to plt.figure()
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
# ------------------------------------------------------------------------------
FIGURE_PROPERTIES = {
    "figsize": (12, 8),       # Figure dimension (width, height) in inches.
    "dpi": 200,               # Dots per inch.
    "facecolor": "white",     # Background color.
    "edgecolor": "none",      # Border color.
    "linewidth": 0.0,         # Border width.
    "frameon": True,          # If False, suppress drawing the figure background patch.
    # "tight_layout": True,   # Adjust subplot params so that subplots are formatted nicely. 
                              # (Better to use plt.tight_layout() at the end)
}

# ------------------------------------------------------------------------------
# Axes Properties
# These properties are passed to ax.set()
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html
# ------------------------------------------------------------------------------
AXES_PROPERTIES = {
    "title": "Simulation Data", # Set a title for the axes.
    "xlabel": "Time [s]",       # Set the label for the x-axis.
    "ylabel": "Value",          # Set the label for the y-axis.
    "xlim": None,               # Set the x-axis view limits. (e.g., (0, 10))
    "ylim": None,               # Set the y-axis view limits.
    "xscale": "linear",         # Set the x-axis scale. {"linear", "log", "symlog", "logit", ...}
    "yscale": "linear",         # Set the y-axis scale. {"linear", "log", "symlog", "logit", ...}
    # "grid": True,             # Grid control is handled separately below
}

# ------------------------------------------------------------------------------
# Grid Properties
# Control grid appearance. Used if ENABLE_GRID is True.
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.grid.html
# ------------------------------------------------------------------------------
ENABLE_GRID = True
GRID_PROPERTIES = {
    "which": "major",    # "major", "minor", or "both"
    "axis": "both",      # "both", "x", or "y"
    "color": "gray",
    "linestyle": "--",
    "linewidth": 0.5,
    "alpha": 0.5,
}

# ------------------------------------------------------------------------------
# Line Properties
# Define what to plot. Each item in the list represents a line/curve on the plot.
# Each dictionary matches the arguments for ax.plot().
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
#
# Available Data Columns (based on standard RigidBody6DOF logging):
# - t: Time
# 
# For each body (e.g., 'capsule', 'canopy'):
# - Position: {body}.p_x, {body}.p_y, {body}.p_z
# - Orientation (Quaternion): {body}.q_x, {body}.q_y, {body}.q_z, {body}.q_w
# - Velocity: {body}.v_x, {body}.v_y, {body}.v_z
# - Angular Velocity: {body}.w_x, {body}.w_y, {body}.w_z
# - Total Force: {body}.f_x, {body}.f_y, {body}.f_z
# - Total Torque: {body}.tau_x, {body}.tau_y, {body}.tau_z
# - Specific Forces (if available, e.g. from aerislab/dynamics/body.py): 
#   {body}.f_{category}_x, ...
#   e.g., capsule.f_gravity_z, canopy.f_aerodynamics_z
# ------------------------------------------------------------------------------
LINE_PROPERTIES = [
    {
        "x": "t",                   # Column name for X-axis data
        "y": "capsule.p_z",         # Column name for Y-axis data
        "label": "Capsule Altitude",# Label for the legend
        "color": "blue",            # Line color
        "linestyle": "-",           # Line style: '-', '--', '-.', ':', etc.
        "linewidth": 2.0,           # Line width
        "marker": None,             # Marker style: 'o', 'x', '*', etc.
        "markersize": 6,            # Marker size
        "alpha": 1.0,               # Transparency (0.0 to 1.0)
    },
    {
        "x": "t",
        "y": "capsule.v_z",
        "label": "Capsule Vertical Velocity",
        "color": "red",
        "linestyle": "--",
        "linewidth": 1.5,
    },
]

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

def main():
    # 1. Load Data
    print(f"Loading log file: {LOG_FILE}")
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file not found at {LOG_FILE}")
        print("Please check the LOG_FILE variable in the CONFIGURATION section.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows.")
    # Print available columns to help the user choose what to plot
    print("Available columns:")
    for col in df.columns:
        print(f"  - {col}")

    # 2. Filter Data (Time Range)
    if TIME_RANGE:
        print(f"Filtering data for time range: {TIME_RANGE}")
        df = df[(df['t'] >= TIME_RANGE[0]) & (df['t'] <= TIME_RANGE[1])]

    if df.empty:
        print("Error: No data found in the specified range.")
        sys.exit(1)

    # 3. Create Figure and Axes
    print("Creating plot...")
    fig = plt.figure(**FIGURE_PROPERTIES)
    ax = fig.add_subplot(111)

    # 4. Plot Lines
    for props in LINE_PROPERTIES:
        # Create a copy to avoid modifying the global dictionary configuration
        p = props.copy()
        
        # Extract data columns
        x_col = p.pop("x") # Remove 'x' from props to pass the rest to ax.plot
        y_col = p.pop("y") # Remove 'y' from props

        if x_col not in df.columns:
            print(f"Warning: Column '{x_col}' not found. Skipping line '{p.get('label', 'unnamed')}'.")
            continue
        if y_col not in df.columns:
            print(f"Warning: Column '{y_col}' not found. Skipping line '{p.get('label', 'unnamed')}'.")
            continue

        x_data = df[x_col]
        y_data = df[y_col]

        ax.plot(x_data, y_data, **p)

    # 5. Apply Axes Properties
    # Filter out properties that are None
    clean_axes_props = {k: v for k, v in AXES_PROPERTIES.items() if v is not None}
    
    ax.set(**clean_axes_props)
    
    if ENABLE_GRID:
        ax.grid(True, **GRID_PROPERTIES)

    ax.legend(loc='best')

    # 6. Save Plot
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    
    # Optional: Show plot if running in an environment that supports it
    # plt.show()

if __name__ == "__main__":
    main()
