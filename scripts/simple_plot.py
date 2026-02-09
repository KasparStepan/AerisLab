"""
Simplified Plotting Script for AerisLab

A concise script to plot simulation results.
Modify the variables below to change the log file or plot appearance.
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
LOG_FILE = "output/knacke_demo_20260209_091450/logs/simulation.csv"
OUTPUT_FILE = "output/custom_plots/knacke_capsule_height.png"
FONT_SIZE = 12 # Adjust font size here

# Apply font size
plt.rcParams.update({'font.size': FONT_SIZE})

# Load Data
print(f"Loading {LOG_FILE}...")
df = pd.read_csv(LOG_FILE)

# Print available columns for reference
# print("Columns:", list(df.columns))

# --- Plotting ---
# Create figure and axes
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

# Add your lines here!
# Usage: ax.plot(x_data, y_data, label="Label", color="color", linestyle="--")
ax.plot(df['t'], df['capsule.p_z'], label='Payload Altitude', color='blue')
#ax.plot(df['t'], df['capsule.v_z'], label='Vertical Velocity', color='red', linestyle='--')

# Customize Axes
ax.set_title("Payload Altitude | Knacke parachute model")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Height [m]")
ax.minorticks_on()
ax.set_xlim(0, 35)
ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.5)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
ax.legend()

# Save
plt.tight_layout()
plt.savefig(OUTPUT_FILE)
print(f"Saved plot to {OUTPUT_FILE}")
