import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Physics Parameters
g = 9.81
v0 = 20
theta = np.radians(45)
t_max = (2 * v0 * np.sin(theta)) / g

# 2. Plot Setup
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25) # Make room for the slider

# Pre-calculate trajectory for the background "ghost" line
t_all = np.linspace(0, t_max, 100)
x_all = v0 * np.cos(theta) * t_all
y_all = v0 * np.sin(theta) * t_all - 0.5 * g * t_all**2
ax.plot(x_all, y_all, 'k--', alpha=0.3) # Trajectory path

# The actual moving point/line
line, = ax.plot([0], [0], 'ro', markersize=10)
ax.set_xlim(0, max(x_all) + 1)
ax.set_ylim(0, max(y_all) + 1)

# 3. Add Slider
ax_time = plt.axes([0.2, 0.1, 0.65, 0.03]) # [left, bottom, width, height]
s_time = Slider(ax_time, 'Time (s)', 0, t_max, valinit=0)

# 4. Update Function
def update(val):
    t = s_time.val
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    line.set_data([x], [y]) # Update point position
    fig.canvas.draw_idle()   # Redraw the figure

s_time.on_changed(update)

plt.show()