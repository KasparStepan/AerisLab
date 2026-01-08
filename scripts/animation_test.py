import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Physics Setup
g = 9.81
v0 = 20
theta = np.radians(45)
t_max = (2 * v0 * np.sin(theta)) / g
t_steps = np.linspace(0, t_max, 100)

# Calculate all positions beforehand
x_data = v0 * np.cos(theta) * t_steps
y_data = v0 * np.sin(theta) * t_steps - 0.5 * g * t_steps**2

# 2. Plotting Setup
fig, ax = plt.subplots()
ax.set_xlim(0, max(x_data) + 2)
ax.set_ylim(0, max(y_data) + 2)
line, = ax.plot([], [], 'o-', lw=2) # The object that moves

# 3. The Update Function
def update(frame):
    # Update the data for the 'line' object
    line.set_data(x_data[:frame], y_data[:frame])
    return line,

# 4. Create Animation
ani = FuncAnimation(fig, update, frames=len(t_steps), interval=20, blit=True)

plt.title("Kinematics: Projectile Motion")
plt.xlabel("Distance (m)")
plt.ylabel("Height (m)")
plt.show()