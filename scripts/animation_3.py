import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# 1. Physics Setup
g = 9.81
v0 = 20
theta = np.radians(45)
t_max = (2 * v0 * np.sin(theta)) / g
t_steps = np.linspace(0, t_max, 200)

# 2. Figure Setup
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.3) # Space for slider and buttons

x_all = v0 * np.cos(theta) * t_steps
y_all = v0 * np.sin(theta) * t_steps - 0.5 * g * t_steps**2
ax.plot(x_all, y_all, 'k--', alpha=0.2)
point, = ax.plot([0], [0], 'ro', markersize=10)

ax.set_xlim(0, max(x_all) + 1)
ax.set_ylim(0, max(y_all) + 1)

# 3. UI Widgets Setup
ax_slider = plt.axes([0.2, 0.15, 0.65, 0.03])
s_time = Slider(ax_slider, 'Time', 0, t_max, valinit=0)

ax_play  = plt.axes([0.2, 0.05, 0.1, 0.05])
ax_pause = plt.axes([0.35, 0.05, 0.1, 0.05])
ax_reset = plt.axes([0.5, 0.05, 0.1, 0.05])

btn_play  = Button(ax_play, '▶ Play')
btn_pause = Button(ax_pause, '⏸ Pause')
btn_reset = Button(ax_reset, '↺ Reset')

# 4. State Variables
is_manual = False
current_t = 0

def update_plot(t):
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    point.set_data([x], [y])
    s_time.set_val(t) # Move slider with animation

def anim_update(frame):
    global current_t
    if not is_manual:
        current_t += (t_max / 200) # Increment time
        if current_t > t_max: current_t = 0
        update_plot(current_t)
    return point,

# 5. Event Listeners
def play(event):
    global is_manual
    is_manual = False

def pause(event):
    global is_manual
    is_manual = True

def reset(event):
    global current_t
    current_t = 0
    update_plot(0)

def slider_change(val):
    global is_manual, current_t
    if is_manual: # Only update via slider if paused
        current_t = val
        update_plot(current_t)

btn_play.on_clicked(play)
btn_pause.on_clicked(pause)
btn_reset.on_clicked(reset)
s_time.on_changed(slider_change)

ani = FuncAnimation(fig, anim_update, interval=30, cache_frame_data=False)

plt.show()