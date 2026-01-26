import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# 1. Physics Setup
g = 9.81
v0 = 20
theta = np.radians(45)
t_max = (2 * v0 * np.sin(theta)) / g
dt_base = t_max / 200  # Time increment per frame

# 2. Figure and Plot Setup
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.35)

# Draw the theoretical path (dashed line)
t_path = np.linspace(0, t_max, 100)
x_path = v0 * np.cos(theta) * t_path
y_path = v0 * np.sin(theta) * t_path - 0.5 * g * t_path**2
ax.plot(x_path, y_path, 'k--', alpha=0.3, label='Path')

# The moving object
point, = ax.plot([0], [0], 'ro', markersize=10, label='Projectile')

ax.set_xlim(0, max(x_path) + 2)
ax.set_ylim(0, max(y_path) + 2)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Height (m)")
ax.legend()

# 3. UI Widgets Setup
# [left, bottom, width, height]
ax_speed  = plt.axes([0.2, 0.22, 0.6, 0.03])
ax_time   = plt.axes([0.2, 0.17, 0.6, 0.03])
ax_play   = plt.axes([0.2, 0.05, 0.15, 0.05])
ax_pause  = plt.axes([0.4, 0.05, 0.15, 0.05])
ax_reset  = plt.axes([0.6, 0.05, 0.15, 0.05])

s_speed = Slider(ax_speed, 'Speed (x)', 0.1, 5.0, valinit=1.0)
s_time  = Slider(ax_time, 'Time (s)', 0, t_max, valinit=0)

# Use plain text to avoid Glyph/Font errors
btn_play  = Button(ax_play, 'PLAY')
btn_pause = Button(ax_pause, 'PAUSE')
btn_reset = Button(ax_reset, 'RESET')

# 4. Logic State
is_paused = True
current_t = 0

def update_visuals(t):
    """Updates the position of the point and the slider handle."""
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    point.set_data([x], [y])

    # Crucial: Temporarily disable slider events to prevent recursion crash
    s_time.eventson = False
    s_time.set_val(t)
    s_time.eventson = True

    fig.canvas.draw_idle()

def anim_step(frame):
    """The main loop for the animation."""
    global current_t
    if not is_paused:
        # Progress time based on speed slider
        current_t += dt_base * s_speed.val

        # Loop back to start if it hits the end
        if current_t >= t_max:
            current_t = 0

        update_visuals(current_t)
    return point,

# 5. Event Callbacks
def press_play(event):
    global is_paused
    is_paused = False

def press_pause(event):
    global is_paused
    is_paused = True

def press_reset(event):
    global current_t, is_paused
    is_paused = True
    current_t = 0
    update_visuals(0)

def manual_time_change(val):
    """Triggered when the user drags the slider."""
    global current_t
    # We only want the slider to manually override if the animation is paused
    if is_paused:
        current_t = val
        # Update point directly (don't call update_visuals to avoid set_val loop)
        x = v0 * np.cos(theta) * val
        y = v0 * np.sin(theta) * val - 0.5 * g * val**2
        point.set_data([x], [y])
        fig.canvas.draw_idle()

# Connect UI elements to functions
btn_play.on_clicked(press_play)
btn_pause.on_clicked(press_pause)
btn_reset.on_clicked(press_reset)
s_time.on_changed(manual_time_change)

# 6. Start Animation
# interval is in milliseconds
ani = FuncAnimation(fig, anim_step, interval=20, cache_frame_data=False)

plt.title("Kinematics Visualization: Projectile Motion")
plt.show()
