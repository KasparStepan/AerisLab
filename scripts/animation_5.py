
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider


class ProjectilePhysics:
    """
    Handles the kinematics and dynamics of the system.
    Pure math. No visualization logic here.
    """
    def __init__(self, v0: float, theta_deg: float, g: float = 9.81):
        self.g = g
        self.update_parameters(v0, theta_deg)

    def update_parameters(self, v0: float, theta_deg: float):
        """Recalculate constants based on new initial conditions."""
        self.v0 = v0
        self.theta_rad = np.radians(theta_deg)

        # Pre-calculate constants for optimization
        self.vx = self.v0 * np.cos(self.theta_rad)
        self.vy_initial = self.v0 * np.sin(self.theta_rad)

        # Flight duration: T = 2*v0*sin(theta) / g
        self.t_max = (2 * self.vy_initial) / self.g

        # Max height for plot scaling: H = (v0*sin(theta))^2 / 2g
        self.h_max = (self.vy_initial ** 2) / (2 * self.g)
        self.x_max = self.vx * self.t_max

    def get_position(self, t: float) -> tuple[float, float]:
        """Returns (x, y) coordinates at time t."""
        x = self.vx * t
        y = self.vy_initial * t - 0.5 * self.g * (t ** 2)
        # Prevent math going below ground due to floating point offsets
        return x, max(0.0, y)

    def get_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Generates the full path for static plotting."""
        t_space = np.linspace(0, self.t_max, num_points)
        x = self.vx * t_space
        y = self.vy_initial * t_space - 0.5 * self.g * (t_space ** 2)
        return x, y

class ProjectileSimApp:
    """
    Manages the GUI, Animation Loop, and User Interactions.
    """
    def __init__(self):
        # 1. Initialize State
        self.physics = ProjectilePhysics(v0=20.0, theta_deg=45.0)
        self.is_paused = False
        self.current_t = 0.0
        self.dt = self.physics.t_max / 200  # Time step
        self.playback_speed = 1.0

        # 2. Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.35)
        self.fig.canvas.manager.set_window_title('Aerospace Kinematics Simulator')

        # 3. Initialize Visual Elements
        self._init_plot_elements()
        self._init_widgets()

        # 4. Start Animation
        self.ani = FuncAnimation(
            self.fig, self._anim_loop, interval=20, cache_frame_data=False
        )

    def _init_plot_elements(self):
        """Initialize line objects and limits."""
        # Static theoretical path (dashed)
        xs, ys = self.physics.get_trajectory()
        self.line_path, = self.ax.plot(xs, ys, 'k--', alpha=0.3, label='Theoretical Trajectory')

        # Dynamic projectile
        self.point_projectile, = self.ax.plot([], [], 'ro', markersize=10, label='Projectile')

        # Axis setup
        self.ax.set_title("Kinematics Visualization")
        self.ax.set_xlabel("Distance [m]")
        self.ax.set_ylabel("Altitude [m]")
        self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle=':', alpha=0.6)

        # Initial View Limits
        self._update_axis_limits()

    def _update_axis_limits(self):
        """Dynamic resizing of the view."""
        self.ax.set_xlim(-1, self.physics.x_max * 1.1)
        self.ax.set_ylim(-1, self.physics.h_max * 1.2)

    def _init_widgets(self):
        """Setup Sliders and Buttons."""
        # Define positions: [left, bottom, width, height]
        self.ax_speed = plt.axes([0.2, 0.22, 0.6, 0.03])
        self.ax_time  = plt.axes([0.2, 0.17, 0.6, 0.03])

        # Sliders
        self.s_speed = Slider(self.ax_speed, 'Speed Factor', 0.1, 5.0, valinit=1.0)
        self.s_time  = Slider(self.ax_time, 'Time [s]', 0, self.physics.t_max, valinit=0)

        # Buttons
        self.btn_play = Button(plt.axes([0.2, 0.05, 0.15, 0.05]), 'PLAY')
        self.btn_pause = Button(plt.axes([0.4, 0.05, 0.15, 0.05]), 'PAUSE')
        self.btn_reset = Button(plt.axes([0.6, 0.05, 0.15, 0.05]), 'RESET')

        # Event Connections
        self.s_speed.on_changed(self._on_speed_change)
        self.s_time.on_changed(self._on_manual_time_drag)
        self.btn_play.on_clicked(self._play)
        self.btn_pause.on_clicked(self._pause)
        self.btn_reset.on_clicked(self._reset)

    # --- Logic & Updates ---

    def _update_visuals(self, t: float):
        """Update the plot data for a specific time t."""
        # Get physics state
        x, y = self.physics.get_position(t)

        # Update graphics
        self.point_projectile.set_data([x], [y])

        # Update UI (Block events to prevent recursion loops)
        self.s_time.eventson = False
        self.s_time.set_val(t)
        self.s_time.eventson = True

        self.fig.canvas.draw_idle()

    def _anim_loop(self, frame):
        """The main ticking clock."""
        if self.is_paused:
            return self.point_projectile,

        # Calculate delta time based on speed slider
        self.current_t += self.dt * self.playback_speed

        # Loop logic
        if self.current_t > self.physics.t_max:
            self.current_t = 0

        self._update_visuals(self.current_t)
        return self.point_projectile,

    # --- Event Callbacks ---

    def _on_speed_change(self, val):
        self.playback_speed = val

    def _on_manual_time_drag(self, val):
        """User drags the time slider."""
        # We override the animation time with the slider value
        self.current_t = val
        self._update_visuals(val)
        # Optional: Auto-pause when dragging for better UX
        self.is_paused = True

    def _play(self, event):
        self.is_paused = False

    def _pause(self, event):
        self.is_paused = True

    def _reset(self, event):
        self.is_paused = True
        self.current_t = 0
        self._update_visuals(0)

    def show(self):
        plt.show()

if __name__ == "__main__":
    app = ProjectileSimApp()
    app.show()
