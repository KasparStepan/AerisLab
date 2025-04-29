from fall_simulator.dynamics import RigidBody6DOF
from fall_simulator.integration import rk4_step
from fall_simulator.utils import plot_trajectory, plot_position_vs_time
import numpy as np

def main():
    dt = 0.01  # Time step (s)
    total_time = 60.0  # Total simulation time (s)
    steps = int(total_time / dt)

    mass = 80.0
    inertia_tensor = np.diag([10.0, 10.0, 5.0])
    initial_position = [0.0, 0.0, 1000.0]
    initial_velocity = [5.0, 0.0, 0.0]
    initial_orientation = [1.0, 0.0, 0.0, 0.0]
    initial_angular_velocity = [0.0, 0.0, 0.0]

    body = RigidBody6DOF(
        mass,
        inertia_tensor,
        initial_position,
        initial_velocity,
        initial_orientation,
        initial_angular_velocity
    )

    trajectory = []
    times = []

    for step in range(steps):
        t = step * dt

        # Deploy parachute
        if body.position[2] < 500.0 and not body.parachute_deployed:
            print(f"Parachute deployed at altitude: {body.position[2]:.2f} m at time {t:.2f} s")
            body.parachute_deployed = True

        # RK4 Integration
        current_state = body.get_state()
        new_state = rk4_step(current_state, lambda s: body.derivative(), dt)
        body.set_state(new_state)

        # Store data
        trajectory.append(body.position.copy())
        times.append(t)

        # Stop if object reaches the ground
        if body.position[2] <= 0.0:
            print(f"Object has landed at time {t:.2f} s")
            break

    trajectory = np.array(trajectory)
    times = np.array(times)

    # Plot results
    plot_trajectory(trajectory)
    plot_position_vs_time(times, trajectory)

if __name__ == "__main__":
    main()
