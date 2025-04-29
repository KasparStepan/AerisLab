from fall_simulator.dynamics import RigidBody6DOF
from fall_simulator.integration import rk4_step
from fall_simulator.utils import plot_trajectory
import numpy as np

def main():
    # Simulation parameters
    dt = 0.01  # Time step (s)
    total_time = 60.0  # Total simulation time (s)
    steps = int(total_time / dt)

    # Initial conditions
    mass = 80.0  # kg
    inertia_tensor = np.diag([10.0, 10.0, 5.0])  # kg·m²
    initial_position = [0.0, 0.0, 1000.0]  # 1000 meters high
    initial_velocity = [5.0, 0.0, 0.0]  # initial horizontal speed
    initial_orientation = [1.0, 0.0, 0.0, 0.0]  # unit quaternion
    initial_angular_velocity = [0.0, 0.0, 0.0]  # no initial spin

    body = RigidBody6DOF(
        mass,
        inertia_tensor,
        initial_position,
        initial_velocity,
        initial_orientation,
        initial_angular_velocity
    )

    trajectory = []

    for _ in range(steps):
        # Deploy parachute below 500m
        if body.position[2] < 500.0 and not body.parachute_deployed:
            print("Parachute deployed at altitude:", body.position[2])
            body.parachute_deployed = True

        # RK4 integration
        current_state = body.get_state()
        new_state = rk4_step(current_state, lambda s: body.derivative(), dt)
        body.set_state(new_state)

        trajectory.append(body.position.copy())

        # Stop simulation if object hits ground
        if body.position[2] <= 0.0:
            print("Object has landed.")
            break

    trajectory = np.array(trajectory)
    plot_trajectory(trajectory)

if __name__ == "__main__":
    main()
