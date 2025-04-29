from fall_simulator.dynamics.rigid_body import RigidBody6DOF
from fall_simulator.utils.plotting import plot_trajectory
import numpy as np

def main():
    # Simulation parameters
    dt = 0.01  # time step (s)
    total_time = 10.0  # total simulation time (s)
    steps = int(total_time / dt)

    # Initial conditions
    mass = 80.0  # kg
    inertia_tensor = np.diag([10.0, 10.0, 5.0])  # kg.m²
    initial_position = [0.0, 0.0, 1000.0]  # m
    initial_velocity = [5.0, 0.0, 0.0]  # m/s
    initial_orientation = [1.0, 0.0, 0.0, 0.0]  # unit quaternion
    initial_angular_velocity = [0.0, 0.0, 0.0]  # rad/s

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
        body.update(dt)
        trajectory.append(body.position.copy())

    trajectory = np.array(trajectory)
    plot_trajectory(trajectory)

if __name__ == "__main__":
    main()