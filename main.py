from fall_simulator.dynamics import RigidBody6DOF
from fall_simulator.integration import rk4_step
from fall_simulator.multibody import Cable
from fall_simulator.utils import plot_trajectory, plot_position_vs_time
import numpy as np

def main():
    dt = 0.01  # Time step (s)
    total_time = 60.0  # seconds
    steps = int(total_time / dt)

    # Payload (heavy)
    payload_mass = 80.0  # kg
    payload_inertia = np.diag([10.0, 10.0, 5.0])
    payload = RigidBody6DOF(
        mass=payload_mass,
        inertia_tensor=payload_inertia,
        position=[0.0, 0.0, 1000.0],
        velocity=[5.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        area=0.5,  # small frontal area
        drag_coefficient=0.5
    )

    # Parachute (light, high drag)
    parachute_mass = 5.0  # kg
    parachute_inertia = np.diag([1.0, 1.0, 0.5])
    parachute = RigidBody6DOF(
        mass=parachute_mass,
        inertia_tensor=parachute_inertia,
        position=[0.0, 0.0, 995.0],  # slightly above payload
        velocity=[5.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        area=15.0,  # big area
        drag_coefficient=0.0  # 0 at start, inflate later
    )

    # Cable connecting payload and parachute
    cable = Cable(
        rest_length=5.0,     # meters
        stiffness=100.0,     # N/m
        damping=20.0         # N·s/m
    )

    trajectory_payload = []
    trajectory_parachute = []
    times = []

    for step in range(steps):
        t = step * dt

        # Parachute inflation (simulate slow inflation)
        if t > 2.0:  # after 2 seconds
            parachute.drag_coefficient = min(1.5, parachute.drag_coefficient + 0.02)

        # Compute cable force
        force_on_payload = cable.compute_force(payload.position, payload.velocity, parachute.position, parachute.velocity)
        force_on_parachute = -force_on_payload

        # RK4 integration
        payload_state = payload.get_state()
        parachute_state = parachute.get_state()

        # Lambda for each body's derivative with external forces
        payload_next_state = rk4_step(payload_state, lambda s: payload.derivative(force_on_payload), dt)
        parachute_next_state = rk4_step(parachute_state, lambda s: parachute.derivative(force_on_parachute), dt)

        payload.set_state(payload_next_state)
        parachute.set_state(parachute_next_state)

        # Record trajectories
        trajectory_payload.append(payload.position.copy())
        trajectory_parachute.append(parachute.position.copy())
        times.append(t)

        # Stop simulation when payload touches ground
        if payload.position[2] <= 0.0:
            print(f"Payload landed at time {t:.2f} seconds")
            break

    trajectory_payload = np.array(trajectory_payload)
    trajectory_parachute = np.array(trajectory_parachute)
    times = np.array(times)

    # Plot results
    plot_trajectory(trajectory_payload)
    plot_position_vs_time(times, trajectory_payload)

if __name__ == "__main__":
    main()
