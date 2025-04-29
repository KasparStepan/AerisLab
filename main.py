from fall_simulator.dynamics import RigidBody6DOF
from fall_simulator.integration import rk4_step
from fall_simulator.multibody import Cable
from fall_simulator.utils import (
    plot_trajectory,
    plot_position_vs_time,
    plot_energy_vs_time,
    plot_acceleration_vs_time,
    animate_multibody_3d,
)
import numpy as np

def main():
    dt = 0.01
    total_time = 60.0
    steps = int(total_time / dt)

    # Create payload
    payload = RigidBody6DOF(
        mass=80.0,
        inertia_tensor=np.diag([10.0, 10.0, 5.0]),
        position=[0.0, 0.0, 1000.0],
        velocity=[5.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        area=0.5,
        drag_coefficient=0.5
    )

    # Create parachute
    parachute = RigidBody6DOF(
        mass=5.0,
        inertia_tensor=np.diag([1.0, 1.0, 0.5]),
        position=[0.0, 0.0, 995.0],
        velocity=[5.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        area=15.0,
        drag_coefficient=0.0  # parachute initially closed
    )

    # Cable between payload and parachute
    cable = Cable(rest_length=5.0, stiffness=100.0, damping=20.0)

    # Data storage
    trajectory_payload = []
    trajectory_parachute = []
    times = []

    energies_kinetic = []
    energies_potential = []
    energies_spring = []

    velocities_payload = []

    for step in range(steps):
        t = step * dt

        # Store velocity for acceleration calculation
        velocities_payload.append(payload.velocity.copy())

        # Deploy parachute based on payload vertical velocity
        v_threshold = 20.0  # m/s
        if parachute.drag_coefficient == 0.0 and payload.velocity[2] < -v_threshold:
            print(f"Parachute deployed at t = {t:.2f} s, altitude = {payload.position[2]:.2f} m")
            parachute.drag_coefficient = 0.1

        # Gradual inflation of parachute
        if parachute.drag_coefficient > 0.0 and parachute.drag_coefficient < 1.5:
            parachute.drag_coefficient = min(1.5, parachute.drag_coefficient + 0.02)

        # Cable forces
        force_on_payload = cable.compute_force(payload.position, payload.velocity,
                                               parachute.position, parachute.velocity)
        force_on_parachute = -force_on_payload

        # RK4 integration
        payload_state = payload.get_state()
        parachute_state = parachute.get_state()

        next_payload_state = rk4_step(payload_state, lambda s: payload.derivative(force_on_payload), dt)
        next_parachute_state = rk4_step(parachute_state, lambda s: parachute.derivative(force_on_parachute), dt)

        payload.set_state(next_payload_state)
        parachute.set_state(next_parachute_state)

        # Store trajectories
        trajectory_payload.append(payload.position.copy())
        trajectory_parachute.append(parachute.position.copy())
        times.append(t)

        # Energies
        ke = 0.5 * payload.mass * np.linalg.norm(payload.velocity)**2 + \
             0.5 * parachute.mass * np.linalg.norm(parachute.velocity)**2

        g = 9.81
        pe = payload.mass * g * payload.position[2] + parachute.mass * g * parachute.position[2]

        stretch = np.linalg.norm(parachute.position - payload.position) - cable.rest_length
        spring_energy = 0.5 * cable.stiffness * stretch**2

        energies_kinetic.append(ke)
        energies_potential.append(pe)
        energies_spring.append(spring_energy)

        if payload.position[2] <= 0.0:
            print(f"Payload landed at {t:.2f} seconds")
            break

    # Convert to arrays
    trajectory_payload = np.array(trajectory_payload)
    trajectory_parachute = np.array(trajectory_parachute)
    velocities_payload = np.array(velocities_payload)
    times = np.array(times)

    # Compute accelerations numerically
    accelerations_payload = np.diff(velocities_payload, axis=0) / dt
    times_accel = (times[:-1] + times[1:]) / 2

    # Plot results
    plot_trajectory(trajectory_payload, label="Payload Trajectory")
    plot_trajectory(trajectory_parachute, label="Parachute Trajectory")
    plot_position_vs_time(times, trajectory_payload)
    plot_energy_vs_time(times, np.array(energies_kinetic), np.array(energies_potential), np.array(energies_spring))
    plot_acceleration_vs_time(times_accel, accelerations_payload)
    animate_multibody_3d(trajectory_payload, trajectory_parachute)

if __name__ == "__main__":
    main()
