from fall_simulator.dynamics import RigidBody6DOF
from fall_simulator.integration import fully_implicit_step
from fall_simulator.multibody import Cable
from fall_simulator.utils import *
import numpy as np


def compute_parachute_drag_coefficient(t_since_deploy, v):
    Cd_steady = 0.8
    Cd_peak = 2 * Cd_steady  # 1.6
    n = 0.8
    D = 4.0

    # Avoid division by zero
    v = max(abs(v), 0.1)

    t_inflation = n * D / v
    t_breathing = 0.2

    if t_since_deploy < t_inflation:
        # Stage 1: Inflation
        s = t_since_deploy / t_inflation
        s = np.clip(s, 0.0, 1.0)
        Cd = Cd_peak * (3 * s**2 - 2 * s**3)
    elif t_since_deploy < t_inflation + t_breathing:
        # Stage 2: Breathing
        s = (t_since_deploy - t_inflation) / t_breathing
        s = np.clip(s, 0.0, 1.0)
        Cd = Cd_peak - (Cd_peak - Cd_steady) * (s**2)
    else:
        # Stage 3: Steady
        Cd = Cd_steady

    return Cd


def main():
    dt = 0.01
    total_time = 60.0
    steps = int(total_time / dt)

    # Create payload
    payload = RigidBody6DOF(
        mass=80.0,
        inertia_tensor=np.diag([10.0, 10.0, 5.0]),
        position=[0.0, 0.0, 4000.0],
        velocity=[0.0, 0.0, -50.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        area=0.5,
        drag_coefficient=0.05
    )

    # Create parachute
    parachute = RigidBody6DOF(
        mass=5.0,
        inertia_tensor=np.diag([1.0, 1.0, 0.5]),
        position=[0.0, 0.0, 3995.0],
        velocity=[0.0, 0.0, -50.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        area=1.414 * 18,
        drag_coefficient=0.0  # parachute initially closed
    )

    cable = Cable(rest_length=5.0, stiffness=0.0, damping=100.0, active=True)


    trajectory_payload = []
    trajectory_parachute = []
    times = []

    energies_kinetic = []
    energies_potential = []
    energies_spring = []

    velocities_payload = []

    # New state variables for parachute model
    parachute_deployed = False
    deploy_time = None

    for step in range(steps):
        t = step * dt
        velocities_payload.append(payload.velocity.copy())

        v_threshold =60.0  # m/s
        v_vertical = payload.velocity[2]

        # Check for deployment trigger
        if not parachute_deployed and v_vertical < -v_threshold:
            parachute_deployed = True
            deploy_time = t
            print(f"Parachute deployed at t = {t:.2f} s, altitude = {payload.position[2]:.2f} m")

        # Update drag coefficient based on deployment stage
        if parachute_deployed:
            time_since_deploy = t - deploy_time
            v_mag = np.linalg.norm(parachute.velocity)
            parachute.drag_coefficient = compute_parachute_drag_coefficient(time_since_deploy, v_mag)

        # Cable forces
        force_on_payload = cable.compute_force(payload.position, payload.velocity,
                                               parachute.position, parachute.velocity)
        force_on_parachute = -force_on_payload

        # RK4 integration
        payload_state = payload.get_state()
        parachute_state = parachute.get_state()

        next_payload_state = fully_implicit_step(payload_state,lambda s: payload.derivative(force_on_payload),dt,payload.mass)

        next_parachute_state = fully_implicit_step(parachute_state,lambda s: parachute.derivative(force_on_parachute),dt,parachute.mass)

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
    plot_position_vs_time(times, trajectory_payload, save_path="outputs/position_vs_time.png")
    plot_energy_vs_time(times, np.array(energies_kinetic), np.array(energies_potential), np.array(energies_spring), save_path="outputs/energy_vs_time.png")
    plot_acceleration_vs_time(times_accel, accelerations_payload, save_path="outputs/acceleration_vs_time.png")
    plot_velocity_vs_time(times, velocities_payload, save_path="outputs/velocity_vs_time.png")
    create_pdf_report("outputs/flight_report.pdf")


if __name__ == "__main__":
    main()