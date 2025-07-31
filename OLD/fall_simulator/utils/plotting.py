import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def plot_trajectory(trajectory, label="Trajectory", save_path=None):
    """
    Plots 3D trajectory of a body.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=label)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=135)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_position_vs_time(times, trajectory, save_path=None):
    """
    Plot X, Y, Z position components vs time.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(times, trajectory[:, 0], label='X', color='blue')
    axs[0].set_ylabel('X (m)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(times, trajectory[:, 1], label='Y', color='green')
    axs[1].set_ylabel('Y (m)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(times, trajectory[:, 2], label='Z', color='red')
    axs[2].set_ylabel('Z (m)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle('Position vs Time')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_energy_vs_time(times, kinetic, potential, spring_energy, save_path=None):
    """
    Plot energies over time: KE, PE, Spring, and total.
    """
    total = kinetic + potential + spring_energy

    plt.figure(figsize=(10, 6))
    plt.plot(times, kinetic, label='Kinetic Energy', color='blue')
    plt.plot(times, potential, label='Potential Energy', color='red')
    plt.plot(times, spring_energy, label='Spring Energy', color='green')
    plt.plot(times, total, label='Total Energy', color='black', linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energy vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_acceleration_vs_time(times, accelerations, save_path=None):
    """
    Plot X, Y, Z acceleration components vs time.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(times, accelerations[:, 0], label='Ax', color='blue')
    axs[0].set_ylabel('Ax (m/s²)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(times, accelerations[:, 1], label='Ay', color='green')
    axs[1].set_ylabel('Ay (m/s²)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(times, accelerations[:, 2], label='Az', color='red')
    axs[2].set_ylabel('Az (m/s²)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle('Payload Acceleration vs Time')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def animate_multibody_3d(trajectory_payload, trajectory_parachute, interval=30):
    """
    Animate payload and parachute motion in 3D.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    max_height = max(np.max(trajectory_payload[:, 2]), np.max(trajectory_parachute[:, 2]))
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([0, max_height + 100])

    payload_dot, = ax.plot([], [], [], 'bo', label='Payload')
    parachute_dot, = ax.plot([], [], [], 'ro', label='Parachute')
    cable_line, = ax.plot([], [], [], 'k--', linewidth=1)

    def update(frame):
        px, py, pz = trajectory_payload[frame]
        qx, qy, qz = trajectory_parachute[frame]

        payload_dot.set_data([px], [py])
        payload_dot.set_3d_properties([pz])

        parachute_dot.set_data([qx], [qy])
        parachute_dot.set_3d_properties([qz])

        cable_line.set_data([px, qx], [py, qy])
        cable_line.set_3d_properties([pz, qz])

        return payload_dot, parachute_dot, cable_line

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory_payload),
                                  interval=interval, blit=True)
    ax.set_title('3D Multibody Animation')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_velocity_vs_time(times, velocities, save_path=None):
    """
    Plot X, Y, Z velocity components vs time.

    Parameters:
    - times: np.ndarray of time points
    - velocities: np.ndarray (N, 3) of velocities
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(times, velocities[:, 0], label='Vx', color='blue')
    axs[0].set_ylabel('Vx (m/s)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(times, velocities[:, 1], label='Vy', color='green')
    axs[1].set_ylabel('Vy (m/s)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(times, velocities[:, 2], label='Vz', color='red')
    axs[2].set_ylabel('Vz (m/s)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle('Payload Velocity vs Time')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
