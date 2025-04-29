import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(trajectory):
    """
    Plots 3D trajectory of the object.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="3D Trajectory")
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D Trajectory of Falling Object')
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=135)
    plt.show()

def plot_position_vs_time(times, trajectory):
    """
    Plots X, Y, Z positions versus time.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(times, trajectory[:, 0], label='X Position', color='blue')
    axs[0].set_ylabel('X (m)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(times, trajectory[:, 1], label='Y Position', color='green')
    axs[1].set_ylabel('Y (m)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(times, trajectory[:, 2], label='Z Position', color='red')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Z (m)')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle('Position Components vs Time')
    plt.show()
