import os
from aerislab.visualization.plotting import (
    plot_trajectory_3d,
    plot_velocity_and_acceleration,
    plot_forces,
)

def main():
    media_dir = os.path.join(os.path.dirname(__file__), "media")
    logs_dir = os.path.join(media_dir, "logs")
    plots_dir = os.path.join(media_dir, "plots")

    # Adjust path/body as needed
    csv_fixed = os.path.join(logs_dir, "parachute_fixed.csv")
    body = "payload"  # or "canopy"

    # Save all plots into 'plots/' directory
    os.makedirs(plots_dir, exist_ok=True)
    plot_trajectory_3d(csv_fixed, body, save_path=os.path.join(plots_dir, f"{body}_traj.png"), show=False)
    plot_velocity_and_acceleration(csv_fixed, body, save_path=os.path.join(plots_dir, f"{body}_vel_acc.png"), show=False)
    plot_forces(csv_fixed, body, save_path=os.path.join(plots_dir, f"{body}_forces.png"), show=False)
    print(f"Saved plots to: {plots_dir}")

if __name__ == "__main__":
    main()
