import os
from aerislab.visualization.plotting import (
    plot_trajectory_3d,
    plot_velocity_and_acceleration,
    plot_forces,
)

def main():
    # Adjust path/body as needed
    csv_fixed = os.path.join("logs", "parachute_fixed.csv")
    body = "payload"  # or "canopy"

    # Save all plots into 'plots/' directory
    os.makedirs("plots", exist_ok=True)
    plot_trajectory_3d(csv_fixed, body, save_path=os.path.join("plots", f"{body}_traj.png"), show=False)
    plot_velocity_and_acceleration(csv_fixed, body, save_path=os.path.join("plots", f"{body}_vel_acc.png"), show=False)
    plot_forces(csv_fixed, body, save_path=os.path.join("plots", f"{body}_forces.png"), show=False)
    print("Saved plots to: plots/")

if __name__ == "__main__":
    main()
