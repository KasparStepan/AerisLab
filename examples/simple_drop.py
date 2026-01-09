"""
Example script demonstrating a simple simulation of a body falling under gravity with drag.
"""
import os
import sys
import numpy as np

# Add src to path so we can import aerislab without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from aerislab.core import World, HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity, Drag

def main():
    # 1. Configure the simulation world
    # We set a ground plane at z=0. The simulation will stop when the payload hits the ground.
    media_dir = os.path.join(os.path.dirname(__file__), "media")
    logs_dir = os.path.join(media_dir, "logs")
    plots_dir = os.path.join(media_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    output_file = os.path.join(logs_dir, "simple_drop.csv")
    world = World(ground_z=0.0, log_enabled=True, log_file=output_file)

    # 2. Create a rigid body
    # Arguments: name, mass, inertia, position (p), orientation (q)
    # Mass = 1.0 kg
    # Inertia = Identity
    # Position = 100m height
    # Orientation = Identity quaternion [x, y, z, w]
    body = RigidBody6DOF(
        "payload",
        1.0,
        np.eye(3),
        np.array([0.0, 0.0, 100.0]),
        np.array([0.0, 0.0, 0.0, 1.0])
    )
    world.add_body(body)

    # 3. Add Forces
    # Gravity pointing down
    gravity = Gravity(np.array([0.0, 0.0, -9.81]))
    world.add_global_force(gravity)

    # Drag (Quadratic)
    # F = -0.5 * rho * Cd * A * v^2
    drag = Drag(rho=1.225, Cd=0.47, area=0.1, mode='quadratic')
    world.add_global_force(drag)

    # 4. Run Simulation
    # We use the HybridSolver (Fixed-step)
    solver = HybridSolver()
    dt = 0.01  # 10ms step
    duration = 20.0 # Max duration

    print(f"Starting simulation. Output: {output_file}")
    world.run(solver, duration=duration, dt=dt)

    if world.t_touchdown:
        print(f"Touchdown detected at t = {world.t_touchdown:.4f} s")
    else:
        print("Simulation finished (max duration reached).")

    print(f"Results saved to {output_file}")

    # Generate plots
    world.save_plots(output_file, plots_dir=plots_dir)

if __name__ == "__main__":
    main()
