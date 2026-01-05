# Parachute fixed-step example using world.save_plots() for quick visuals.

from __future__ import annotations
import os
import numpy as np
import time

# Robust imports for module/script execution
try:
    from ..AerisLab import (
        World, RigidBody6DOF, Gravity, Drag, ParachuteDrag, RigidTetherJoint,
        HybridSolver, CSVLogger
    )
except ImportError:
    from aerislab import (
        World, RigidBody6DOF, Gravity, Drag, ParachuteDrag, RigidTetherJoint,
        HybridSolver, CSVLogger
    )


def build_world() -> World:
    """Create payload + canopy linked by a rigid tether (distance constraint)."""
    w = World(ground_z=0.0, payload_index=0)

    # Bodies
    I_sphere = (2 / 5) * 10.0 * 0.2**2 * np.eye(3)  # crude sphere inertia for payload
    payload = RigidBody6DOF(
        "payload", mass=10.0, inertia_tensor_body=I_sphere,
        position=np.array([0.0, 0.0, 200.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    canopy = RigidBody6DOF(
        "canopy", mass=2.0, inertia_tensor_body=0.1 * np.eye(3),
        position=np.array([0.0, 0.0, 205.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    pidx = w.add_body(payload)
    cidx = w.add_body(canopy)
    w.payload_index = pidx

    # Global gravity
    w.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))
    # Per-body forces
    payload.per_body_forces.append(Drag(rho=1.225, Cd=0.15, area=0.3, mode="quadratic"))
    canopy.per_body_forces.append(ParachuteDrag(rho=1.225, Cd=1.5, area=5, activation_velocity=30))

    # Rigid tether (fixed distance between attachment points)
    tether = RigidTetherJoint(pidx, cidx, attach_i_local=[0, 0, 0], attach_j_local=[0, 0, 0], length=5.0)
    w.add_constraint(tether.attach(w.bodies))
    return w


def main():
    world = build_world()

    # CSV logger
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", "parachute_fixed.csv")
    world.set_logger(CSVLogger(csv_path))

    # Fixed-step solver (Baumgarte stabilization)
    solver = HybridSolver(alpha=5.0, beta=0.2)

    # Run until touchdown (no contact modeling; stop on z <= ground_z)
    dt = 0.01
    world.run(solver, duration=200.0, dt=dt)

    # Report
    print(f"Fixed-step finished: t={world.t:.6f}s, touchdown≈{world.t_touchdown}")
    print(f"CSV: {csv_path}")

    # One-liner plots from CSV
    os.makedirs("plots_fixed", exist_ok=True)
    world.save_plots(csv_path, bodies=["payload", "canopy"], plots_dir="plots_fixed", show=False)
    print("Plots saved under: plots_fixed/")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Elapsed time: {end - start:.3f}s")
