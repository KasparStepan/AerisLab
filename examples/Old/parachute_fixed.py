"""
Parachute-payload system with fixed-step integration.

Demonstrates:
- Two-body system with constraint
- Parachute deployment logic
- Spring tether forces
- Fixed-step solver with Baumgarte stabilization
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Drag, Gravity, ParachuteDrag
from aerislab.dynamics.joints import RigidTetherJoint


def build_parachute_system() -> World:
    """Build two-body parachute-payload system."""
    world = World.with_logging(
        name="parachute_fixed_step",
        ground_z=0.0,
        payload_index=0,
        auto_save_plots=True
    )

    # Payload (sphere)
    payload_mass = 10.0
    payload_radius = 0.2
    I_payload = (2/5) * payload_mass * payload_radius**2 * np.eye(3)

    payload = RigidBody6DOF(
        name="payload",
        mass=payload_mass,
        inertia_tensor_body=I_payload,
        position=np.array([0.0, 0.0, 300.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        radius=payload_radius
    )

    # Canopy (light, small body)
    canopy_mass = 2.0
    I_canopy = 0.1 * np.eye(3)
    tether_length = 10.0

    canopy = RigidBody6DOF(
        name="canopy",
        mass=canopy_mass,
        inertia_tensor_body=I_canopy,
        position=np.array([0.0, 0.0, 300.0 + tether_length]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        radius=1.0
    )

    # Add bodies
    payload_idx = world.add_body(payload)
    canopy_idx = world.add_body(canopy)
    world.payload_index = payload_idx

    # Forces
    world.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))

    # Payload drag (small)
    payload.per_body_forces.append(
        Drag(rho=1.225, Cd=0.47, area=np.pi * payload_radius**2)
    )

    # Parachute drag (activates at velocity threshold)
    canopy.per_body_forces.append(
        ParachuteDrag(
            rho=1.225,
            Cd=1.5,
            area=15.0,  # 15 m² deployed area
            activation_velocity=30.0,  # Deploy at 30 m/s
            activation_altitude=200.0,  # Or at 2000m altitude
            gate_sharpness=50.0,  # Smooth deployment
            area_collapsed=0.01  # Small collapsed area
        )
    )

    # Rigid tether constraint
    tether = RigidTetherJoint(
        body_i=payload_idx,
        body_j=canopy_idx,
        attach_i_local=np.zeros(3),
        attach_j_local=np.zeros(3),
        length=tether_length
    )
    world.add_constraint(tether.attach(world.bodies))

    return world


def main():
    """Run parachute simulation."""
    print("=" * 60)
    print("Parachute Drop Test (Fixed-Step)")
    print("=" * 60)

    world = build_parachute_system()

    # Print configuration
    print("\nConfiguration:")
    print(f"  Initial altitude: {world.bodies[0].p[2]:.1f} m")
    print(f"  Payload mass: {world.bodies[0].mass:.1f} kg")
    print(f"  Canopy mass: {world.bodies[1].mass:.1f} kg")
    print(f"  Tether length: {np.linalg.norm(world.bodies[1].p - world.bodies[0].p):.1f} m")

    # Solver with strong Baumgarte stabilization
    solver = HybridSolver(alpha=10.0, beta=2.0)

    print("\nRunning simulation...")
    print("  Solver: Fixed-step semi-implicit Euler")
    print("  Time step: 0.01 s")
    print(f"  Baumgarte: α={solver.alpha}, β={solver.beta}")

    start = time.time()
    world.run(solver, duration=500.0, dt=0.01)
    elapsed = time.time() - start

    # Results
    print("\nResults:")
    print(f"  Simulation completed: {world.t:.3f} s")
    print(f"  Computation time: {elapsed:.3f} s")
    print(f"  Performance: {world.t/elapsed:.1f}x realtime")

    if world.t_touchdown:
        print(f"\n  Touchdown: {world.t_touchdown:.3f} s")
        print(f"  Payload velocity: {np.linalg.norm(world.bodies[0].v):.2f} m/s")
        print(f"  Canopy velocity: {np.linalg.norm(world.bodies[1].v):.2f} m/s")

        # Check constraint violation
        dist = np.linalg.norm(world.bodies[1].p - world.bodies[0].p)
        print(f"  Tether length: {dist:.4f} m (constraint violation: {abs(dist - 10.0):.4f} m)")

    print(f"\nOutput: {world.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
