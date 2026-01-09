"""
Simple drop test: Single body free fall with drag.

Demonstrates:
- Basic World setup with logging
- Fixed-step integration
- Automatic plot generation
"""
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity, Drag


def main():
    """Run simple drop simulation."""
    print("=" * 60)
    print("Simple Drop Test")
    print("=" * 60)
    
    # Create world with automatic logging
    world = World.with_logging(
        name="simple_drop",
        ground_z=0.0,
        payload_index=0,
        auto_save_plots=True  # Plots generated automatically
    )
    
    # Create sphere body
    mass = 10.0
    radius = 0.2
    I_sphere = (2/5) * mass * radius**2 * np.eye(3)
    
    payload = RigidBody6DOF(
        name="payload",
        mass=mass,
        inertia_tensor_body=I_sphere,
        position=np.array([0.0, 0.0, 1000.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        radius=radius
    )
    
    world.add_body(payload)
    
    # Add forces
    world.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))
    payload.per_body_forces.append(
        Drag(rho=1.225, Cd=0.47, area=np.pi * radius**2, mode="quadratic")
    )
    
    # Print initial conditions
    print(f"\nInitial Conditions:")
    print(f"  Altitude: {payload.p[2]:.1f} m")
    print(f"  Mass: {mass:.1f} kg")
    print(f"  Drag area: {np.pi * radius**2:.4f} m²")
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver = HybridSolver(alpha=5.0, beta=1.0)
    
    start = time.time()
    world.run(solver, duration=200.0, dt=0.01)
    elapsed = time.time() - start
    
    # Results
    print(f"\nResults:")
    print(f"  Simulation time: {world.t:.3f} s")
    print(f"  Wall clock time: {elapsed:.3f} s")
    print(f"  Speed: {world.t/elapsed:.1f}x realtime")
    
    if world.t_touchdown:
        print(f"  Touchdown time: {world.t_touchdown:.3f} s")
        print(f"  Final velocity: {np.linalg.norm(payload.v):.2f} m/s")
    
    # Energy diagnostic
    energy = world.get_energy()
    print(f"\nFinal Energy:")
    print(f"  Kinetic: {energy['kinetic']:.2f} J")
    print(f"  Potential: {energy['potential']:.2f} J")
    print(f"  Total: {energy['total']:.2f} J")
    
    print(f"\nOutput saved to: {world.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
