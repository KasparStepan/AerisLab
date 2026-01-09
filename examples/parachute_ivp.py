"""
Parachute-payload system with adaptive IVP integration.

Demonstrates:
- Variable-step stiff solver (Radau)
- Terminal event detection
- High-accuracy integration
- Comparison with fixed-step method
"""
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aerislab.core.simulation import World
from aerislab.core.solver import HybridIVPSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity, Drag, ParachuteDrag
from aerislab.dynamics.joints import RigidTetherJoint


def build_parachute_system() -> World:
    """Build two-body parachute-payload system."""
    world = World.with_logging(
        name="parachute_ivp",
        ground_z=0.0,
        payload_index=0,
        auto_save_plots=True
    )
    
    # Payload
    payload_mass = 10.0
    payload_radius = 0.2
    I_payload = (2/5) * payload_mass * payload_radius**2 * np.eye(3)
    
    payload = RigidBody6DOF(
        name="payload",
        mass=payload_mass,
        inertia_tensor_body=I_payload,
        position=np.array([0.0, 0.0, 3000.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        radius=payload_radius
    )
    
    # Canopy
    canopy_mass = 2.0
    I_canopy = 0.1 * np.eye(3)
    tether_length = 10.0
    
    canopy = RigidBody6DOF(
        name="canopy",
        mass=canopy_mass,
        inertia_tensor_body=I_canopy,
        position=np.array([0.0, 0.0, 3000.0 + tether_length]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        radius=1.0
    )
    
    # Add bodies
    payload_idx = world.add_body(payload)
    canopy_idx = world.add_body(canopy)
    world.payload_index = payload_idx
    
    # Forces
    world.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))
    
    payload.per_body_forces.append(
        Drag(rho=1.225, Cd=0.47, area=np.pi * payload_radius**2)
    )
    
    canopy.per_body_forces.append(
        ParachuteDrag(
            rho=1.225,
            Cd=1.5,
            area=15.0,
            activation_velocity=30.0,
            activation_altitude=2000.0,
            gate_sharpness=50.0,  # Smooth for IVP solver
            area_collapsed=0.01
        )
    )
    
    # Rigid tether
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
    """Run parachute simulation with IVP solver."""
    print("=" * 60)
    print("Parachute Drop Test (Adaptive IVP)")
    print("=" * 60)
    
    world = build_parachute_system()
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Initial altitude: {world.bodies[0].p[2]:.1f} m")
    print(f"  Payload mass: {world.bodies[0].mass:.1f} kg")
    print(f"  Canopy mass: {world.bodies[1].mass:.1f} kg")
    
    # High-accuracy IVP solver
    solver = HybridIVPSolver(
        method="Radau",  # Implicit Runge-Kutta (stiff)
        rtol=1e-6,       # Relative tolerance
        atol=1e-8,       # Absolute tolerance
        alpha=10.0,      # Baumgarte position
        beta=2.0,        # Baumgarte velocity
        max_step=0.5     # Maximum step size
    )
    
    print(f"\nRunning simulation...")
    print(f"  Solver: {solver.method} (adaptive)")
    print(f"  Tolerances: rtol={solver.rtol:.0e}, atol={solver.atol:.0e}")
    print(f"  Baumgarte: α={solver.alpha}, β={solver.beta}")
    
    start = time.time()
    sol = world.integrate_to(solver, t_end=500.0)
    elapsed = time.time() - start
    
    # Results
    print(f"\nResults:")
    print(f"  Status: {'Success' if sol.success else 'Failed'} ({sol.message})")
    print(f"  Simulation time: {world.t:.3f} s")
    print(f"  Computation time: {elapsed:.3f} s")
    print(f"  Performance: {world.t/elapsed:.1f}x realtime")
    print(f"  Time steps: {len(sol.t)}")
    print(f"  Avg step size: {world.t / len(sol.t):.4f} s")
    
    if world.t_touchdown:
        print(f"\n  Touchdown: {world.t_touchdown:.3f} s")
        print(f"  Payload velocity: {np.linalg.norm(world.bodies[0].v):.2f} m/s")
        print(f"  Canopy velocity: {np.linalg.norm(world.bodies[1].v):.2f} m/s")
        
        # Constraint check
        dist = np.linalg.norm(world.bodies[1].p - world.bodies[0].p)
        print(f"  Tether length: {dist:.6f} m (error: {abs(dist - 10.0):.6f} m)")
    
    # Terminal event info
    if hasattr(sol, 't_events') and len(sol.t_events[0]) > 0:
        print(f"\n  Terminal event triggered at t={sol.t_events[0][0]:.6f} s")
    
    print(f"\nOutput: {world.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
