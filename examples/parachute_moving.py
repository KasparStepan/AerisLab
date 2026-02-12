"""
Parachute-payload system with initial horizontal velocity.

Demonstrates:
- Initial velocity in Y direction
- Trajectory visualization with horizontal drift
- Based on parachute_ivp.py
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


def build_moving_parachute_system(initial_velocity_y: float = 20.0) -> World:
    """
    Build parachute-payload system with initial horizontal velocity.
    
    Parameters
    ----------
    initial_velocity_y : float
        Initial velocity in the Y direction [m/s].
    """
    world = World.with_logging(
        name="parachute_moving",
        ground_z=0.0,
        payload_index=0,
        auto_save_plots=True
    )
    
    # Payload
    payload_mass = 10.0
    payload_radius = 0.2
    I_payload = (2/5) * payload_mass * payload_radius**2 * np.eye(3)
    
    # Initial conditions with Y velocity
    initial_velocity = np.array([0.0, initial_velocity_y, 0.0])
    
    payload = RigidBody6DOF(
        name="payload",
        mass=payload_mass,
        inertia_tensor_body=I_payload,
        position=np.array([0.0, 0.0, 300.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # Quaternion [x, y, z, w]
        linear_velocity=initial_velocity,  # <-- Initial velocity
        radius=payload_radius
    )
    
    # Canopy - same initial velocity to keep tether constraint happy
    canopy_mass = 2.0
    I_canopy = 0.1 * np.eye(3)
    tether_length = 10.0
    
    canopy = RigidBody6DOF(
        name="canopy",
        mass=canopy_mass,
        inertia_tensor_body=I_canopy,
        position=np.array([0.0, 0.0, 300.0 + tether_length]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity=initial_velocity,  # Same velocity for constraint compatibility
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
            activation_altitude=200.0,
            gate_sharpness=50.0,
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
    """Run parachute simulation with initial horizontal velocity."""
    print("=" * 60)
    print("Parachute Drop Test (Moving Start)")
    print("=" * 60)
    
    initial_vy = 20.0  # 20 m/s in Y direction
    world = build_moving_parachute_system(initial_velocity_y=initial_vy)
    
    print(f"\nConfiguration:")
    print(f"  Initial altitude: {world.bodies[0].p[2]:.1f} m")
    print(f"  Initial velocity: [0, {initial_vy}, 0] m/s")
    print(f"  Payload mass: {world.bodies[0].mass:.1f} kg")
    print(f"  Canopy mass: {world.bodies[1].mass:.1f} kg")
    
    solver = HybridIVPSolver(
        method="Radau",
        rtol=1e-6,
        atol=1e-8,
        alpha=10.0,
        beta=2.0,
        max_step=0.5
    )
    
    print(f"\nRunning simulation...")
    
    start = time.time()
    sol = world.integrate_to(solver, t_end=500.0)
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Status: {'Success' if sol.success else 'Failed'}")
    print(f"  Simulation time: {world.t:.3f} s")
    print(f"  Computation time: {elapsed:.3f} s")
    print(f"  Time steps: {len(sol.t)}")
    
    if world.t_touchdown:
        final_pos = world.bodies[0].p
        print(f"\n  Touchdown at t={world.t_touchdown:.3f} s")
        print(f"  Final position: [{final_pos[0]:.1f}, {final_pos[1]:.1f}, {final_pos[2]:.1f}] m")
        print(f"  Horizontal drift (Y): {final_pos[1]:.1f} m")
        print(f"  Impact velocity: {np.linalg.norm(world.bodies[0].v):.2f} m/s")
    
    print(f"\nOutput: {world.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
