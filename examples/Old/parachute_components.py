#!/usr/bin/env python3
"""
Parachute system example using the component architecture.

Demonstrates:
- Component base class with composition pattern
- Parachute deployment state machine
- System for managing multi-component assemblies
- Tether constraint between payload and canopy

This example shows the recommended way to build complex simulations
using the component architecture introduced in v0.2.0.
"""

import numpy as np

from aerislab.components import Parachute, Payload, System
from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint
from aerislab.dynamics.forces import Gravity


def build_recovery_system() -> tuple[World, System]:
    """
    Build a parachute recovery system using component architecture.

    Returns
    -------
    tuple[World, System]
        Configured World and System ready for simulation
    """
    # Create World with logging
    world = World.with_logging(
        name="parachute_components_example",
        ground_z=0.0,
        auto_save_plots=True,
    )

    # Create System to manage components
    system = System(name="recovery_system")

    # -------------------------------------------------------------------------
    # Payload Component
    # -------------------------------------------------------------------------
    payload_mass = 10.0  # kg
    payload_radius = 0.2  # m

    payload_body = RigidBody6DOF(
        name="payload",
        mass=payload_mass,
        inertia_tensor_body=(2 / 5) * payload_mass * payload_radius**2 * np.eye(3),
        position=np.array([0.0, 0.0, 1000.0]),  # Start at 1000m altitude
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        radius=payload_radius,
    )

    payload = Payload(
        name="payload",
        body=payload_body,
        Cd=0.47,  # Sphere drag coefficient
        area=np.pi * payload_radius**2,
    )
    payload_idx = system.add_component(payload)

    # -------------------------------------------------------------------------
    # Parachute Component
    # -------------------------------------------------------------------------
    canopy_mass = 2.0  # kg
    canopy_radius = 1.0  # m (for visualization)

    canopy_body = RigidBody6DOF(
        name="canopy",
        mass=canopy_mass,
        inertia_tensor_body=0.1 * np.eye(3),  # Low inertia for canopy
        position=np.array([0.0, 0.0, 1010.0]),  # Above payload
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        radius=canopy_radius,
    )

    parachute = Parachute(
        name="main_chute",
        body=canopy_body,
        Cd=1.5,  # Parachute drag coefficient
        area=15.0,  # Deployed area [m²]
        activation_altitude=800.0,  # Deploy at 800m
        activation_velocity=40.0,  # Or when speed exceeds 40 m/s
        gate_sharpness=40.0,  # Smooth deployment transition
    )
    canopy_idx = system.add_component(parachute)

    # -------------------------------------------------------------------------
    # Tether Constraint
    # -------------------------------------------------------------------------
    tether_length = 10.0  # m

    # Create constraint (using indices within the system)
    # Note: constraint.attach() needs the bodies list
    tether = DistanceConstraint(
        world_bodies=system.get_bodies(),
        body_i=payload_idx,
        body_j=canopy_idx,
        attach_i_local=np.zeros(3),  # Center of payload
        attach_j_local=np.zeros(3),  # Center of canopy
        length=tether_length,
    )
    system.add_constraint(tether)

    # -------------------------------------------------------------------------
    # Add System to World
    # -------------------------------------------------------------------------
    world.add_system(system)

    # Add global gravity
    world.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))

    # Set payload as termination monitor
    world.payload_index = 0  # First body added to world

    return world, system


def run_simulation(world: World, system: System) -> None:
    """Run the simulation and report results."""
    print("=" * 60)
    print("Parachute Component Architecture Example")
    print("=" * 60)
    print(f"\n{system.summary()}\n")

    # Create solver
    solver = HybridSolver(alpha=5.0, beta=1.0)

    # Run simulation
    print("Starting simulation...")
    world.run(solver, duration=200.0, dt=0.01)

    # Report results
    print("\n" + "=" * 60)
    print("Simulation Results")
    print("=" * 60)

    # Get parachute component
    parachute = system.get_component("main_chute")
    if parachute is not None and hasattr(parachute, "deployment_state"):
        state_name = parachute.deployment_state.name
        deploy_time = parachute.deployment_time
        print(f"Parachute state: {state_name}")
        if deploy_time is not None:
            print(f"Deployment time: {deploy_time:.3f}s")

    # Report touchdown
    if world.t_touchdown is not None:
        print(f"Touchdown time: {world.t_touchdown:.3f}s")

    # Final positions
    print("\nFinal component states:")
    for comp in system.components:
        alt = comp.position[2]
        vel = np.linalg.norm(comp.velocity)
        print(f"  {comp.name}: altitude={alt:.2f}m, speed={vel:.2f}m/s")

    print("\n" + "=" * 60)
    print(f"Output saved to: {world.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    world, system = build_recovery_system()
    run_simulation(world, system)
