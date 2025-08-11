"""
Fixed-step example: payload + parachute (both rigid) with a rigid tether.
"""
from __future__ import annotations
import numpy as np
from hybridsim import (
    World, RigidBody6DOF, Gravity, Drag,
    RigidTetherJoint, GroundProjection, CSVLogger, HybridSolver, SolverSettings
)

def main():
    solver = HybridSolver(SolverSettings())
    world = World(dt=0.005, solver=solver, contact_model=GroundProjection(ground_z=0.0))
    world.logger = CSVLogger("parachute_fixed.csv")

    payload = RigidBody6DOF(
        name="payload", mass=1.5,
        inertia_tensor_body=np.diag([0.2, 0.2, 0.2]),
        position=[0, 0, 20], orientation=[0, 0, 0, 1],
        linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0],
        radius=0.15
    )

    chute = RigidBody6DOF(
        name="parachute", mass=1.0,
        inertia_tensor_body=np.diag([0.1, 0.1, 0.1]),
        position=[0, 0, 21], orientation=[0, 0, 0, 1],
        linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0],
        radius=0.25
    )

    world.add_body(payload); world.add_body(chute)
    world.add_global_force(Gravity([0, 0, -9.81]))
    payload.forces.append(Drag(rho=1.2, Cd=0.47, area=0.1))
    chute.forces.append(Drag(rho=1.2, Cd=1.5, area=np.pi * (1.2**2)))

    # High-level joint API: rigid tether of 1 m (COM-to-COM for demo)
    RigidTetherJoint(payload, chute, np.zeros(3), np.zeros(3), length=1.0).attach(world)

    world.run(duration=5.0)
    print("Saved log to parachute_fixed.csv")

if __name__ == "__main__":
    main()
