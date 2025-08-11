"""
Example using variable-step Radau integration with contact events.
"""
from __future__ import annotations
import numpy as np
from hybridsim import (
    World, RigidBody6DOF, Gravity, Drag,
    DistanceConstraint, GroundProjection,
    HybridSolver, SolverSettings,
    HybridIVPSolver, IVPSettings,
    CSVLogger
)

def main():
    fixed = HybridSolver(SolverSettings(baumgarte_alpha=0.0, baumgarte_beta=0.0))
    world = World(dt=0.005, solver=fixed, contact_model=GroundProjection(ground_z=0.0))
    world.logger = CSVLogger("parachute_ivp.csv")

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

    # Rigid 1 m tether (COMs)
    world.add_constraint(DistanceConstraint(payload, chute, np.zeros(3), np.zeros(3), L=1.0))

    # Variable-step solver: Radau, stiff, with small tolerances
    ivp = HybridIVPSolver(
        settings=world.solver.settings,
        ivp=IVPSettings(method="Radau", rtol=1e-6, atol=1e-9, max_step=None)
    )

    world.integrate_to(world.time + 5.0, ivp)

    print("Saved log to parachute_ivp.csv")

if __name__ == "__main__":
    main()
