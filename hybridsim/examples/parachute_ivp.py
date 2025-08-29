from __future__ import annotations
import numpy as np
from pathlib import Path
from hybridsim import *

def main():
    world = World(ground_z=0.0)

    I_diag_payload = np.diag([0.4, 0.4, 0.2])
    I_diag_canopy  = np.diag([0.05, 0.05, 0.02])

    payload = RigidBody6DOF(
        name="payload", mass=20.0, I_body=I_diag_payload,
        p=np.array([0.0, 0.0, 200.0]), q=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    canopy = RigidBody6DOF(
        name="canopy", mass=2.0, I_body=I_diag_canopy,
        p=np.array([0.0, 0.0, 205.0]), q=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    i_payload = world.add_body(payload)
    i_canopy  = world.add_body(canopy)
    world.set_payload(i_payload)

    world.global_forces.append(Gravity())
    payload.forces.append(Drag(rho=1.225, Cd=1.0, area=0.05))
    canopy.forces.append(Drag(rho=1.225, Cd=1.6, area=40.0))

    RigidTetherJoint(
        i_payload, i_canopy,
        r_i_b=np.zeros(3), r_j_b=np.zeros(3),
        length=5.0
    ).attach(world)

    log = CSVLogger(Path(__file__).with_suffix(".csv"))
    world.set_logger(log)

    # IVP solver with ground terminal event
    ivp = HybridIVPSolver(method="Radau", rtol=1e-6, atol=1e-9, alpha=5.0, beta=1.0)
    world.integrate_to(t_end=20.0, ivp=ivp)

    log.close()
    print("Finished IVP run at t =", world.time)

if __name__ == "__main__":
    main()
