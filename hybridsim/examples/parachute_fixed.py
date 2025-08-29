from __future__ import annotations
import numpy as np
from pathlib import Path
from hybridsim import *

def main():
    world = World(ground_z=0.0)

    # Bodies
    I_diag_payload = np.diag([0.4, 0.4, 0.2])
    I_diag_canopy  = np.diag([0.05, 0.05, 0.02])

    payload = RigidBody6DOF(
        name="payload", mass=20.0, I_body=I_diag_payload,
        p=np.array([0.0, 0.0, 200.0]), q=np.array([1.0, 0.0, 0.0, 0.0]),
        v=np.array([0.0, 0.0, 0.0]), w=np.zeros(3), radius=0.2,
    )
    canopy = RigidBody6DOF(
        name="canopy", mass=2.0, I_body=I_diag_canopy,
        p=np.array([0.0, 0.0, 205.0]), q=np.array([1.0, 0.0, 0.0, 0.0]),
        v=np.zeros(3), w=np.zeros(3), radius=4.0,
    )

    i_payload = world.add_body(payload)
    i_canopy  = world.add_body(canopy)
    world.set_payload(i_payload)

    # Forces
    world.global_forces.append(Gravity(g=np.array([0.0, 0.0, -9.81])))
    # Drag: small on payload, large on canopy
    payload.forces.append(Drag(rho=1.225, Cd=1.0, area=0.05, mode="quadratic"))
    # area schedule for canopy (simple linear "inflation" first 2s)
    def area_schedule(t: float | None): 
        if t is None: 
            return 10.0
        return float(50.0 * min(max(t, 0.0), 2.0) / 2.0 + 5.0)
    canopy.forces.append(Drag(rho=1.225, Cd=1.6, area=area_schedule, mode="quadratic"))

    # Joint: rigid tether of length 5 m from payload top to canopy center
    RigidTetherJoint(
        i_payload, i_canopy,
        r_i_b=np.array([0.0, 0.0, +0.0]),
        r_j_b=np.array([0.0, 0.0, +0.0]),
        length=5.0
    ).attach(world)

    # Logger
    log = CSVLogger(Path(__file__).with_suffix(".csv"))
    world.set_logger(log)

    # Termination predicate: payload z <= 0
    world.set_termination(lambda w: w.bodies[w.payload_index].p[2] <= w.ground_z)

    # Run
    solver = HybridSolver(alpha=5.0, beta=1.0)  # gentle Baumgarte
    world.run(duration=10.0, dt=0.005, solver=solver)

    log.close()
    print("Finished fixed-step run at t =", world.time)

if __name__ == "__main__":
    main()
