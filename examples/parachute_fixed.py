# Example: two bodies with a rigid tether, gravity + drag.
import numpy as np
from hybridsim import (
    World, RigidBody6DOF, Gravity, Drag,
    RigidTetherJoint, HybridSolver, CSVLogger
)

def main():
    world = World(ground_z=0.0, payload_index=0)

    # Bodies
    I_sphere = (2/5) * 10.0 * 0.2**2 * np.eye(3)  # crude
    payload = RigidBody6DOF("payload", mass=10.0, inertia_tensor_body=I_sphere,
                            position=np.array([0.0, 0.0, 200.0]),
                            orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    canopy  = RigidBody6DOF("canopy", mass=2.0, inertia_tensor_body=0.1*np.eye(3),
                            position=np.array([0.0, 0.0, 205.0]),
                            orientation=np.array([1.0, 0.0, 0.0, 0.0]))

    payload_idx = world.add_body(payload)
    canopy_idx = world.add_body(canopy)
    world.payload_index = payload_idx

    # Forces
    world.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))

    # Time-varying canopy area to mimic inflation
    def area_schedule(t, body):
        return min(15.0, 0.5 + 1.5*t)  # m^2
    payload.per_body_forces.append(Drag(rho=1.225, Cd=1.0, area=0.3, mode="quadratic"))
    canopy.per_body_forces.append(Drag(rho=1.225, Cd=1.5, area=area_schedule, mode="quadratic"))

    # Joint: rigid tether of fixed length between attachment points
    tether = RigidTetherJoint(payload_idx, canopy_idx, attach_i_local=[0,0,0], attach_j_local=[0,0,0], length=5.0)
    world.add_constraint(tether.attach(world.bodies))

    # Logger
    world.set_logger(CSVLogger("logs/parachute_fixed.csv"))

    # Integrate fixed-step until touchdown
    solver = HybridSolver(alpha=5.0, beta=2.0)   # modest stabilization
    dt = 0.01
    world.run(solver, duration=120.0, dt=dt)

    print(f"Stopped at t={world.t:.3f}s, touchdown≈{world.t_touchdown:.3f}s")

if __name__ == "__main__":
    main()
