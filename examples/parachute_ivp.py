# Same physical setup as fixed-step, but with SciPy IVP (Radau) + terminal event.
import numpy as np
from hybridsim import (
    World, RigidBody6DOF, Gravity, Drag,
    RigidTetherJoint, HybridIVPSolver, CSVLogger
)

def main():
    world = World(ground_z=0.0, payload_index=0)

    I_sphere = (2/5) * 10.0 * 0.2**2 * np.eye(3)
    payload = RigidBody6DOF("payload", mass=10.0, inertia_tensor_body=I_sphere,
                            position=np.array([0.0, 0.0, 200.0]),
                            orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    canopy  = RigidBody6DOF("canopy", mass=2.0, inertia_tensor_body=0.1*np.eye(3),
                            position=np.array([0.0, 0.0, 205.0]),
                            orientation=np.array([1.0, 0.0, 0.0, 0.0]))

    pidx = world.add_body(payload)
    cidx = world.add_body(canopy)
    world.payload_index = pidx

    world.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))

    def area_schedule(t, body):
        return min(15.0, 0.5 + 1.5*t)
    payload.per_body_forces.append(Drag(rho=1.225, Cd=1.0, area=0.3, mode="quadratic"))
    canopy.per_body_forces.append(Drag(rho=1.225, Cd=1.5, area=area_schedule, mode="quadratic"))

    tether = RigidTetherJoint(pidx, cidx, attach_i_local=[0,0,0], attach_j_local=[0,0,0], length=5.0)
    world.add_constraint(tether.attach(world.bodies))

    world.set_logger(CSVLogger("logs/parachute_ivp.csv"))

    ivp = HybridIVPSolver(method="Radau", rtol=1e-6, atol=1e-8, alpha=5.0, beta=2.0)
    sol = world.integrate_to(ivp, t_end=200.0)

    print(f"IVP finished at t={world.t:.6f}s, touchdown≈{world.t_touchdown}")

if __name__ == "__main__":
    main()
