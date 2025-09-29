# Same physical setup as fixed-step, but with SciPy IVP (Radau) + terminal event.
import numpy as np
import time
import os
# Robust imports for module/script execution
try:
    from ..hybridsim import (
        World, RigidBody6DOF, Gravity, Drag, ParachuteDrag, RigidTetherJoint, HybridIVPSolver, CSVLogger
    )
except ImportError:
    from hybridsim import (
        World, RigidBody6DOF, Gravity, Drag, ParachuteDrag, RigidTetherJoint, HybridIVPSolver, CSVLogger
    )

def build_world() -> World:
    w = World(ground_z=0.0, payload_index=0)

    I_sphere = (2/5) * 10.0 * 0.2**2 * np.eye(3)
    payload = RigidBody6DOF("payload", mass=10.0, inertia_tensor_body=I_sphere,
                            position=np.array([0.0, 0.0, 200.0]),
                            orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    canopy  = RigidBody6DOF("canopy", mass=2.0, inertia_tensor_body=0.1*np.eye(3),
                            position=np.array([0.0, 0.0, 205.0]),
                            orientation=np.array([1.0, 0.0, 0.0, 0.0]))

    pidx = w.add_body(payload)
    cidx = w.add_body(canopy)
    w.payload_index = pidx

    # Forces
    w.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))

    # Canopy inflation schedule
    def area_schedule(t, body):
        t = 0.0 if t is None else float(t)
        return min(15.0, 0.5 + 1.5 * t)

    payload.per_body_forces.append(Drag(rho=1.225, Cd=1.0, area=0.3, mode="quadratic"))
    canopy.per_body_forces.append(ParachuteDrag(rho=1.225, Cd=1.5, area=5))

    # Rigid tether (distance constraint)
    tether = RigidTetherJoint(pidx, cidx, [0,0,0], [0,0,0], length=5.0)
    w.add_constraint(tether.attach(w.bodies))
    return w


def main():
    world = build_world()

    # CSV logger
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", "parachute_ivp.csv")
    world.set_logger(CSVLogger(csv_path))

    # Diagnostics
    z0 = world.bodies[world.payload_index].p[2]
    print(f"Initial payload z0 = {z0:.6f} m; ground_z = {world.ground_z:.6f} m")

    # IVP solver (stiff, adaptive)
    ivp = HybridIVPSolver(method="Radau", rtol=1e-6, atol=1e-8, alpha=5.0, beta=2.0, max_step=1.0)

    # Integrate to t_end; terminal event stops at touchdown
    t_end = 200.0
    sol = world.integrate_to(ivp, t_end=t_end)

    # Report
    if sol is not None and getattr(sol, "t", None) is not None and sol.t.size > 0:
        print(f"IVP finished: status={getattr(sol, 'status', 'NA')} (0=success, 1=event)")
        print(f"t span: [{sol.t[0]:.6f}, {sol.t[-1]:.6f}] ; "
              f"events: {[len(e) for e in sol.t_events] if getattr(sol, 't_events', None) else 'None'}")
    print(f"World time t={world.t:.6f}s, touchdown={world.t_touchdown}")

    # One-liner plots from CSV
    os.makedirs("plots", exist_ok=True)
    world.save_plots(csv_path, bodies=["payload", "canopy"], plots_dir="plots", show=False)
    print(f"CSV: {csv_path}\nPlots saved under: plots/")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Elapsed time: {end - start:.3f}s")
