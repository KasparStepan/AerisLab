# Same physical setup as fixed-step, but with SciPy IVP (Radau) + terminal event.
import numpy as np
import time
import os
import sys

# Robust imports for module/script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from aerislab.core import World, HybridIVPSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity, Drag, ParachuteDrag
from aerislab.dynamics.joints import RigidTetherJoint
from aerislab.logger import CSVLogger

def build_world() -> World:
    w = World(ground_z=0.0, payload_index=0, log_enabled=False)

    I_sphere = (2/5) * 10.0 * 0.2**2 * np.eye(3)
    payload = RigidBody6DOF("payload", 10.0, I_sphere, np.array([0.0, 0.0, 200.0]),
                            np.array([0.0, 0.0, 0.0, 1.0]))
    canopy  = RigidBody6DOF("canopy", 2.0, 0.1*np.eye(3), np.array([0.0, 0.0, 205.0]),
                            np.array([0.0, 0.0, 0.0, 1.0]))

    pidx = w.add_body(payload)
    cidx = w.add_body(canopy)
    w.payload_index = pidx

    # Forces
    w.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))

    payload.per_body_forces.append(Drag(rho=1.225, Cd=1.0, area=0.3, mode="quadratic"))
    canopy.per_body_forces.append(ParachuteDrag(rho=1.225, Cd=1.5, area=5, activation_velocity=25))

    # Rigid tether (distance constraint)
    tether = RigidTetherJoint(pidx, cidx, [0, 0, 0], [0, 0, 0], length=5.0)
    w.add_constraint(tether.attach(w.bodies))
    return w


def main():
    world = build_world()

    # CSV logger
    media_dir = os.path.join(os.path.dirname(__file__), "media")
    logs_dir = os.path.join(media_dir, "logs")
    plots_dir = os.path.join(media_dir, "plots_ivp")

    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, "parachute_ivp.csv")
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
    os.makedirs(plots_dir, exist_ok=True)
    world.save_plots(csv_path, bodies=["payload", "canopy"], plots_dir=plots_dir, show=False)
    print(f"CSV: {csv_path}\nPlots saved under: {plots_dir}")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Elapsed time: {end - start:.3f}s")
