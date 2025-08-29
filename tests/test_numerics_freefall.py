import numpy as np
from hybridsim import *

def test_freefall_vertical_matches_analytic():
    g = -9.81
    w = World(ground_z=-1e9)  # disable termination
    z0, v0 = 100.0, -5.0
    b = RigidBody6DOF("payload", 2.0, np.eye(3),
                      p=np.array([0,0,z0]), q=np.array([1,0,0,0]),
                      v=np.array([0,0,v0]))
    i = w.add_body(b); w.set_payload(i)
    w.global_forces.append(Gravity(g=np.array([0,0,g])))
    solver = HybridSolver()
    dt = 1e-3
    T = 0.5
    steps = int(T/dt)
    for _ in range(steps):
        w.step(dt, solver)
    z_exact = z0 + v0*T + 0.5*g*T*T
    assert abs(w.bodies[i].p[2] - z_exact) < 5e-3  # modest tolerance
