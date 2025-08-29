import numpy as np
from hybridsim import *

def test_fixed_step_stops_on_ground():
    w = World(ground_z=0.0)
    b = RigidBody6DOF("payload", 1.0, np.eye(3), p=np.array([0,0,2.0]), q=np.array([1,0,0,0]))
    i = w.add_body(b); w.set_payload(i)
    w.global_forces.append(Gravity())
    solver = HybridSolver()
    w.run(duration=10.0, dt=0.01, solver=solver)
    assert w.time < 10.0  # terminated early
    assert w.bodies[i].p[2] <= 0.0 + 1e-12
