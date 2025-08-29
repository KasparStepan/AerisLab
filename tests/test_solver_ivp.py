import numpy as np
import pytest
from hybridsim import *

@pytest.mark.skipif("scipy" not in globals(), reason="SciPy not imported here")
def test_ivp_terminates_on_ground():
    w = World(ground_z=0.0)
    b = RigidBody6DOF("payload", 1.0, np.eye(3),
                      p=np.array([0,0,10.0]), q=np.array([1,0,0,0]), v=np.zeros(3))
    i = w.add_body(b); w.set_payload(i)
    w.global_forces.append(Gravity())
    ivp = HybridIVPSolver(method="Radau", rtol=1e-8, atol=1e-10)
    sol = w.integrate_to(5.0, ivp=ivp)
    assert w.time <= 5.0 + 1e-6
    assert w.bodies[i].p[2] <= 0.0 + 1e-6
