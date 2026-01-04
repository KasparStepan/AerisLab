import numpy as np
import pytest

@pytest.mark.skipif(__import__("importlib").util.find_spec("scipy") is None, reason="SciPy required")
def test_free_fall_matches_kinematics():
    from AerisLab import World, RigidBody6DOF, Gravity, HybridIVPSolver
    I = np.eye(3)
    z0 = 20.0
    w = World(ground_z=-1000.0, payload_index=0)
    b = RigidBody6DOF("b", 1.0, I, np.array([0,0,z0]), np.array([1,0,0,0]))
    w.add_body(b)
    w.add_global_force(Gravity(np.array([0,0,-9.81])))

    ivp = HybridIVPSolver(method="Radau", rtol=1e-8, atol=1e-10, max_step=np.inf)
    t_end = 1.234
    sol = w.integrate_to(ivp, t_end=t_end)
    z = w.bodies[0].p[2]
    z_exact = z0 - 0.5*9.81*t_end**2
    assert abs(z - z_exact) < 1e-5
