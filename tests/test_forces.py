import numpy as np
from hybridsim import RigidBody6DOF, Gravity, Drag

def test_gravity_direction_and_magnitude():
    b = RigidBody6DOF("b", mass=3.0, I_body=np.eye(3), p=np.zeros(3), q=np.array([1,0,0,0]))
    g = Gravity(g=np.array([0.0, 0.0, -9.81]))
    b.clear_forces()
    g.apply(b, t=0.0)
    assert np.allclose(b.F, np.array([0.0, 0.0, -29.43]), atol=1e-12)

def test_drag_opposes_velocity():
    b = RigidBody6DOF("b", mass=1.0, I_body=np.eye(3), p=np.zeros(3), q=np.array([1,0,0,0]), v=np.array([10.0,0.0,0.0]))
    d = Drag(rho=1.2, Cd=1.0, area=0.5, mode="quadratic")
    b.clear_forces()
    d.apply(b, t=0.0)
    assert b.F[0] < 0.0 and abs(b.F[1]) < 1e-12 and abs(b.F[2]) < 1e-12
