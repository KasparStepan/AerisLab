import numpy as np
from hybridsim import RigidBody6DOF, Drag

def test_drag_opposes_velocity():
    b = RigidBody6DOF("b", 1.0, np.eye(3), [0,0,0], [0,0,0,1], [1,2,3], [0,0,0])
    d = Drag(rho=1.2, Cd=1.0, area=0.5, model="quadratic")
    b.clear_forces()
    d.apply(b)
    assert np.dot(b.force, b.linear_velocity) <= 0.0 + 1e-12
