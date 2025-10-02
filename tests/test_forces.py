import numpy as np
import pytest
from AerisLab import RigidBody6DOF, Gravity, Drag

def test_gravity_force_direction():
    b = RigidBody6DOF("b", 2.0, np.eye(3), np.zeros(3), np.array([1,0,0,0]))
    g = Gravity(np.array([0, 0, -9.81]))
    b.clear_forces()
    g.apply(b)
    assert np.allclose(b.f, np.array([0,0,-19.62]))

def test_drag_opposes_velocity_quadratic():
    b = RigidBody6DOF("b", 1.0, np.eye(3), np.zeros(3), np.array([1,0,0,0]))
    b.v = np.array([3.0, 4.0, 0.0])  # speed 5
    d = Drag(rho=1.225, Cd=1.0, area=2.0, mode="quadratic")
    b.clear_forces()
    d.apply(b, t=0.0)
    assert np.dot(b.f, b.v) <= 0.0 + 1e-12
    # zero velocity -> zero drag
    b.v[:] = 0.0
    b.clear_forces()
    d.apply(b, t=0.0)
    assert np.allclose(b.f, 0.0)
