"""
Tests for force models.
"""
import pytest
import numpy as np
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity, Drag, ParachuteDrag, Spring


class MockRigidBody:
    """Mock body for testing forces without full RigidBody6DOF."""
    def __init__(self, v=(0, 0, 0), p=(0, 0, 0), mass=1.0):
        self.v = np.array(v, dtype=float)
        self.p = np.array(p, dtype=float)
        self.mass = mass
        self.applied_forces = []
        self.f = np.zeros(3)
        self.tau = np.zeros(3)
        
    def apply_force(self, f, point_world=None):
        self.applied_forces.append((f, point_world))
        self.f += f
        
    def clear_forces(self):
        self.applied_forces = []
        self.f = np.zeros(3)
        self.tau = np.zeros(3)


def test_gravity_applies_correct_force():
    g_vec = np.array([0.0, 0.0, -9.81])
    gravity = Gravity(g_vec)
    body = MockRigidBody(mass=2.0)
    
    gravity.apply(body, t=0.0)
    
    # Should apply F = m * g
    expected = 2.0 * g_vec
    np.testing.assert_array_almost_equal(body.f, expected)


def test_drag_quadratic_mode():
    drag = Drag(rho=1.225, Cd=0.5, area=1.0, mode="quadratic")
    body = MockRigidBody(v=[10, 0, 0])
    
    drag.apply(body, t=0.0)
    
    # F_drag = -0.5 * rho * Cd * A * |v| * v
    # = -0.5 * 1.225 * 0.5 * 1.0 * 10 * [10,0,0]
    # = -30.625 * [1,0,0]
    expected_mag = 0.5 * 1.225 * 0.5 * 1.0 * 10**2
    assert abs(body.f[0] + expected_mag) < 1e-6
    assert abs(body.f[1]) < 1e-10
    assert abs(body.f[2]) < 1e-10


def test_drag_linear_mode():
    drag = Drag(mode="linear", k_linear=2.0)
    body = MockRigidBody(v=[5, 0, 0])
    
    drag.apply(body, t=0.0)
    
    # F_drag = -k * v = -2.0 * [5,0,0]
    expected = -2.0 * np.array([5, 0, 0])
    np.testing.assert_array_almost_equal(body.f, expected)


def test_parachute_activation_and_transition():
    """Test parachute activation logic."""
    para = ParachuteDrag(
        rho=1.0, Cd=1.0, area=10.0,
        activation_velocity=20.0,
        gate_sharpness=100.0,
        area_collapsed=0.1  # Small collapsed area
    )
    
    # 1. Below activation velocity
    body = MockRigidBody(v=[10, 0, 0])  # 10 < 20
    para.apply(body, t=0.0)
    # Should have small force from collapsed area
    force_inactive = np.linalg.norm(body.f)
    
    # 2. Above activation velocity
    body = MockRigidBody(v=[25, 0, 0])  # 25 > 20
    para.apply(body, t=0.0)
    force_active = np.linalg.norm(body.f)
    
    # Active force should be significantly larger
    assert force_active > force_inactive * 5


def test_spring_force():
    I = np.eye(3)
    body_a = RigidBody6DOF(
        "a", 1.0, I,
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 1])
    )
    body_b = RigidBody6DOF(
        "b", 1.0, I,
        np.array([0, 0, 12]),  # 12m apart, rest = 10m
        np.array([0, 0, 0, 1])
    )
    
    spring = Spring(
        body_a, body_b,
        np.zeros(3), np.zeros(3),
        k=100.0, c=0.0, rest_length=10.0
    )
    
    spring.apply_pair(t=0.0)
    
    # Extension = 12 - 10 = 2m
    # Force = k * extension = 100 * 2 = 200N (on each body)
    # Direction: body_a pulled up (+z), body_b pulled down (-z)
    assert abs(body_a.f[2] - 200.0) < 1e-6
    assert abs(body_b.f[2] + 200.0) < 1e-6


def test_spring_damping():
    I = np.eye(3)
    body_a = RigidBody6DOF(
        "a", 1.0, I,
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 1]),
        linear_velocity=np.array([0, 0, 0])
    )
    body_b = RigidBody6DOF(
        "b", 1.0, I,
        np.array([0, 0, 10]),
        np.array([0, 0, 0, 1]),
        linear_velocity=np.array([0, 0, 5])  # Moving away
    )
    
    spring = Spring(
        body_a, body_b,
        np.zeros(3), np.zeros(3),
        k=0.0, c=10.0, rest_length=10.0
    )
    
    spring.apply_pair(t=0.0)
    
    # Relative velocity = 5 m/s (separating)
    # Damping force = -c * v_rel = -10 * 5 = -50N (resisting separation)
    # body_a gets pulled up, body_b gets pulled down
    assert abs(body_a.f[2] - 50.0) < 1e-6
    assert abs(body_b.f[2] + 50.0) < 1e-6


def test_callable_area_drag():
    """Test drag with callable area function."""
    def growing_area(t, body):
        return 1.0 + 0.5 * t  # Grows over time
    
    drag = Drag(rho=1.0, Cd=1.0, area=growing_area)
    body = MockRigidBody(v=[10, 0, 0])
    
    # At t=0, area=1.0
    drag.apply(body, t=0.0)
    force_t0 = np.linalg.norm(body.f)
    
    # At t=2, area=2.0
    body.clear_forces()
    drag.apply(body, t=2.0)
    force_t2 = np.linalg.norm(body.f)
    
    # Force should double
    assert abs(force_t2 / force_t0 - 2.0) < 0.01
