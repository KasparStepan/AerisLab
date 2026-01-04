import pytest
import numpy as np
from AerisLab.forces import Gravity, Drag, ParachuteDrag, Spring

# Mock RigidBody6DOF for testing purposes
class MockRigidBody:
    def __init__(self, mass=10.0, p=None, v=None, w=None):
        self.mass = float(mass)
        self.p = np.array(p if p is not None else [0, 0, 0], dtype=np.float64)
        self.v = np.array(v if v is not None else [0, 0, 0], dtype=np.float64)
        self.w = np.array(w if w is not None else [0, 0, 0], dtype=np.float64)
        self.applied_forces = []  # List of (force_vector, point_world)

    def apply_force(self, force, point_world=None):
        self.applied_forces.append((np.array(force), point_world))

    def rotation_world(self):
        # Return identity matrix for simplicity
        return np.eye(3)

def test_gravity():
    g_vec = [0, 0, -9.81]
    gravity = Gravity(g_vec)
    body = MockRigidBody(mass=2.0)
    
    gravity.apply(body)
    
    assert len(body.applied_forces) == 1
    f, p = body.applied_forces[0]
    # Force should be mass * g
    np.testing.assert_allclose(f, [0, 0, -19.62])
    assert p is None

def test_drag_quadratic_const():
    # F = -0.5 * rho * Cd * A * |v| * v
    drag = Drag(rho=1.0, Cd=2.0, area=0.5, mode='quadratic')
    body = MockRigidBody(v=[10, 0, 0])
    
    drag.apply(body)
    
    # Calculation: -0.5 * 1.0 * 2.0 * 0.5 * 10 * [10, 0, 0]
    # = -0.5 * 100 = -50
    assert len(body.applied_forces) == 1
    f, _ = body.applied_forces[0]
    np.testing.assert_allclose(f, [-50, 0, 0])

def test_drag_linear():
    # F = -k * v
    drag = Drag(mode='linear', k_linear=5.0)
    body = MockRigidBody(v=[0, -2, 0])
    
    drag.apply(body)
    
    # Calculation: -5.0 * [0, -2, 0] = [0, 10, 0]
    assert len(body.applied_forces) == 1
    f, _ = body.applied_forces[0]
    np.testing.assert_allclose(f, [0, 10, 0])

def test_drag_callable():
    # Test that Cd and Area can be functions of time
    # Area = t
    drag = Drag(rho=1.0, Cd=1.0, area=lambda t, b: t, mode='quadratic')
    body = MockRigidBody(v=[1, 0, 0])
    
    drag.apply(body, t=2.0)
    
    # F = -0.5 * 1 * 1 * 2.0 * 1 * [1, 0, 0] = [-1, 0, 0]
    f, _ = body.applied_forces[0]
    np.testing.assert_allclose(f, [-1, 0, 0])

def test_parachute_activation_and_transition():
    para = ParachuteDrag(
        rho=1.0, Cd=1.0, area=10.0, 
        activation_velocity=20.0,
        gate_sharpness=100.0, # Sharp transition for easier testing
        area_collapsed=0.0
    )
    body = MockRigidBody(v=[10, 0, 0]) # Below activation
    
    # 1. Check inactive state
    para.apply(body, t=0.0)
    f, _ = body.applied_forces[0]
    np.testing.assert_allclose(f, [0, 0, 0])
    assert not para.activation_status
    
    # 2. Trigger activation
    body.v = np.array([25.0, 0, 0])
    para.apply(body, t=1.0)
    assert para.activation_status
    assert para.activation_time == 1.0
    
    # At t=activation_time, tanh(0)=0, gate=0.5. Area = 0.5 * 10 = 5.0
    # F = -0.5 * 1 * 1 * 5.0 * 25 * [25, 0, 0] = -1562.5 * [1, 0, 0]
    f, _ = body.applied_forces[1]
    np.testing.assert_allclose(f, [-1562.5, 0, 0])
    
    # 3. Check full deployment (t >> activation_time)
    para.apply(body, t=10.0)
    # gate -> 1.0. Area -> 10.0
    # F = -0.5 * 1 * 1 * 10.0 * 25 * 25 = -3125
    f, _ = body.applied_forces[2]
    np.testing.assert_allclose(f, [-3125, 0, 0])

def test_parachute_callable_area():
    # Ensure the fix for callable area works
    para = ParachuteDrag(area=lambda t, b: 10.0, activation_velocity=5.0)
    body = MockRigidBody(v=[10, 0, 0])
    
    # Should not raise TypeError
    para.apply(body, t=0.0)
    assert para.activation_status

def test_parachute_altitude_activation():
    # Test activation by altitude (e.g. deploying main chute at 1000m)
    para = ParachuteDrag(
        activation_velocity=9999.0, # Set high so velocity doesn't trigger it
        activation_altitude=1000.0
    )
    # Case 1: Above altitude -> No activation
    body = MockRigidBody(p=[0, 0, 1500.0], v=[10, 0, 0])
    para.apply(body, t=0.0)
    assert not para.activation_status

    # Case 2: Below altitude -> Activation
    body.p = np.array([0.0, 0.0, 900.0])
    para.apply(body, t=1.0)
    assert para.activation_status
    assert para.activation_time == 1.0

def test_spring():
    b1 = MockRigidBody(p=[0, 0, 0], v=[0, 0, 0])
    b2 = MockRigidBody(p=[2, 0, 0], v=[0, 0, 0])
    
    # Spring L0=1.0, current dist=2.0. Extension=1.0.
    # k=10. Force magnitude = 10 * 1 = 10.
    # Direction on b1 is towards b2 (+x).
    spring = Spring(b1, b2, [0,0,0], [0,0,0], k=10.0, c=0.0, rest_length=1.0)
    spring.apply_pair()
    
    f1, _ = b1.applied_forces[0]
    f2, _ = b2.applied_forces[0]
    np.testing.assert_allclose(f1, [10, 0, 0])
    np.testing.assert_allclose(f2, [-10, 0, 0])