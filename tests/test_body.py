import numpy as np
import pytest
from aerislab.dynamics.body import RigidBody6DOF

@pytest.fixture
def body_default():
    """Fixture for a default body at rest with identity orientation."""
    mass = 10.0
    I = np.diag([1.0, 2.0, 3.0])
    p = np.zeros(3)
    # Identity quaternion (scalar-last): [0, 0, 0, 1]
    q = np.array([0.0, 0.0, 0.0, 1.0])
    return RigidBody6DOF("test_body", mass, I, p, q)

def test_initialization(body_default):
    b = body_default
    assert b.name == "test_body"
    assert b.mass == 10.0
    assert b.inv_mass == 0.1
    assert np.allclose(b.I_body, np.diag([1.0, 2.0, 3.0]))
    assert np.allclose(b.p, np.zeros(3))
    assert np.allclose(b.q, [0, 0, 0, 1])
    assert np.allclose(b.v, np.zeros(3))
    assert np.allclose(b.w, np.zeros(3))
    assert b.radius == 0.0

def test_initialization_with_velocity():
    mass = 1.0
    I = np.eye(3)
    p = np.zeros(3)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([0.1, 0.2, 0.3])
    b = RigidBody6DOF("moving", mass, I, p, q, linear_velocity=v, angular_velocity=w)
    
    assert np.allclose(b.v, v)
    assert np.allclose(b.w, w)

def test_clear_forces(body_default):
    b = body_default
    b.f = np.array([1.0, 1.0, 1.0])
    b.tau = np.array([1.0, 1.0, 1.0])
    b.clear_forces()
    assert np.allclose(b.f, np.zeros(3))
    assert np.allclose(b.tau, np.zeros(3))

def test_rotation_world_identity(body_default):
    # q=[0,0,0,1] -> Identity matrix
    R = body_default.rotation_world()
    assert np.allclose(R, np.eye(3))

def test_rotation_world_90z():
    # 90 deg around Z
    # q = [0, 0, sin(45), cos(45)] = [0, 0, 1/sqrt(2), 1/sqrt(2)]
    val = 1.0 / np.sqrt(2.0)
    q = np.array([0.0, 0.0, val, val])
    b = RigidBody6DOF("rot", 1.0, np.eye(3), np.zeros(3), q)
    
    R = b.rotation_world()
    # Expected rotation matrix for 90 deg about Z:
    # [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    assert np.allclose(R, expected, atol=1e-7)

def test_inertia_world(body_default):
    # Identity rotation -> I_world = I_body
    I_w = body_default.inertia_world()
    assert np.allclose(I_w, body_default.I_body)

def test_inertia_world_rotated():
    # I_body = diag(1, 2, 3)
    # Rotate 90 deg around X
    # q = [sin(45), 0, 0, cos(45)] = [0.707, 0, 0, 0.707]
    val = 1.0 / np.sqrt(2.0)
    q = np.array([val, 0.0, 0.0, val])
    I_body = np.diag([1.0, 2.0, 3.0])
    b = RigidBody6DOF("rot", 1.0, I_body, np.zeros(3), q)
    
    # R_x(90) = [[1,0,0],[0,0,-1],[0,1,0]]
    # I_world = R I R^T
    # Result should swap I_yy and I_zz in the diagonal frame
    I_w = b.inertia_world()
    expected = np.diag([1.0, 3.0, 2.0])
    assert np.allclose(I_w, expected, atol=1e-7)

def test_mass_matrix_world(body_default):
    M = body_default.mass_matrix_world()
    assert M.shape == (6, 6)
    # Top-left 3x3 is mass * I
    assert np.allclose(M[:3, :3], 10.0 * np.eye(3))
    # Bottom-right 3x3 is I_world (which is I_body here)
    assert np.allclose(M[3:, 3:], np.diag([1.0, 2.0, 3.0]))

def test_apply_force_pure(body_default):
    f = np.array([5.0, 0.0, 0.0])
    body_default.apply_force(f)
    assert np.allclose(body_default.f, f)
    assert np.allclose(body_default.tau, np.zeros(3))

def test_apply_force_with_torque(body_default):
    f = np.array([5.0, 0.0, 0.0])
    # Apply at (0, 1, 0). Body at (0,0,0).
    # r = (0,1,0). tau = r x f = (0,1,0)x(5,0,0) = (0, 0, -5)
    pt = np.array([0.0, 1.0, 0.0])
    body_default.apply_force(f, point_world=pt)
    
    assert np.allclose(body_default.f, f)
    assert np.allclose(body_default.tau, np.array([0.0, 0.0, -5.0]))

def test_integrate_semi_implicit(body_default):
    b = body_default
    dt = 0.1
    a_lin = np.array([10.0, 0.0, 0.0])
    a_ang = np.array([0.0, 0.0, 0.0])
    
    # v_new = 0 + 10*0.1 = 1.0
    # p_new = 0 + 1.0*0.1 = 0.1
    b.integrate_semi_implicit(dt, a_lin, a_ang)
    
    assert np.allclose(b.v, [1.0, 0.0, 0.0])
    assert np.allclose(b.p, [0.1, 0.0, 0.0])
    assert np.allclose(b.q, [0.0, 0.0, 0.0, 1.0]) # No rotation

def test_integrate_semi_implicit_rotation(body_default):
    b = body_default
    dt = 0.1
    a_lin = np.zeros(3)
    a_ang = np.array([0.0, 0.0, 10.0]) # alpha
    b.integrate_semi_implicit(dt, a_lin, a_ang)
    assert np.allclose(b.w, [0.0, 0.0, 1.0])
    
    # qdot = 0.5 * q * [0, 0, 1, 0] (scalar-last pure vector)
    # q = [0, 0, 0, 1]
    # qdot = 0.5 * [0, 0, 1, 0] = [0, 0, 0.5, 0]
    # q_new = [0, 0, 0, 1] + [0, 0, 0.05, 0] = [0, 0, 0.05, 1]
    expected_q = np.array([0.0, 0.0, 0.05, 1.0])
    expected_q /= np.linalg.norm(expected_q)
    assert np.allclose(b.q, expected_q)