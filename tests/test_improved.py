"""
Extended tests with improved coverage.
"""
import numpy as np
import pytest

from aerislab.core.solver import HybridSolver, assemble_system, solve_kkt
from aerislab.dynamics.body import RigidBody6DOF, quat_normalize
from aerislab.dynamics.constraints import DistanceConstraint


def test_quat_normalize_zero():
    """Test that zero quaternion returns identity."""
    q_zero = np.zeros(4)
    with pytest.warns(RuntimeWarning, match="Zero-norm quaternion"):
        q_norm = quat_normalize(q_zero)

    np.testing.assert_array_equal(q_norm, [0, 0, 0, 1])


def test_assemble_system_no_constraints():
    """Test system assembly without constraints."""
    I = np.eye(3) * 0.1
    body1 = RigidBody6DOF(
        "b1", 1.0, I,
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 1])
    )
    body2 = RigidBody6DOF(
        "b2", 1.0, I,
        np.array([0, 0, 5]),
        np.array([0, 0, 0, 1])
    )

    bodies = [body1, body2]
    constraints = []

    Minv, J, F, rhs, v = assemble_system(bodies, constraints, alpha=0, beta=0)

    assert Minv.shape == (12, 12)
    assert J.shape == (0, 12)
    assert F.shape == (12,)
    assert rhs.shape == (0,)


def test_solve_kkt_unconstrained():
    """Test KKT solver with no constraints."""
    Minv = np.eye(2)
    J = np.zeros((0, 2))
    F = np.array([1.0, 0.0])
    rhs = np.zeros(0)

    a, lam = solve_kkt(Minv, J, F, rhs)

    np.testing.assert_array_almost_equal(a, [1.0, 0.0])
    assert lam.shape == (0,)


def test_solve_kkt_constrained():
    """Test KKT solver with constraint."""
    Minv = np.eye(2)
    J = np.array([[1.0, -1.0]])
    F = np.array([1.0, -1.0])
    rhs = np.array([0.0])

    a, lam = solve_kkt(Minv, J, F, rhs)

    # Constraint forces should make accelerations equal
    assert abs(a[0] - a[1]) < 1e-10


def test_fixed_solver_energy_conservation():
    """Test fixed solver conserves energy approximately."""
    I = np.eye(3) * 0.1
    body = RigidBody6DOF(
        "test", 1.0, I,
        np.array([0, 0, 10]),
        np.array([0, 0, 0, 1]),
        linear_velocity=np.array([0, 0, -1])
    )

    bodies = [body]
    constraints = []
    solver = HybridSolver(alpha=0, beta=0)

    # Initial energy
    KE0 = body.kinetic_energy()
    PE0 = body.mass * 9.81 * body.p[2]
    E0 = KE0 + PE0

    # Integrate
    for _ in range(100):
        body.clear_forces()  # CRITICAL: Clear forces each step!
        body.apply_force(np.array([0, 0, -9.81 * body.mass]))
        solver.step(bodies, constraints, dt=0.01)

    # Final energy
    KE1 = body.kinetic_energy()
    PE1 = body.mass * 9.81 * body.p[2]
    E1 = KE1 + PE1

    # Energy should be approximately conserved (within 5%)
    assert abs(E1 - E0) / E0 < 0.05


def test_baumgarte_stabilization():
    """Test Baumgarte parameters reduce constraint drift."""
    I = np.eye(3) * 0.1
    body1 = RigidBody6DOF(
        "b1", 1.0, I,
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 1])
    )
    body2 = RigidBody6DOF(
        "b2", 1.0, I,
        np.array([0, 0, 5.1]),  # Slightly violated
        np.array([0, 0, 0, 1])
    )

    bodies = [body1, body2]
    constraint = DistanceConstraint(
        bodies, 0, 1,
        np.zeros(3), np.zeros(3),
        length=5.0
    )

    # Without stabilization
    Minv, J, F, rhs0, v = assemble_system(bodies, [constraint], alpha=0, beta=0)

    # With stabilization
    Minv, J, F, rhs1, v = assemble_system(bodies, [constraint], alpha=10, beta=1)

    # Stabilized RHS should have correction term
    assert abs(rhs1[0]) > abs(rhs0[0])
