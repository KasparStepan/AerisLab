"""
Comprehensive solver tests with numerical validation.
"""
import pytest
import numpy as np
from aerislab.core.solver import (
    HybridSolver,
    HybridIVPSolver,
    assemble_system,
    solve_kkt
)
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint


@pytest.fixture
def two_body_system():
    """Create simple two-body system."""
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
    
    return [body1, body2]


def test_assemble_system_no_constraints(two_body_system):
    """Test system assembly without constraints."""
    bodies = two_body_system
    constraints = []
    
    Minv, J, F, rhs, v = assemble_system(bodies, constraints, alpha=0, beta=0)
    
    assert Minv.shape == (12, 12)
    assert J.shape == (0, 12)
    assert F.shape == (12,)
    assert rhs.shape == (0,)
    assert v.shape == (12,)


def test_assemble_system_with_constraint(two_body_system):
    """Test system assembly with distance constraint."""
    bodies = two_body_system
    
    constraint = DistanceConstraint(
        bodies, 0, 1,
        np.zeros(3), np.zeros(3),
        length=5.0
    )
    constraints = [constraint]
    
    Minv, J, F, rhs, v = assemble_system(bodies, constraints, alpha=5.0, beta=1.0)
    
    assert Minv.shape == (12, 12)
    assert J.shape == (1, 12)
    assert F.shape == (12,)
    assert rhs.shape == (1,)


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


def test_ivp_solver_invalid_method():
    """Test IVP solver rejects invalid method."""
    with pytest.raises(ValueError, match="Method must be one of"):
        HybridIVPSolver(method="InvalidMethod")


def test_ivp_solver_invalid_tolerances():
    """Test IVP solver rejects invalid tolerances."""
    with pytest.raises(ValueError, match="Tolerances must be positive"):
        HybridIVPSolver(rtol=-1e-6)
    
    with pytest.raises(ValueError, match="Tolerances must be positive"):
        HybridIVPSolver(atol=0.0)


def test_baumgarte_stabilization():
    """Test Baumgarte parameters reduce constraint drift."""
    bodies = [
        RigidBody6DOF(
            "b1", 1.0, np.eye(3),
            np.array([0, 0, 0]),
            np.array([0, 0, 0, 1])
        ),
        RigidBody6DOF(
            "b2", 1.0, np.eye(3),
            np.array([0, 0, 5.1]),  # Slightly violated constraint
            np.array([0, 0, 0, 1])
        )
    ]
    
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
