"""
Momentum-Conservation Tests for the Coupled (Constrained) System.

The existing suite checks angular-momentum conservation for a single spinning
body. These tests close the gap for *internal constraint forces*: two bodies
joined by a distance constraint, with no external forces, form an isolated
system. Total linear momentum must be conserved to machine precision, because
the KKT constraint force is internal (Jᵀλ produces equal-and-opposite forces).

A drift in total linear momentum is the signature of a Newton's-third-law
violation in the constraint coupling — a bug pure trajectory checks miss.
"""

import numpy as np

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint


def _build_dumbbell(m: float, d: float):
    """Two equal masses on a rigid distance constraint (a spinning, drifting dumbbell)."""
    body1 = RigidBody6DOF(
        name="m1",
        mass=m,
        inertia_tensor_body=np.eye(3) * 0.01,
        position=np.array([-d, 0.0, 0.0]),
        # common drift (+x) plus opposite transverse velocities (spin about COM):
        linear_velocity=np.array([0.5, 1.0, 0.0]),
        orientation=np.array([0, 0, 0, 1]),
    )
    body2 = RigidBody6DOF(
        name="m2",
        mass=m,
        inertia_tensor_body=np.eye(3) * 0.01,
        position=np.array([d, 0.0, 0.0]),
        linear_velocity=np.array([0.5, -1.0, 0.0]),
        orientation=np.array([0, 0, 0, 1]),
    )
    world = World(ground_z=-1e9, payload_index=1)
    world.add_body(body1)
    world.add_body(body2)
    # No gravity, no global forces: the system is isolated.
    world.add_constraint(
        DistanceConstraint(
            world_bodies=world.bodies,
            body_i=0,
            body_j=1,
            attach_i_local=np.zeros(3),
            attach_j_local=np.zeros(3),
            length=2 * d,
        )
    )
    world.set_termination_callback(lambda w: False)
    return world, body1, body2


def _linear_momentum(bodies) -> np.ndarray:
    return sum(b.mass * b.v for b in bodies)


def _angular_momentum_about(point, bodies) -> np.ndarray:
    L = np.zeros(3)
    for b in bodies:
        L += b.mass * np.cross(b.p - point, b.v)
        L += b.inertia_world() @ b.w
    return L


class TestLinearMomentumConservation:
    """Total Σ mᵢvᵢ conserved to machine precision for the isolated dumbbell."""

    def test_linear_momentum_conserved(self):
        m, d = 1.0, 1.0
        world, b1, b2 = _build_dumbbell(m, d)
        bodies = [b1, b2]

        p0 = _linear_momentum(bodies)
        solver = HybridSolver(alpha=5.0, beta=1.0)
        for _ in range(2000):
            world.step(solver, 0.001)
        p1 = _linear_momentum(bodies)

        drift = np.linalg.norm(p1 - p0)
        assert drift < 1e-9, (
            f"Linear momentum drifted by {drift:.2e} (p0={p0}, p1={p1}). "
            f"Constraint forces should be internal (equal & opposite)."
        )

    def test_com_velocity_constant(self):
        """COM velocity stays at the initial drift (no net external impulse)."""
        m, d = 1.0, 1.0
        world, b1, b2 = _build_dumbbell(m, d)
        bodies = [b1, b2]

        total_mass = sum(b.mass for b in bodies)
        v_com0 = _linear_momentum(bodies) / total_mass

        solver = HybridSolver(alpha=5.0, beta=1.0)
        for _ in range(2000):
            world.step(solver, 0.001)
        v_com1 = _linear_momentum(bodies) / total_mass

        assert np.allclose(v_com1, v_com0, atol=1e-9), (
            f"COM velocity changed: {v_com0} -> {v_com1}"
        )


class TestAngularMomentumConservation:
    """
    Angular momentum about the (drifting) center of mass is conserved for the
    isolated system. This is a softer check than linear momentum: Baumgarte
    feedback and the first-order integrator introduce a small bounded error.
    """

    def test_angular_momentum_about_com(self):
        m, d = 1.0, 1.0
        world, b1, b2 = _build_dumbbell(m, d)
        bodies = [b1, b2]

        total_mass = sum(b.mass for b in bodies)

        def com():
            return sum(b.mass * b.p for b in bodies) / total_mass

        L0 = _angular_momentum_about(com(), bodies)
        solver = HybridSolver(alpha=5.0, beta=1.0)
        for _ in range(2000):
            world.step(solver, 0.001)
        L1 = _angular_momentum_about(com(), bodies)

        rel_error = np.linalg.norm(L1 - L0) / max(np.linalg.norm(L0), 1e-12)
        assert rel_error < 0.01, (
            f"Angular momentum about COM changed by {rel_error * 100:.3f}% "
            f"(L0={L0}, L1={L1})"
        )
