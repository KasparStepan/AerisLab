"""
World-Anchor (fixed body / World.WORLD) Verification Tests.

Phase 1 of the constraint rework introduces a genuinely immovable body
(``fixed=True``) and the implicit ``World.WORLD`` frame it powers. These tests
pin down two things:

1. A ``fixed=True`` body absorbs no acceleration from applied or constraint
   forces (it is a true clamp, not a heavy approximation).

2. A pendulum built against ``World.WORLD`` produces the *same* motion and the
   *same* tether tension as the legacy heavy-anchor (``mass=1e9``) approach —
   and does so without the ~m_bob/m_anchor reduced-mass bias.

The constraint-force (tension) cross-check reuses the same assemble/solve path
exercised by ``test_multipliers.py``.
"""

import numpy as np

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver, assemble_system, solve_kkt
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint
from aerislab.dynamics.forces import Gravity


def _bob(L, theta0, m=1.0):
    return RigidBody6DOF(
        name="bob",
        mass=m,
        inertia_tensor_body=np.eye(3) * 0.01,
        position=np.array([L * np.sin(theta0), 0.0, -L * np.cos(theta0)]),
        orientation=np.array([0, 0, 0, 1]),
    )


def _tension(world: World, body_index: int) -> float:
    """Recover |Jᵀλ| on a body at the current state (α=β=0, pure dynamic)."""
    for b in world.bodies:
        b.clear_forces()
    for b in world.bodies:
        for fb in b.per_body_forces:
            fb.apply(b, world.t)
    Minv, J, F, rhs, _ = assemble_system(world.bodies, world.constraints, 0.0, 0.0)
    _, lam = solve_kkt(Minv, J, F, rhs)
    Fc = (J.T @ lam)[6 * body_index: 6 * body_index + 3]
    return float(np.linalg.norm(Fc))


class TestFixedBodyIsImmovable:
    """A fixed body must not accelerate, no matter the force applied."""

    def test_zero_inverse_mass_matrix(self):
        anchor = RigidBody6DOF(
            "a", mass=0.0, inertia_tensor_body=np.eye(3),
            position=np.zeros(3), orientation=np.array([0, 0, 0, 1]), fixed=True,
        )
        assert anchor.fixed
        assert anchor.inv_mass == 0.0
        assert np.allclose(anchor.inv_mass_matrix_world(), 0.0)

    def test_fixed_body_does_not_move_under_gravity(self):
        world = World(ground_z=-1e9, payload_index=0)
        anchor_idx = world.WORLD
        anchor = world.bodies[anchor_idx]
        # Even with gravity applied directly, a clamped body stays put.
        anchor.per_body_forces.append(Gravity(np.array([0, 0, -9.81])))
        solver = HybridSolver(alpha=0.0, beta=0.0)
        for _ in range(100):
            world.step(solver, 0.01)
        assert np.allclose(anchor.p, 0.0)
        assert np.allclose(anchor.v, 0.0)


class TestWorldAnchorMatchesHeavyAnchor:
    """Pendulum on World.WORLD ≡ pendulum on a mass=1e9 anchor."""

    L, g, theta0 = 1.5, 9.81, 0.3

    def _heavy(self):
        pivot = RigidBody6DOF(
            "pivot", mass=1e9, inertia_tensor_body=np.eye(3) * 1e9,
            position=np.zeros(3), orientation=np.array([0, 0, 0, 1]),
        )
        w = World(ground_z=-1e9, payload_index=1)
        w.add_body(pivot)
        bob = _bob(self.L, self.theta0)
        w.add_body(bob)
        bob.per_body_forces.append(Gravity(np.array([0, 0, -self.g])))
        w.add_constraint(DistanceConstraint(w.bodies, 0, 1, np.zeros(3), np.zeros(3), self.L))
        w.set_termination_callback(lambda w: False)
        return w, bob

    def _world(self):
        w = World(ground_z=-1e9, payload_index=0)
        bob = _bob(self.L, self.theta0)
        w.add_body(bob)
        bob.per_body_forces.append(Gravity(np.array([0, 0, -self.g])))
        w.add_constraint(DistanceConstraint(w.bodies, w.WORLD, 0, np.zeros(3), np.zeros(3), self.L))
        w.set_termination_callback(lambda w: False)
        return w, bob

    def test_trajectories_match(self):
        wh, bh = self._heavy()
        ww, bw = self._world()
        solver = HybridSolver(alpha=5.0, beta=1.0)
        for _ in range(500):
            wh.step(solver, 0.002)
            ww.step(solver, 0.002)
        # Same swing to tight tolerance (heavy anchor differs only by ~1e-9 bias).
        assert np.allclose(bh.p, bw.p, atol=1e-6), f"{bh.p} vs {bw.p}"
        assert np.allclose(bh.v, bw.v, atol=1e-6)

    def test_tension_matches_and_is_unbiased(self):
        wh, bh = self._heavy()
        ww, bw = self._world()
        # Initial-state tension for a released pendulum: T = m g cos(theta0).
        expected = 1.0 * self.g * np.cos(self.theta0)
        t_heavy = _tension(wh, body_index=1)
        t_world = _tension(ww, body_index=0)
        assert np.isclose(t_world, expected, rtol=1e-9), f"{t_world} != {expected}"
        # World anchor is at least as accurate as the heavy-mass hack.
        assert abs(t_world - expected) <= abs(t_heavy - expected) + 1e-12
