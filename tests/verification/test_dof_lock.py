"""
DOF-Lock (translational, world-frame) Verification Tests.

`DOFLockConstraint` adds one KKT row per locked world axis. These tests check
the three things that matter for composing locks onto a body:

1. Functional   — a locked axis stays put under an out-of-plane disturbance.
2. Quantitative — the lock's Lagrange multiplier equals the disturbance it
                  cancels (the reaction force is physically correct).
3. Composition  — DistanceConstraint + lock-y under a lateral force reproduces
                  the *pure planar* pendulum that never felt the lateral force.

A no-lock control confirms the disturbance actually matters.
"""

import numpy as np

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver, assemble_system, solve_kkt
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint, DOFLockConstraint
from aerislab.dynamics.forces import Gravity


def _bob(L, theta0, m=1.0):
    return RigidBody6DOF(
        name="bob", mass=m, inertia_tensor_body=np.eye(3) * 0.01,
        position=np.array([L * np.sin(theta0), 0.0, -L * np.cos(theta0)]),
        orientation=np.array([0, 0, 0, 1]),
    )


class TestLockHoldsAxis:
    """A lateral (+y) force is fully absorbed by a y-lock; without it, y drifts."""

    L, g, theta0, Fy = 1.5, 9.81, 0.3, 4.0

    def _pendulum(self, with_lock: bool, lateral: bool):
        w = World(ground_z=-1e9, payload_index=0)
        bob = _bob(self.L, self.theta0)
        w.add_body(bob)
        bob.per_body_forces.append(Gravity(np.array([0, 0, -self.g])))
        if lateral:
            bob.per_body_forces.append(Gravity(np.array([0, self.Fy, 0])))  # const +y accel
        w.add_constraint(DistanceConstraint(w.bodies, w.WORLD, 0, np.zeros(3), np.zeros(3), self.L))
        if with_lock:
            w.add_constraint(
                DOFLockConstraint(w.bodies, 0, w.WORLD, np.zeros(3), np.zeros(3),
                                  locked_translation=(False, True, False))
            )
        w.set_termination_callback(lambda w: False)
        return w, bob

    def test_locked_axis_stays_zero(self):
        w, bob = self._pendulum(with_lock=True, lateral=True)
        solver = HybridSolver(alpha=5.0, beta=1.0)
        max_y = 0.0
        for _ in range(500):
            w.step(solver, 0.002)
            max_y = max(max_y, abs(bob.p[1]))
        assert max_y < 1e-6, f"y-lock failed: drifted to {max_y}"

    def test_control_without_lock_drifts(self):
        w, bob = self._pendulum(with_lock=False, lateral=True)
        solver = HybridSolver(alpha=5.0, beta=1.0)
        for _ in range(500):
            w.step(solver, 0.002)
        assert abs(bob.p[1]) > 1e-2, "control: lateral force should move y"

    def test_in_plane_motion_matches_planar_reference(self):
        # (b) distance + lock-y WITH lateral force  ==  (a) pure planar pendulum, no lateral.
        wb, bb = self._pendulum(with_lock=True, lateral=True)
        wa, ba = self._pendulum(with_lock=False, lateral=False)
        solver = HybridSolver(alpha=5.0, beta=1.0)
        for _ in range(500):
            wb.step(solver, 0.002)
            wa.step(solver, 0.002)
        assert np.allclose(bb.p[[0, 2]], ba.p[[0, 2]], atol=1e-6), f"{bb.p} vs {ba.p}"


class TestLockReactionForce:
    """The lock multiplier must equal the lateral force it cancels."""

    def test_reaction_equals_applied_lateral_force(self):
        L, g, m, Fy = 1.0, 9.81, 1.0, 7.5
        w = World(ground_z=-1e9, payload_index=0)
        bob = _bob(L, 0.0, m)            # hanging straight down, at rest
        w.add_body(bob)
        bob.per_body_forces.append(Gravity(np.array([0, 0, -g])))
        bob.per_body_forces.append(Gravity(np.array([0, Fy, 0])))  # +y push of magnitude m*Fy
        w.add_constraint(DistanceConstraint(w.bodies, w.WORLD, 0, np.zeros(3), np.zeros(3), L))
        lock = DOFLockConstraint(w.bodies, 0, w.WORLD, np.zeros(3), np.zeros(3),
                                 locked_translation=(False, True, False))
        w.add_constraint(lock)

        for b in w.bodies:
            b.clear_forces()
        for b in w.bodies:
            for fb in b.per_body_forces:
                fb.apply(b, 0.0)
        Minv, J, F, rhs, _ = assemble_system(w.bodies, w.constraints, 0.0, 0.0)
        _, lam = solve_kkt(Minv, J, F, rhs)
        # Reaction on the bob from the lock row is the last constraint row.
        Fc_lock = (J[-1:].T @ lam[-1:])[0:3]   # world force from the y-lock alone
        # Must oppose the applied +y force of magnitude m*Fy.
        assert np.isclose(Fc_lock[1], -m * Fy, rtol=1e-9), f"{Fc_lock[1]} != {-m * Fy}"
        assert np.allclose(Fc_lock[[0, 2]], 0.0, atol=1e-9)
