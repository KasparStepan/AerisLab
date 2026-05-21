"""
Constraint-Validation Framework Tests (Phase 4).

`World.validate_constraints()` is the static safety net for composing
constraints. It must:

1. Pass a well-posed system (pendulum on WORLD).
2. Catch the rank-deficient "lock the swing axis" mistake we discussed
   (lock z on a vertical pendulum) at build time — not as a frozen bob later.
3. Catch a duplicate / redundant lock (same DOF locked twice).
4. Catch a velocity-inconsistent initial condition.
5. Flag a constraint that acts only on fixed bodies (singular row source).
"""

import warnings

import numpy as np
import pytest

from aerislab.core.simulation import World
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint, DOFLockConstraint
from aerislab.dynamics.forces import Gravity


def _pendulum_world():
    w = World(ground_z=-1e9, payload_index=0)
    bob = RigidBody6DOF("bob", 1.0, np.eye(3) * 0.01,
                        np.array([0.0, 0.0, -1.0]), np.array([0, 0, 0, 1]))
    w.add_body(bob)
    bob.per_body_forces.append(Gravity(np.array([0, 0, -9.81])))
    w.add_constraint(DistanceConstraint(w.bodies, w.WORLD, 0, np.zeros(3), np.zeros(3), 1.0))
    return w, bob


def test_well_posed_pendulum_passes():
    w, _ = _pendulum_world()
    rep = w.validate_constraints()
    assert rep["ok"], rep["issues"]
    assert rep["deficiency"] == 0


def test_lock_swing_axis_is_flagged():
    # Bob hangs straight down (z=-1). Locking z conflicts with |d|=L as it swings:
    # the distance row and the z-lock row become linearly dependent at this pose.
    w, _ = _pendulum_world()
    w.add_constraint(DOFLockConstraint(w.bodies, 0, w.WORLD, np.zeros(3), np.zeros(3),
                                       locked_translation=(False, False, True)))  # lock z
    rep = w.validate_constraints()
    assert not rep["ok"]
    assert rep["deficiency"] >= 1
    assert any("rank-deficient" in s for s in rep["issues"])


def test_strict_mode_raises():
    w, _ = _pendulum_world()
    w.add_constraint(DOFLockConstraint(w.bodies, 0, w.WORLD, np.zeros(3), np.zeros(3),
                                       locked_translation=(False, False, True)))
    with pytest.raises(RuntimeError, match="rank-deficient"):
        w.validate_constraints(strict=True)


def test_duplicate_lock_is_flagged():
    w, _ = _pendulum_world()
    for _ in range(2):  # lock y twice -> redundant
        w.add_constraint(DOFLockConstraint(w.bodies, 0, w.WORLD, np.zeros(3), np.zeros(3),
                                           locked_translation=(False, True, False)))
    rep = w.validate_constraints()
    assert not rep["ok"]
    assert rep["deficiency"] >= 1


def test_velocity_inconsistent_start_is_flagged():
    w = World(ground_z=-1e9, payload_index=0)
    # Bob starts with a y-velocity but we lock y=0 -> J·v != 0 at t0.
    bob = RigidBody6DOF("bob", 1.0, np.eye(3) * 0.01,
                        np.array([0.0, 0.0, -1.0]), np.array([0, 0, 0, 1]),
                        linear_velocity=np.array([0.0, 0.5, 0.0]))
    w.add_body(bob)
    w.add_constraint(DOFLockConstraint(w.bodies, 0, w.WORLD, np.zeros(3), np.zeros(3),
                                       locked_translation=(False, True, False)))
    rep = w.validate_constraints()
    assert not rep["ok"]
    assert any("velocity-inconsistent" in s for s in rep["issues"])


def test_constraint_between_two_fixed_bodies_is_flagged():
    w = World(ground_z=-1e9, payload_index=0)
    anchor = RigidBody6DOF("anchor2", 0.0, np.eye(3),
                           np.array([1.0, 0.0, 0.0]), np.array([0, 0, 0, 1]), fixed=True)
    a_idx = w.add_body(anchor)
    # Distance between WORLD and another fixed body: acts only on immovable DOFs.
    w.add_constraint(DistanceConstraint(w.bodies, w.WORLD, a_idx, np.zeros(3), np.zeros(3), 1.0))
    rep = w.validate_constraints()
    assert not rep["ok"]
    assert any("fixed" in s for s in rep["issues"])


def test_run_emits_warning_on_bad_set():
    from aerislab.core.solver import HybridSolver
    w, _ = _pendulum_world()
    w.add_constraint(DOFLockConstraint(w.bodies, 0, w.WORLD, np.zeros(3), np.zeros(3),
                                       locked_translation=(False, False, True)))
    w.set_termination_callback(lambda w: False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w.run(HybridSolver(alpha=5.0, beta=1.0), duration=0.01, dt=0.005, log_interval=0)
    assert any("validation found issues" in str(r.message) for r in rec)
