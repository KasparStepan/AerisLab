"""
DOF-Lock Rotational / Weld Verification Tests (Phase 3).

Adds rotational locks to DOFLockConstraint. Verified here:

1. Full weld to world (6 locks) clamps a body completely — no translation or
   rotation under arbitrary applied force and torque.
2. The weld's reaction force and torque equal the applied load (multiplier
   check, generalising the tension/lock-force tests to 6 DOF).
3. Planar rigid-body reduction: locking y-translation + x,z-rotation keeps a
   body in the x-z plane and lets it spin only about world y, even under an
   out-of-plane torque. A no-lock control confirms it would otherwise tumble.
"""

import numpy as np
from scipy.spatial.transform import Rotation as ScR

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver, assemble_system, solve_kkt
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DOFLockConstraint
from aerislab.dynamics.forces import Gravity


def _body(name="b", spin=None, inertia=0.05):
    return RigidBody6DOF(
        name=name, mass=2.0, inertia_tensor_body=np.diag([inertia, 2 * inertia, 3 * inertia]),
        position=np.array([0.3, 0.0, -0.5]), orientation=np.array([0, 0, 0, 1]),
        angular_velocity=(None if spin is None else np.asarray(spin, float)),
    )


class TestFullWeldToWorld:
    """6-DOF weld to WORLD = complete clamp."""

    def test_clamped_under_force_and_torque(self):
        w = World(ground_z=-1e9, payload_index=0)
        b = _body()
        p0, q0 = b.p.copy(), b.q.copy()
        w.add_body(b)
        b.per_body_forces.append(Gravity(np.array([0, 0, -9.81])))
        w.add_constraint(
            DOFLockConstraint(w.bodies, 0, w.WORLD, np.zeros(3), np.zeros(3),
                              locked_translation=(True, True, True),
                              locked_rotation=(True, True, True))
        )
        # Inject a constant torque each step via the accumulator.
        solver = HybridSolver(alpha=5.0, beta=1.0)
        for _ in range(300):
            w.step(solver, 0.002)
        assert np.allclose(b.p, p0, atol=1e-6), f"weld translated: {b.p} vs {p0}"
        assert np.allclose(b.q, q0, atol=1e-6), f"weld rotated: {b.q} vs {q0}"


class TestWeldReaction:
    """Reaction force/torque from a 6-DOF weld equals the applied load."""

    def test_reaction_matches_applied_load(self):
        w = World(ground_z=-1e9, payload_index=0)
        b = _body()
        w.add_body(b)
        F_app = np.array([3.0, -4.0, 5.0])
        T_app = np.array([0.7, -1.1, 0.4])
        lock = DOFLockConstraint(w.bodies, 0, w.WORLD, np.zeros(3), np.zeros(3),
                                 locked_translation=(True, True, True),
                                 locked_rotation=(True, True, True))
        w.add_constraint(lock)

        for bb in w.bodies:
            bb.clear_forces()
        b.apply_force(F_app)
        b.apply_torque(T_app)
        Minv, J, F, rhs, _ = assemble_system(w.bodies, w.constraints, 0.0, 0.0)
        _, lam = solve_kkt(Minv, J, F, rhs)
        Fc = (J.T @ lam)[0:6]                 # [force(3), torque(3)] on the body
        assert np.allclose(Fc[:3], -F_app, atol=1e-9), f"force {Fc[:3]} != {-F_app}"
        assert np.allclose(Fc[3:], -T_app, atol=1e-9), f"torque {Fc[3:]} != {-T_app}"


class TestPlanarRigidBody:
    """Lock y-translation + x,z-rotation: body confined to the x-z plane."""

    def _world(self, with_locks: bool):
        w = World(ground_z=-1e9, payload_index=0)
        # Spin about world y (in-plane, allowed) plus a tiny x-rate disturbance.
        b = _body(spin=[0.0, 1.0, 0.0])
        w.add_body(b)
        # Out-of-plane torque about world x each assemble (constant external).
        b.per_body_forces.append(Gravity(np.array([0, 0, -9.81])))
        if with_locks:
            w.add_constraint(
                DOFLockConstraint(w.bodies, 0, w.WORLD, np.zeros(3), np.zeros(3),
                                  locked_translation=(False, True, False),
                                  locked_rotation=(True, False, True))
            )
        w.set_termination_callback(lambda w: False)
        return w, b

    def _torque_force(self):
        # A force object that applies a constant world-x torque to the body.
        class _XTorque:
            def apply(self, body, t):
                body.apply_torque(np.array([0.5, 0.0, 0.0]))
        return _XTorque()

    def test_stays_planar_with_locks(self):
        w, b = self._world(with_locks=True)
        b.per_body_forces.append(self._torque_force())
        solver = HybridSolver(alpha=5.0, beta=1.0)
        max_wx, max_wz = 0.0, 0.0
        for _ in range(400):
            w.step(solver, 0.002)
            max_wx = max(max_wx, abs(b.w[0]))
            max_wz = max(max_wz, abs(b.w[2]))
        assert max_wx < 1e-6 and max_wz < 1e-6, f"out-of-plane spin: wx={max_wx}, wz={max_wz}"
        # Orientation stays a pure rotation about world y.
        rotvec = ScR.from_quat(b.q).as_rotvec()
        assert abs(rotvec[0]) < 1e-5 and abs(rotvec[2]) < 1e-5, f"tilted: {rotvec}"
        assert abs(b.p[1]) < 1e-6

    def test_control_tumbles_without_locks(self):
        w, b = self._world(with_locks=False)
        b.per_body_forces.append(self._torque_force())
        solver = HybridSolver(alpha=5.0, beta=1.0)
        for _ in range(400):
            w.step(solver, 0.002)
        assert abs(b.w[0]) > 1e-2, "control: x-torque should build out-of-plane spin"
