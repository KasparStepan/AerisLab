"""
Solver Cross-Validation Tests.

Two independent integration paths must agree on the same problem:
  - HybridSolver      : fixed-step semi-implicit Euler + KKT
  - HybridIVPSolver   : scipy adaptive (Radau) + the same KKT enforcement

Agreement between independent integrators is a strong correctness signal — a bug
in either the fixed-step path, the adaptive path, or the shared assemble/solve
machinery shows up as disagreement. We drive each to high accuracy (small dt /
tight tol) and require the final states to match.
"""

import numpy as np

from aerislab.core.simulation import World
from aerislab.core.solver import HybridIVPSolver, HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint
from aerislab.dynamics.forces import Drag, Gravity

G = 9.81


def _fixed(world, dt, T):
    solver = HybridSolver(alpha=5.0, beta=5.0)
    for _ in range(int(round(T / dt))):
        world.step(solver, dt)


def _ivp(world, T):
    solver = HybridIVPSolver(method="Radau", rtol=1e-10, atol=1e-12, alpha=5.0, beta=5.0)
    world.integrate_to(solver, T)


class TestUnconstrainedAgreement:

    def _projectile(self):
        w = World(ground_z=-1e9, payload_index=0)
        b = RigidBody6DOF("p", 1.0, np.eye(3) * 0.01,
                          position=np.array([0.0, 0.0, 100.0]),
                          linear_velocity=np.array([10.0, 3.0, 0.0]),
                          orientation=np.array([0, 0, 0, 1]))
        w.add_body(b)
        b.per_body_forces.append(Gravity(np.array([0, 0, -G])))
        b.per_body_forces.append(Drag(rho=1.225, Cd=0.47, area=0.02))
        w.set_termination_callback(lambda w: False)
        return w, b

    def test_projectile_with_drag(self):
        wf, bf = self._projectile()
        wi, bi = self._projectile()
        _fixed(wf, dt=2e-4, T=2.0)
        _ivp(wi, T=2.0)
        # Fixed-step (1st order, dt=2e-4) vs essentially-exact Radau: agreement
        # is limited by the fixed-step truncation error (~few mm over ~80 m).
        assert np.allclose(bf.p, bi.p, atol=5e-3), f"pos {bf.p} vs {bi.p}"
        assert np.allclose(bf.v, bi.v, atol=5e-3), f"vel {bf.v} vs {bi.v}"


class TestConstrainedAgreement:

    def _pendulum(self):
        w = World(ground_z=-1e9, payload_index=0)
        L, th0 = 1.0, 0.5
        bob = RigidBody6DOF("bob", 1.0, np.eye(3) * 0.01,
                            position=np.array([L * np.sin(th0), 0.0, -L * np.cos(th0)]),
                            orientation=np.array([0, 0, 0, 1]))
        w.add_body(bob)
        bob.per_body_forces.append(Gravity(np.array([0, 0, -G])))
        w.add_constraint(DistanceConstraint(w.bodies, w.WORLD, 0, np.zeros(3), np.zeros(3), L))
        w.set_termination_callback(lambda w: False)
        return w, bob

    def test_pendulum(self):
        wf, bf = self._pendulum()
        wi, bi = self._pendulum()
        _fixed(wf, dt=1e-4, T=1.5)
        _ivp(wi, T=1.5)
        assert np.allclose(bf.p, bi.p, atol=2e-3), f"pos {bf.p} vs {bi.p}"
