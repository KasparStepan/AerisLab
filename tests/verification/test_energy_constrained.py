"""
Constrained-System Energy Conservation Tests.

An *ideal* (workless) constraint exchanges no energy with the system, so a
pendulum on a rigid distance constraint should conserve total mechanical energy
E = ½m|v|² + m g z. This is a sensitive end-to-end check of the KKT path: a
sign/scale error in J, J̇v, or the multiplier solve would leak energy.

A subtlety specific to this maximal-coordinate solver: with Baumgarte OFF
(α = β = 0) the constraint is only enforced at the acceleration level, so the
tether *length* slowly drifts (index-reduction drift) and the bob sinks —
changing E. Turning stabilization ON holds the geometry and therefore conserves
energy *better*. So the meaningful, true statements are:

  1. With stabilization on, E stays bounded over a long run (no blow-up, no leak).
  2. Stabilization *reduces* the secular energy drift versus the unstabilized case.

Both use semi-implicit (symplectic) Euler, which bounds energy error rather than
dissipating it.
"""

import numpy as np

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint
from aerislab.dynamics.forces import Gravity

G = 9.81
L = 1.0
M = 1.0


def _pendulum(theta0):
    w = World(ground_z=-1e9, payload_index=0)
    bob = RigidBody6DOF(
        "bob", M, np.eye(3) * 0.01,
        position=np.array([L * np.sin(theta0), 0.0, -L * np.cos(theta0)]),
        orientation=np.array([0, 0, 0, 1]),
    )
    w.add_body(bob)
    bob.per_body_forces.append(Gravity(np.array([0, 0, -G])))
    w.add_constraint(DistanceConstraint(w.bodies, w.WORLD, 0, np.zeros(3), np.zeros(3), L))
    w.set_termination_callback(lambda w: False)
    return w, bob


def _energy(bob):
    return 0.5 * M * float(bob.v @ bob.v) + M * G * float(bob.p[2])


def _run_energy_trace(alpha, beta, theta0=0.6, dt=2e-3, n=2500):
    w, bob = _pendulum(theta0)
    solver = HybridSolver(alpha=alpha, beta=beta)
    E = [_energy(bob)]
    for _ in range(n):
        w.step(solver, dt)
        E.append(_energy(bob))
    return np.array(E)


def _max_dev_and_drift(E):
    swing = M * G * L  # energy scale of the swing
    rel_dev = float(np.abs(E - E[0]).max() / swing)
    half = len(E) // 2
    drift = abs(E[half:].mean() - E[:half].mean()) / swing
    return rel_dev, float(drift)


class TestPendulumEnergyConservation:

    def test_energy_bounded_with_stabilization(self):
        # Critically-damped Baumgarte holds the geometry: E bounded, no blow-up.
        E = _run_energy_trace(alpha=5.0, beta=5.0)
        rel_dev, drift = _max_dev_and_drift(E)
        assert rel_dev < 5e-3, f"energy not bounded: max dev {rel_dev:.2e}"
        assert drift < 2e-3, f"secular energy drift {drift:.2e}"

    def test_stabilization_reduces_energy_drift(self):
        # Holding the constraint (Baumgarte on) conserves energy better than the
        # unstabilized index-reduction, which drifts geometrically.
        _, drift_on = _max_dev_and_drift(_run_energy_trace(alpha=5.0, beta=5.0))
        _, drift_off = _max_dev_and_drift(_run_energy_trace(alpha=0.0, beta=0.0))
        assert drift_on < drift_off, f"stabilized drift {drift_on:.2e} !< {drift_off:.2e}"
        # Even unstabilized, energy stays bounded (no runaway) over the run.
        rel_off, _ = _max_dev_and_drift(_run_energy_trace(alpha=0.0, beta=0.0))
        assert rel_off < 2e-2, f"unstabilized energy unbounded: {rel_off:.2e}"
