"""
Baumgarte Stabilization Decay-Rate Tests (quantitative check of Fix #2).

The corrected stabilization forces the constraint error to obey a damped
harmonic oscillator:

    C̈ + 2α·Ċ + β²·C = 0.

(Indeed C̈ = J·a + J̇v = (rhs) + J̇v = -2α·Ċ - β²·C exactly at the continuous
level, independent of mass — the KKT solve enforces J·a = rhs.)

So if we start a constraint *deliberately violated* and at rest, C(t) must
follow the closed-form solution of that ODE. This pins the actual feedback
gains, not just "the other tests still pass":

    critically damped (α = β = ω):  C(t) = C₀ (1 + ω t) e^{-ω t}
    underdamped (α < β):            C(t) = C₀ e^{-α t}[cos ω_d t + (α/ω_d) sin ω_d t],
                                    ω_d = √(β² - α²)

We use a body on the x-axis tethered to WORLD at the origin, with no other
forces, so the motion is purely radial and C is scalar.
"""

import numpy as np

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint

L = 1.0
EPS = 0.02  # 2% initial length violation (small -> linear regime)


def _violated_world():
    w = World(ground_z=-1e9, payload_index=0)
    body = RigidBody6DOF("m", 1.0, np.eye(3) * 0.01,
                         position=np.array([L * (1.0 + EPS), 0.0, 0.0]),
                         orientation=np.array([0, 0, 0, 1]))
    w.add_body(body)
    con = DistanceConstraint(w.bodies, w.WORLD, 0, np.zeros(3), np.zeros(3), L)
    w.add_constraint(con)
    w.set_termination_callback(lambda w: False)
    return w, con


def _trace_C(alpha, beta, dt, T):
    w, con = _violated_world()
    solver = HybridSolver(alpha=alpha, beta=beta)
    n = int(round(T / dt))
    ts = [0.0]
    Cs = [float(con.evaluate()[0])]
    for k in range(n):
        w.step(solver, dt)
        ts.append((k + 1) * dt)
        Cs.append(float(con.evaluate()[0]))
    return np.array(ts), np.array(Cs)


class TestBaumgarteDecay:

    def test_critical_damping_matches_analytic(self):
        w = 4.0
        dt, T = 1e-3, 2.5
        ts, Cs = _trace_C(alpha=w, beta=w, dt=dt, T=T)
        C0 = Cs[0]
        analytic = C0 * (1.0 + w * ts) * np.exp(-w * ts)
        # Relative agreement over the decay (1st-order Euler -> few % at this dt).
        err = np.abs(Cs - analytic) / abs(C0)
        assert err.max() < 0.05, f"critical-damping decay off by {err.max():.3f}"
        # No overshoot for critical damping: C stays the same sign as C0.
        assert np.all(Cs > -1e-4 * abs(C0)), "critical damping should not overshoot"

    def test_underdamped_oscillates_and_matches(self):
        alpha, beta = 1.0, 6.0
        wd = np.sqrt(beta**2 - alpha**2)
        dt, T = 5e-4, 3.0
        ts, Cs = _trace_C(alpha=alpha, beta=beta, dt=dt, T=T)
        C0 = Cs[0]
        analytic = C0 * np.exp(-alpha * ts) * (np.cos(wd * ts) + (alpha / wd) * np.sin(wd * ts))
        err = np.abs(Cs - analytic) / abs(C0)
        assert err.max() < 0.05, f"underdamped decay off by {err.max():.3f}"
        # Underdamped must overshoot through zero (oscillatory return).
        assert Cs.min() < -0.05 * abs(C0), "underdamped response should overshoot"

    def test_critical_decays_faster_than_underdamped(self):
        # Sanity: at equal frequency scale, larger damping settles sooner.
        _, C_crit = _trace_C(alpha=5.0, beta=5.0, dt=1e-3, T=2.0)
        _, C_under = _trace_C(alpha=0.5, beta=5.0, dt=1e-3, T=2.0)
        assert abs(C_crit[-1]) < abs(C_under[-1])
