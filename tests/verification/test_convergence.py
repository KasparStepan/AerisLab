"""
Convergence-Rate (Order Verification) Tests.

The strongest correctness statement available for a numerical integrator is
that it converges to the exact solution at its *claimed order*. These tests
drive each solver on problems with a known analytical solution, refine the
timestep (or tolerance), and measure the observed order of accuracy as the
slope of log(error) vs log(dt).

Expected slopes for this engine:
    HybridSolver (semi-implicit / symplectic Euler) ......... ~1
    HybridIVPSolver(method="RK45")  .......... tolerance-driven (not dt-driven)
    HybridIVPSolver(method="Radau") .......... tolerance-driven (order 5 stages)

Note on adaptive solvers: scipy's RK45/Radau/BDF choose their own internal
step, so halving the *check* step does not measure their order. For those we
instead refine rtol/atol and assert the error decreases and reaches a tight
floor — the meaningful accuracy statement for an adaptive method.

References
----------
- LeVeque, Finite Difference Methods (Ch. on order verification)
- Roache, Verification and Validation in Computational Science (MMS / order)
"""

import numpy as np
import pytest

from aerislab.core.simulation import World
from aerislab.core.solver import HybridIVPSolver, HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint
from aerislab.dynamics.forces import Gravity, Spring

# -----------------------------------------------------------------------------
# Problem builders (known analytical solutions)
# -----------------------------------------------------------------------------

ANCHOR_MASS = 1e9  # effectively immovable; reduced-mass bias ~ m/M ~ 1e-9


def _make_anchor(name: str = "anchor") -> RigidBody6DOF:
    return RigidBody6DOF(
        name=name,
        mass=ANCHOR_MASS,
        inertia_tensor_body=np.eye(3) * ANCHOR_MASS,
        position=np.zeros(3),
        orientation=np.array([0, 0, 0, 1]),
    )


def _build_sho(k: float, m: float, A: float) -> tuple[World, RigidBody6DOF]:
    """1-DOF spring-mass oscillator along x. Solution: x(t) = A cos(ωt), ω=√(k/m)."""
    anchor = _make_anchor()
    mass_body = RigidBody6DOF(
        name="mass",
        mass=m,
        inertia_tensor_body=np.eye(3) * 0.01,
        position=np.array([A, 0.0, 0.0]),
        orientation=np.array([0, 0, 0, 1]),
    )
    world = World(ground_z=-1e9, payload_index=1)
    world.add_body(anchor)
    world.add_body(mass_body)
    world.add_interaction_force(
        Spring(
            body_a=anchor,
            body_b=mass_body,
            attach_a_local=np.zeros(3),
            attach_b_local=np.zeros(3),
            k=k,
            rest_length=0.0,
            c=0.0,
        )
    )
    world.set_termination_callback(lambda w: False)
    return world, mass_body


def _build_pendulum(L: float, g: float, theta0: float) -> tuple[World, RigidBody6DOF]:
    """
    Constrained point-mass pendulum in the x-z plane.

    Small-angle solution: θ(t) = θ₀ cos(ωt), ω = √(g/L), so x(t) ≈ L θ₀ cos(ωt).
    """
    pivot = _make_anchor("pivot")
    bob_pos = np.array([L * np.sin(theta0), 0.0, -L * np.cos(theta0)])
    bob = RigidBody6DOF(
        name="bob",
        mass=1.0,
        inertia_tensor_body=np.eye(3) * 0.01,
        position=bob_pos,
        orientation=np.array([0, 0, 0, 1]),
    )
    world = World(ground_z=-1e9, payload_index=1)
    world.add_body(pivot)
    world.add_body(bob)
    bob.per_body_forces.append(Gravity(np.array([0, 0, -g])))
    world.add_constraint(
        DistanceConstraint(
            world_bodies=world.bodies,
            body_i=0,
            body_j=1,
            attach_i_local=np.zeros(3),
            attach_j_local=np.zeros(3),
            length=L,
        )
    )
    world.set_termination_callback(lambda w: False)
    return world, bob


# -----------------------------------------------------------------------------
# Order-measurement utilities
# -----------------------------------------------------------------------------

def _fixed_step_state(build_fn, solver: HybridSolver, dt: float, T: float) -> float:
    """Run a fixed-step sim to time T and return x(T) (the body's x position)."""
    world, body = build_fn()
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        world.step(solver, dt)
    return float(body.p[0])


def _fixed_step_error(build_fn, ref_fn, solver: HybridSolver, dt: float, T: float) -> float:
    """Run a fixed-step sim to time T and return |x_num(T) - x_ref(T)|."""
    world, body = build_fn()
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        world.step(solver, dt)
    return abs(body.p[0] - ref_fn(n_steps * dt))


def _measure_order(build_fn, ref_fn, solver: HybridSolver, dts, T: float):
    """Return (errors, fitted_order) from a log-log fit of error vs dt."""
    errors = np.array([_fixed_step_error(build_fn, ref_fn, solver, dt, T) for dt in dts])
    # Slope of log(error) vs log(dt) is the observed order of accuracy.
    order = float(np.polyfit(np.log(np.asarray(dts)), np.log(errors), 1)[0])
    return errors, order


# =============================================================================
# Tier 1: fixed-step order verification — UNCONSTRAINED
# =============================================================================

class TestHybridSolverOrderUnconstrained:
    """
    HybridSolver uses semi-implicit (symplectic) Euler → globally 1st order.

    This pins down what the fixed-step solver *actually* is, independent of
    its docstring. An observed slope far from 1 indicates an integrator bug.
    """

    def test_observed_order_is_first(self):
        k, m, A = 100.0, 1.0, 0.5
        omega = np.sqrt(k / m)
        T = 1.0  # ≈ 1.6 periods; generic point so error does not cancel

        solver = HybridSolver(alpha=0.0, beta=0.0)  # no constraints present
        dts = [0.005, 0.0025, 0.00125, 0.000625]

        errors, order = _measure_order(
            _build_sho_factory(k, m, A),
            lambda t: A * np.cos(omega * t),
            solver,
            dts,
            T,
        )

        # Errors must shrink monotonically as dt halves.
        assert np.all(np.diff(errors) < 0), f"non-monotone errors: {errors}"
        # Semi-implicit Euler is first order: slope ≈ 1.
        assert 0.85 <= order <= 1.3, (
            f"Observed order {order:.3f} (errors={errors}). "
            f"Expected ~1 for semi-implicit Euler."
        )


# =============================================================================
# Tier 2: fixed-step order verification — CONSTRAINED (KKT + Baumgarte)
# =============================================================================

class TestHybridSolverOrderConstrained:
    """
    Same integrator on an index-reduced DAE (pendulum as a length constraint).

    Order verification here uses a *self-reference*: a high-accuracy run of the
    same solver at a much finer dt. This isolates the integrator's discretization
    order from any modeling/linearization error.

    NOTE: an earlier version compared against the small-angle *linear* solution
    L·θ₀·cos(ωt). That reference is only accurate to O(θ₀³) (~1e-6 here), so at
    small dt the integration error fell below the linearization floor and the
    measured slope collapsed (to ~ -0.9) — a TEST artifact, not a solver bug.
    Against a proper nonlinear reference the solver converges at ~1st order.
    """

    def test_observed_order_constrained(self):
        L, g, theta0 = 1.0, 9.81, 0.3  # finite angle: genuinely nonlinear pendulum
        T = 1.0

        solver = HybridSolver(alpha=5.0, beta=1.0)
        dts = [0.004, 0.002, 0.001, 0.0005]
        dt_ref = 1.0 / 16000.0  # ~30x finer than the smallest test dt

        build = _build_pendulum_factory(L, g, theta0)
        # High-accuracy self-reference at the final time.
        x_ref = _fixed_step_state(build, solver, dt_ref, T)
        ref_fn = lambda _t: x_ref  # noqa: E731 - compare every dt to the same fine ref

        errors, order = _measure_order(build, ref_fn, solver, dts, T)

        assert np.all(errors > 0), f"degenerate errors: {errors}"
        # Errors must shrink as dt halves (no linearization floor anymore).
        assert np.all(np.diff(errors) < 0), f"non-monotone errors: {errors}"
        assert order >= 0.8, (
            f"Constrained order {order:.3f} below first order (errors={errors}). "
            f"A slope < 1 here points to a constraint-formulation or Baumgarte bug."
        )
        # Informational: surfaces the measured order in -s / -rs pytest output.
        print(f"\n[constrained order] slope={order:.3f}  errors={errors}")


# =============================================================================
# Tier 3: adaptive solver accuracy — tolerance refinement
# =============================================================================

class TestIVPSolverToleranceRefinement:
    """
    Adaptive scipy solvers are tolerance-driven, not dt-driven. Verify that
    tightening rtol/atol monotonically reduces the trajectory error and that a
    tight tolerance reaches a small absolute error.
    """

    @pytest.mark.parametrize("method", ["RK45", "Radau"])
    def test_error_decreases_with_tolerance(self, method):
        k, m, A = 100.0, 1.0, 0.5
        omega = np.sqrt(k / m)
        T = 1.0
        exact = A * np.cos(omega * T)

        tols = [1e-4, 1e-7, 1e-10]
        errors = []
        for tol in tols:
            world, body = _build_sho(k, m, A)
            solver = HybridIVPSolver(method=method, rtol=tol, atol=tol)
            world.integrate_to(solver, T)
            errors.append(abs(body.p[0] - exact))

        errors = np.array(errors)
        assert np.all(np.diff(errors) < 0), (
            f"{method}: error did not decrease with tighter tolerance: {errors}"
        )
        assert errors[-1] < 1e-4, (
            f"{method}: error at tol=1e-10 is {errors[-1]:.2e}, expected < 1e-4."
        )


# -----------------------------------------------------------------------------
# Small factory helpers (so _measure_order can rebuild a fresh world each run)
# -----------------------------------------------------------------------------

def _build_sho_factory(k, m, A):
    return lambda: _build_sho(k, m, A)


def _build_pendulum_factory(L, g, theta0):
    return lambda: _build_pendulum(L, g, theta0)
