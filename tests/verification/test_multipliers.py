"""
Lagrange-Multiplier (Constraint-Force) Verification Tests.

For a maximal-coordinate KKT formulation, the Lagrange multipliers λ are the
physical constraint forces (here, tether tension). These tests assemble the
KKT system at a known configuration and check that the recovered constraint
force matches the closed-form value:

    Pendulum at rest (vertical)        →  T = m g
    Conical pendulum (steady)          →  T = m g / cos θ,  with Ω² = g / (L cos θ)

This directly exercises the Schur-complement path in ``solve_kkt`` against an
analytical answer — the most natural correctness check for a KKT solver.
"""

import numpy as np

from aerislab.core.simulation import World
from aerislab.core.solver import assemble_system, solve_kkt
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint
from aerislab.dynamics.forces import Gravity


def _apply_forces(world: World) -> None:
    """Replicate World.step's force-accumulation phase (no integration)."""
    for b in world.bodies:
        b.clear_forces()
    for b in world.bodies:
        for fb in b.per_body_forces:
            fb.apply(b, world.t)
    for fg in world.global_forces:
        for b in world.bodies:
            fg.apply(b, world.t)
    for fpair in world.interaction_forces:
        fpair.apply_pair(world.t)


def _constraint_force_on(world: World, body_index: int, alpha: float = 0.0,
                         beta: float = 0.0) -> np.ndarray:
    """
    Return the constraint force (F_c = Jᵀλ) acting on a given body.

    With α = β = 0 and a state already on the constraint manifold (C = 0, Jv = 0),
    λ is the pure dynamic constraint force, free of stabilization feedback.
    """
    _apply_forces(world)
    Minv, J, F, rhs, _ = assemble_system(world.bodies, world.constraints, alpha, beta)
    _, lam = solve_kkt(Minv, J, F, rhs)
    Fc = J.T @ lam  # (6N,)
    return Fc[6 * body_index: 6 * body_index + 3]


def _make_pivot() -> RigidBody6DOF:
    return RigidBody6DOF(
        name="pivot",
        mass=1e9,
        inertia_tensor_body=np.eye(3) * 1e9,
        position=np.zeros(3),
        orientation=np.array([0, 0, 0, 1]),
    )


class TestVerticalTension:
    """Pendulum hanging at rest: tether tension must equal m g."""

    def test_tension_equals_mg(self):
        L, g, m = 1.5, 9.81, 2.0

        pivot = _make_pivot()
        bob = RigidBody6DOF(
            name="bob",
            mass=m,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=np.array([0.0, 0.0, -L]),  # straight down, at rest
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

        Fc = _constraint_force_on(world, body_index=1)
        tension = np.linalg.norm(Fc)

        assert np.isclose(tension, m * g, rtol=1e-6), (
            f"Tension {tension:.6f} N != mg {m * g:.6f} N"
        )
        # Force must point up the tether (toward the pivot), i.e. +z.
        assert Fc[2] > 0, f"Constraint force points the wrong way: {Fc}"


class TestConicalTension:
    """
    Conical pendulum in steady circular motion at half-angle θ:
        Ω² = g / (L cos θ),  tension T = m g / cos θ.
    The state is placed exactly on the manifold with a tangential velocity,
    so λ is the pure dynamic (gravity + centripetal) constraint force.
    """

    def test_tension_equals_mg_over_cos_theta(self):
        L, g, m = 1.0, 9.81, 1.0
        theta = np.deg2rad(30.0)

        r = L * np.sin(theta)         # horizontal radius of the cone
        h = L * np.cos(theta)         # depth below the pivot
        Omega = np.sqrt(g / (L * np.cos(theta)))  # circular rate
        speed = Omega * r             # tangential speed (about vertical axis)

        pivot = _make_pivot()
        bob = RigidBody6DOF(
            name="bob",
            mass=m,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=np.array([r, 0.0, -h]),
            # Velocity tangential to the circle (perpendicular to the tether):
            linear_velocity=np.array([0.0, speed, 0.0]),
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

        Fc = _constraint_force_on(world, body_index=1)
        tension = np.linalg.norm(Fc)
        expected = m * g / np.cos(theta)

        assert np.isclose(tension, expected, rtol=1e-6), (
            f"Conical tension {tension:.6f} N != mg/cosθ {expected:.6f} N"
        )
