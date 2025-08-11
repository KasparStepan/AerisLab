import numpy as np
import pytest

from hybridsim import (
    World, RigidBody6DOF, Gravity, DistanceConstraint, GroundProjection,
    HybridSolver, SolverSettings, HybridIVPSolver, IVPSettings
)

scipy = pytest.importorskip("scipy", reason="SciPy required for variable-step tests")

def test_ivp_distance_constraint_and_ground_event():
    world = World(dt=0.01, solver=HybridSolver(SolverSettings()), contact_model=GroundProjection(ground_z=0.0))

    a = RigidBody6DOF("a", 1.0, np.eye(3), [0,0,0.5], [0,0,0,1], [0,0,0], [0,0,0])
    b = RigidBody6DOF("b", 1.0, np.eye(3), [1,0,2.0], [0,0,0,1], [0,0,0], [0,0,0])
    world.add_body(a); world.add_body(b)
    world.add_global_force(Gravity([0,0,-9.81]))
    world.add_constraint(DistanceConstraint(a, b, np.zeros(3), np.zeros(3), L=1.0))

    ivp = HybridIVPSolver(settings=world.solver.settings, ivp=IVPSettings(method="Radau", rtol=1e-6, atol=1e-8))
    world.integrate_to(world.time + 2.0, ivp)

    dist = np.linalg.norm(b.position - a.position)
    assert np.isclose(dist, 1.0, atol=1e-4)

    assert a.position[2] >= -1e-12
    assert b.position[2] >= -1e-12
