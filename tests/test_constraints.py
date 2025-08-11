import numpy as np
from hybridsim import World, RigidBody6DOF, DistanceConstraint, Gravity, HybridSolver, SolverSettings

def test_distance_constraint_basic():
    world = World(dt=0.01, solver=HybridSolver(SolverSettings()))
    a = RigidBody6DOF("a", 1.0, np.eye(3), [0,0,0], [0,0,0,1], [0,0,0], [0,0,0])
    b = RigidBody6DOF("b", 1.0, np.eye(3), [1,0,0], [0,0,0,1], [0,0,0], [0,0,0])
    world.add_body(a); world.add_body(b)
    world.add_constraint(DistanceConstraint(a, b, np.zeros(3), np.zeros(3), L=1.0))
    world.add_global_force(Gravity([0,0,0]))  # no gravity
    for _ in range(10):
        world.step()
    dist = np.linalg.norm(b.position - a.position)
    assert np.isclose(dist, 1.0, atol=1e-6)
