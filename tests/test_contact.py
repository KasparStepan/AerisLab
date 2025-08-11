import numpy as np
from hybridsim import World, RigidBody6DOF, Gravity, GroundProjection, HybridSolver

def test_projection_stops_penetration():
    world = World(dt=0.01, solver=HybridSolver(), contact_model=GroundProjection(ground_z=0.0))
    body = RigidBody6DOF("drop", 1.0, np.eye(3), [0,0,0.1], [0,0,0,1], [0,0,-1], [0,0,0])
    world.add_body(body)
    world.add_global_force(Gravity([0,0,-9.81]))
    for _ in range(200):
        world.step()
    assert body.position[2] >= -1e-12
    assert body.linear_velocity[2] >= -1e-12
