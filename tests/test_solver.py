import numpy as np
from hybridsim import World, RigidBody6DOF, DistanceConstraint, Gravity, HybridSolver, SolverSettings

def test_solver_enforces_velocity_constraint():
    world = World(dt=0.005, solver=HybridSolver(SolverSettings()))
    a = RigidBody6DOF("a", 1.0, np.eye(3), [0,0,0], [0,0,0,1], [1,0,0], [0,0,0])
    b = RigidBody6DOF("b", 1.0, np.eye(3), [1,0,0], [0,0,0,1], [-1,0,0], [0,0,0])
    world.add_body(a); world.add_body(b)
    world.add_constraint(DistanceConstraint(a, b, np.zeros(3), np.zeros(3), L=1.0))
    world.add_global_force(Gravity([0,0,0]))

    world.step()

    from hybridsim.constraints import DistanceConstraint as DC
    c = DC(a, b, np.zeros(3), np.zeros(3), L=1.0)
    class W: bodies=[a,b]
    Jloc = c.jacobian_local(W)
    J = np.zeros((1, 12))
    J[:,0:3] = Jloc[:,0:3]; J[:,3:6] = Jloc[:,3:6]
    J[:,6:9] = Jloc[:,6:9]; J[:,9:12] = Jloc[:,9:12]
    v = np.hstack([a.linear_velocity, a.angular_velocity, b.linear_velocity, b.angular_velocity])
    val = float(J @ v)
    assert abs(val) < 1e-6
