import numpy as np
from aerislab import (
    World, RigidBody6DOF, Gravity, RigidTetherJoint, HybridSolver
)

def test_velocity_level_satisfaction_after_step():
    # Two identical bodies connected by rigid tether
    I = np.eye(3)
    a = RigidBody6DOF("a", 1.0, I, np.array([0,0,1], float), np.array([1,0,0,0]))
    b = RigidBody6DOF("b", 1.0, I, np.array([1,0,1], float), np.array([1,0,0,0]))
    w = World(ground_z=0.0, payload_index=0)
    ia = w.add_body(a)
    ib = w.add_body(b)
    joint = RigidTetherJoint(ia, ib, [0,0,0], [0,0,0], length=1.0)
    w.add_constraint(joint.attach(w.bodies))
    w.add_global_force(Gravity(np.array([0,0,-9.81])))

    solver = HybridSolver(alpha=10.0, beta=5.0)
    dt = 1e-3
    # One step
    stop = w.step(solver, dt)
    assert not stop
    # Check velocity-level satisfaction: ||J v|| ~ 0
    c = w.constraints[0]
    nb = len(w.bodies)
    vstack = np.zeros(6*nb)
    for i, b in enumerate(w.bodies):
        vstack[6*i:6*i+3] = b.v
        vstack[6*i+3:6*i+6] = b.w
    J = c.jacobian()
    val = J @ np.concatenate([vstack[0:6], vstack[6:12]])
    assert np.linalg.norm(val) < 1e-6
