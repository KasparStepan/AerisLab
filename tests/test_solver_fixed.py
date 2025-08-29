import numpy as np
from hybridsim import *

def test_kkt_enforces_velocity_level_constraint_one_step():
    w = World()
    A = RigidBody6DOF("A", 1.0, np.eye(3), p=np.array([0,0,0],float), q=np.array([1,0,0,0]))
    B = RigidBody6DOF("B", 1.0, np.eye(3), p=np.array([1,0,0],float), q=np.array([1,0,0,0]))
    iA, iB = w.add_body(A), w.add_body(B)
    w.constraints.append(DistanceConstraint(iA, iB, np.zeros(3), np.zeros(3), L=1.0))
    solver = HybridSolver(alpha=10.0, beta=1.0)
    # tiny step
    w.step(1e-3, solver)
    # J v ~ 0
    from hybridsim.solver import _assemble_system
    M,Q,J,C,Jv = _assemble_system(w, solver.alpha, solver.beta)
    assert np.linalg.norm(Jv) < 1e-6
