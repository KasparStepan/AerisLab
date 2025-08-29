import numpy as np
from hybridsim import RigidBody6DOF, World, DistanceConstraint, PointWeldConstraint

def test_distance_constraint_value_and_jacobian():
    w = World()
    A = RigidBody6DOF("A", 1.0, np.eye(3), p=np.array([0,0,0],float), q=np.array([1,0,0,0]))
    B = RigidBody6DOF("B", 1.0, np.eye(3), p=np.array([1,0,0],float), q=np.array([1,0,0,0]))
    iA, iB = w.add_body(A), w.add_body(B)
    c = DistanceConstraint(iA, iB, r_i_b=np.zeros(3), r_j_b=np.zeros(3), L=1.0)
    w.constraints.append(c)
    C = c.evaluate(w)
    assert np.allclose(C, 0.0, atol=1e-12)
    J = c.jacobian_local(w)
    assert J.shape == (1, 12)

def test_point_weld_holds_points():
    w = World()
    A = RigidBody6DOF("A", 1.0, np.eye(3), p=np.array([0,0,0],float), q=np.array([1,0,0,0]))
    B = RigidBody6DOF("B", 1.0, np.eye(3), p=np.array([0,1,0],float), q=np.array([1,0,0,0]))
    iA, iB = w.add_body(A), w.add_body(B)
    c = PointWeldConstraint(iA, iB, r_i_b=np.zeros(3), r_j_b=np.array([0,-1,0],float))
    w.constraints.append(c)
    C = c.evaluate(w)
    assert np.allclose(C, np.zeros(3), atol=1e-12)
