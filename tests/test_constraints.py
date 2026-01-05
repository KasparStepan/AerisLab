import numpy as np
from aerislab import RigidBody6DOF, DistanceConstraint, PointWeldConstraint

def _two_bodies():
    I = np.eye(3)
    a = RigidBody6DOF("a", 1.0, I, np.array([0,0,0], float), np.array([1,0,0,0]))
    b = RigidBody6DOF("b", 1.0, I, np.array([1,0,0], float), np.array([1,0,0,0]))
    return [a, b]

def test_distance_constraint_eval_and_J():
    bodies = _two_bodies()
    dc = DistanceConstraint(bodies, 0, 1, [0,0,0], [0,0,0], length=1.0)
    C = dc.evaluate()
    assert C.shape == (1,)
    J = dc.jacobian()
    assert J.shape == (1, 12)
    # at current configuration d = [-1,0,0], so gradient wrt vA is d
    assert np.isclose(J[0,0], -1.0)

def test_weld_constraint_eval_and_J():
    bodies = _two_bodies()
    wc = PointWeldConstraint(bodies, 0, 1, [0,0,0], [1,0,0])
    # pa = [0,0,0], pb = [2,0,0] => residual [-2,0,0]
    C = wc.evaluate()
    assert np.allclose(C, np.array([-2.0, 0.0, 0.0]))
    J = wc.jacobian()
    assert J.shape == (3, 12)
