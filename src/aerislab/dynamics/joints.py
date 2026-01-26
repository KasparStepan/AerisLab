from __future__ import annotations

import numpy as np

from .body import RigidBody6DOF
from .constraints import DistanceConstraint, PointWeldConstraint
from .forces import Spring


class RigidTetherJoint:
    """High-level facade that adds a DistanceConstraint."""
    def __init__(self, body_i: int, body_j: int, attach_i_local, attach_j_local, length: float):
        self.body_i = body_i
        self.body_j = body_j
        self.ri = np.asarray(attach_i_local, dtype=np.float64)
        self.rj = np.asarray(attach_j_local, dtype=np.float64)
        self.length = float(length)

    def attach(self, bodies: list[RigidBody6DOF]):
        return DistanceConstraint(bodies, self.body_i, self.body_j, self.ri, self.rj, self.length)


class WeldJoint:
    """High-level facade that adds a PointWeldConstraint (3 dof)."""
    def __init__(self, body_i: int, body_j: int, attach_i_local, attach_j_local):
        self.body_i = body_i
        self.body_j = body_j
        self.ri = np.asarray(attach_i_local, dtype=np.float64)
        self.rj = np.asarray(attach_j_local, dtype=np.float64)

    def attach(self, bodies: list[RigidBody6DOF]):
        return PointWeldConstraint(bodies, self.body_i, self.body_j, self.ri, self.rj)


class SoftTetherJoint:
    """Facade wrapping a Spring (soft tether)."""
    def __init__(self, body_i: int, body_j: int, attach_i_local, attach_j_local, k: float, c: float, rest_length: float):
        self.body_i = body_i
        self.body_j = body_j
        self.ri = np.asarray(attach_i_local, dtype=np.float64)
        self.rj = np.asarray(attach_j_local, dtype=np.float64)
        self.k = float(k)
        self.c = float(c)
        self.L0 = float(rest_length)

    def attach(self, bodies: list[RigidBody6DOF]):
        a = bodies[self.body_i]
        b = bodies[self.body_j]
        return Spring(a, b, self.ri, self.rj, self.k, self.c, self.L0)
