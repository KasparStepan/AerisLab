from __future__ import annotations

import numpy as np

from .body import RigidBody6DOF

Array = np.ndarray

def skew(v: Array) -> Array:
    """
    Skew-symmetric matrix S(v) s.t. S(v) @ w = v × w.
    v: (3,) -> (3,3)
    """
    vx, vy, vz = v
    return np.array([
        [0.0, -vz,  vy],
        [vz,  0.0, -vx],
        [-vy, vx,  0.0]
    ], dtype=np.float64)

class Constraint:
    """Abstract base for constraints used in KKT solve."""
    def rows(self) -> int: raise NotImplementedError
    def index_map(self) -> list[int]: raise NotImplementedError  # body indices in world order
    def evaluate(self) -> Array: raise NotImplementedError       # C(q)
    def jacobian(self) -> Array: raise NotImplementedError       # J such that Cdot = J v_g (v_g stacks [v, w] per-body)

    # Helper for velocity-level residual:
    def c_dot(self, vstack: Array) -> Array:
        J = self.jacobian()
        return J @ vstack


class DistanceConstraint(Constraint):
    """
    Enforce fixed separation between two attachment points.
    Scalar constraint: C = 0.5 (||d||^2 - L^2) = 0

    Ċ = d · (vA + ωA×ra_w - vB - ωB×rb_w)
      = [d^T, (ra_w × d)^T, -d^T, -(rb_w × d)^T] [vA, ωA, vB, ωB]
    """
    def __init__(
        self,
        world_bodies: list[RigidBody6DOF],
        body_i: int,
        body_j: int,
        attach_i_local: Array,
        attach_j_local: Array,
        length: float,
    ) -> None:
        self.bodies = world_bodies
        self.i = body_i
        self.j = body_j
        self.ri_local = np.asarray(attach_i_local, dtype=np.float64)
        self.rj_local = np.asarray(attach_j_local, dtype=np.float64)
        self.L = float(length)

    def rows(self) -> int: return 1
    def index_map(self) -> list[int]: return [self.i, self.j]

    def _geom(self) -> tuple[Array, Array, Array, Array, Array]:
        bi = self.bodies[self.i]
        bj = self.bodies[self.j]
        Ri = bi.rotation_world()
        Rj = bj.rotation_world()
        ri_w = Ri @ self.ri_local
        rj_w = Rj @ self.rj_local
        pi = bi.p + ri_w
        pj = bj.p + rj_w
        d = pi - pj
        return d, ri_w, rj_w, pi, pj

    def evaluate(self) -> Array:
        d, *_ = self._geom()
        return np.array([0.5*(d @ d - self.L*self.L)], dtype=np.float64)

    def jacobian(self) -> Array:
        d, ri_w, rj_w, *_ = self._geom()
        J = np.zeros((1, 12), dtype=np.float64)  # [v_i, w_i, v_j, w_j]
        J[0, 0:3] = d
        J[0, 3:6] = np.cross(ri_w, d)  # (ri_w × d)
        J[0, 6:9] = -d
        J[0, 9:12] = -np.cross(rj_w, d)
        return J


class PointWeldConstraint(Constraint):
    """
    Enforce coincidence of two attachment points (3 equations):
    C = pa - pb = 0

    Velocity-level:
      Ċ = vA + ωA×ra_w - vB - ωB×rb_w
         = [I, -skew(ra_w), -I,  +skew(rb_w)] [vA, ωA, vB, ωB]
    """
    def __init__(
        self,
        world_bodies: list[RigidBody6DOF],
        body_i: int,
        body_j: int,
        attach_i_local: Array,
        attach_j_local: Array,
    ) -> None:
        self.bodies = world_bodies
        self.i = body_i
        self.j = body_j
        self.ri_local = np.asarray(attach_i_local, dtype=np.float64)
        self.rj_local = np.asarray(attach_j_local, dtype=np.float64)

    def rows(self) -> int: return 3
    def index_map(self) -> list[int]: return [self.i, self.j]

    def _geom(self):
        bi = self.bodies[self.i]
        bj = self.bodies[self.j]
        Ri = bi.rotation_world()
        Rj = bj.rotation_world()
        ri_w = Ri @ self.ri_local
        rj_w = Rj @ self.rj_local
        pi = bi.p + ri_w
        pj = bj.p + rj_w
        return ri_w, rj_w, pi, pj

    def evaluate(self) -> Array:
        ri_w, rj_w, pi, pj = self._geom()
        return (pi - pj).astype(np.float64)

    def jacobian(self) -> Array:
        ri_w, rj_w, *_ = self._geom()
        J = np.zeros((3, 12), dtype=np.float64)
        J[:, 0:3] = np.eye(3)
        J[:, 3:6] = -skew(ri_w)
        J[:, 6:9] = -np.eye(3)
        J[:, 9:12] = skew(rj_w)
        return J
