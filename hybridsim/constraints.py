from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .mathutil import quaternion_to_rotation_matrix, skew

Array = np.ndarray

class Constraint:
    """
    Base class for equality constraints C(q)=0.
    Provide velocity Jacobian J s.t. Cdot = J v.
    """
    def rows(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def index_map(self, world: "World") -> list[int]:  # pragma: no cover
        """Return indices of bodies in world.bodies touched by this constraint."""
        raise NotImplementedError

    def evaluate(self, world: "World") -> Array:  # pragma: no cover
        """Optional: position residual C(q) for diagnostics."""
        raise NotImplementedError

    def jacobian_local(self, world: "World") -> Array:  # pragma: no cover
        """
        Return J_local over involved bodies as blocks [vi, ωi, vj, ωj, ...].
        The solver will map these into the big J using index_map().
        """
        raise NotImplementedError

@dataclass
class DistanceConstraint(Constraint):
    """
    Keep distance between two attachment points equal to L.
    Scalar constraint: |p_b - p_a| - L = 0
    """
    body_a: "RigidBody6DOF"
    body_b: "RigidBody6DOF"
    r_a_local: Array
    r_b_local: Array
    L: float

    def rows(self) -> int:
        return 1

    def index_map(self, world: "World") -> list[int]:
        return [world.bodies.index(self.body_a), world.bodies.index(self.body_b)]

    def _points(self) -> tuple[Array, Array, Array, Array]:
        Ra = quaternion_to_rotation_matrix(self.body_a.orientation)
        Rb = quaternion_to_rotation_matrix(self.body_b.orientation)
        ra = Ra @ self.r_a_local
        rb = Rb @ self.r_b_local
        pa = self.body_a.position + ra
        pb = self.body_b.position + rb
        return pa, pb, ra, rb

    def evaluate(self, world: "World") -> Array:
        pa, pb, *_ = self._points()
        return np.array([np.linalg.norm(pb - pa) - self.L])

    def jacobian_local(self, world: "World") -> Array:
        pa, pb, ra, rb = self._points()
        d = pb - pa
        nrm = np.linalg.norm(d)
        n = np.array([1.0, 0.0, 0.0]) if nrm < 1e-12 else d / nrm
        Jv_a = -n.reshape(1, 3)
        Jw_a = -n.reshape(1, 3) @ skew(ra)
        Jv_b = +n.reshape(1, 3)
        Jw_b = +n.reshape(1, 3) @ skew(rb)
        return np.hstack([Jv_a, Jw_a, Jv_b, Jw_b])  # (1, 12)

@dataclass
class PointWeldConstraint(Constraint):
    """Enforce p_a == p_b (3 equations)."""
    body_a: "RigidBody6DOF"
    body_b: "RigidBody6DOF"
    r_a_local: Array
    r_b_local: Array

    def rows(self) -> int:
        return 3

    def index_map(self, world: "World") -> list[int]:
        return [world.bodies.index(self.body_a), world.bodies.index(self.body_b)]

    def _points(self) -> tuple[Array, Array, Array, Array]:
        Ra = quaternion_to_rotation_matrix(self.body_a.orientation)
        Rb = quaternion_to_rotation_matrix(self.body_b.orientation)
        ra = Ra @ self.r_a_local
        rb = Rb @ self.r_b_local
        pa = self.body_a.position + ra
        pb = self.body_b.position + rb
        return pa, pb, ra, rb

    def evaluate(self, world: "World") -> Array:
        pa, pb, *_ = self._points()
        return pb - pa

    def jacobian_local(self, world: "World") -> Array:
        _, _, ra, rb = self._points()
        I = np.eye(3)
        Ja = np.hstack([-I, -skew(ra)])
        Jb = np.hstack([+I, +skew(rb)])
        return np.hstack([Ja, Jb])  # (3, 12)
