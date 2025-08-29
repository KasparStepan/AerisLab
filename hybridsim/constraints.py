from __future__ import annotations
import numpy as np
from .mathutil import skew

class Constraint:
    """Abstract constraint interface for rigid equality constraints C(q) = 0.
    Each constraint returns:
        - rows(): number of scalar equations m
        - index_map(world): list of involved body indices
        - evaluate(world): C(q) ∈ R^m (for diagnostics/Baumgarte)
        - jacobian_local(world): velocity Jacobian wrt [v, ω] of involved bodies,
          stacked horizontally as (m, 6*nb). World assembles global J.
    """
    def rows(self) -> int: raise NotImplementedError
    def index_map(self, world) -> list[int]: raise NotImplementedError
    def evaluate(self, world) -> np.ndarray: raise NotImplementedError
    def jacobian_local(self, world) -> np.ndarray: raise NotImplementedError

class DistanceConstraint(Constraint):
    """Keep two attachment points at fixed distance L.

    Let A on body i at r_i^B, B on body j at r_j^B.
    World points: x_i = p_i + R_i r_i, x_j = p_j + R_j r_j.
    Define d = x_j - x_i, constraint C = 0.5 (||d||^2 - L^2) = 0 [scalar].

    Velocity Jacobian (wrt [v_i, ω_i, v_j, ω_j]):
        ∂C/∂v_i = -d^T
        ∂C/∂v_j = +d^T
        ∂C/∂ω_i = -d^T [r_i]_x
        ∂C/∂ω_j = +d^T [r_j]_x
    """
    def __init__(self, body_i_idx: int, body_j_idx: int,
                 r_i_b: np.ndarray, r_j_b: np.ndarray, L: float) -> None:
        self.i = int(body_i_idx)
        self.j = int(body_j_idx)
        self.r_i_b = np.array(r_i_b, dtype=np.float64)
        self.r_j_b = np.array(r_j_b, dtype=np.float64)
        self.L = float(L)

    def rows(self) -> int: return 1

    def index_map(self, world) -> list[int]: return [self.i, self.j]

    def evaluate(self, world) -> np.ndarray:
        bi, bj = world.bodies[self.i], world.bodies[self.j]
        Ri, Rj = bi.rotation_world(), bj.rotation_world()
        ri_w, rj_w = Ri @ self.r_i_b, Rj @ self.r_j_b
        xi, xj = bi.p + ri_w, bj.p + rj_w
        d = xj - xi
        return np.array([0.5 * (d @ d - self.L * self.L)], dtype=np.float64)

    def jacobian_local(self, world) -> np.ndarray:
        bi, bj = world.bodies[self.i], world.bodies[self.j]
        Ri, Rj = bi.rotation_world(), bj.rotation_world()
        ri_w, rj_w = Ri @ self.r_i_b, Rj @ self.r_j_b
        xi, xj = bi.p + ri_w, bj.p + rj_w
        d = xj - xi
        dT = d.reshape(1, 3)
        Ji = np.hstack([-dT, -dT @ skew(ri_w)])
        Jj = np.hstack([+dT, +dT @ skew(rj_w)])
        return np.hstack([Ji, Jj])  # (1, 12)

class PointWeldConstraint(Constraint):
    """Enforce two attachment points to coincide in world: x_j - x_i = 0 (3 eq).
    Velocity Jacobian blocks:
        for body i: [-I, -[r_i]_x]
        for body j: [+I, +[r_j]_x]
    """
    def __init__(self, body_i_idx: int, body_j_idx: int,
                 r_i_b: np.ndarray, r_j_b: np.ndarray) -> None:
        self.i = int(body_i_idx)
        self.j = int(body_j_idx)
        self.r_i_b = np.array(r_i_b, dtype=np.float64)
        self.r_j_b = np.array(r_j_b, dtype=np.float64)

    def rows(self) -> int: return 3

    def index_map(self, world) -> list[int]: return [self.i, self.j]

    def evaluate(self, world) -> np.ndarray:
        bi, bj = world.bodies[self.i], world.bodies[self.j]
        Ri, Rj = bi.rotation_world(), bj.rotation_world()
        ri_w, rj_w = Ri @ self.r_i_b, Rj @ self.r_j_b
        xi, xj = bi.p + ri_w, bj.p + rj_w
        return (xj - xi).astype(np.float64)

    def jacobian_local(self, world) -> np.ndarray:
        bi, bj = world.bodies[self.i], world.bodies[self.j]
        Ri, Rj = bi.rotation_world(), bj.rotation_world()
        ri_w, rj_w = Ri @ self.r_i_b, Rj @ self.r_j_b
        Ji = np.hstack([-np.eye(3), -skew(ri_w)])
        Jj = np.hstack([+np.eye(3), +skew(rj_w)])
        return np.hstack([Ji, Jj])  # (3, 12)
