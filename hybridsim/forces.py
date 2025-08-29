from __future__ import annotations
import numpy as np
from typing import Protocol, Callable

class Force(Protocol):
    """Per-body force interface."""
    def apply(self, body, t: float | None = None) -> None: ...  # noqa: E701

class Gravity:
    """Uniform gravity. Adds m*g at COM."""
    def __init__(self, g: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=np.float64)) -> None:
        self.g = np.array(g, dtype=np.float64)

    def apply(self, body, t: float | None = None) -> None:
        body.apply_force(body.mass * self.g)

class Drag:
    """Aerodynamic drag.
    Modes:
      - 'quadratic': F = -0.5 * ρ * Cd * A * ||v|| * v (world frame)
      - 'linear'   : F = -k * v
    Area can be a float or a callable A(t) for time-varying canopy area.

    Args:
        rho: air density [kg/m^3]
        Cd: drag coefficient [-]
        area: float or Callable[[float|None], float]
        mode: 'quadratic' or 'linear'
        k: linear drag coefficient [N·s/m] if mode='linear'
    """
    def __init__(
        self,
        rho: float = 1.225,
        Cd: float = 1.0,
        area: float | Callable[[float | None], float] = 0.0,
        mode: str = "quadratic",
        k: float = 0.0,
    ) -> None:
        self.rho = float(rho)
        self.Cd = float(Cd)
        self.area = area
        self.mode = mode
        self.k = float(k)

    def _area(self, t: float | None) -> float:
        if callable(self.area):
            return float(self.area(t))
        return float(self.area)

    def apply(self, body, t: float | None = None) -> None:
        v = body.v
        if self.mode == "quadratic":
            A = self._area(t)
            speed = float(np.linalg.norm(v))
            if speed > 0.0 and A > 0.0:
                F = -0.5 * self.rho * self.Cd * A * speed * v
                body.apply_force(F)
        elif self.mode == "linear":
            body.apply_force(-self.k * v)
        else:
            raise ValueError(f"Unknown drag mode: {self.mode}")

class Spring:
    """Soft tether between two bodies (not part of rigid DAE).
    Hooke + line damping along the current line of centers.

    Args:
        body_i, body_j: RigidBody6DOF instances
        r_i_b, r_j_b  : (3,) attachment points in body frames
        k             : stiffness [N/m]
        c             : damping [N·s/m]
        L0            : rest length [m]
    """
    def __init__(
        self, body_i, body_j,
        r_i_b: np.ndarray, r_j_b: np.ndarray,
        k: float, c: float, L0: float
    ) -> None:
        self.body_i = body_i
        self.body_j = body_j
        self.r_i_b = np.array(r_i_b, dtype=np.float64)
        self.r_j_b = np.array(r_j_b, dtype=np.float64)
        self.k = float(k)
        self.c = float(c)
        self.L0 = float(L0)

    def apply(self, t: float | None = None) -> None:
        """Apply equal/opposite forces to both bodies."""
        bi, bj = self.body_i, self.body_j
        Ri, Rj = bi.rotation_world(), bj.rotation_world()
        ri_w, rj_w = Ri @ self.r_i_b, Rj @ self.r_j_b
        xi, xj = bi.p + ri_w, bj.p + rj_w
        vi_pt = bi.v + np.cross(bi.w, ri_w)
        vj_pt = bj.v + np.cross(bj.w, rj_w)

        d = xj - xi
        L = float(np.linalg.norm(d))
        if L < 1e-12:
            return
        n = d / L
        v_rel_n = (vj_pt - vi_pt).dot(n)
        Fmag = -self.k * (L - self.L0) - self.c * v_rel_n
        F = Fmag * n

        bi.apply_force(-F, point_world=xi)
        bj.apply_force(+F, point_world=xj)
