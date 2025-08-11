from __future__ import annotations
import numpy as np
from dataclasses import dataclass

Array = np.ndarray

class Force:
    def apply(self, body: "RigidBody6DOF", t: float | None = None) -> None:
        raise NotImplementedError

@dataclass
class Gravity(Force):
    g: Array = np.array([0.0, 0.0, -9.81], dtype=float)

    def __post_init__(self):
        # Coerce to float ndarray even if user passed a list/tuple
        self.g = np.asarray(self.g, dtype=float)

    def apply(self, body: "RigidBody6DOF", t: float | None = None) -> None:
        if body.mass > 0:
            body.apply_force(body.mass * self.g)

@dataclass
class Drag(Force):
    rho: float
    Cd: float
    area: float
    model: str = "quadratic"

    def apply(self, body: "RigidBody6DOF", t: float | None = None) -> None:
        v = body.linear_velocity
        speed = np.linalg.norm(v)
        if speed < 1e-12:
            return
        if self.model == "quadratic":
            f = -0.5 * self.rho * self.Cd * self.area * speed * v
        elif self.model == "linear":
            f = -self.Cd * v
        else:
            raise ValueError(f"Unknown drag model: {self.model}")
        body.apply_force(f)

# (Spring unchanged)


@dataclass
class Spring:
    """
    Optional soft spring (legacy/auxiliary) between two bodies at attachment points.
    Not part of the DAE constraint path.
    """
    body_a: "RigidBody6DOF"
    body_b: "RigidBody6DOF"
    r_a_local: Array
    r_b_local: Array
    rest_length: float
    k: float
    c: float

    def apply(self) -> None:
        from .mathutil import quaternion_to_rotation_matrix
        Ra = quaternion_to_rotation_matrix(self.body_a.orientation)
        Rb = quaternion_to_rotation_matrix(self.body_b.orientation)
        pa = self.body_a.position + Ra @ self.r_a_local
        pb = self.body_b.position + Rb @ self.r_b_local
        d = pb - pa
        L = np.linalg.norm(d)
        if L < 1e-12:
            return
        n = d / L
        # spring
        fs = self.k * (L - self.rest_length) * n
        # damping along the line
        va = self.body_a.linear_velocity + np.cross(self.body_a.angular_velocity, Ra @ self.r_a_local)
        vb = self.body_b.linear_velocity + np.cross(self.body_b.angular_velocity, Rb @ self.r_b_local)
        rel_v = vb - va
        fd = self.c * np.dot(rel_v, n) * n
        f = fs + fd
        self.body_a.apply_force(-f, point_world=pa)
        self.body_b.apply_force(+f, point_world=pb)
