from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from .mathutil import normalize_quaternion, quaternion_multiply, quaternion_to_rotation_matrix

Array = np.ndarray

@dataclass
class RigidBody6DOF:
    """
    6-DoF rigid body with:
      - state: position p, quaternion q=[x,y,z,w], linear vel v, angular vel ω
      - properties: mass m, body-frame inertia I_body
      - runtime accumulators: force (R^3), torque (R^3)
      - per-body force *objects* in `forces` (list of callables with .apply)
    """
    name: str
    mass: float
    inertia_tensor_body: Array  # 3x3
    position: Array
    orientation: Array          # [x, y, z, w]
    linear_velocity: Array
    angular_velocity: Array
    radius: float = 0.1

    # accumulators used each step
    force: Array = field(default_factory=lambda: np.zeros(3))
    torque: Array = field(default_factory=lambda: np.zeros(3))

    # list of per-body forces (e.g., Drag(...), custom forces)
    forces: list = field(default_factory=list)

    # integrator scratch (filled by solver)
    _a_lin: Array = field(default_factory=lambda: np.zeros(3))
    _a_ang: Array = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.orientation = normalize_quaternion(np.array(self.orientation, dtype=float))
        self.linear_velocity = np.array(self.linear_velocity, dtype=float)
        self.angular_velocity = np.array(self.angular_velocity, dtype=float)
        self.inertia_tensor_body = np.array(self.inertia_tensor_body, dtype=float)
        self.inv_mass = 0.0 if self.mass == 0 else 1.0 / self.mass
        self.inv_inertia_body = (np.linalg.inv(self.inertia_tensor_body)
                                 if self.mass != 0 else np.zeros((3, 3)))

    def clear_forces(self) -> None:
        self.force[:] = 0.0
        self.torque[:] = 0.0

    def apply_force(self, force: Array, point_world: Array | None = None) -> None:
        f = np.array(force, dtype=float)
        self.force += f
        if point_world is not None:
            r = np.array(point_world, dtype=float) - self.position
            self.torque += np.cross(r, f)

    def apply_torque(self, torque: Array) -> None:
        self.torque += np.array(torque, dtype=float)

    def rotation_world(self) -> Array:
        return quaternion_to_rotation_matrix(self.orientation)

    def inertia_world(self) -> Array:
        R = self.rotation_world()
        return R @ self.inertia_tensor_body @ R.T

    def mass_matrix_world(self) -> Array:
        M = np.zeros((6, 6))
        M[:3, :3] = np.eye(3) * self.mass
        M[3:, 3:] = self.inertia_world()
        return M

    def generalized_force(self) -> Array:
        return np.hstack([self.force, self.torque])

    def integrate_semi_implicit(self, dt: float) -> None:
        self.linear_velocity += self._a_lin * dt
        self.position += self.linear_velocity * dt
        self.angular_velocity += self._a_ang * dt
        dq = quaternion_multiply(self.orientation, np.hstack(([0.0], self.angular_velocity))) * 0.5
        self.orientation = normalize_quaternion(self.orientation + dq * dt)
