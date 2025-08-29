from __future__ import annotations
import numpy as np
from .mathutil import q_normalize, q_to_R

class RigidBody6DOF:
    """Rigid body with 6-DoF state (p, q, v, ω). Quaternion is scalar-first [w,x,y,z].

    Frames:
        - World W
        - Body B at COM

    State variables (world frame unless noted):
        p  : (3,) position of COM [m]
        q  : (4,) unit quaternion (B->W)
        v  : (3,) linear velocity [m/s]
        w  : (3,) angular velocity [rad/s]

    Properties:
        mass      : scalar [kg]
        I_body    : (3,3) inertia in body frame about COM [kg·m²]
        radius    : convenience for drag etc. [m]

    Accumulators (cleared each step):
        F         : (3,) resultant force [N]
        T         : (3,) resultant torque about COM in world frame [N·m]

    Integration:
        Semi-implicit Euler for (v, w), explicit quaternion update + renormalize.
    """
    __slots__ = (
        "name", "p", "q", "v", "w",
        "mass", "I_body", "I_body_inv", "radius",
        "F", "T",
        "a_lin", "a_ang",
        "forces"  # per-body forces
    )

    def __init__(
        self,
        name: str,
        mass: float,
        I_body: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
        v: np.ndarray | None = None,
        w: np.ndarray | None = None,
        radius: float = 0.0,
    ) -> None:
        self.name = name
        self.mass = float(mass)
        self.I_body = np.array(I_body, dtype=np.float64)
        self.I_body_inv = np.linalg.inv(self.I_body)
        self.radius = float(radius)

        self.p = np.array(p, dtype=np.float64)
        self.q = q_normalize(np.array(q, dtype=np.float64))
        self.v = np.zeros(3, dtype=np.float64) if v is None else np.array(v, dtype=np.float64)
        self.w = np.zeros(3, dtype=np.float64) if w is None else np.array(w, dtype=np.float64)

        self.F = np.zeros(3, dtype=np.float64)
        self.T = np.zeros(3, dtype=np.float64)
        self.a_lin = np.zeros(3, dtype=np.float64)
        self.a_ang = np.zeros(3, dtype=np.float64)

        self.forces: list = []

    # --- Kinematics / dynamics helpers ---
    def rotation_world(self) -> np.ndarray:
        """Rotation matrix R_BW: body->world."""
        return q_to_R(self.q)

    def inertia_world(self) -> np.ndarray:
        """World-frame inertia at current orientation: I_W = R I_body Rᵀ."""
        R = self.rotation_world()
        return R @ self.I_body @ R.T

    def mass_matrix_world(self) -> np.ndarray:
        """Generalized mass/inertia matrix M_i = diag(m I3, I_W)."""
        M = np.zeros((6, 6), dtype=np.float64)
        M[:3, :3] = self.mass * np.eye(3)
        M[3:, 3:] = self.inertia_world()
        return M

    def clear_forces(self) -> None:
        """Zero accumulators F, T."""
        self.F.fill(0.0)
        self.T.fill(0.0)

    def apply_force(self, F: np.ndarray, point_world: np.ndarray | None = None) -> None:
        """Apply force F at a world point. If point is None, acts at COM (no torque).

        Args:
            F: (3,) force in world frame [N]
            point_world: (3,) application point in world frame [m]
        """
        F = np.asarray(F, dtype=np.float64)
        self.F += F
        if point_world is not None:
            r = np.asarray(point_world, dtype=np.float64) - self.p
            self.T += np.cross(r, F)

    def apply_torque(self, T: np.ndarray) -> None:
        """Apply torque in world frame."""
        self.T += np.asarray(T, dtype=np.float64)

    def generalized_force(self) -> np.ndarray:
        """Return generalized force (6,), including gyroscopic bias term on torque:
           Q = [ F,
                 T - w × (I w) ]
        """
        Iw = self.inertia_world() @ self.w
        tau_eff = self.T - np.cross(self.w, Iw)
        return np.hstack([self.F, tau_eff])

    # --- Integration ---
    def integrate_semi_implicit(self, dt: float) -> None:
        """Semi-implicit Euler for (v, w) with quaternion renormalization."""
        self.v += self.a_lin * dt
        self.w += self.a_ang * dt
        self.p += self.v * dt

        # q' = 0.5 * q ⊗ [0, w]; explicit Euler is fine for small dt
        w0, x, y, z = self.q
        wx, wy, wz = self.w
        qdot = 0.5 * np.array([
            -x*wx - y*wy - z*wz,
             w0*wx + y*wz - z*wy,
             w0*wy - x*wz + z*wx,
             w0*wz + x*wy - y*wx,
        ], dtype=np.float64)
        self.q = q_normalize(self.q + qdot * dt)
