from __future__ import annotations
import numpy as np

EPS = 1e-12

def skew(v: np.ndarray) -> np.ndarray:
    """Return 3×3 skew-symmetric matrix [v]_x such that [v]_x w = v × w.
    Args:
        v: (3,) vector.
    Returns:
        (3,3) skew-symmetric matrix.
    """
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -vz,  vy],
                     [vz,  0.0, -vx],
                     [-vy, vx,  0.0]], dtype=np.float64)

def q_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize unit quaternion (scalar-first [w, x, y, z])."""
    n = np.linalg.norm(q)
    if n < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return (q / n).astype(np.float64)

def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, scalar-first quaternions.
    Args:
        q1, q2: (4,), [w, x, y, z]
    Returns:
        q = q1 ⊗ q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def omega_to_qdot(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion derivative from angular velocity (world frame) using q' = 0.5 q ⊗ [0, ω].
    Args:
        q: (4,) unit quaternion [w, x, y, z]
        omega: (3,) angular velocity in world frame [rad/s].
    Returns:
        qdot: (4,)
    """
    w, x, y, z = q
    wx, wy, wz = omega
    # q ⊗ [0, ω]
    return 0.5 * np.array([
        - x*wx - y*wy - z*wz,
         w*wx + y*wz - z*wy,
         w*wy - x*wz + z*wx,
         w*wz + x*wy - y*wx
    ], dtype=np.float64)

def q_to_R(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from unit quaternion (scalar-first).
    Args:
        q: (4,) unit quaternion [w, x, y, z]
    Returns:
        R: (3,3) rotation matrix, maps body->world
    """
    w, x, y, z = q_normalize(q)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [ww + xx - yy - zz, 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),     ww - xx - yy + zz]
    ], dtype=np.float64)
