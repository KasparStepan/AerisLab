from __future__ import annotations
import numpy as np

Array = np.ndarray

# All quaternions use scalar-first convention q = [w, x, y, z]

def quat_normalize(q: Array) -> Array:
    """Return unit quaternion (float64). q shape: (4,)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_mul(q1: Array, q2: Array) -> Array:
    """
    Hamilton product q = q1 ⊗ q2 (scalar-first).
    q1, q2: (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_to_rotmat(q: Array) -> Array:
    """
    Rotation matrix R(q) mapping body->world (3,3).
    q: (4,), unit quaternion (scalar-first).
    """
    q = quat_normalize(q)
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    return np.array([
        [ww + xx - yy - zz, 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),     ww - xx - yy + zz]
    ], dtype=np.float64)


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


def quat_derivative(q: Array, omega: Array) -> Array:
    """
    qdot = 0.5 * q ⊗ [0, ω]
    q: (4,), ω: (3,) -> (4,)
    """
    w = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float64)
    return 0.5 * quat_mul(q, w)
