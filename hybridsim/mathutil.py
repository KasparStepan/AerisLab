from __future__ import annotations
import numpy as np

Array = np.ndarray

def normalize_quaternion(q: Array) -> Array:
    """Normalize quaternion [x, y, z, w] to unit length."""
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero quaternion cannot be normalized.")
    return q / n

def quaternion_multiply(q1: Array, q2: Array) -> Array:
    """Hamilton product q1 ⊗ q2, both [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)

def quaternion_to_rotation_matrix(q: Array) -> Array:
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=float)

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Array:
    """XYZ Euler to quaternion [x,y,z,w]."""
    cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5);  sr = np.sin(roll * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([x, y, z, w], dtype=float)

def quaternion_to_euler(q: Array) -> Array:
    """Quaternion [x,y,z,w] -> XYZ Euler."""
    x, y, z, w = q
    roll  = np.arctan2(2*(y*w + x*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(2*(w*y - x*z))
    yaw   = np.arctan2(2*(x*w + y*z), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=float)

def skew(v: Array) -> Array:
    """Skew-symmetric matrix S(v) such that S(v) @ w = v × w."""
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=float)
