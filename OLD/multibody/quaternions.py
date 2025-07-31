import numpy as np

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_derivative(q, omega):
    # q: [x, y, z, w], omega: [wx, wy, wz]
    wx, wy, wz = omega
    qx, qy, qz, qw = q
    dq = 0.5 * np.array([
        qw * wx + qy * wz - qz * wy,
        qw * wy - qx * wz + qz * wx,
        qw * wz + qx * wy - qy * wx,
        -qx * wx - qy * wy - qz * wz
    ])
    return dq
