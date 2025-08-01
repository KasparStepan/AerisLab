import numpy as np

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_derivative(q, omega):
    wx, wy, wz = omega
    qx, qy, qz, qw = q
    dq = 0.5 * np.array([
        qw * wx + qy * wz - qz * wy,
        qw * wy - qx * wz + qz * wx,
        qw * wz + qx * wy - qy * wx,
        -qx * wx - qy * wy - qz * wz
    ])
    return dq

def quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)]
    ])

def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax(np.diag(R))
        j = (i + 1) % 3
        k = (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        qw = (R[k, j] - R[j, k]) / (2.0 * s)
        qx = s / 4.0
        qy = (R[i, j] + R[j, i]) / (2.0 * s)
        qz = (R[i, k] + R[k, i]) / (2.0 * s)
    return np.array([qx, qy, qz, qw])

def quaternion_multiply(q1, q2):
    q1x, q1y, q1z, q1w = q1
    q2x, q2y, q2z, q2w = q2
    return np.array([
        q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y,
        q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x,
        q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w,
        q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
    ])

def quaternion_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quaternion_inverse(q):
    norm_sq = np.dot(q, q)
    if norm_sq == 0:
        raise ValueError("Cannot compute inverse of a zero quaternion.")
    return quaternion_conjugate(q) / norm_sq

def quaternion_slerp(q1, q2, t):
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = (1 - t) * q1 + t * q2
        return normalize_quaternion(result)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    s1 = np.sin(theta_0 - theta_t) / sin_theta_0
    s2 = sin_theta_t / sin_theta_0
    return s1 * q1 + s2 * q2

def quaternion_to_euler(q):
    qx, qy, qz, qw = q
    roll = np.arctan2(2 * (qy * qw + qx * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(2 * (qw * qy - qx * qz))
    yaw = np.arctan2(2 * (qx * qw + qy * qz), 1 - 2 * (qy**2 + qz**2))
    return np.array([roll, pitch, yaw])

def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qx, qy, qz, qw])