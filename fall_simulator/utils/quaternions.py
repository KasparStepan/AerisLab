import numpy as np

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_derivative(q, omega):
    omega_quat = np.array([0.0, *omega])
    dqdt = 0.5 * quaternion_multiply(q, omega_quat)
    return dqdt

def normalize_quaternion(q):
    return q / np.linalg.norm(q)
