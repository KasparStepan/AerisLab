import numpy as np
from math_utils import quaternion_to_rotation_matrix

class Joint:
    def __init__(self, body1, body2, local_point1, local_point2, joint_type='ball_socket'):
        self.body1 = body1
        self.body2 = body2
        self.local_point1 = np.array(local_point1, dtype=np.float64)
        self.local_point2 = np.array(local_point2, dtype=np.float64)
        self.joint_type = joint_type

    def get_attachment_points(self):
        R1 = quaternion_to_rotation_matrix(self.body1.orientation)
        R2 = quaternion_to_rotation_matrix(self.body2.orientation)
        p1 = self.body1.position + R1 @ self.local_point1
        p2 = self.body2.position + R2 @ self.local_point2
        return p1, p2

    def apply_constraint(self):
        if self.joint_type == 'ball_socket':
            p1, p2 = self.get_attachment_points()
            r1 = p1 - self.body1.position
            r2 = p2 - self.body2.position
            v1 = self.body1.linear_velocity + np.cross(self.body1.angular_velocity, r1)
            v2 = self.body2.linear_velocity + np.cross(self.body2.angular_velocity, r2)
            rel_vel = v2 - v1
            C = p2 - p1
            beta = 0.1
            bias = -beta * C / 0.01
            if self.body1.inv_mass == 0 and self.body2.inv_mass == 0:
                return
            I1_inv = np.linalg.inv(self.body1.get_inertia_world()) if self.body1.inv_mass != 0 else np.zeros((3, 3))
            I2_inv = np.linalg.inv(self.body2.get_inertia_world()) if self.body2.inv_mass != 0 else np.zeros((3, 3))
            K = np.zeros((3, 3))
            for i in range(3):
                e = np.zeros(3); e[i] = 1
                cross_r1_e = np.cross(r1, e)
                cross_r2_e = np.cross(r2, e)
                K[:, i] = (self.body1.inv_mass + self.body2.inv_mass) * e + \
                          np.cross(I1_inv @ cross_r1_e, r1) + np.cross(I2_inv @ cross_r2_e, r2)
            lambda_ = np.linalg.solve(K + 1e-6 * np.eye(3), -(rel_vel + bias))
            impulse = lambda_
            self.body1.apply_impulse(-impulse, p1)
            self.body2.apply_impulse(impulse, p2)