import numpy as np
from math_utils import normalize_quaternion, quaternion_to_rotation_matrix

class RigidBody6DOF:
    def __init__(self, name, mass, inertia_tensor, position, orientation, linear_velocity, angular_velocity, radius=0.1):
        self.name = name
        self.mass = mass
        self.inv_mass = 0.0 if mass == 0 else 1.0 / mass
        self.inertia_tensor_body = np.array(inertia_tensor, dtype=np.float64)
        self.inv_inertia_tensor_body = np.linalg.inv(self.inertia_tensor_body) if mass != 0 else np.zeros((3, 3), dtype=np.float64)
        self.position = np.array(position, dtype=np.float64)
        self.orientation = normalize_quaternion(np.array(orientation, dtype=np.float64))
        self.linear_velocity = np.array(linear_velocity, dtype=np.float64)
        self.angular_velocity = np.array(angular_velocity, dtype=np.float64)
        self.radius = radius
        self.force = np.zeros(3, dtype=np.float64)
        self.torque = np.zeros(3, dtype=np.float64)
        self.forces = []

    def apply_force(self, force, point_world=None):
        self.force += np.array(force, dtype=np.float64)
        if point_world is not None:
            point_world = np.array(point_world, dtype=np.float64)
            r = point_world - self.position
            self.torque += np.cross(r, force)

    def apply_impulse(self, impulse, point_world=None):
        if self.inv_mass != 0:
            self.linear_velocity += impulse * self.inv_mass
        if point_world is not None:
            r = point_world - self.position
            torque = np.cross(r, impulse)
            if not np.allclose(self.inv_inertia_tensor_body, 0):
                Iw = self.get_inertia_world()
                self.angular_velocity += np.linalg.inv(Iw) @ torque

    def clear_forces(self):
        self.force.fill(0)
        self.torque.fill(0)

    def get_inertia_world(self):
        R = quaternion_to_rotation_matrix(self.orientation)
        return R @ self.inertia_tensor_body @ R.T