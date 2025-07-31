import numpy as np
from .math_utils import normalize_quaternion, quaternion_multiply,quaternion_to_rotation_matrix

class RigidBody6DOF:
    def __init__(self,name, mass, inertia_tensor, position, orientation, linear_velocity, do all, radius=0.1):

        """
        Initialize a rigid body.
        :param orientation: quaternion as [w,x,y,z]
        """

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



    def apply_force(self, force, point_world=None):
        """Apply a force to the body, optionally at a specific point in world coordinates."""
        self.force += np.array(force, dtype=np.float64)
        if point_world is not None:
            point_world = np.array(point_world, dtype=np.float64)
            r = point_world - self.position
            self.torque += np.cross(r, force)

    def clear_forces(self):
        """Clear the accumulated forces and torques."""
        self.force.fill(0)
        self.torque.fill(0)

    def get_inertia_world(self):
        """Compute the inertia tensor in world coordinates."""
        R = quaternion_to_rotation_matrix(self.orientation)
        return R @ self.inertia_tensor_body @ R.T



