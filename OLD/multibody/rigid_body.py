import numpy as np
from scipy.spatial.transform import Rotation as R
from .quaternions import quaternion_derivative, normalize_quaternion

class RigidBody6DOF:
    def __init__(self, mass, inertia_matrix, position, velocity, orientation, angular_velocity, area=1.0, drag_coefficient=0.5):
        self.mass = mass
        self.inertia_matrix = inertia_matrix  # 3x3 matrix

        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.orientation = R.from_quat(orientation)
        self.angular_velocity = np.array(angular_velocity, dtype=float)

        self.area = area
        self.drag_coefficient = drag_coefficient
        self.air_density = 1.225

        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    def apply_force(self, force, point=None):
        self.force += force
        if point is not None:
            r = point - self.position
            self.torque += np.cross(r, force)

    def reset_forces(self):
        self.force[:] = 0
        self.torque[:] = 0

    def update(self, dt, method='semi_euler'):
        from .integration import integrate
        integrate(self, dt, method)

    def _semi_implicit_euler(self, dt):
        acceleration = self.force / self.mass
        angular_acceleration = np.linalg.inv(self.inertia_matrix) @ self.torque

        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        self.angular_velocity += angular_acceleration * dt
        delta_orientation = R.from_rotvec(self.angular_velocity * dt)
        self.orientation = delta_orientation * self.orientation

        self.reset_forces()

    def get_state_vector(self):
        return np.hstack((
            self.position,
            self.velocity,
            self.orientation.as_quat(),
            self.angular_velocity
        ))

    def set_state_vector(self, state):
        self.position = state[0:3]
        self.velocity = state[3:6]
        self.orientation = R.from_quat(state[6:10])
        self.angular_velocity = state[10:13]

    def compute_state_derivative(self):
        acceleration = self.force / self.mass
        angular_acceleration = np.linalg.inv(self.inertia_matrix) @ self.torque
        q_deriv = quaternion_derivative(self.orientation.as_quat(), self.angular_velocity)
        return np.hstack((self.velocity, acceleration, q_deriv, angular_acceleration))

