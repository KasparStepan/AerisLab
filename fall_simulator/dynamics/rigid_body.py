import numpy as np
from fall_simulator.utils.quaternions import quaternion_derivative, normalize_quaternion

class RigidBody6DOF:
    def __init__(self, mass, inertia_tensor, position, velocity, orientation, angular_velocity):
        self.mass = mass
        self.inertia = inertia_tensor

        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.orientation = np.array(orientation, dtype=float)
        self.angular_velocity = np.array(angular_velocity, dtype=float)

        self.gravity = np.array([0.0, 0.0, -9.81])

    def update(self, dt):
        # Forces
        force = self.mass * self.gravity
        acceleration = force / self.mass

        # Translational update
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # Torques (no external torques yet)
        torque = np.array([0.0, 0.0, 0.0])
        angular_acceleration = np.linalg.inv(self.inertia) @ (torque - np.cross(self.angular_velocity, self.inertia @ self.angular_velocity))

        # Rotational update
        self.angular_velocity += angular_acceleration * dt
        dqdt = quaternion_derivative(self.orientation, self.angular_velocity)
        self.orientation += dqdt * dt
        self.orientation = normalize_quaternion(self.orientation)
