import numpy as np
from fall_simulator.utils.quaternions import quaternion_derivative, normalize_quaternion
from fall_simulator.dynamics.forces import gravity_force, drag_force

class RigidBody6DOF:
    def __init__(self, mass, inertia_tensor, position, velocity, orientation, angular_velocity, area=1.0, drag_coefficient=0.5):
        """
        General Rigid Body class for payload or parachute.
        """
        self.mass = mass
        self.inertia = inertia_tensor

        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.orientation = np.array(orientation, dtype=float)
        self.angular_velocity = np.array(angular_velocity, dtype=float)

        self.gravity = np.array([0.0, 0.0, -9.81])
        self.area = area
        self.drag_coefficient = drag_coefficient

    def derivative(self, external_force=np.zeros(3)):
        """
        Computes the time derivative of the full state vector under external forces.
        """
        # Gravity
        f_gravity = gravity_force(self.mass)

        # Aerodynamic drag
        f_drag = drag_force(self.velocity, 1.225, self.drag_coefficient, self.area)

        # Total force
        total_force = f_gravity + f_drag + external_force

        # Translational dynamics
        deriv_position = self.velocity
        deriv_velocity = total_force / self.mass

        # Rotational dynamics (no torques modeled yet)
        deriv_orientation = quaternion_derivative(self.orientation, self.angular_velocity)
        deriv_angular_velocity = np.zeros(3)

        return np.hstack((deriv_position, deriv_velocity, deriv_orientation, deriv_angular_velocity))

    def get_state(self):
        """
        Returns the current full state vector.
        """
        return np.hstack((self.position, self.velocity, self.orientation, self.angular_velocity))

    def set_state(self, state):
        """
        Sets the object's full state vector.
        """
        self.position = state[0:3]
        self.velocity = state[3:6]
        self.orientation = state[6:10]
        self.angular_velocity = state[10:13]
        self.orientation = normalize_quaternion(self.orientation)
