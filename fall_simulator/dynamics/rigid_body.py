import numpy as np
from fall_simulator.utils.quaternions import quaternion_derivative, normalize_quaternion
from fall_simulator.dynamics.forces import gravity_force, parachute_drag_force

class RigidBody6DOF:
    def __init__(self, mass, inertia_tensor, position, velocity, orientation, angular_velocity):
        self.mass = mass
        self.inertia = inertia_tensor
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.orientation = np.array(orientation, dtype=float)
        self.angular_velocity = np.array(angular_velocity, dtype=float)
        self.gravity = np.array([0.0, 0.0, -9.81])

        # 🚨 Add this
        self.parachute_deployed = False


    def derivative(self):
        """
        Computes the time derivative of the full state vector with gravity and parachute drag.
        """
        # Forces
        f_gravity = gravity_force(self.mass)
        f_parachute = parachute_drag_force(self.velocity, self.parachute_deployed)

        total_force = f_gravity + f_parachute

        # Translational dynamics
        deriv_position = self.velocity
        deriv_velocity = total_force / self.mass

        # Rotational dynamics (no torques yet)
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

    def update_euler(self, dt):
        """
        Updates the state using explicit Euler method (legacy method).
        """
        derivative = self.derivative()
        new_state = self.get_state() + derivative * dt
        self.set_state(new_state)
