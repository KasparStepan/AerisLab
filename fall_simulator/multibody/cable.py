import numpy as np

class Cable:
    def __init__(self, rest_length, stiffness, damping):
        """
        Represents an elastic cable between two bodies.

        Parameters:
        - rest_length: float, natural length of the cable (m)
        - stiffness: float, spring constant (N/m)
        - damping: float, damping coefficient (N·s/m)
        """
        self.rest_length = rest_length
        self.stiffness = stiffness
        self.damping = damping

    def compute_force(self, pos_a, vel_a, pos_b, vel_b):
        """
        Computes the force applied on body A by the cable connected to body B.

        Parameters:
        - pos_a: np.ndarray, position of body A
        - vel_a: np.ndarray, velocity of body A
        - pos_b: np.ndarray, position of body B
        - vel_b: np.ndarray, velocity of body B

        Returns:
        - np.ndarray, force vector applied to body A (N)
        """
        delta_pos = pos_b - pos_a
        delta_vel = vel_b - vel_a

        distance = np.linalg.norm(delta_pos)
        if distance == 0.0:
            direction = np.zeros(3)
        else:
            direction = delta_pos / distance

        stretch = distance - self.rest_length
        relative_speed = np.dot(delta_vel, direction)

        spring_force = self.stiffness * stretch
        damping_force = self.damping * relative_speed

        total_force = (spring_force + damping_force) * direction
        return total_force
