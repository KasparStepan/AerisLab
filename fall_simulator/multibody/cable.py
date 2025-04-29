import numpy as np

class Cable:
    def __init__(self, rest_length, stiffness, damping):
        self.rest_length = rest_length
        self.stiffness = stiffness
        self.damping = damping

    def compute_force(self, pos_a, vel_a, pos_b, vel_b):
        delta_pos = pos_b - pos_a
        delta_vel = vel_b - vel_a
        distance = np.linalg.norm(delta_pos)
        if distance == 0.0:
            return np.zeros(3)
        direction = delta_pos / distance
        stretch = distance - self.rest_length
        relative_speed = np.dot(delta_vel, direction)
        spring_force = self.stiffness * stretch
        damping_force = self.damping * relative_speed
        return (spring_force + damping_force) * direction
