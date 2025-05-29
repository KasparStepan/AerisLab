import numpy as np

class Cable:
    def __init__(self, rest_length, stiffness=0.0, damping=0.0, mode='spring'):
        """
        Cable between two bodies.

        mode:
            'spring'     - classic spring-damper (default)
            'constraint' - ideal inextensible connector (only transmits tension if stretched)
        """
        self.rest_length = rest_length
        self.stiffness = stiffness
        self.damping = damping
        self.mode = mode

    def compute_force(self, pos_a, vel_a, pos_b, vel_b):
        delta_pos = pos_b - pos_a
        delta_vel = vel_b - vel_a
        distance = np.linalg.norm(delta_pos)

        if distance == 0.0:
            return np.zeros(3)

        direction = delta_pos / distance

        if self.mode == 'spring':
            stretch = distance - self.rest_length
            relative_speed = np.dot(delta_vel, direction)
            spring_force = self.stiffness * stretch
            damping_force = self.damping * relative_speed
            return (spring_force + damping_force) * direction

        elif self.mode == 'constraint':
            # Only push if distance exceeds rest length
            if distance > self.rest_length:
                # Approximate constraint force (project relative motion)
                relative_speed = np.dot(delta_vel, direction)
                constraint_force = -relative_speed * direction * self.damping
                return constraint_force
            else:
                return np.zeros(3)

        else:
            raise ValueError(f"Unknown cable mode: {self.mode}")
