import numpy as np
from .math_utils import normalize_quaternion, quaternion_to_rotation_matrix

class GravityForce:
    def __init__(self, gravity_vector=np.array([0,0,-9.81])):
        """Initialize the gravity force with a gravity vector."""
        self.gravity_vector = np.array(gravity_vector, dtype=np.float64)

    def __str__(self):
        return f"GravityForce(gravity_vector={self.gravity_vector})"

    def apply(self, body):
        if body.mass != 0:
            F = body.mass * self.gravity_vector
            body.apply_force(F)
            print(f"[FORCE] Gravity on {body.name}: {F}")

class DragForce:
    def __init__(self, rho, Cd, area, model = 'quadratic'):
        self.rho = rho  # Density of the fluid
        self.Cd = Cd
        self.area = area
        self.model = model

    def __str__(self):
        return f"DragForce(rho={self.rho}, Cd={self.Cd}, area={self.area}, model={self.model})"

    def apply(self, body):
        v=body.linear_velocity
        speed = np.linalg.norm(v)

        if speed > 1e-6:
            if self.model == 'quadratic':
                drag_force = -0.5 * self.rho * self.Cd * self.area * speed * (v/speed)
                
            elif self.model == 'linear':
                drag_force = -self.Cd * speed * (v/speed)
            else:
                raise ValueError("Unknown drag model: {}".format(self.model))
            
            body.apply_force(drag_force)
            print(f"[FORCE2] Drag on {body.name}: {drag_force},real")    



class MomentumFluxForce:
    def __init__(self, rho, area, normal):
        self.rho = rho; self.A = area
        n = np.array(normal, float)
        self.normal = n / np.linalg.norm(n)
        self.time = 0.0
    
    def apply(self, body):
        v = body.linear_velocity
        v_n = np.dot(v,self.normal)
        if v_n < 0:  # incoming
            mass_flow = self.rho * self.A * abs(v_n)
            force = mass_flow * v
            body.apply_force(force)
            print(f"[FORCE] Momentum-flux on {body.name}: {force}")


class SpringForce:
    def __init__(self, body1, body2, local_point1, local_point2, rest_length, stiffness, damping):
        self.body1 = body1
        self.body2 = body2
        self.local_point1 = np.array(local_point1, dtype=np.float64)
        self.local_point2 = np.array(local_point2, dtype=np.float64)
        self.rest_length = rest_length
        self.k = stiffness
        self.d = damping

    def apply(self):
        # World positions of attachment points
        R1 = quaternion_to_rotation_matrix(self.body1.orientation)
        R2 = quaternion_to_rotation_matrix(self.body2.orientation)
        p1 = self.body1.position + R1 @ self.local_point1
        p2 = self.body2.position + R2 @ self.local_point2
        d = p2 - p1
        dist = np.linalg.norm(d)
        if dist > 1e-6:
            dir = d / dist
            # Spring force
            f_spring = self.k * (dist - self.rest_length) * dir
            # Damping force based on relative velocity
            v1 = self.body1.linear_velocity + np.cross(self.body1.angular_velocity, R1 @ self.local_point1)
            v2 = self.body2.linear_velocity + np.cross(self.body2.angular_velocity, R2 @ self.local_point2)
            rel_vel = v2 - v1
            f_damping = self.d * np.dot(rel_vel, dir) * dir
            total_force = f_spring + f_damping
            self.body1.apply_force(-total_force, p1)
            self.body2.apply_force(total_force, p2)