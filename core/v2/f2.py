
import numpy as np

class GravityForce:
    def __init__(self, gravity_vector=np.array([0, 0, -9.81])):
        self.gravity_vector = np.array(gravity_vector, dtype=np.float64)

    def apply(self, body):
        if body.mass != 0:
            F = body.mass * self.gravity_vector
            body.apply_force(F)
            print(f"[FORCE] Gravity on {body.name}: {F}")

class DragForce:
    def __init__(self, rho, Cd, area, model='quadratic'):
        self.rho = rho
        self.Cd = Cd
        self.area = area
        self.model = model

    def apply(self, body):
        v = body.linear_velocity
        speed = np.linalg.norm(v)
        if speed > 1e-6:
            if self.model == 'quadratic':
                drag_force = -0.5 * self.rho * self.Cd * self.area * speed * (v / speed)
            elif self.model == 'linear':
                drag_force = -self.Cd * speed * (v / speed)
            else:
                raise ValueError(f"Unknown drag model: {self.model}")
            body.apply_force(drag_force)
            print(f"[FORCE] Drag on {body.name}: {drag_force}")