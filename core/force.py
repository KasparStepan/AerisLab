import numpy as np
from .math_utils import normalize_quaternion

class GravityForce:
    def __init__(self, gravity_vector=np.array([0,0,-9.81])):
        """Initialize the gravity force with a gravity vector."""
        self.gravity_vector = np.array(gravity_vector, dtype=np.float64)

    def apply(self, body):
        if body.mass == 0:
            F = body.mass * self.gravity_vector
            body.apply_force(F)
            print(f"[FORCE] Gravity on {body.name}: {F}")

class DragForce:
    def __init__(self, rho, Cd, area, model = 'quadratic'):
        self.rho = rho  # Density of the fluid
        self.Cd = Cd
        self.area = area
        self.model = model

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
            print(f"[FORCE] Drag on {body.name}: {drag_force}")    



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