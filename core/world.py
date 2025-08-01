import numpy as np
from .integrators import integrate_rigid_body
from .data_logger import Logger

class World:
    def __init__(self, dt, integrator='semi'):
        self.bodies = []
        self.global_forces = []
        self.interaction_forces = []
        self.dt = dt
        self.time = 0.0
        self.integrator = integrator
        self.logger = Logger()

    def add_body(self, body):
        self.bodies.append(body)

    def add_global_force(self, force):
        self.global_forces.append(force)
        print(f"Added global force: {force}")

    def add_interaction_force(self, force):
        self.interaction_forces.append(force)

    def step(self):
        # Clear forces
        for body in self.bodies:
            body.clear_forces()
        # Apply global forces
        for force in self.global_forces:
            for body in self.bodies:
                if body.position[2] > 0:
                    # Only apply forces to bodies above ground
                    print(f"Applying global force {force} to body {body}")
                    force.apply(body)
                else:
                    body.linear_velocity[2] = 0  # Stop bodies below ground
        # Apply body-specific forces
        for body in self.bodies:
            for force in body.forces:
                force.apply(body)
                print(f"Applied drag force {force} to body {body}")
        # Apply interaction forces
        for force in self.interaction_forces:
            print(f"Applying interaction force: {force}")
            force.apply()
        # Integrate
        for body in self.bodies:
            integrate_rigid_body(body, self.dt, self.integrator)
        self.time += self.dt
        for body in self.bodies:
            print(f"[DEBUG] {body.name}: force={body.force}, torque={body.torque}")
        self.logger.log_bodies(self.time, self.bodies)

    def run(self, duration):
        end_time = self.time + duration
        while self.time < end_time:
            self.step()