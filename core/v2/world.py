from .visualization import Visualization
from .integrators import integrate_rigid_body

class World:
    def __init__(self, dt, integrator='semi'):
        self.bodies = []
        self.global_forces = []
        self.joints = []
        self.dt = dt
        self.time = 0.0
        self.integrator = integrator
        self.visualization = Visualization()

    def add_body(self, body):
        self.bodies.append(body)
        self.visualization.add_body(body)

    def add_global_force(self, force):
        self.global_forces.append(force)

    def add_joint(self, joint):
        self.joints.append(joint)
        self.visualization.add_joint(joint)

    def step(self):
        for body in self.bodies:
            body.clear_forces()
        for force in self.global_forces:
            for body in self.bodies:
                force.apply(body)
        for body in self.bodies:
            for force in body.forces:
                force.apply(body)
        for joint in self.joints:
            joint.apply_constraint()
        for body in self.bodies:
            integrate_rigid_body(body, self.dt, self.integrator)
        self.time += self.dt
        self.visualization.update(self.bodies, self.joints, self.time)

    def run(self, duration):
        self.visualization.run(self, duration)