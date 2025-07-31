from core.math_utils import kinetic_energy, potential_energy
from core.collision import sphere_sphere_collision
from integrators.rigid_integrators import integrate_rigid_body

class World:
    def __init__(self):
        self.bodies=[]; self.joints=[]; self.forces=[]
        self.logs={'time':[],'energy':[],'positions':{}}
        self.integrator='semi'; self.gravity=[0,0,-9.81]
        self.use_kane=False

    def add_body(self,b): self.bodies.append(b)
    def add_joint(self,j): self.joints.append(j)
    def add_force(self,f,body=None): self.forces.append((f,body))

    def step(self, dt):
        for b in self.bodies: b.sim_dt = dt
        if not self.use_kane:
            for fg,bod in self.forces:
                if bod is None: [fg.apply(b) for b in self.bodies]
                else: fg.apply(bod)
            for j in self.joints: j.apply_constraint_forces()
            for b in self.bodies: integrate_rigid_body(b, dt, scheme=self.integrator)
            for i in range(len(self.bodies)):
                for j in range(i+1,len(self.bodies)):
                    sphere_sphere_collision(self.bodies[i], self.bodies[j])
        else:
            # Placeholder: compute partials, generalized forces/inertia, then solve
            pass

        t0=self.logs['time'][-1] if self.logs['time'] else 0.0
        self.logs['time'].append(t0+dt)
        E=sum(kinetic_energy(b)+potential_energy(b,self.gravity) for b in self.bodies)
        self.logs['energy'].append(E)
        for b in self.bodies:
            self.logs['positions'].setdefault(b.name,[]).append(b.position.copy())
