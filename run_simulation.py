import numpy as np
import matplotlib.pyplot as plt
from sim.world import World
from core.body import RigidBody6DOF
from core.joints import RevoluteJoint
from core.force import GravityForce, DragForce, MomentumFluxForce
from sim.visualization import draw

world = World()
b1 = RigidBody6DOF("A",5,np.eye(3),position=[0,0,1000],orientation=[1,0,0,0],linear_velocity=[0,0,0],angular_velocity=[0,0,0])
world.add_body(b1)

world.add_force(GravityForce())

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
dt = 0.01

for i in range(5000):
    world.step(dt)
    if i % 10 == 0:
        draw(world, ax)
        plt.pause(0.001)
plt.show()

plt.figure()
plt.plot(world.logs['time'], world.logs['energy'])
plt.xlabel('Time'); plt.ylabel('Energy')
plt.show()
