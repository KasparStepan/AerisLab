import matplotlib.pyplot as plt
import numpy as np
from core.math_utils import kinetic_energy
from core.math_utils import quaternion_to_rotation_matrix
from mpl_toolkits.mplot3d import Axes3D

def draw(world, ax):
    ax.cla()
    maxT = max(kinetic_energy(b) for b in world.bodies)+1e-6
    for b in world.bodies:
        T = kinetic_energy(b)
        color = plt.cm.viridis(T/maxT)
        ax.scatter(*b.position, color=color, s=50)
        traj = world.logs['positions'][b.name]
        traj = np.array(traj)
        ax.plot(traj[:,0], traj[:,1], traj[:,2], '--', linewidth=1)
    for j in world.joints:
        pa = j.body_a.position + quaternion_to_rotation_matrix(j.body_a.orientation) @ j.pA
        pb = j.body_b.position + quaternion_to_rotation_matrix(j.body_b.orientation) @ j.pB
        ax.plot([pa[0],pb[0]],[pa[1],pb[1]],[pa[2],pb[2]], 'k-')
    energy = world.logs['energy'][-1] if world.logs['energy'] else 0.0
    ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
    ax.set_title(f"Total energy = {energy:.3f}")
