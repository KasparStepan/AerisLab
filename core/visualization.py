import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class Visualization:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.spheres = []  # List of (body, sphere_plot) tuples
        self.lines = []    # List of (joint, line_plot) tuples
        self.time_text = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)

    def add_body(self, body):
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = body.radius * np.outer(np.cos(u), np.sin(v))
        y = body.radius * np.outer(np.sin(u), np.sin(v))
        z = body.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        sphere = self.ax.plot_wireframe(x, y, z, color='b')
        self.spheres.append((body, sphere))

    def add_joint(self, joint):
        line, = self.ax.plot([], [], [], 'r-')
        self.lines.append((joint, line))

    def update(self, bodies, joints, time):
        # Update spheres
        self.ax.collections.clear()  # Remove all wireframes (spheres)
        new_spheres = []
        for body, _ in self.spheres:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = body.radius * np.outer(np.cos(u), np.sin(v)) + body.position[0]
            y = body.radius * np.outer(np.sin(u), np.sin(v)) + body.position[1]
            z = body.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + body.position[2]
            sphere = self.ax.plot_wireframe(x, y, z, color='b')
            new_spheres.append((body, sphere))
        self.spheres = new_spheres

        # Update lines
        for joint, line in self.lines:
            p1, p2 = joint.get_attachment_points()
            line.set_data_3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

        # Update time text
        self.time_text.set_text(f"Time: {time:.2f} s")

        # Adjust axes
        if bodies:
            positions = np.array([body.position for body in bodies])
            if positions.size > 0:
                max_range = np.max(np.ptp(positions, axis=0)) / 2
                mid = np.mean(positions, axis=0)
                self.ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                self.ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                self.ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

    def animate(self, frame, world):
        world.step()
        self.update(world.bodies, world.joints, world.time)
        return [sphere[1] for sphere in self.spheres] + [line[1] for line in self.lines] + [self.time_text]

    def run(self, world, duration):
        frames = int(duration / world.dt)
        ani = FuncAnimation(self.fig, self.animate, fargs=(world,), frames=frames, interval=world.dt * 1000, blit=True)
        plt.show()
