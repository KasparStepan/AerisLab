import numpy as np

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Drag, Gravity

# Create world with automatic logging
world = World.with_logging(
    name="my_first_simulation",
    ground_z=0.0,
    auto_save_plots=True
)

# Create a spherical body
mass = 10.0
radius = 0.2
I_sphere = (2/5) * mass * radius**2 * np.eye(3)

payload = RigidBody6DOF(
    name="payload",
    mass=mass,
    inertia_tensor_body=I_sphere,
    position=np.array([0.0, 0.0, 1000.0]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0])
)

world.add_body(payload)

# Add forces
world.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))
payload.per_body_forces.append(
    Drag(rho=1.225, Cd=0.47, area=np.pi * radius**2)
)

# Run simulation
solver = HybridSolver(alpha=5.0, beta=1.0)
world.run(solver, duration=200.0, dt=0.01)

# Results automatically saved to: output/my_first_simulation_TIMESTAMP/
