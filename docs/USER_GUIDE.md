# AerisLab User Guide

Complete guide to using AerisLab for physics simulations.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [World and Simulation](#world-and-simulation)
3. [Rigid Bodies](#rigid-bodies)
4. [Forces](#forces)
5. [Constraints](#constraints)
6. [Solvers](#solvers)
7. [Data Logging](#data-logging)
8. [Visualization](#visualization)
9. [Advanced Topics](#advanced-topics)

## Core Concepts

### The Simulation Loop

Every AerisLab simulation follows this pattern:

```text
Create World → 2. Add Bodies → 3. Add Forces → 4. Add Constraints → 5. Run Solver
```

### Coordinate System

AerisLab uses a right-handed coordinate system:
- **X**: East (horizontal)
- **Y**: North (horizontal)
- **Z**: Up (vertical)

All quantities use SI units (meters, kilograms, seconds, radians).

### State Representation

Each rigid body has 13 degrees of freedom:
- Position: (x, y, z) - 3 DOF
- Orientation: quaternion (qx, qy, qz, qw) - 4 DOF
- Linear velocity: (vx, vy, vz) - 3 DOF
- Angular velocity: (ωx, ωy, ωz) - 3 DOF

## World and Simulation

### Creating a World

**Option 1: With automatic logging (recommended)**

```python
from aerislab.core.simulation import World

world = World.with_logging(
    name="my_simulation",
    ground_z=0.0,
    payload_index=0,
    auto_save_plots=True
)
```
Option 2: Manual configuration

```python
world = World(ground_z=0.0, payload_index=0)
world.enable_logging("my_simulation")
```
Option 3: No logging

```python
world = World(ground_z=0.0, payload_index=0)
# Useful for performance testing or parameter sweeps
```
World Parameters
```python
World(
    ground_z=0.0,           # Ground altitude [m]
    payload_index=0,        # Which body to monitor for termination
    simulation_name=None,   # Name for output organization
    output_dir="output",    # Base output directory
    auto_timestamp=True,    # Add timestamp to folder names
    auto_save_plots=False   # Generate plots automatically
)
```
Termination Conditions
Default: Simulation stops when payload touches ground

```python
# Payload crosses ground_z going downward → simulation stops
```
Custom termination:

```python
def stop_at_velocity(world):
    """Stop when velocity exceeds threshold."""
    v = np.linalg.norm(world.bodies.v)
    return v > 100.0  # Stop at 100 m/s

world.set_termination_callback(stop_at_velocity)
```
No termination:

```python
# Run for fixed duration regardless of state
world.set_termination_callback(lambda w: False)
```
Rigid Bodies
Creating Bodies
```python
from aerislab.dynamics.body import RigidBody6DOF
import numpy as np

# Define inertia tensor (in body frame)
I_body = np.diag([Ixx, Iyy, Izz])  # Principal moments

body = RigidBody6DOF(
    name="my_body",
    mass=10.0,
    inertia_tensor_body=I_body,
    position=np.array([0.0, 0.0, 100.0]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    linear_velocity=np.array([0.0, 0.0, 0.0]),
    angular_velocity=np.array([0.0, 0.0, 0.0]),
    radius=0.5  # For visualization only
)

world.add_body(body)
```
Common Inertia Tensors
Solid sphere:

```python
I = (2/5) * m * r**2 * np.eye(3)
```
Hollow sphere:

```python
I = (2/3) * m * r**2 * np.eye(3)
```
Solid cylinder (axis = z):

```python
I = np.diag([
    (1/12) * m * (3*r**2 + h**2),  # Ixx
    (1/12) * m * (3*r**2 + h**2),  # Iyy
    (1/2) * m * r**2                # Izz
])
```
Box (sides a, b, c):

```python
I = (1/12) * m * np.diag([
    b**2 + c**2,  # Ixx
    a**2 + c**2,  # Iyy
    a**2 + b**2   # Izz
])
```
Point mass (for small/negligible rotation):

```python
I = 0.01 * np.eye(3)  # Small but non-zero
```
Accessing Body State
```python
# Position
x, y, z = body.p

# Velocity
vx, vy, vz = body.v
speed = np.linalg.norm(body.v)

# Orientation (quaternion)
qx, qy, qz, qw = body.q

# Rotation matrix
R = body.rotation_world()  # 3x3 matrix

# Energy
KE = body.kinetic_energy()  # Joules
```
Forces
Gravity
```python
from aerislab.dynamics.forces import Gravity

gravity = Gravity(g=np.array([0.0, 0.0, -9.81]))
world.add_global_force(gravity)
```
Custom gravity:

```python
# Moon gravity
gravity_moon = Gravity(g=np.array([0.0, 0.0, -1.62]))

# Horizontal "gravity" (wind acceleration)
wind = Gravity(g=np.array([5.0, 0.0, 0.0]))
```
Aerodynamic Drag
Quadratic drag (standard):

```python
from aerislab.dynamics.forces import Drag

drag = Drag(
    rho=1.225,              # Air density [kg/m³]
    Cd=0.47,                # Drag coefficient [-]
    area=0.1,               # Reference area [m²]
    mode="quadratic"        # F = -0.5 * ρ * Cd * A * |v| * v
)
body.per_body_forces.append(drag)
```
Linear drag (low Reynolds number):

```python
drag_linear = Drag(
    mode="linear",
    k_linear=0.5  # Damping coefficient [N·s/m]
)
```
Time-varying drag:

```python
def area_function(t, body):
    """Area grows over time."""
    return 1.0 + 0.1 * t  # Starts at 1 m², grows by 0.1 m²/s

drag = Drag(rho=1.225, Cd=0.47, area=area_function)
```
Parachute Drag
```python
from aerislab.dynamics.forces import ParachuteDrag

parachute = ParachuteDrag(
    rho=1.225,
    Cd=1.5,                      # Parachute drag coefficient
    area=15.0,                   # Deployed area [m²]
    activation_velocity=30.0,    # Deploy at 30 m/s
    activation_altitude=2000.0,  # OR deploy at 2000m (whichever first)
    gate_sharpness=50.0,         # Smoothness of deployment
    area_collapsed=0.01          # Collapsed area [m²]
)
canopy.per_body_forces.append(parachute)
```
Activation logic:

Deploys when velocity ≥ activation_velocity OR altitude ≤ activation_altitude

Area transitions smoothly from area_collapsed to area over ~0.1 seconds

gate_sharpness controls transition steepness (higher = faster deployment)

Spring/Tether Forces
```python
from aerislab.dynamics.forces import Spring

# Soft tether between two bodies
spring = Spring(
    body_a=payload,
    body_b=canopy,
    attach_a_local=np.zeros(3),  # Attachment point in body A frame
    attach_b_local=np.zeros(3),  # Attachment point in body B frame
    k=1000.0,                    # Stiffness [N/m]
    c=50.0,                      # Damping [N·s/m]
    rest_length=10.0             # Unstretched length [m]
)
world.add_interaction_force(spring)
```
Force equation:

```text
F = -k * (|d| - L₀) * d_hat - c * (v_rel · d_hat) * d_hat
```
Custom Forces
Implement the Force protocol:

```python
class CustomForce:
    def apply(self, body, t=None):
        """Apply custom force to body."""
        # Example: sinusoidal force
        if t is not None:
            f = np.array([10.0 * np.sin(2 * np.pi * t), 0.0, 0.0])
            body.apply_force(f)

custom = CustomForce()
body.per_body_forces.append(custom)
```
Constraints
Constraints enforce geometric relationships between bodies using Lagrange multipliers.

Distance Constraint
Maintains fixed distance between two attachment points:

```python
from aerislab.dynamics.constraints import DistanceConstraint

constraint = DistanceConstraint(
    world_bodies=world.bodies,
    body_i=0,                    # Index of first body
    body_j=1,                    # Index of second body
    attach_i_local=np.zeros(3), # Attachment point on body i
    attach_j_local=np.zeros(3), # Attachment point on body j
    length=10.0                  # Fixed distance [m]
)
world.add_constraint(constraint)
```
Point Weld Constraint
Enforces that two points remain coincident:

```python
from aerislab.dynamics.constraints import PointWeldConstraint

weld = PointWeldConstraint(
    world_bodies=world.bodies,
    body_i=0,
    body_j=1,
    attach_i_local=np.zeros(3),
    attach_j_local=np.zeros(3)
)
world.add_constraint(weld)
```
Difference from distance constraint:

Distance: 1 equation (scalar distance)

Weld: 3 equations (full 3D position)

Joint Helpers
Convenience wrappers for common constraint patterns:

```python
from aerislab.dynamics.joints import RigidTetherJoint

# Rigid tether (distance constraint)
tether = RigidTetherJoint(
    body_i=payload_idx,
    body_j=canopy_idx,
    attach_i_local=np.zeros(3),
    attach_j_local=np.zeros(3),
    length=10.0
)
world.add_constraint(tether.attach(world.bodies))
```
Constraint Stabilization
Constraints are enforced at velocity level with Baumgarte stabilization:

```text
ṙhs = -(1 + β) * J*v - α * C
```
α (alpha): Position error correction [1/s]. Typical: 1-10

β (beta): Velocity error correction [-]. Typical: 0.1-2

Higher values provide stronger enforcement but may cause instability.

Solvers
Fixed-Step Solver
Semi-implicit Euler integration with fixed time step:

```python
from aerislab.core.solver import HybridSolver

solver = HybridSolver(
    alpha=5.0,  # Baumgarte position correction
    beta=1.0    # Baumgarte velocity correction
)

world.run(solver, duration=100.0, dt=0.01)
```
Advantages:

Fast

Predictable performance

Good for real-time visualization

Disadvantages:

Fixed accuracy (determined by dt)

Must choose dt carefully

Recommended dt values:

Slow dynamics: 0.01 - 0.1 s

Moderate dynamics: 0.001 - 0.01 s

Fast dynamics: 0.0001 - 0.001 s

Adaptive-Step IVP Solver
Variable time step with automatic error control:

```python
from aerislab.core.solver import HybridIVPSolver

solver = HybridIVPSolver(
    method="Radau",     # Radau, BDF, RK45
    rtol=1e-6,          # Relative tolerance
    atol=1e-8,          # Absolute tolerance
    max_step=0.5,       # Maximum step size [s]
    alpha=10.0,         # Baumgarte position
    beta=2.0            # Baumgarte velocity
)

sol = world.integrate_to(solver, t_end=100.0)
```
Advantages:

Automatic accuracy control

Handles stiff systems

Efficient for smooth dynamics

Disadvantages:

Slower than fixed-step

Unpredictable computation time

Method selection:

"Radau": Best for stiff systems (recommended)

"BDF": Very stiff systems

"RK45": Non-stiff systems (explicit)

Choosing a Solver
Use Case	Recommended Solver
Quick visualization	Fixed-step, dt=0.01
Publication-quality data	IVP, Radau, rtol=1e-6
Real-time simulation	Fixed-step, dt=0.01-0.1
Stiff constraints	IVP, Radau, high α/β
Parameter sweeps	Fixed-step (faster)
Data Logging
Automatic Logging
```python
world = World.with_logging(
    name="my_sim",
    auto_save_plots=True  # Plots generated automatically
)

# Logging happens automatically during run()
world.run(solver, duration=100.0, dt=0.01)
```
Manual Logging Control
```python
world = World()
# ... add bodies, forces ...

# Enable logging when ready
world.enable_logging("my_sim")

world.run(solver, duration=100.0, dt=0.01)

# Manually generate plots
world.save_plots(bodies=["payload", "canopy"])
```
Custom Logger
```python
from aerislab.logger import CSVLogger

logger = CSVLogger(
    filepath="custom_output.csv",
    buffer_size=1000,
    fields=["p", "v"]  # Only log position and velocity
)

world.logger = logger
```
CSV Format
The CSV file has this structure:

```text
t,body1.p_x,body1.p_y,body1.p_z,body1.v_x,body1.v_y,body1.v_z,...
0.0,0.0,0.0,100.0,0.0,0.0,0.0,...
0.01,0.0,0.0,99.995,0.0,0.0,-0.098,...
```
Columns:

t: Time [s]

<body>.<field>_<component>: State variables

Available fields:

p: Position [m]

q: Quaternion [-]

v: Linear velocity [m/s]

w: Angular velocity [rad/s]

f: Force [N]

tau: Torque [N·m]

Reading Logs
```python
import pandas as pd

df = pd.read_csv("output/my_sim_TIMESTAMP/logs/simulation.csv")

# Extract payload altitude
altitude = df['payload.p_z']

# Plot velocity magnitude
velocity_mag = np.sqrt(df['payload.v_x']**2 + 
                       df['payload.v_y']**2 + 
                       df['payload.v_z']**2)
plt.plot(df['t'], velocity_mag)
```
Visualization
Automatic Plots
With auto_save_plots=True, three plots are generated per body:

3D Trajectory - Spatial path through 3D space

Velocity & Acceleration - Time series of kinematics

Forces - Time series of applied forces

Manual Plotting
```python
world.save_plots(
    bodies=["payload", "canopy"],  # Which bodies to plot
    show=False                      # Display interactively?
)
```
Custom Plotting
```python
from aerislab.visualization.plotting import (
    plot_trajectory_3d,
    plot_velocity_and_acceleration,
    plot_forces
)

csv_path = "output/my_sim_TIMESTAMP/logs/simulation.csv"

plot_trajectory_3d(
    csv_path,
    "payload",
    save_path="custom_trajectory.png",
    show=True
)
```
Standalone Plotting Script
For re-plotting existing data:

```bash
python examples/plot_parachute_logs.py path/to/simulation.csv --bodies payload canopy
```
Advanced Topics
Energy Tracking
```python
energy = world.get_energy()

print(f"Kinetic: {energy['kinetic']:.2f} J")
print(f"Potential: {energy['potential']:.2f} J")
print(f"Total: {energy['total']:.2f} J")
```
Note: Potential energy only accounts for gravity.

Multiple Simulations
```python
# Parameter sweep
for mass in [5.0, 10.0, 15.0]:
    world = World.with_logging(f"mass_{mass}kg")
    
    body = RigidBody6DOF(
        "payload", mass, I, position, orientation
    )
    world.add_body(body)
    
    # ... add forces ...
    
    world.run(solver, duration=100.0, dt=0.01)
```
Batch Processing
```python
import multiprocessing

def run_simulation(params):
    mass, altitude = params
    world = World.with_logging(f"sim_m{mass}_h{altitude}")
    # ... setup and run ...
    return world.t_touchdown

params_list = [(5, 1000), (10, 1000), (5, 2000), (10, 2000)]

with multiprocessing.Pool() as pool:
    results = pool.map(run_simulation, params_list)
```
Performance Optimization
For long simulations:

```python
# Disable logging during run
world = World()
# ... setup ...
world.run(solver, duration=1000.0, dt=0.01)

# Enable logging and re-run short segment for data
world.enable_logging("final_segment")
world.run(solver, duration=10.0, dt=0.01)
```
For parameter sweeps:

```python
# Turn off plots, generate manually later
world = World.with_logging("sweep", auto_save_plots=False)
```
Optimize buffer size:

```python
logger = CSVLogger(filepath="log.csv", buffer_size=5000)
```
Debugging
Check constraint violations:

```python
from aerislab.dynamics.constraints import DistanceConstraint

# After simulation
constraint = world.constraints
violation = constraint.evaluate()
print(f"Constraint error: {violation}")
```
Monitor forces:

```python
# After each step
print(f"Payload force: {world.bodies.f}")
print(f"Payload torque: {world.bodies.tau}")
```
Energy drift:

```python
E0 = world.get_energy()
world.run(solver, duration=10.0, dt=0.01)
E1 = world.get_energy()
drift = (E1['total'] - E0['total']) / E0['total']
print(f"Energy drift: {drift*100:.2f}%")
```
Best Practices
✅ DO:

Use SI units consistently

Start with simple models, add complexity gradually

Check plots to verify physical behavior

Use realistic parameter values

Test with small time steps first

Write tests for custom forces/constraints

❌ DON'T:

Mix unit systems

Use unrealistic masses or dimensions

Ignore constraint violations

Set time step too large

Forget to enable logging for important runs

Use magic numbers (define constants)

Next Steps
Read PHYSICS.md for mathematical theory

See EXAMPLES.md for detailed example walkthroughs

Check API_REFERENCE.md for complete API

Review TROUBLESHOOTING.md for common issues