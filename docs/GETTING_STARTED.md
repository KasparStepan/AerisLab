# Getting Started with AerisLab

This guide will walk you through creating your first physics simulation with AerisLab in about 10 minutes.

## Prerequisites

- Python 3.10 or higher
- Basic understanding of Python programming
- Familiarity with NumPy (helpful but not required)

## Installation

### Step 1: Get the Code

```bash
git clone https://github.com/KasparStepan/aerislab.git
cd aerislab
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Step 3: Install AerisLab

```bash
# Install in development mode
pip install -e .

# Install with development tools (optional)
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "import aerislab; print('AerisLab installed successfully!')"
```

## Your First Simulation
Let's simulate a sphere dropping from 100 meters with air resistance.

### Create Your Script
Create a file called my_first_sim.py:

```python
import numpy as np
from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity, Drag

def main():
    # 1. Create the simulation world
    world = World.with_logging(
        name="my_first_simulation",
        ground_z=0.0,
        auto_save_plots=True
    )
    
    # 2. Define physical properties
    mass = 5.0  # kg
    radius = 0.15  # m
    I_sphere = (2/5) * mass * radius**2 * np.eye(3)
    
    # 3. Create the body
    sphere = RigidBody6DOF(
        name="sphere",
        mass=mass,
        inertia_tensor_body=I_sphere,
        position=np.array([0.0, 0.0, 100.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    world.add_body(sphere)
    
    # 4. Add forces
    world.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))
    sphere.per_body_forces.append(
        Drag(rho=1.225, Cd=0.47, area=np.pi * radius**2)
    )
    
    # 5. Create solver and run
    solver = HybridSolver(alpha=5.0, beta=1.0)
    world.run(solver, duration=15.0, dt=0.01)
    
    # 6. Print results
    print(f"\n{'='*60}")
    print(f"Simulation completed!")
    print(f"Time: {world.t:.3f} s")
    if world.t_touchdown:
        print(f"Touchdown: {world.t_touchdown:.3f} s")
        print(f"Impact velocity: {np.linalg.norm(sphere.v):.2f} m/s")
    print(f"Results saved to: {world.output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
```

### Run Your Simulation

```bash
python my_first_sim.py
```

You should see output like:

```text
[World] Logging enabled: output/my_first_simulation_20260109_150000
        Logs: output/my_first_simulation_20260109_150000/logs
        Plots: output/my_first_simulation_20260109_150000/plots
[World] Simulation terminated at t=4.517s
        Touchdown detected at t=4.517s
[World] Generating plots for: sphere
[World] Plots saved to: output/my_first_simulation_20260109_150000/plots

============================================================
Simulation completed!
Time: 4.517 s
Touchdown: 4.517 s
Impact velocity: 21.34 m/s
Results saved to: output/my_first_simulation_20260109_150000
============================================================
```

### View Your Results
Check the output directory:

```text
output/my_first_simulation_20260109_150000/
├── logs/
│   └── simulation.csv          # Time-series data
└── plots/
    ├── sphere_trajectory_3d.png
    ├── sphere_velocity_acceleration.png
    └── sphere_forces.png
```

Open the PNG files to see visualizations of your simulation!

## Understanding the Code

### 1. World Creation
```python
world = World.with_logging(
    name="my_first_simulation",  # Name for output folder
    ground_z=0.0,                 # Ground altitude (meters)
    auto_save_plots=True          # Generate plots automatically
```
The World is the main container. It manages bodies, forces, time, and output.

### 2. Physical Properties
```python
mass = 5.0  # kg
radius = 0.15  # m
I_sphere = (2/5) * mass * radius**2 * np.eye(3)
```
For a solid sphere, the moment of inertia is I = (2/5)mr² in each principal axis.

### 3. Body Definition
```python
sphere = RigidBody6DOF(
    name="sphere",
    mass=mass,
    inertia_tensor_body=I_sphere,
    position=np.array([0.0, 0.0, 100.0]),  # [x, y, z] in meters
    orientation=np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion [x, y, z, w]
)
```
State variables:

Position: [x, y, z] in world frame

Orientation: Quaternion [qx, qy, qz, qw] (scalar-last convention)

Velocity: Automatically initialized to zero

Angular velocity: Automatically initialized to zero

### 4. Forces
```python
# Gravity (applies to all bodies)
world.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))

# Drag (specific to this body)
sphere.per_body_forces.append(
    Drag(rho=1.225, Cd=0.47, area=np.pi * radius**2)
)
```
Global forces: Applied to every body (e.g., gravity)
Per-body forces: Applied to specific bodies (e.g., drag, parachutes)

### 5. Solver Configuration
```python
solver = HybridSolver(alpha=5.0, beta=1.0)
```
Parameters:

alpha: Position correction strength (typical: 1-10)

beta: Velocity correction strength (typical: 0.1-2)

Higher values = stronger constraint enforcement but potential instability.

### 6. Running the Simulation
```python
world.run(solver, duration=15.0, dt=0.01)
```
Parameters:

solver: The numerical integrator to use

duration: Maximum simulation time (seconds)

dt: Time step size (seconds)

Simulation stops early if the body hits the ground (z = ground_z).

## Next Steps

### Add More Complexity
Try these modifications to your simulation:

1. Add initial velocity:

```python
sphere = RigidBody6DOF(
    name="sphere",
    mass=mass,
    inertia_tensor_body=I_sphere,
    position=np.array([0.0, 0.0, 100.0]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    linear_velocity=np.array([10.0, 0.0, 0.0])  # 10 m/s horizontal
)
```

2. Change drag coefficient:

```python
# Less drag (more aerodynamic)
Drag(rho=1.225, Cd=0.2, area=np.pi * radius**2)

# More drag (less aerodynamic)
Drag(rho=1.225, Cd=1.2, area=np.pi * radius**2)
```

3. Drop from higher altitude:

```python
position=np.array([0.0, 0.0, 1000.0])  # 1000 meters
```

### Explore Examples
Check the examples/ directory for more complex simulations:

simple_drop.py - Similar to this tutorial

parachute_fixed.py - Two-body system with constraint

parachute_ivp.py - High-accuracy adaptive integration

### Read More Documentation
USER_GUIDE.md - Comprehensive features and usage

API_REFERENCE.md - Complete API documentation

PHYSICS.md - Mathematical theory behind the simulation

EXAMPLES.md - Detailed example explanations

## Common Issues

### Import Error
```text
ModuleNotFoundError: No module named 'aerislab'
```
Solution: Make sure you installed with `pip install -e .` and activated virtual environment.

### Simulation Runs Forever
Solution: Check that `ground_z` is set correctly. Default termination happens when body crosses ground plane.

### Poor Results / Instability
Solution: Try:

Smaller time step: dt=0.005

Lower Baumgarte parameters: alpha=2.0, beta=0.5

Check that forces are correct magnitude

### No Plots Generated
Solution: Ensure `auto_save_plots=True` and simulation has run (not just created).

## Tips for Success
💡 Start simple - Get basic simulation working before adding complexity

💡 Check units - All values must be in SI units (meters, kilograms, seconds)

💡 Use realistic values - Unrealistic masses or areas cause numerical issues

💡 Visualize early - Generate plots to verify behavior makes sense

💡 Read error messages - AerisLab provides detailed validation errors

## Getting Help
Check TROUBLESHOOTING.md

Review FAQ.md

Open an issue on GitHub

## What's Next?
Now that you have a working simulation, explore:

Multiple bodies - Add more objects to your world

Constraints - Connect bodies with joints

Custom forces - Create your own force models

Advanced solvers - Try adaptive IVP integration

Data analysis - Process CSV logs with pandas

Happy simulating! 🚀