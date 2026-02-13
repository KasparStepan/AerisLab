# 🪂 AerisLab — Complete User Guide

> **AerisLab** is a high-fidelity Python physics engine for **6-DOF rigid body dynamics** with holonomic constraints, designed for aerospace engineering — especially parachute-payload trajectory simulation.

---

## Table of Contents

- [🏗️ Architecture Overview](#️-architecture-overview)
- [⚡ Quick Start](#-quick-start)
- [🧱 Core Concepts](#-core-concepts)
  - [Rigid Bodies](#1-rigid-bodies-rigidbody6dof)
  - [Forces](#2-forces)
  - [Constraints](#3-constraints)
  - [Solvers](#4-solvers)
  - [World (Orchestrator)](#5-world-orchestrator)
- [🧩 Component Architecture](#-component-architecture)
  - [Payload](#payload)
  - [Parachute](#parachute)
  - [System](#system)
- [🎯 Scenario API](#-scenario-api)
- [📊 Visualization](#-visualization)
- [📁 Project Structure](#-project-structure)
- [🔧 Configuration & Solver Tuning](#-configuration--solver-tuning)
- [📚 Further Reading](#-further-reading)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          Scenario API                           │
│              (Fluent interface for quick setup)                  │
├──────────────────────────┬──────────────────────────────────────┤
│      Component Layer     │          Core Layer                  │
│                          │                                      │
│  ┌──────────┐            │   ┌───────────────────────┐          │
│  │ Payload  │            │   │        World           │          │
│  │ Parachute│    ◄───────┼── │  (orchestrator)        │          │
│  │ System   │            │   └───────────┬───────────┘          │
│  └──────────┘            │               │                      │
│                          │   ┌───────────▼───────────┐          │
│                          │   │  HybridSolver (fixed)  │          │
│                          │   │  HybridIVPSolver (IVP) │          │
│                          │   └───────────────────────┘          │
├──────────────────────────┴──────────────────────────────────────┤
│                        Dynamics Layer                            │
│                                                                  │
│  RigidBody6DOF    Forces (Gravity, Drag, Spring, ParachuteDrag) │
│  Constraints (Distance, PointWeld)    Joints (RigidTether)      │
├──────────────────────────────────────────────────────────────────┤
│                   Utilities & Visualization                      │
│                                                                  │
│  CSVLogger    Plotting (3D trajectory, forces, velocity, ...)   │
│  Orientation utils    Dashboard (Streamlit)                      │
└──────────────────────────────────────────────────────────────────┘
```

**Three ways to use AerisLab** (from simplest to most control):

| Level | API | Best For |
|-------|-----|----------|
| **High-level** | `Scenario` | Quick setups, parameter sweeps |
| **Mid-level** | `Component` + `System` + `World` | Custom parachute systems |
| **Low-level** | `RigidBody6DOF` + `Solver` directly | Research, custom physics |

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/KasparStepan/aerislab.git
cd aerislab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Hello World — Drop a Payload (10 lines)

```python
from aerislab.api.scenario import Scenario
from aerislab.components.standard import Payload

capsule = Payload(
    name="capsule", mass=10.0, radius=0.5,
    Cd=0.5, position=[0, 0, 1000]
)

Scenario(name="simple_drop") \
    .add_system([capsule]) \
    .enable_plotting(show=True) \
    .run(duration=30.0)
```

### Parachute Recovery System

```python
from aerislab.api.scenario import Scenario
from aerislab.components.standard import Payload, Parachute

payload = Payload(name="instrument", mass=50.0, radius=0.4,
                  position=[0, 0, 2000])

parachute = Parachute(name="main_chute", mass=5.0, diameter=12.0,
                       model="knacke", activation_altitude=1500,
                       position=[0, 0, 2000.5])

scenario = Scenario(name="recovery_system")
scenario.add_system([payload, parachute], name="recovery")
scenario.connect(payload, parachute, type="tether", length=10.0)
scenario.enable_plotting(show=True)
scenario.run(duration=60.0)
```

---

## 🧱 Core Concepts

### 1. Rigid Bodies (`RigidBody6DOF`)

The fundamental building block. Each body has **6 degrees of freedom** — 3 translational + 3 rotational.

```python
from aerislab.dynamics.body import RigidBody6DOF
import numpy as np

body = RigidBody6DOF(
    name="payload",
    mass=10.0,                                    # kg
    inertia_tensor_body=np.diag([1.0, 1.0, 1.0]), # kg·m² (body frame)
    position=np.array([0.0, 0.0, 1000.0]),         # [x, y, z] in meters
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),    # quaternion [qx, qy, qz, qw]
)
```

**State variables:**

| Variable | Type | Description |
|----------|------|-------------|
| `body.p` | `[3]` | Position vector [m] |
| `body.q` | `[4]` | Orientation quaternion `[qx, qy, qz, qw]` |
| `body.v` | `[3]` | Linear velocity [m/s] |
| `body.w` | `[3]` | Angular velocity [rad/s] |
| `body.f` | `[3]` | Net force accumulator [N] |
| `body.tau` | `[3]` | Net torque accumulator [N·m] |

**Key methods:**
- `apply_force(F)` / `apply_torque(tau)` — Add force/torque (world frame)
- `apply_force_body(F)` — Add force in body frame
- `clear_forces()` — Reset `f` and `tau` to zero
- `mass_matrix_world()` — 6×6 generalized mass matrix
- `inv_mass_matrix_world()` — Inverse mass matrix (for KKT solver)

---

### 2. Forces

Forces implement the `Force` protocol — they have an `apply(body, t)` method that adds forces/torques to a body.

| Force | Description | Parameters |
|-------|-------------|------------|
| **`Gravity`** | Constant gravitational force | `g=[0, 0, -9.81]` |
| **`Drag`** | Aerodynamic drag (quadratic or linear) | `rho`, `Cd`, `area`, `mode` |
| **`ParachuteDrag`** | Deployment-aware drag with smooth activation | `Cd`, `area`, `activation_altitude`, `gate_sharpness` |
| **`Spring`** | Hookean spring + damping between two points | `k`, `c`, `L0` |

```python
from aerislab.dynamics.forces import Gravity, Drag

gravity = Gravity(g=np.array([0, 0, -9.81]))
drag = Drag(rho=1.225, Cd=0.5, area=1.0, mode="quadratic")
```

**How forces are categorized in `World`:**

```
Global Forces ──→ applied to ALL bodies    (e.g., gravity)
Per-body Forces ──→ applied to ONE body    (e.g., drag on payload)
Interaction Forces ──→ between TWO bodies  (e.g., spring)
```

---

### 3. Constraints

Constraints enforce rigid connections between bodies via the **KKT (Karush-Kuhn-Tucker)** solver.

| Constraint | DOFs Removed | Use Case |
|------------|--------------|----------|
| **`DistanceConstraint`** | 1 | Tethers, rods |
| **`PointWeldConstraint`** | 3 | Rigid joints |

```python
from aerislab.dynamics.constraints import DistanceConstraint

rod = DistanceConstraint(
    world_bodies=world.bodies,
    body_i=0, body_j=1,               # body indices
    attach_i_local=np.zeros(3),        # attachment point (body frame)
    attach_j_local=np.zeros(3),
    length=5.0,                        # constraint distance [m]
)
```

**Baumgarte stabilization** (`alpha`, `beta`) prevents numerical constraint drift over time.

---

### 4. Solvers

Two solver strategies are available:

#### Fixed-Step: `HybridSolver`

```python
from aerislab.core.solver import HybridSolver

solver = HybridSolver(alpha=5.0, beta=1.0)
world.run(solver, duration=10.0, dt=0.01)
```

- Uses **semi-implicit Euler** integration
- You choose the step size `dt`
- Fast and predictable, but needs small `dt` for accuracy

#### Variable-Step: `HybridIVPSolver`

```python
from aerislab.core.solver import HybridIVPSolver

solver = HybridIVPSolver(method="Radau", rtol=1e-6, atol=1e-8)
world.integrate_to(solver, t_end=100.0)
```

- Uses `scipy.integrate.solve_ivp` under the hood
- **Automatically** adapts step size for accuracy
- Best for stiff problems (parachute deployment)
- Methods: `"Radau"` (stiff), `"RK45"` (non-stiff), `"BDF"`, etc.

---

### 5. World (Orchestrator)

The `World` class ties everything together. It manages bodies, forces, constraints, and runs the simulation loop.

```python
from aerislab.core.simulation import World

world = World(ground_z=0.0, payload_index=0)

# Populate
world.add_body(body)
world.add_global_force(gravity)
world.add_constraint(rod)

# Enable logging
world.enable_logging("my_simulation")

# Run
world.run(solver, duration=10.0, dt=0.01)
```

**Simulation loop (`step()`):**

```
1. Clear force accumulators     ← reset F, τ to zero
2. Update component states      ← deployment logic, actuation
3. Apply forces (system → per-body → global → interaction)
4. solver.step()                ← KKT + integration
5. Log state to CSV
6. Check termination            ← ground contact?
```

**Ground detection** uses linear interpolation for sub-step accuracy:

```python
# Automatic: stops when payload z ≤ ground_z
world = World(ground_z=0.0, payload_index=0)

# Custom: any condition
world.set_termination_callback(
    lambda w: np.linalg.norm(w.bodies[0].v) > 100.0  # speed limit
)
```

---

## 🧩 Component Architecture

For aerospace systems, the **component layer** provides a higher-level API that wraps `RigidBody6DOF` with domain-specific behavior.

### Payload

A simple body with drag. Automatically creates the rigid body and drag force.

```python
from aerislab.components.standard import Payload

payload = Payload(
    name="capsule",
    mass=50.0,         # kg
    radius=0.4,        # m (for drag & inertia)
    Cd=0.5,            # drag coefficient
    position=[0, 0, 1000],
    velocity=[0, 0, 0],
)
```

### Parachute

Includes a **deployment state machine** with smooth drag activation:

```
STOWED ──→ DEPLOYING ──→ DEPLOYED
              ↓
           FAILED (future)
```

```python
from aerislab.components.standard import Parachute

chute = Parachute(
    name="main_chute",
    mass=5.0,
    diameter=12.0,              # m (canopy diameter)
    model="knacke",             # inflation model
    activation_altitude=1500,   # deploy below 1500m
    position=[0, 0, 1000.5],
)
```

Deployment triggers when **either** condition is met:
- Altitude ≤ `activation_altitude`
- Speed ≥ `activation_velocity`

### System

Groups components and their inter-connections:

```python
from aerislab.components.system import System

system = System("recovery")
system.add_component(payload)
system.add_component(parachute)
system.add_constraint(tether)

world.add_system(system)  # registers all bodies + constraints
```

---

## 🎯 Scenario API

The **highest-level** API — a fluent interface for common simulation patterns:

```python
from aerislab.api.scenario import Scenario

scenario = Scenario(name="drop_test", output_dir="output")
scenario.add_system([payload, parachute])
scenario.connect(payload, parachute, type="tether", length=10.0)
scenario.set_initial_state(altitude=2000, velocity=[0, 0, 0])
scenario.configure_solver(preset="accurate")  # or "fast", "stiff", "default"
scenario.enable_plotting(show=True)
scenario.run(duration=60.0)
```

**Solver presets:**

| Preset | Method | rtol | atol | Best For |
|--------|--------|------|------|----------|
| `"default"` | Radau | 1e-6 | 1e-8 | General use |
| `"fast"` | RK45 | 1e-3 | 1e-6 | Quick iterations |
| `"accurate"` | Radau | 1e-9 | 1e-12 | Validation runs |
| `"stiff"` | Radau | 1e-6 | 1e-8 | Parachute deployment |

---

## 📊 Visualization

### Automatic Plotting

```python
# Enable auto-plots at end of simulation
world = World.with_logging("my_sim", auto_save_plots=True)
```

Generates these plots automatically in `output/<sim_name>/plots/`:

| Plot | Shows |
|------|-------|
| `*_trajectory_3d.png` | 3D flight path |
| `*_velocity_acceleration.png` | Speed & acceleration vs. time |
| `*_forces.png` | Force & torque magnitudes vs. time |
| `*_force_breakdown.png` | Individual force contributions |

### Manual Plotting

```python
from aerislab.visualization.plotting import (
    plot_trajectory_3d,
    plot_velocity_and_acceleration,
    plot_forces,
    plot_force_breakdown,
    compare_trajectories,
)

plot_trajectory_3d("output/sim/logs/simulation.csv", "Payload", show=True)

# Compare multiple runs
compare_trajectories(
    ["run1/simulation.csv", "run2/simulation.csv"],
    body_name="Payload",
    labels=["Case 1", "Case 2"],
    show=True
)
```

---

## 📁 Project Structure

```
AerisLab/
├── src/aerislab/
│   ├── __init__.py              # Public API exports
│   ├── logger.py                # CSVLogger (buffered CSV writer)
│   │
│   ├── core/                    # Simulation engine
│   │   ├── simulation.py        # World class (orchestrator)
│   │   └── solver.py            # HybridSolver, HybridIVPSolver, KKT
│   │
│   ├── dynamics/                # Physics primitives
│   │   ├── body.py              # RigidBody6DOF
│   │   ├── forces.py            # Gravity, Drag, ParachuteDrag, Spring
│   │   ├── constraints.py       # DistanceConstraint, PointWeldConstraint
│   │   └── joints.py            # RigidTetherJoint
│   │
│   ├── components/              # Domain-level abstractions
│   │   ├── base.py              # Component base class
│   │   ├── standard.py          # Payload, Parachute (convenience wrappers)
│   │   ├── payload.py           # Payload component
│   │   ├── parachute.py         # Parachute with deployment state machine
│   │   └── system.py            # System (multi-component assembly)
│   │
│   ├── api/
│   │   └── scenario.py          # Scenario (fluent high-level API)
│   │
│   ├── models/
│   │   └── aerodynamics/        # Aerodynamic models
│   │
│   ├── visualization/
│   │   ├── plotting.py          # Matplotlib plot functions
│   │   └── dashboard.py         # Streamlit interactive dashboard
│   │
│   └── utils/
│       ├── orientation.py       # Quaternion utilities
│       ├── validation.py        # Input validation helpers
│       └── io.py                # File I/O utilities
│
├── examples/
│   ├── scenarios/               # Ready-to-run example scripts
│   │   ├── 01_simple_drop.py
│   │   ├── 02_parachute_system.py
│   │   └── 03_scenario_options.py
│   │
│   └── architecture/            # Deep-dive explanation notebooks
│       ├── solver_walkthrough.ipynb
│       ├── hybrid_solver_explained.ipynb
│       ├── hybrid_ivp_solver_explained.ipynb
│       └── simulation_explained.ipynb
│
├── tests/                       # Test suite (>90% coverage)
├── pyproject.toml               # Build config & dependencies
└── README.md
```

---

## 🔧 Configuration & Solver Tuning

### Baumgarte Stabilization

Prevents constraint drift in long simulations:

```python
solver = HybridSolver(alpha=5.0, beta=1.0)
#                     ^^^^^^      ^^^^^
#                     damping    stiffness
```

- **`alpha`** (damping term) — controls how fast constraint velocity errors decay
- **`beta`** (stiffness term) — controls how fast constraint position errors are corrected
- Higher values = more aggressive correction, but can cause oscillations
- Recommended starting point: `alpha=5.0, beta=1.0`

### Step Size Guidelines

| Scenario | Recommended `dt` | Solver |
|----------|-----------------|--------|
| Free fall (simple) | 0.01 s | `HybridSolver` |
| Constrained systems | 0.001–0.01 s | `HybridSolver` |
| Parachute deployment | Adaptive | `HybridIVPSolver("Radau")` |
| High-accuracy validation | Adaptive | `HybridIVPSolver("Radau", rtol=1e-9)` |

### Output Organization

All simulation results are automatically organized:

```
output/
  my_sim_20260213_143000/    ← timestamped to prevent overwrites
    logs/
      simulation.csv         ← full state history
    plots/
      Payload_trajectory_3d.png
      Payload_velocity_acceleration.png
      Payload_forces.png
```

---

## 📚 Further Reading

| Resource | Description |
|----------|-------------|
| `examples/architecture/solver_walkthrough.ipynb` | Line-by-line KKT matrix construction |
| `examples/architecture/hybrid_solver_explained.ipynb` | Fixed-step solver math & code |
| `examples/architecture/hybrid_ivp_solver_explained.ipynb` | Adaptive solver with `scipy.solve_ivp` |
| `examples/architecture/simulation_explained.ipynb` | World orchestrator deep-dive |
| `examples/scenarios/` | Ready-to-run simulation scripts |

---

<p align="center">
<b>AerisLab v0.2.0</b> · MIT License · Python 3.10+<br>
<i>Designed for aerospace engineering research at Brno University of Technology</i>
</p>
