# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AerisLab is a Python-based 6-DOF rigid body dynamics engine for aerospace recovery system simulations (parachute-payload systems). It solves constrained multi-body dynamics using KKT formulation with Baumgarte stabilization.

## Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file or test
pytest tests/test_body.py
pytest tests/test_body.py::test_quaternion_normalization

# Run excluding slow tests
pytest -m "not slow"

# Lint
ruff check src/

# Type check
mypy src/aerislab/

# Run an example
python examples/scenarios/01_simple_drop.py
```

The venv is at `.venv/` — use `.venv/bin/python` and `.venv/bin/pytest` if not activated.

## Architecture

**Layered design (bottom-up):**

1. **Physics primitives** (`dynamics/`): `RigidBody6DOF` (quaternion-based), `Force` protocol (Gravity, Drag, ParachuteDrag, Spring), `Constraint` types (Distance, PointWeld), joint facades.

2. **Simulation engine** (`core/`): `World` orchestrates bodies/forces/constraints/logging. `HybridSolver` does fixed-step semi-implicit Euler + KKT. `HybridIVPSolver` wraps scipy adaptive solvers (Radau, RK45, BDF) with the same KKT constraint enforcement.

3. **Component layer** (`components/`): `Component` ABC wraps RigidBody6DOF via composition (HAS-A, not IS-A). `Payload` and `Parachute` add domain logic. `System` groups components + constraints. `standard.py` provides self-configuring convenience components.

4. **Scenario API** (`api/scenario.py`): Fluent interface for quick simulation setup with solver presets.

**Key data flow per timestep:**
- Clear forces → Update component states (deployment logic) → Apply forces (component, global, interaction) → Assemble KKT system (M, J, F, rhs) → Solve via Schur complement → Integrate (semi-implicit Euler or scipy IVP) → Log → Check termination

**Parachute models** (`models/aerodynamics/parachute_models.py`): 6 inflation models (Knacke, French-Huckins, continuous inflation, mass-flow balance, porosity-corrected, added mass). The `AdvancedParachute` class is a force that implements the `Force` protocol.

**Constraints** use velocity-level enforcement: `J·a = rhs` where rhs includes Baumgarte stabilization terms `-(1+β)·J·v - α·C`.

## Conventions

- SI units throughout (meters, kg, seconds, radians)
- Quaternions are scalar-last: `[qx, qy, qz, qw]` (scipy convention)
- Physics variable names are intentionally uppercase (`I`, `J`, `F`, `M`, `Cd`) — ruff rules N806, N803, E741 are suppressed for this reason
- Line length: 100 (ruff configured)
- Docstrings: NumPy style
- Source layout: `src/aerislab/` (src-layout pattern)
