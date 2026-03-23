# AerisLab — Software Evaluation

**Evaluator perspective:** Senior Python developer & computational scientist
**Codebase version:** 0.2.0 (commit ef58d37)
**Date:** 2026-03-23

---

## 1. What Is AerisLab?

AerisLab (**A**erospace & **E**ngineering **R**esearch **I**ntegrated **S**imulator) is a Python-based 6-DOF rigid body dynamics engine designed for aerospace recovery system simulations — primarily parachute-payload systems.

It solves the constrained multi-body dynamics problem:

```
M · a = F + Jᵀ · λ       (equations of motion)
J · a = rhs               (constraint equations)
```

using a KKT (Karush-Kuhn-Tucker) formulation with Baumgarte stabilization.

### Core capabilities

- **6-DOF rigid body dynamics** with quaternion orientation (avoids gimbal lock)
- **Holonomic constraints** (distance, point-weld) solved via Schur complement
- **Dual solvers**: fixed-step semi-implicit Euler and adaptive scipy IVP (Radau, RK45, BDF)
- **Parachute inflation models**: 6 physics-based models (Knacke, French-Huckins, Pflanz, porosity-corrected, etc.)
- **Component architecture**: Payload, Parachute, System composition
- **Automatic output management**: timestamped directories, CSV logging, built-in plotting

### Typical use case

Simulating a payload dropping from altitude with one or more parachutes deploying at configurable conditions, connected via tethers/constraints, and analyzing descent trajectory, velocities, forces, and opening shock loads.

---

## 2. Architecture Overview

```
aerislab/
├── api/scenario.py            # High-level fluent API (Scenario)
├── components/                # Domain layer
│   ├── base.py                #   Component ABC (composition pattern)
│   ├── payload.py             #   Payload component
│   ├── parachute.py           #   Parachute with deployment state machine
│   ├── system.py              #   Multi-component assembly manager
│   └── standard.py            #   Self-configuring convenience components
├── core/                      # Simulation engine
│   ├── simulation.py          #   World orchestrator (~765 LOC)
│   └── solver.py              #   KKT solver + integrators (~635 LOC)
├── dynamics/                  # Physics primitives
│   ├── body.py                #   RigidBody6DOF (~464 LOC)
│   ├── forces.py              #   Gravity, Drag, ParachuteDrag, Spring (~462 LOC)
│   ├── constraints.py         #   Distance & PointWeld constraints (~136 LOC)
│   └── joints.py              #   Joint facades (tether, weld, soft) (~50 LOC)
├── models/aerodynamics/
│   └── parachute_models.py    #   Advanced inflation models (~500+ LOC)
├── visualization/plotting.py  #   Trajectory, velocity, force plots (~517 LOC)
├── logger.py                  #   Buffered CSV logger (~236 LOC)
└── utils/                     #   Validation & I/O helpers
```

**Total source:** ~4,660 LOC
**Total tests:** ~115 tests across 13 unit test + 6 verification modules

### Layered design

```
┌────────────────────────────────────┐
│         Scenario API (fluent)      │  ← User-facing
├────────────────────────────────────┤
│    Components (Payload, Parachute) │  ← Domain logic
├────────────────────────────────────┤
│  World ←→ Solver (KKT + Euler/IVP)│  ← Simulation engine
├────────────────────────────────────┤
│  RigidBody6DOF, Forces, Constraints│  ← Physics primitives
└────────────────────────────────────┘
```

Users can work at any level — from the high-level `Scenario` API down to manually constructing `RigidBody6DOF` objects and stepping the solver.

---

## 3. Strengths

### 3.1 Scientific / Domain

- **Correct physics formulation.** The KKT constrained dynamics with Baumgarte stabilization is a well-established approach used in production multibody dynamics codes. The Schur complement solver is the right choice for systems with few constraints relative to DOFs.

- **Multiple parachute inflation models.** Having 6 different models (simple drag, Knacke, continuous inflation, mass-flow balance, French-Huckins, porosity-corrected) in a single package is valuable for comparing approaches and validating against experimental data. References to primary literature (Wolf 1973, French & Huckins 1964, Knacke 1992) are cited.

- **Dual solver strategy.** Offering both fixed-step semi-implicit Euler (fast, stable, predictable) and adaptive stiff solvers (Radau, BDF via scipy) is a pragmatic design — users can prototype quickly and then switch to high-accuracy integration.

- **Quaternion orientation.** Avoids gimbal lock and singularities inherent to Euler angles. Exponential map integration option is a nice addition for long-duration simulations.

- **Verification test suite.** Having dedicated verification tests (energy conservation, pendulum period, rotation consistency, kinematics) goes beyond typical unit testing and shows scientific rigor.

### 3.2 Software Engineering

- **Clean separation of concerns.** The layered architecture (physics primitives → engine → components → API) is well-structured. Each layer can be used independently.

- **Composition over inheritance.** The `Component` HAS-A `RigidBody6DOF` design is the right pattern. The docstrings explicitly justify this choice, which is good for future maintainers.

- **Type hints throughout.** Full PEP 484 type annotations on all public APIs, using modern syntax (`list[X]`, `X | None`). Makes the codebase navigable with IDEs.

- **Thorough docstrings.** NumPy-style docstrings with Parameters, Returns, Notes, Examples, and References sections. Units are documented consistently (SI). This is above average for research code.

- **Input validation.** Physical parameters are validated at construction (positive mass, positive-definite inertia, valid quaternion, etc.) with meaningful error messages. Warnings for edge cases (near-zero mass, ill-conditioned constraints).

- **Protocol-based forces.** Using `typing.Protocol` for the `Force` interface enables structural subtyping — custom forces don't need to inherit from anything, just implement `apply()`.

- **Sensible project config.** `pyproject.toml` is well-configured: ruff with physics-aware lint ignores (N806 for `I`, `J`, `F`; E741 for `l`), pytest strict markers, coverage exclusions.

- **`__slots__` on RigidBody6DOF.** Good for memory efficiency when simulating many bodies.

- **Buffered CSV logger.** Reducing I/O syscalls with configurable buffer size is a pragmatic optimization.

### 3.3 Usability

- **Multiple API levels.** The `Scenario` fluent API is great for quick setups; the `World`/`System`/`Component` API gives full control; and the raw `RigidBody6DOF`/`Solver` API allows custom integration loops.

- **Auto-managed output.** Timestamped output directories with organized `logs/` and `plots/` subdirectories prevent accidental overwrites during parameter studies.

- **Self-configuring components.** `standard.Payload` and `standard.Parachute` auto-create their rigid bodies with sensible defaults, lowering the barrier for new users.

---

## 4. Issues & Bugs Found

### 4.1 Critical Bug: Double Gravity in Scenario API

**File:** `src/aerislab/api/scenario.py:36-38`

```python
self.world.add_global_force(Gravity(np.array([0, 0, -9.81])))
self.world.add_global_force(Gravity(np.array([0, 0, -9.81])))  # DUPLICATE
```

Gravity is added **twice**, meaning every simulation run through the `Scenario` API experiences `g = -19.62 m/s²`. This silently doubles all gravitational forces, producing physically incorrect results. Every example in `examples/scenarios/` uses `Scenario`, so all scenario-based simulations are affected.

**Impact:** High — scientifically incorrect results from the primary user-facing API.

### 4.2 Test Failures (11 of 115)

Current test suite: **104 passed, 11 failed**.

Failure categories:
- **5 failures in `test_forces.py`:** `TypeError: MockRigidBody...` — mock objects are out of sync with the `apply_force(label=...)` signature change. The tests use mocks that don't accept the `label` keyword argument.
- **3-4 failures in `test_integration_example.py`:** `AssertionError` (typo in test?) and file-creation assertions — likely related to the double-gravity bug or output path assumptions.
- **1 failure in `test_constraints.py`:** Pendulum period test — may indicate a numerical accuracy issue or the test tolerance is too tight.
- **1 failure in `test_rotation.py`:** Dzhanibekov effect test — complex rotational dynamics verification failing.

The mock-related failures suggest the `label` parameter was added to `apply_force()` but the test mocks were not updated. This is a test maintenance issue, not a physics bug.

### 4.3 Version Mismatch

`pyproject.toml` declares `version = "0.1.0"` but `__init__.py` has `__version__ = "0.2.0"`. These should be in sync. Consider using a single source of truth (e.g., `importlib.metadata`).

### 4.4 Duplicate `# Logging` Comment

`__init__.py:77-78` has a duplicated `# Logging` comment in `__all__`. Minor but sloppy.

---

## 5. What Could Be Improved

### 5.1 High Priority (affects correctness or usability)

| Issue | Details |
|-------|---------|
| **Fix double gravity bug** | `scenario.py` line 38 is a copy-paste duplicate |
| **Fix broken tests** | Update mocks in `test_forces.py` to match current `apply_force` signature; investigate verification test failures |
| **Atmosphere model** | Air density (`rho`) is hardcoded at sea-level (1.225 kg/m³). For drops from 2+ km, this introduces 20-30% error in drag. Implement at minimum ISA (International Standard Atmosphere) |
| **No wind model** | All aerodynamic forces assume zero wind. Even a simple constant-wind vector would significantly increase realism |
| **CI pipeline is minimal** | Only runs `pytest` on Python 3.11. Should also run ruff, mypy, and test on 3.10/3.12 |

### 5.2 Medium Priority (code quality and maintainability)

| Issue | Details |
|-------|---------|
| **`World` class is too large** | 765 LOC doing orchestration, logging, plotting, and output management. Consider extracting `SimulationRunner`, `OutputManager` |
| **`print()` statements everywhere** | World, Scenario, Parachute all use `print()` for status. Replace with Python `logging` module so users can control verbosity |
| **No configuration file support** | Simulations are defined purely in Python scripts. YAML/JSON/TOML config support would enable parameter studies without code changes |
| **`Constraint` is not a true ABC** | `constraints.py` defines `Constraint` with `raise NotImplementedError` methods instead of using `@abstractmethod`. This means errors are caught at runtime, not at class definition |
| **Hardcoded attachment point logic** | `Scenario.connect()` assumes vertical stacking and guesses attachment points from radius. This will silently produce wrong results for non-trivial geometries |
| **No serialization / checkpoint** | Cannot save/restore simulation state mid-run. Important for long simulations and debugging |
| **`hasattr` checks for force logging** | `simulation.py:456` uses `hasattr(fb, "last_force")` — fragile duck typing. The `Force` protocol should include this if force breakdown logging is a feature |

### 5.3 Lower Priority (nice-to-have)

| Issue | Details |
|-------|---------|
| **No CLI entry point** | Package has no `[project.scripts]` — users must run Python files directly. A `aerislab run config.yaml` CLI would improve usability |
| **Empty `docs/` directory** | No external documentation beyond README. For a research tool, having usage guides and theory documentation would help adoption |
| **No animation / 3D visualization** | Only static matplotlib plots. A simple 3D animation (even matplotlib `FuncAnimation`) would be valuable for debugging and presentations |
| **Sparse `__init__.py` for subpackages** | `components/__init__.py`, `models/__init__.py` etc. could re-export key symbols for cleaner imports |
| **No benchmarking in CI** | `pytest-benchmark` is a dev dependency but no benchmark tests exist and CI doesn't track performance |
| **Ground interaction model** | Simulation just terminates on ground contact. No bounce, friction, or impact dynamics |

---

## 6. Numerical / Scientific Considerations

### What's done well

- **Baumgarte stabilization** prevents constraint drift (position-level and velocity-level correction terms).
- **Condition number monitoring** in `solve_kkt()` warns about ill-conditioned constraint systems.
- **Least-squares fallback** when the constraint system is singular — graceful degradation.
- **Epsilon guards** throughout (velocity, distance, quaternion) prevent division-by-zero.
- **Semi-implicit Euler** is symplectic for unconstrained systems — good energy behavior.
- **Energy diagnostic** (`get_energy()`) available for conservation checks.

### What needs attention

- **No constraint drift monitoring at runtime.** Baumgarte stabilization reduces drift but doesn't eliminate it. Logging the constraint violation `C(q)` over time would help users detect problems.
- **Mass matrix is re-computed every RHS evaluation** in the IVP solver. For systems with many bodies, caching the inverse mass matrix (which only changes with orientation) could significantly improve performance.
- **IVP solver re-applies all forces for logging** after integration — this doubles the force computation cost. Consider storing forces during integration instead.
- **No adaptive Baumgarte parameters.** Fixed α, β work for one system but may need tuning for different setups. Auto-tuning or at least guidelines would help.
- **Quaternion normalization during IVP.** The solver normalizes quaternions inside the RHS function (`quat_normalize`), which adds a non-smooth operation that stiff solvers may struggle with. A quaternion constraint approach (||q||² = 1 as an additional constraint) would be more mathematically rigorous.

---

## 7. Code Metrics Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Total source LOC | ~4,660 | Compact, manageable |
| Test count | 115 (104 pass, 11 fail) | Good coverage intent, needs maintenance |
| Test pass rate | 90.4% | Needs fixing before any release |
| Dependencies (runtime) | 4 (numpy, scipy, pandas, matplotlib) | Lean, all standard scientific Python |
| Python version | 3.10+ | Good — modern syntax without cutting-edge requirement |
| Type hints | Comprehensive | Strong IDE support |
| Docstrings | Thorough (NumPy style) | Above average for research code |
| Linting | Ruff configured | Good, with physics-aware ignores |
| CI | GitHub Actions (minimal) | Needs expansion |

---

## 8. Comparison to Alternatives

| Feature | AerisLab | JSBSim | OpenDSSA | PyDy |
|---------|----------|--------|----------|------|
| Language | Python | C++ (Python bindings) | Fortran/Python | Python |
| Focus | Recovery systems | Full flight dynamics | Parachute-specific | General multibody |
| 6-DOF | Yes | Yes | Yes | Yes |
| Constraints | KKT solver | Built-in | Limited | Kane's method |
| Parachute models | 6 inflation models | Basic | Advanced | None |
| Ease of use | High (pure Python) | Medium | Low | Medium |
| Performance | Moderate | High | High | Moderate |
| Maturity | Alpha (v0.2) | Production | Research | Stable |

AerisLab occupies a useful niche: **more accessible than Fortran/C++ tools, more domain-specific than general multibody libraries, and richer parachute physics than flight simulators.**

---

## 9. Recommendations for Future Development

### Immediate (before any scientific use)
1. Fix the double-gravity bug in `Scenario`
2. Fix all 11 failing tests
3. Sync version numbers

### Short-term (next development cycle)
4. Add ISA atmosphere model (density varies with altitude)
5. Replace `print()` with `logging` module
6. Add constraint violation monitoring/logging
7. Expand CI (ruff, mypy, Python 3.10-3.12)

### Medium-term (towards v1.0)
8. Configuration file support (YAML/TOML)
9. Wind model (at minimum constant, ideally turbulence profiles)
10. CLI entry point
11. Extract output management from `World`
12. Add 3D animation capability
13. Documentation site (Sphinx or MkDocs)

### Long-term (research features)
14. Monte Carlo / parameter sweep framework
15. Multi-stage parachute systems (drogue → main)
16. Deformable body support (canopy shape changes)
17. Ground interaction model
18. Real-time visualization

---

## 10. Overall Assessment

AerisLab is a **well-architected early-stage simulation tool** with solid physics foundations and clean Python engineering. The layered architecture, composition patterns, and comprehensive docstrings demonstrate thoughtful design. The parachute inflation models represent genuine domain expertise.

The main concerns are:
- A **critical bug** (double gravity) in the primary user API that produces scientifically wrong results
- **11 failing tests** indicating the codebase has evolved faster than the test suite
- Missing **atmosphere and wind models** that limit scientific applicability above sea-level

For a v0.2 alpha, the code quality is strong. With the bugs fixed and an atmosphere model added, this would be a credible tool for parachute recovery system analysis in a research context.

**Rating: 7/10** — Solid foundation, needs bug fixes and atmosphere model before scientific use.
