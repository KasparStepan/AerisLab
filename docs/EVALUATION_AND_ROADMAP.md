# AerisLab — Senior Engineering & Scientific Evaluation

**Evaluator perspective:** Senior Python developer & computational scientist
**Codebase version:** 0.2.0 (branch `ClaudeHelp`, commit `0403dbc`)
**Date evaluated:** 2026-04-29
**PhD context:** Recovery system simulation with planned ML-based aerodynamic surrogates from FSI data and a path to becoming a more general multibody/multiphysics engine.

> A previous, more concise evaluation (`docs/SOFTWARE_EVALUATION.md`, dated 2026-03-23) is still useful as a snapshot. This document supersedes it with deeper analysis and adds (i) ML/FSI integration architecture, (ii) extensibility to other physics, (iii) a PhD-grade roadmap, and (iv) reproducibility/publication concerns.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Context & Goals](#2-project-context--goals)
3. [Codebase Snapshot](#3-codebase-snapshot)
4. [Architecture Assessment](#4-architecture-assessment)
5. [Code Quality & Engineering Practices](#5-code-quality--engineering-practices)
6. [Scientific & Numerical Correctness](#6-scientific--numerical-correctness)
7. [Concrete Issues & Bugs](#7-concrete-issues--bugs)
8. [Architectural Recommendations](#8-architectural-recommendations)
9. [Roadmap](#9-roadmap)
10. [ML/FSI Integration Plan](#10-mlfsi-integration-plan)
11. [Path to a General Multiphysics Engine](#11-path-to-a-general-multiphysics-engine)
12. [PhD-Specific: Reproducibility, Validation & Publication Readiness](#12-phd-specific-reproducibility-validation--publication-readiness)
13. [Risks & Trade-offs](#13-risks--trade-offs)
14. [Final Assessment](#14-final-assessment)

---

## 1. Executive Summary

**AerisLab is a clean, well-architected v0.2 alpha** with a strong physics core (KKT-constrained 6-DOF rigid body dynamics) and a thoughtful layered API. Tests are now passing (115/115 on `ClaudeHelp`), the previously-known critical bug (double gravity) has been fixed, and the parachute inflation model library is genuinely valuable for recovery-system research.

The codebase is in a healthy place to **support a PhD**, but to evolve it into a tool that is

1. publishable as a methodology contribution,
2. capable of consuming neural-network surrogates trained on FSI data, and
3. extensible to other physics (deformable bodies, fluid coupling, control loops),

it needs **structural changes** rather than incremental fixes. The most important ones are:

- Decouple **physical models** from **forces** (introduce a `Model` abstraction layer the way `dynamics/forces.py` is currently used).
- Adopt an **explicit state-vector / state-derivative interface** at the `World` level so neural-network surrogates can be plugged in without touching the integrator. Today the IVP solver re-applies forces twice and rebuilds the KKT system every RHS call — this will become both a correctness and performance bottleneck.
- Introduce **atmosphere, wind, and aerodynamic-coefficient providers** as first-class citizens (the path to ISA, Mach effects, and ML surrogate Cd(state)).
- Resolve the **Payload/Parachute API duplication** between `components/payload.py` + `components/parachute.py` and `components/standard.py` — these are two parallel implementations of the same idea with incompatible constructors.

Strengths — the things you should **not** lose during refactoring — are: the layered design, the composition-over-inheritance pattern, the dual fixed-step / IVP solvers, the rigorous verification suite (energy, pendulum period, Dzhanibekov), and the comprehensive parachute inflation model catalogue.

**Overall grade: 7.5/10** (up from 7/10 in the prior evaluation, primarily because all tests now pass and the critical bug was fixed). It is **not yet** ready to be the methodological backbone of a PhD without the structural changes described in §8 and §10.

---

## 2. Project Context & Goals

| Dimension | Current state | Long-term goal |
|---|---|---|
| Domain | Parachute–payload recovery systems | Generic constrained multibody + multi-fidelity aerodynamics |
| Aerodynamics | Analytical inflation models (Knacke, French-Huckins, etc.) | Neural-network surrogates trained on the user's FSI simulations (state → drag, possibly more) |
| Constraints | Rigid distance + point-weld | Joints (revolute, prismatic), suspension-line networks, soft constraints |
| Bodies | Rigid 6-DOF | + deformable / reduced-order canopy, possibly modal models |
| Atmosphere | None (constant ρ) | ISA + custom profiles + wind fields |
| Use cases | Single drop scenarios | Monte Carlo, design optimisation, parameter sweeps, validation against flight test data |
| Output | CSV + matplotlib | + structured (Parquet/HDF5/Zarr), animations, dashboards, automated reports |

This evaluation is calibrated to the **long-term goal**, not to today's narrow use case. A library that only does parachute drops is not where you want to spend a PhD.

---

## 3. Codebase Snapshot

```
src/aerislab/
├── api/scenario.py              # Fluent high-level API
├── components/                  # Domain objects
│   ├── base.py                  #   Component ABC (composition pattern)
│   ├── payload.py               #   Canonical Payload (takes body= arg)
│   ├── parachute.py             #   Parachute with deployment state machine
│   ├── system.py                #   Multi-component assembly
│   └── standard.py              #   PARALLEL Payload/Parachute (auto-builds body)
├── core/
│   ├── simulation.py            #   World orchestrator (765 LOC)
│   └── solver.py                #   KKT + Euler/IVP (635 LOC)
├── dynamics/
│   ├── body.py                  #   RigidBody6DOF (464 LOC)
│   ├── forces.py                #   Gravity, Drag, ParachuteDrag, Spring
│   ├── constraints.py           #   Distance, PointWeld
│   └── joints.py                #   Tether/weld/soft facades
├── models/aerodynamics/
│   └── parachute_models.py      #   6 inflation models + factories (1224 LOC)
├── visualization/plotting.py    #   matplotlib plots
├── logger.py                    #   Buffered CSV
└── utils/{validation,io}.py
```

**Code metrics (ClaudeHelp branch):**

| Metric | Value |
|---|---|
| Source LOC (counted) | ~3,300 (without `parachute_models.py` and `plotting.py`) |
| Source LOC total | ~5,300 |
| Tests | 115 (all passing) |
| Test LOC | ~3,900 |
| Source-to-test ratio | ~1.4 — healthy |
| Ruff issues | 283 (almost all whitespace; ~45 substantive) |
| Mypy errors | 20 (all in `parachute_models.py`, `plotting.py`, `scenario.py`) |
| Runtime deps | numpy, scipy, pandas, matplotlib |
| Python | 3.10+ |

The repo also has substantial dead weight:

- `examples/Old/` — 10 superseded demos
- `scripts/` — 10 ad-hoc plotting/animation scripts that are not part of the package or tests
- `simulation.csv` checked into the repo root
- `examples/scenarios/*.ipynb` — committed notebooks (heavyweight diffs)

This is normal during early development, but should be cleaned before the next milestone (see roadmap).

---

## 4. Architecture Assessment

### 4.1 Layering — strong

The four-layer design is excellent:

```
Scenario (fluent API)
    └─→ Components / System (domain objects)
            └─→ World + Solver (engine)
                    └─→ RigidBody6DOF + Force + Constraint (primitives)
```

Each layer can be used in isolation: a researcher who needs custom physics can drop down to `RigidBody6DOF` and run the solver manually; a student can stay at `Scenario`. This is the right structure for a research tool.

### 4.2 Composition over inheritance — strong

`Component` HAS-A `RigidBody6DOF`, not IS-A. The docstrings explicitly justify this, which is excellent. It also means the body can later be swapped for a deformable variant without rewriting the component layer — a key enabler for the long-term roadmap.

### 4.3 Force protocol — strong but incomplete

`Force` is a `typing.Protocol` with a single `apply(body, t)` method. Using structural subtyping rather than nominal inheritance is the right call.

**Gap:** there is no separation between *what the force is* (a model) and *how it acts on a body* (the application mechanics). Today, `ParachuteDrag` and `AdvancedParachute` both contain inflation-model logic, drag-coefficient logic, deployment-trigger logic, and the body-application logic in one class. When you later want a *neural-network drag coefficient* parameterised on velocity, altitude, deployment phase, etc., you will have to:

- subclass `Drag` (or copy it), or
- pass a callable `Cd=` (which works for scalar but not for vectorised batches), or
- write yet another parallel class.

This is the architectural change of largest leverage: **introduce a `Model` layer** between `Force` and the underlying physics (see §8 and §10).

### 4.4 KKT solver — solid

Schur complement is correct for the regime (few constraints relative to DOFs), Baumgarte stabilization is implemented, condition-number monitoring exists, and there's a least-squares fallback for singular constraint systems. The graceful-degradation behaviour is mature.

### 4.5 Dual integrators — strong

Having both fixed-step semi-implicit Euler and adaptive scipy IVP is pragmatic. The fixed-step solver is symplectic (good energy behaviour); the IVP solver has Radau/BDF for stiffness handling during parachute opening shock. This is a thoughtful design.

### 4.6 Component / API duplication — **architectural smell**

There are two `Payload` classes:

- `aerislab.components.payload.Payload` — takes `body=` (a pre-built `RigidBody6DOF`)
- `aerislab.components.standard.Payload` — takes `mass=`, `radius=`, `position=` and auto-builds the body

And two `Parachute` paths:

- `aerislab.components.parachute.Parachute` — uses `ParachuteDrag` + `DeploymentState` state machine
- `aerislab.components.standard.Parachute` — uses `AdvancedParachute` directly, **bypasses** the deployment state machine

The `__init__.py` of `components/` exports the canonical ones; `examples/scenarios/` import from `standard.py`; tests import from the canonical location. The result is two parallel APIs that diverge in capability:

| Feature | `components.Parachute` | `standard.Parachute` |
|---|---|---|
| Deployment state machine | Yes | No |
| Inflation model | `ParachuteDrag` (single tanh-gate) | `AdvancedParachute` (6 models) |
| User-facing constructor | Awkward (needs body) | Convenient |
| Used by examples | No | Yes |
| Used by tests | Yes | No |

**This is the single highest-priority refactor.** Pick one design (the `standard` constructor ergonomics + the canonical state machine + the advanced inflation models) and delete the other. See §8.1.

### 4.7 World — too big

`World` (765 LOC) is doing orchestration, logging, plotting, output management, energy diagnostics, ground termination, and progress printing. By v1.0 this should be 4–5 collaborating objects. See §8.2.

### 4.8 IVP solver re-application — performance bug-in-waiting

In `HybridIVPSolver.integrate()`:

1. Inside the RHS function, all forces are applied, the KKT system is assembled, accelerations computed.
2. After integration, **the RHS evaluation is repeated** for each logged time point so the logger can capture forces.

This doubles the force-computation cost, and worse, when you add a neural-network surrogate that takes ~1 ms per call, this overhead becomes severe. Caching forces during integration (or storing the multipliers and gradient-of-forces) is much better. See §8.4.

---

## 5. Code Quality & Engineering Practices

### 5.1 What is good

- **Type hints everywhere** with modern syntax (`list[X]`, `X | None`).
- **NumPy-style docstrings** on public APIs with Parameters/Returns/Notes/References.
- **Consistent SI units** documented in module-level docstrings.
- **`__slots__` on `RigidBody6DOF`** — good for memory and cache.
- **`pyproject.toml`** is well-structured: physics-aware ruff ignores (N806 for `I`/`J`/`F`/`M`), strict pytest markers, coverage exclusions.
- **Verification test suite** separate from unit tests — `tests/verification/` mirrors NASA-STD-7009A spirit (energy conservation, pendulum period, Dzhanibekov, terminal velocity). This is **above average for research code**.
- **Input validation** at constructors with meaningful errors and warnings.
- **CSV logger uses buffering** — pragmatic.

### 5.2 What is weak

- **Whitespace hygiene.** 248 W293 (blank line with whitespace) + 13 W291 (trailing whitespace) issues. These are auto-fixable; running `ruff check --fix` should be a pre-commit hook.
- **Mypy errors (20 in 3 files).** Mostly `Optional[float]` flowing into `_compute_normalized_time` etc. without a guard. None are correctness bugs in practice, but they signal that the type contract is loose.
- **`print()` everywhere.** `World`, `Scenario`, `Parachute` all `print(...)` for status. This should be `logging` so users can silence/redirect. Once you run Monte Carlo (1000s of simulations), `print` traffic alone matters.
- **`Constraint` is not a real ABC.** It uses `raise NotImplementedError` instead of `@abstractmethod`. Subclasses without `rows()`/`evaluate()`/`jacobian()` will fail at runtime instead of class-definition time.
- **`hasattr(fb, "last_force")` duck typing in `simulation.py:456`.** Fragile. Either standardise via a Protocol method or remove the feature.
- **`Scenario.connect(type="tether")` guesses attachment points** from `body.radius`. For non-trivial geometries (offset attachments, multi-line risers) this will silently produce wrong results.
- **Mutable default args** (B006) in two places — minor but well-known footgun.
- **No `logging.getLogger(__name__)` pattern.** Once present, swapping to JSON logs, file logs, or Rich console output becomes trivial.
- **No structured error types.** Everything is `ValueError`/`RuntimeError`. Custom exceptions (`ConstraintSingularError`, `IntegrationFailedError`) help users programmatically respond to failure modes, which matters for batch / Monte Carlo runs.

### 5.3 What is missing entirely

- **CI.** A `.github/workflows/test.yml` exists (single Python version, just pytest). Should expand to: ruff + mypy + pytest on 3.10/3.11/3.12; coverage upload; benchmark regression check.
- **Pre-commit hooks.** No `.pre-commit-config.yaml`. With ruff + mypy this is a 5-minute add and prevents whitespace drift.
- **Reproducibility infra.** No seeded RNG, no provenance metadata in CSV outputs (git SHA, package version, parameters, timestamp, atmosphere model used). Critical for PhD work — see §12.
- **Configuration files.** All scenarios are pure Python. A YAML/TOML loader is essential for parameter studies and external collaboration.
- **CLI entry point.** No `[project.scripts]` in `pyproject.toml`. `aerislab run scenario.yaml` would be a much better story than `python examples/scenarios/02_parachute_system.py`.
- **Continuous benchmarking.** `pytest-benchmark` is in dev deps but no benchmark tests exist.

---

## 6. Scientific & Numerical Correctness

### 6.1 What's done right

- **Quaternion orientation.** Avoids gimbal lock; scalar-last `[x,y,z,w]` matches scipy. Exponential-map integration is available as an option.
- **Gyroscopic torque** `-ω × (I_world ω)` is correctly added to the generalized force in `assemble_system()` (line 113). This is correct rigid-body Newton-Euler. Easy to get wrong.
- **Baumgarte stabilization** with both position (α) and velocity (β) terms.
- **Semi-implicit Euler** is symplectic for unconstrained systems → bounded energy drift.
- **Schur complement KKT solve** is the appropriate approach for sparse-constraint regimes.
- **Constraint Jacobians** for `DistanceConstraint` and `PointWeldConstraint` are correct (with the `r × d` and `-skew(r)` patterns).
- **Verification suite** tests free fall, pendulum period, spring period, terminal velocity, angular momentum conservation, and the **Dzhanibekov effect** (intermediate-axis instability) — the last one is a real test of 3D rotational integrity.

### 6.2 What needs attention

#### Numerical

- **Quaternion normalization inside the IVP RHS** (`solver.py:509, 549`) introduces a non-smooth operation. Stiff solvers using BDF or Radau jacobian estimation will see spurious gradients near `||q|| = 1 ± ε`. The mathematically clean approach is to add `||q||² = 1` as a holonomic constraint and let the KKT solver enforce it. This is also what production multibody codes (Adams, MBDyn) do.
- **Mass matrix is recomputed every RHS call** (in `inv_mass_matrix_world` per body). For *N* bodies and *K* RHS evaluations per step (Radau does 3+), this is `3KN` rotation-matrix multiplications. Cache the inverse mass matrix and invalidate only when q changes appreciably.
- **No constraint-violation logging.** Baumgarte reduces drift but does not eliminate it — for long simulations or stiff configurations, `C(q)` should be logged so users can detect drift.
- **Fixed Baumgarte `α`, `β`** are user-tunable but undocumented for non-trivial systems. A diagnostic (`solver.diagnose(world)` returning ω_C, ζ_C estimates) would help users pick values.
- **`HybridSolver.step()` re-uses `assemble_system` then explicitly applies constraint forces for logging** — the second pass solves nothing new but logs the multiplier-derived force. Fine for now, but tightly coupled to logger's structure.

#### Physical model fidelity

- **No atmosphere model.** `ρ = 1.225 kg/m³` is hardcoded. At 5 km altitude the actual density is ~0.74 — a 40% drag-force error. ISA atmosphere should be implemented before any quantitative validation against flight data.
- **No wind model.** `body.v` is treated as airspeed. With non-zero wind, true airspeed is `v - v_wind(p, t)`, not `v`. This is a 5-line change with huge impact on realism.
- **No Mach / Reynolds dependency for Cd.** Acceptable for round parachutes at subsonic speeds, but the moment you simulate higher-speed (drogue) parachutes or any winged decelerator, this is binding.
- **Added-mass model is applied as a post-hoc force, not in the mass matrix.** Physically the air entrained by the canopy modifies the effective inertia (`M_eff = M + M_a`), and `M_a` should appear in the LHS of the dynamics. As a force, it produces qualitatively similar peak loads but is not formally correct. For ML/FSI integration this matters: an FSI surrogate naturally outputs `M_a(state)` and `F_aero(state)` separately.
- **Parachute is rigid.** The canopy is a rigid body with constant inertia. This is fine for trajectory studies but precludes any interaction with shape change (skirt collapse, canopy breathing, glide). FSI data will give time-varying inertia and force application points; the architecture must allow this.
- **No coupled multi-body inertia for the parachute–payload system.** The system is two rigid bodies linked by a constraint, but the canopy "is" only its rigid component — the entrained air mass that *does* couple via added-mass effects is approximated rather than co-simulated.

### 6.3 Verification & validation gaps

- The verification suite covers analytical solutions but **not numerical convergence rates**. Adding a "halve dt, verify error halves accordingly" test for each integrator would catch order-of-accuracy regressions immediately.
- No comparison to **published parachute test data** (NASA TN, JSC datasets). Even one sample case (e.g., Apollo drogue chute) reproduced from literature would be enormously valuable for credibility.
- No **regression baseline** for parachute models. If you change `_force_continuous_inflation` next year, you need a snapshot of expected force-time curves to detect silent regressions.

---

## 7. Concrete Issues & Bugs

### 7.1 Bugs (functional defects)

| # | Severity | Location | Issue |
|---|---|---|---|
| B1 | High | `components/standard.py:91-122` (parallel `Parachute`) | Bypasses `DeploymentState` state machine — examples using `standard.Parachute` never expose deployment status to logging or tests |
| B2 | Medium | `api/scenario.py:118-123` | `connect(type="tether")` infers attachment points from `body.radius`, which silently produces garbage for non-spherical or offset attachments |
| B3 | Medium | `core/solver.py` IVP path | After integration, all forces are re-applied per logged sample — doubles compute, and in `solver.py:619` constraint forces are recomputed via a fresh `assemble_system` + `solve_kkt`. Both correctness-fragile and slow |
| B4 | Low | `dynamics/forces.py:230` | `ParachuteDrag.activation_status` is mutable state on a force object. If a user reuses the object across runs (Monte Carlo), state leaks across runs. Add a `.reset()` method or make the activation state immutable per call |
| B5 | Low | `core/simulation.py:564-632` | `integrate_to` chunked path silently ignores `t_events` from earlier chunks if a new `integrate()` is called — touchdown detection in chunked mode is fragile |

### 7.2 Code-smell issues

| # | Location | Issue |
|---|---|---|
| C1 | `dynamics/constraints.py:21-30` | `Constraint` uses `raise NotImplementedError` instead of `@abstractmethod` — runtime failures instead of class-definition failures |
| C2 | `core/simulation.py:455` | `if hasattr(fb, "last_force")` — fragile duck typing for force breakdown |
| C3 | Throughout | `print(...)` for status messages — should be `logging` |
| C4 | `examples/Old/`, `scripts/` | 20+ legacy files cluttering the repo |
| C5 | `simulation.csv` | A 237-byte CSV file at the repo root, accidentally committed |
| C6 | `examples/scenarios/*.ipynb` | Notebooks committed with output cells (large/messy diffs) |
| C7 | `models/__init__.py:9` | Lists "future modules" — fine, but they are comments, not stubs |
| C8 | `pyproject.toml:30` | Optional `viz` dep group was removed but `streamlit`/`plotly` are still referenced in deleted `dashboard.py` (gone on `ClaudeHelp`); ensure no straggling imports |

### 7.3 Type / lint health

- **Mypy:** 20 errors in 3 files. All in `parachute_models.py` (`Optional[float]` not narrowed before arithmetic), `plotting.py` (`plt.cm.tab10` resolution), and `scenario.py` (kwargs typed as `dict[str, object]`). Fixable in <1 hour.
- **Ruff:** 283 errors, ~245 are whitespace (auto-fixable in seconds). The remaining ~40 substantive issues should be addressed.

---

## 8. Architectural Recommendations

This section is the heart of the document. These are **structural** changes, not patches.

### 8.1 Resolve the components / standard duplication

**Pick one API.** Recommended:

```python
# components/payload.py — keep as the canonical ABC-style component
class Payload(Component):
    @classmethod
    def from_basic(cls, name, mass, radius, Cd=0.47, ...):
        """Convenience constructor (replaces standard.Payload)."""
        body = RigidBody6DOF(name=f"{name}_body", mass=mass, ...)
        return cls(name=name, body=body, Cd=Cd, ...)
```

Then delete `standard.py`. Migrate examples in one commit.

The `Parachute` story is more involved because `standard.Parachute` is the only path that exposes `AdvancedParachute`'s 6 inflation models. The fix:

1. Make `components.Parachute` accept any `InflationModel` (not only the simplistic `ParachuteDrag`).
2. Have its constructor accept `model: InflationModel | str = "knacke"` and resolve string → `AdvancedParachute(model_type=...)` internally.
3. Keep the `DeploymentState` state machine in the canonical `components.Parachute`.
4. Delete `standard.Parachute`.

### 8.2 Split `World`

`World` should be ~250 LOC, doing only orchestration. Extract:

- `OutputManager` (output dir creation, plot generation, manifest writing)
- `SimulationRunner` (the loop in `run()`/`integrate_to()` with progress, log_interval, termination)
- `EnergyDiagnostics` (`get_energy()`, plus future `momentum`, `constraint_violation`)
- `TerminationPolicy` (callback chain — ground contact, timeout, custom)

This is a refactor in 4 small PRs.

### 8.3 Introduce a `Model` layer

Create `models/` as a true sibling to `dynamics/`:

```
models/
├── atmosphere/         # ISA, custom, exponential
│   └── isa.py
├── wind/               # constant, profile, turbulent
├── aerodynamics/
│   ├── drag_models.py        # SphereCd, FlatPlateCd, CdTable, NeuralCd
│   ├── parachute_models.py   # existing AdvancedParachute (unchanged)
│   └── added_mass.py
├── deployment/
│   └── reefing.py
└── materials/          # fabric properties, line stretch
```

Each model is a Protocol-implementing object that the appropriate `Force` consumes. For example:

```python
class DragForce:
    def __init__(self, drag_model: DragModel, atmosphere: AtmosphereModel,
                 wind: WindModel | None = None, area_model: AreaModel | None = None):
        ...
    def apply(self, body, t):
        rho = self.atmosphere.density(body.p[2])
        v_air = body.v - (self.wind.velocity(body.p, t) if self.wind else 0)
        Cd = self.drag_model(state=body, t=t)  # could be ML surrogate
        A = self.area_model(state=body, t=t) if self.area_model else self.area
        f = -0.5 * rho * Cd * A * np.linalg.norm(v_air) * v_air
        body.apply_force(f, label="aerodynamic")
```

This gives you:

- **ISA / wind for free** (just plug different `AtmosphereModel`/`WindModel`).
- **ML drag coefficients** as a `NeuralCd(DragModel)` that wraps a torch / JAX model.
- **Decoupling** that lets the same `DragForce` serve a sphere, a parachute, or a wing.

### 8.4 Make the `World ↔ Solver` interface explicit (state-vector based)

Today the IVP solver mutates `world.bodies` inside the RHS function. This is fine for one solver but couples everything to one interface. Move to:

```python
class World:
    def state(self) -> NDArray:    # pack y from all bodies
    def set_state(self, y) -> None:  # unpack y back into bodies
    def state_derivative(self, t, y) -> NDArray:  # the RHS

class Solver(Protocol):
    def integrate(self, world: World, t_end: float) -> Trajectory: ...
```

Then:

- Adding new integrators (RK4 with substepping for ML inference, position-Verlet, exponential integrators) becomes trivial.
- The neural surrogate calls happen inside `state_derivative`, just like every other force — uniform.
- A future JAX-based engine for differentiable simulation can plug in by replacing `state_derivative` only.

This is the single highest-leverage refactor for the long-term roadmap.

### 8.5 Fix the IVP logging path

After integration:

- Either store `(t, y, forces, lambdas)` tuples during the RHS evaluation in a buffer, or
- Use `dense_output=True` and recompute *only the inexpensive things* (positions, velocities) for logging.

The current pattern of re-running force application + KKT for each logged sample is both 2× redundant and a correctness bug nucleus.

### 8.6 Replace `print()` with `logging`

Module-level `logger = logging.getLogger(__name__)`, keep `print` for examples only. Use Rich for prettier console rendering (optional dep).

### 8.7 First-class configuration

Add `aerislab.config.Scenario.from_yaml(path)` plus a YAML schema for:

```yaml
name: knacke_main_chute
duration: 60.0
solver: { preset: stiff, rtol: 1e-7 }
atmosphere: { model: isa, ground_temp: 288.15 }
wind: { model: constant, velocity: [5, 0, 0] }
components:
  - !Payload   { name: capsule, mass: 50, radius: 0.4, position: [0,0,2000] }
  - !Parachute { name: main, mass: 5, diameter: 12, model: continuous_inflation,
                 activation_altitude: 1500 }
connections:
  - { from: capsule, to: main, type: tether, length: 10.0 }
```

This is the bedrock for parameter studies and external collaboration — writing 200 YAML files is much easier than writing 200 Python scripts.

### 8.8 Output: structured + provenance

- Replace CSV-only with a default of **Parquet** (10× smaller, columnar, native pandas/Polars/DuckDB), keeping CSV as an option.
- For large runs and trajectory ensembles, support **HDF5** or **Zarr**.
- Every output directory should have a `manifest.json`:

```json
{
  "git_sha": "0403dbc",
  "package_version": "0.2.0",
  "python": "3.11.4",
  "scenario": {...},   // full scenario config
  "started": "2026-04-29T14:32:00Z",
  "finished": "2026-04-29T14:32:48Z",
  "rng_seed": 42,
  "warnings": [...]
}
```

Without this, three years from now you cannot reproduce your own thesis figures.

---

## 9. Roadmap

Phased to align with PhD milestones.

### Phase 0 — Stabilize (1–2 weeks)

- [ ] Run `ruff check --fix` (eliminates 245 whitespace issues at once).
- [ ] Fix the 20 mypy errors.
- [ ] Add `.pre-commit-config.yaml` (ruff + mypy + black-compatible).
- [ ] Expand CI: 3.10/3.11/3.12, ruff, mypy, pytest, coverage upload.
- [ ] Delete `examples/Old/`, `scripts/` (keep the 5 useful ones in `examples/scripts/`).
- [ ] Remove `simulation.csv` from repo root; add to `.gitignore`.
- [ ] Strip notebook outputs (use `nbstripout`).

### Phase 1 — Resolve duplication & smells (1 month)

- [ ] Unify `Payload` API; delete `standard.py`.
- [ ] Unify `Parachute` API; expose all 6 inflation models through canonical `components.Parachute`.
- [ ] Replace `print()` with `logging.getLogger(__name__)` everywhere in `src/`.
- [ ] Convert `Constraint` to true ABC (`@abstractmethod`).
- [ ] Add custom exceptions (`ConstraintSingularError`, `IntegrationFailedError`, `DeploymentError`).
- [ ] Add constraint-violation logging.

### Phase 2 — Physical models (1–2 months)

- [ ] Implement `models.atmosphere.ISA` with altitude-dependent ρ, T, p.
- [ ] Implement `models.wind.{Constant, AltitudeProfile}` and integrate into `Drag`/`ParachuteDrag`.
- [ ] Refactor forces to consume `AtmosphereModel`/`WindModel`.
- [ ] Add convergence-rate verification tests (halve dt → halve error).
- [ ] Reproduce one published parachute case (Apollo drogue or T-10) for literature validation.

### Phase 3 — Architectural refactor (2–3 months)

- [ ] Extract `OutputManager`, `SimulationRunner`, `TerminationPolicy` from `World`.
- [ ] Introduce explicit `World.state() / set_state() / state_derivative()` interface.
- [ ] Refactor `HybridIVPSolver` to consume the new interface.
- [ ] Fix IVP logging redundancy (single-pass).
- [ ] Cache inverse mass matrix; invalidate on quaternion change.
- [ ] Make added mass enter the LHS (`M + M_a`), not as a force.
- [ ] Replace CSV default with Parquet; add `manifest.json` provenance.

### Phase 4 — ML/FSI integration (3–6 months) — see §10

- [ ] Define `AeroSurrogate` Protocol.
- [ ] Implement `NeuralDragModel` reading PyTorch/ONNX models.
- [ ] Build the FSI → training data pipeline (state featurisation, target labelling).
- [ ] Add training script + config + reproducibility metadata.
- [ ] Add uncertainty propagation (ensembles or MC dropout).
- [ ] Validate against held-out FSI cases.

### Phase 5 — Generalisation (6–12 months) — see §11

- [ ] Add joint types (revolute, prismatic, universal).
- [ ] Add Modal-superposition deformable canopy (or reduced-order model).
- [ ] Add ground-interaction model (contact, friction).
- [ ] Monte Carlo / parameter sweep framework.
- [ ] Full Sphinx documentation site.
- [ ] CLI entry point.

### Phase 6 — v1.0 & publication (timed to thesis)

- [ ] Methodology paper.
- [ ] Open-source release with DOI (Zenodo).
- [ ] Software Impacts / JOSS submission.

---

## 10. ML/FSI Integration Plan

This is the most distinctive part of the user's PhD vision. It deserves its own design.

### 10.1 Goal

Train neural networks on FSI simulation data so that during a multibody simulation, the parachute's instantaneous **drag force vector** (and ideally **moment**, **added-mass tensor**, and **deformation modes**) is produced by the network as a function of:

- Current velocity vector (or velocity in body frame)
- Altitude (→ ρ via atmosphere)
- Inflation phase / time-since-deployment / current effective area
- Optionally: deformation state, line tension, history (RNN/Transformer)

The network replaces or augments the analytical inflation models.

### 10.2 Design

#### Surrogate Protocol

```python
class AeroSurrogate(Protocol):
    """Maps state → aerodynamic quantities."""
    def predict(self, state: AeroState) -> AeroOutput: ...

@dataclass
class AeroState:
    v_air: np.ndarray   # (3,) airspeed in body frame
    rho: float
    T: float            # temperature
    deployment_phase: float   # 0..1
    inflation_progress: float
    history: np.ndarray | None  # for sequence models

@dataclass
class AeroOutput:
    F: np.ndarray     # (3,) aerodynamic force, body frame
    M: np.ndarray     # (3,) moment, body frame
    M_added: np.ndarray | None   # (3,3) added-mass tensor (optional)
    Cd_eff: float     # effective Cd, for diagnostic logging
```

Then a force class:

```python
class NeuralAeroForce:
    def __init__(self, surrogate: AeroSurrogate, atmosphere, wind=None):
        self.surrogate = surrogate
        ...
    def apply(self, body, t):
        v_air = self._compute_airspeed(body, t)
        state = AeroState(v_air=v_air, rho=self.atmosphere.density(body.p[2]), ...)
        out = self.surrogate.predict(state)
        F_world = body.rotation_world() @ out.F
        body.apply_force(F_world, point_world=body.p, label="aero_neural")
        if out.M is not None:
            body.apply_torque(body.rotation_world() @ out.M)
```

#### Three reference implementations

1. **`AnalyticalSurrogate(model=AdvancedParachute(...))`** — wrap existing models in the protocol so the new code path is exercised by today's tests.
2. **`OnnxSurrogate(model_path="parachute.onnx")`** — production-friendly, avoids torch/jax import costs in user code.
3. **`TorchSurrogate(model=...)`** — for development and gradient propagation if you want differentiable simulation.

#### Critical engineering concerns

- **Frame consistency.** FSI simulations operate in the canopy body frame; the neural net should output body-frame quantities to keep the network rotation-equivariant. Apply `body.rotation_world()` only at the boundary.
- **Non-dimensionalisation.** Train on `(v/v_ref, ρ/ρ₀, area/area₀)` — not raw SI — so the network generalises across scales and altitudes.
- **History dependency.** Parachute opening shock has memory (canopy is filling). A purely state-conditioned MLP will be wrong during inflation. Either include `(t - t_deploy)` and `dA/dt` as features, or use an LSTM/GRU/Transformer.
- **Out-of-distribution detection.** If the simulator queries the network with `(v, ρ)` outside the FSI training envelope, the network will silently extrapolate. Add an OOD detector (e.g., Mahalanobis distance to training set, or a calibrated uncertainty estimator) and either log a warning or fall back to the analytical model.
- **Performance.** A naive PyTorch call from inside the IVP RHS is 1–10 ms; with Radau evaluating 3–5 stages per step over 60 s simulated time at ms steps, that is 150–1500 s of pure inference. Mitigations:
  - **Batch in time** when using fixed-step (call once per step for all stages).
  - **ONNX Runtime** instead of torch — typically 5–10× faster for inference.
  - **TensorRT / OpenVINO** for production.
  - **Precompile** with `torch.jit.script` or `torch.compile`.
  - **Ensure GIL is not held** if you use multiprocessing for Monte Carlo.

#### Training pipeline (separate package or `aerislab.ml/`)

```
aerislab/ml/
├── data/
│   ├── fsi_loader.py      # Read FSI simulation outputs (CFD format-agnostic)
│   ├── featurise.py       # state → (X, y) tensors with non-dimensionalisation
│   └── augment.py         # symmetry augmentation (mirror, rotation)
├── models/
│   ├── mlp.py
│   ├── lstm.py
│   └── transformer.py
├── train.py               # Hydra/typer-based config-driven training
├── evaluate.py            # held-out FSI cases, plots vs analytical model
└── infer.py               # ONNX/TorchScript export
```

Use **Hydra** or **Typer** for config-driven training (matches the YAML scenario approach in §8.7). Use **MLflow** or **Weights & Biases** for experiment tracking — important for the PhD, every run is reproducible and citable.

#### Validation & uncertainty

- Hold out **trajectories**, not random samples. A network that interpolates between `t=1.0s` and `t=1.05s` of the same FSI run will look great but generalise poorly.
- Report metrics at each phase (inflation peak, steady descent) separately.
- Use **deep ensembles** (5–10 networks with different seeds) for uncertainty bands. Cheap, well-calibrated, easy to integrate.
- Compare **integrated trajectory error** (terminal velocity, opening time, peak load) to the analytical model on held-out FSI cases. The end-user metric is descent prediction, not pointwise drag.

### 10.3 What this enables

- **Multi-fidelity workflows.** Use ML for the bulk of trajectory studies, switch to analytical for sanity checks.
- **Inverse design.** Train the model differentiably and gradient-optimise canopy parameters.
- **Real-time / hardware-in-loop** (post-PhD).
- **Genuine methodological contribution** — coupling FSI surrogates with constrained multibody dynamics is publishable (AIAA, Aerospace Science & Tech, JCP, JOSS).

---

## 11. Path to a General Multiphysics Engine

The user wants to use AerisLab beyond parachutes. This means the design must accommodate physics it does not yet have. Below is the minimum set of architectural seats reserved.

### 11.1 More joint / constraint types

Add to `dynamics/constraints.py`:

- `RevoluteJoint` (1 DOF — for hinges, control surfaces)
- `PrismaticJoint` (1 DOF — for sliders)
- `UniversalJoint` (2 DOF)
- `BallJoint` (3 DOF rotational, equivalent to keeping translational locked)

The KKT solver doesn't change — the Jacobian library expands. Each is ~30 LOC.

### 11.2 Beyond rigid bodies

Two viable paths:

- **Reduced-order modal models** for canopy: keep `RigidBody6DOF` for the gross motion, add modal coordinates for deformation. The modal forces come from FSI training (the same pipeline as §10).
- **Simple lumped-mass particle systems** for suspension lines: a tree of particles connected by stiff Spring + linear damping. Solve them at a sub-step inside the integrator if needed.

The `Component` abstraction is already general enough to wrap either.

### 11.3 Control surfaces & actuators

A `Controller` Protocol consuming `World` state and writing torques/forces:

```python
class Controller(Protocol):
    def update(self, t: float, dt: float, world: World) -> None: ...
```

Plug in PID, LQR, or RL policies later.

### 11.4 Multi-physics coupling

For light coupling (one-way from CFD → multibody) the `AeroSurrogate` story is already enough.

For heavy coupling (running an external solver in lockstep with the multibody integrator), reserve the `World.state_derivative()` boundary (§8.4) so the future "callout to FEniCS / OpenFOAM / your FSI solver" is a single clean call.

### 11.5 Differentiable simulation (long-term)

If you eventually want gradient-based optimisation of recovery-system designs:

- Either build a JAX backend (substantial; everything must be reimplemented in JAX primitives), or
- Use **tinygrad / PyTorch with `torch.func`** to autodiff through the existing implementation (pragmatic).

This is post-thesis but the `state_derivative` boundary already forecloses on it.

---

## 12. PhD-Specific: Reproducibility, Validation & Publication Readiness

Things that are *technically* niceties but, for a PhD, are not optional:

### 12.1 Reproducibility

- **Seed every RNG.** Currently no RNG is seeded because there is no stochasticity. The moment you add Monte Carlo, atmospheric turbulence, or an ML model with dropout, deterministic seeds become essential.
- **Pin dependencies.** `pyproject.toml` says `numpy>=1.21`. For thesis figures, use a `requirements.lock` (or `uv pip compile`) snapshot per chapter.
- **Provenance metadata.** Every output directory must contain git SHA + package version + config + seed. Without this, a reviewer asking "can you regenerate Figure 4.7?" gets an unsatisfying answer.
- **Docker / Apptainer image** with a tagged release per published result. Expensive but pays off in year 4.
- **Output is a manifest, not a CSV.** Already discussed in §8.8; for a PhD this matters a lot.

### 12.2 Validation hierarchy (per ASME V&V 10-2006 / NASA-STD-7009A)

You already have **verification** (analytical solutions). Add:

- **Code verification by manufactured solutions** — pick a non-trivial dynamics test case, manufacture an exact solution, demonstrate convergence.
- **Calculation verification** — convergence study at every published result.
- **Validation** — comparison to experimental data. Required for any ML claim. Even one wind-tunnel or drop-test dataset reproduced is gold.

### 12.3 Documentation & dissemination

- **Sphinx site** with API + theory + tutorials. MkDocs is lighter; either is fine.
- **Worked examples notebook gallery** (sphinx-gallery or jupyter-book).
- **JOSS / Software Impacts** publication after v1.0 — a few weeks of writing for a citable software contribution.
- **Zenodo DOI** per release.
- **Cite the correct things** in the docs: KKT formulation (Featherstone 2008 *Rigid Body Dynamics Algorithms*), Baumgarte (1972), the parachute references (Knacke 1992; Wolf 1973; French & Huckins 1964) — already done, keep doing.

### 12.4 Performance & scale

- **Benchmark the IVP path.** With 2 bodies and Radau, what is steps-per-second? This is your headroom budget for ML inference cost.
- **Profiling baseline.** Run `cProfile` on the parachute scenario, save the SVG flame graph. Re-run after each refactor.
- **Embarrassingly-parallel Monte Carlo.** `multiprocessing.Pool` over scenarios is sufficient; no need for Dask until N > 10⁴.

### 12.5 What reviewers will ask

Pre-empt:

- "How does drift behave over a 600 s simulation with constraints?" → constraint-violation log.
- "What's the convergence rate?" → automated test, plotted.
- "How does the surrogate degrade for inputs outside training data?" → OOD detector + uncertainty plot.
- "Can I reproduce Figure X?" → provenance manifest + Docker image.

---

## 13. Risks & Trade-offs

- **Refactor budget.** §8 lists ~6 months of part-time engineering. A PhD has finite time. Prioritise refactors that **unblock scientific contributions** (the `Model` layer, the state-vector interface) and defer the cosmetic ones.
- **Premature generalisation.** Trying to build a "general physics engine" before you have a clear second use case will cost you 6 months and gain nothing. Implement the second physics (e.g., a winged decelerator) and let the abstractions emerge from real demand.
- **ML before physics is wrong.** Do not start training neural networks until the atmosphere model, wind, and validation case are in place — without them, you cannot interpret the surrogate's errors. Pure ML on a wrong baseline is the most common mistake in CFD-ML papers.
- **Scope creep into general MBD.** PyDy, MuJoCo, MBDyn, and Adams already exist. Your contribution is *coupling FSI-derived ML aerodynamics to constrained multibody dynamics in an open, scriptable Python package*. Stay close to that.
- **Compatibility breakage.** The duplication and the `Model` layer refactor will break user code. There is no user code yet — do them now, before publication. After v1.0, semver discipline becomes necessary.

---

## 14. Final Assessment

| Axis | Score | Comment |
|---|---|---|
| Architecture | 7/10 | Layered design + composition is excellent; component duplication is the main blemish |
| Code quality | 7/10 | Type hints, docstrings, validation are solid; whitespace and `print` hygiene weak |
| Numerical correctness | 7.5/10 | Core physics correct; needs atmosphere/wind, IVP path tightening, constraint drift logging |
| Testing | 7/10 | Verification suite is unusually good for research code; needs convergence-rate tests + literature replication |
| Engineering practices | 6/10 | CI minimal, no pre-commit, no provenance metadata, `print` everywhere |
| Extensibility for ML/FSI | 5/10 | Possible but not designed for it; the `Model` layer + state-vector interface are gating refactors |
| PhD readiness | 6/10 | Verification rigor is there; reproducibility infra is not |
| **Overall** | **7.5/10** | **Solid foundation, ready for the PhD with the refactors in §8 and §10.** |

### What to do this month

1. `ruff check --fix` + add pre-commit (1 hour).
2. Fix 20 mypy errors (2 hours).
3. Delete `examples/Old/`, `scripts/` legacy, `simulation.csv` (10 minutes).
4. Strip notebook outputs (30 minutes).
5. Replace `print()` with `logging` (half a day).
6. Implement ISA atmosphere model (one day).
7. Resolve `Payload` / `Parachute` duplication (one day).

After that, you will have **a clean v0.3** to build the `Model` layer, ML integration, and the rest of the roadmap on. Without those refactors, every neural network and atmosphere change will fight the existing structure.

---

## Appendix A — Recommended Library Choices

| Need | Recommendation |
|---|---|
| Logging | stdlib `logging` + optional `rich` for nicer console |
| Config | `pydantic` v2 for typed configs + YAML loader |
| CLI | `typer` (or `click`) |
| Output | `pyarrow` (Parquet) + `h5py` (HDF5) optional |
| Plotting | keep matplotlib, add `plotly` optional for interactive |
| Animation | `matplotlib.animation` (basic) or `pyvista` (for nice 3D) |
| ML inference | `onnxruntime` for production; `torch` for development |
| ML training | `pytorch` + `pytorch-lightning` + `hydra-core` + `wandb` or `mlflow` |
| Profiling | `py-spy` (sampling), `scalene` (memory + CPU + GPU) |
| Docs | `mkdocs-material` (lightweight) or `sphinx` + `myst-parser` |
| Testing | keep pytest, add `hypothesis` for property-based tests |
| Pre-commit | `pre-commit` + ruff + mypy + nbstripout |
| Packaging | `hatch` or stay with setuptools |

## Appendix B — One-page action checklist

- [ ] Run `ruff check --fix` and commit.
- [ ] Add `.pre-commit-config.yaml`.
- [ ] Expand CI matrix to 3.10/3.11/3.12 with ruff + mypy.
- [ ] Delete `examples/Old/` and `scripts/`.
- [ ] Remove `simulation.csv` from repo root.
- [ ] Run `nbstripout --install` and clean notebook diffs.
- [ ] Replace every `print()` in `src/` with `logger.info/warning/...`.
- [ ] Fix the 20 mypy errors (`Optional[float]` narrowing; matplotlib type stubs).
- [ ] Make `Constraint` a true ABC.
- [ ] Add `manifest.json` writer to `OutputManager`.
- [ ] Implement `models.atmosphere.ISA`.
- [ ] Implement `models.wind.Constant`.
- [ ] Refactor `Drag` / `ParachuteDrag` to consume `AtmosphereModel` + `WindModel`.
- [ ] Resolve `Payload` / `Parachute` duplication.
- [ ] Add convergence-rate verification tests.
- [ ] Reproduce one published parachute case.
- [ ] Define `AeroSurrogate` Protocol.
- [ ] Refactor `World` to expose `state()` / `set_state()` / `state_derivative()`.

---

*Evaluation prepared for: Štěpán Kaspar (kaspar.stepan.cz@gmail.com), VUT Brno PhD project.*
