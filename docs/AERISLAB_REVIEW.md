# AerisLab — Comprehensive Engineering & Scientific Review

**Reviewer perspective:** Senior Python developer & computational scientist — independent re-evaluation
**Codebase version:** 0.2.0
**Branch / commit reviewed:** `ClaudeHelp` @ `8ef845a` (HEAD)
**Date:** 2026-05-13
**Supersedes:** `SOFTWARE_EVALUATION.md` (2026-03-23) and `EVALUATION_AND_ROADMAP.md` (2026-04-29)

This document is the single up-to-date reference. The two prior documents remain useful as historical snapshots but several of their findings are now stale (some claims are no longer true, some new bugs have appeared). All claims here were re-verified against `HEAD` by reading the current sources, running `pytest`, `ruff`, `mypy`, `coverage`, and executing live smoke tests.

A user-stated constraint shapes this review: **the in-house solvers are kept** (no rebase onto Project Chrono / MuJoCo / MBDyn). The architectural recommendations are aimed at making the in-house engine the best version of itself, not at replacing it.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Ground-Truth Status (Re-verified)](#2-ground-truth-status-re-verified)
3. [New Critical Bugs Found This Pass](#3-new-critical-bugs-found-this-pass)
4. [Standing Bugs from Prior Docs (Status Update)](#4-standing-bugs-from-prior-docs-status-update)
5. [Architectural Assessment](#5-architectural-assessment)
6. [Subsystem Deep-Dive](#6-subsystem-deep-dive)
7. [Numerical & Physics Correctness](#7-numerical--physics-correctness)
8. [Tests, V&V, and Reproducibility](#8-tests-vv-and-reproducibility)
9. [Engineering Hygiene](#9-engineering-hygiene)
10. [Short-Term Roadmap (next ~4 weeks)](#10-short-term-roadmap-next-4-weeks)
11. [Medium-Term Refactors (3–6 months)](#11-medium-term-refactors-36-months)
12. [Long-Term Capabilities & Aerospace/Space Expansion](#12-long-term-capabilities--aerospacespace-expansion)
13. [ML / FSI Integration Plan (Refined)](#13-ml--fsi-integration-plan-refined)
14. [Risks, Trade-offs, and What to Avoid](#14-risks-trade-offs-and-what-to-avoid)
15. [Action Checklist](#15-action-checklist)

---

## 1. Executive Summary

AerisLab is a clean, thoughtfully-layered v0.2 alpha with a strong physics core and a coherent vision. Since the prior evaluation, real progress has been made — the double-gravity bug is fixed, all 115 tests now pass, and the version mismatch is resolved.

**However, a fresh diagnostic pass surfaced two new severity-critical bugs** that make the primary user-facing API silently produce physically wrong results, plus several high-impact correctness issues that affect the IVP solver path (the default in `Scenario`). The engineering rigor of the verification suite (energy, pendulum, Dzhanibekov) coexists with smoke-test scenarios that are quietly broken — a strong sign that the unit tests are not exercising the paths users actually run.

The architecture is in a healthy place to support a thesis but needs three structural moves to scale to the long-term vision (FSI/ML aerodynamic surrogates → general aerospace recovery → potentially space dynamics):

1. **Decouple `World` from `Solver` via an explicit state-vector boundary.** Today the IVP solver mutates body state inside the RHS, recomputes mass matrices unnecessarily, and re-runs the full force pipeline twice for logging. This is the gating refactor for both performance and ML pluggability.
2. **Introduce a `Model` layer between `Force` and the underlying physics.** Atmosphere, wind, drag-coefficient providers, inflation models, and ML surrogates must all be composable instead of baked into individual `Force` subclasses. This is the gating refactor for FSI/ML integration and for any quantitative scientific use above sea level.
3. **Resolve the `components/` vs. `components/standard/` duplication.** Every `examples/` script uses `standard.Parachute`, which silently bypasses the deployment state machine. Pick one canonical API.

If you do those three refactors, plus fix the three critical correctness bugs, AerisLab becomes a credible v0.3 platform on which the rest of the roadmap (atmosphere, wind, ML, additional joints, deformable canopy, etc.) is straightforward incremental work.

**Overall grade: 7/10.** The grade did not move up from the prior evaluation despite real progress, because the new critical bugs offset the bug fixes and the architectural smells remain. With the §15 checklist completed, the natural next state is 8/10 and a credible v0.3 release.

---

## 2. Ground-Truth Status (Re-verified)

Numbers below come from running tools on `8ef845a` in a fresh `venv` with `pip install -e . pytest pytest-cov pytest-benchmark ruff mypy`.

| Metric | Value | Notes |
|---|---|---|
| Tests collected | **115** | Up from prior baseline |
| Tests passing | **115 / 115** | Was 104/115 in 2026-03-23 doc |
| Test runtime | 36 s | Reasonable |
| Source LOC (counted) | **~5,895** | `find src -name '*.py' \| xargs wc -l` |
| Test LOC | **~3,621** | Source-to-test ratio ≈ 1.6 — healthy |
| Largest module | `models/aerodynamics/parachute_models.py` (1,223 LOC) | ~270 LOC of `example_*` functions live inside this source file |
| Second-largest | `core/simulation.py` (764 LOC) | `World` is still doing too much |
| Third | `core/solver.py` (633 LOC) | Both fixed-step and IVP solvers in one file |
| Coverage (overall) | **75%** | Honest |
| Coverage gaps | `components/standard.py` 0%, `utils/io.py` 0%, `utils/validation.py` 0%, `api/scenario.py` 22% | The user-facing path has the worst coverage |
| Ruff issues | **283** total: 248 `W293` (whitespace-on-blank-line), 13 `W291`, 6 `F401`, 6 `I001`, 5 `UP037`, 2 `B006` (mutable defaults), 1 `B904`, 1 `F541`, 1 `UP035` | Auto-fix removes 238 in one command |
| Mypy errors | **20** in 3 files (`parachute_models.py`, `plotting.py`, `scenario.py`) | None are correctness bugs in practice; all are loose type contracts |
| Version | **0.2.0** in both `pyproject.toml` and `__init__.py` | Mismatch fixed |
| CI status | **Broken** | `pyproject.toml` `dev` extra includes `types-numpy`, which does not exist on PyPI; `pip install -e ".[dev]"` fails. The single CI workflow runs exactly that command. So CI almost certainly hasn't run successfully in some time. See §3 / HIGH-3. |

The `.venv/` referenced in `CLAUDE.md` does not exist; the actual virtualenv is `venv/`. Minor doc drift.

`simulation.csv` (237 bytes) is still committed to the repo root.

`examples/Old/` (10 superseded files) and `scripts/` (10 ad-hoc plotting/animation scripts) are still present.

`examples/scenarios/*.ipynb` are still committed with output cells.

---

## 3. New Critical Bugs Found This Pass

These were not flagged in either prior document. They were surfaced by reading the current code and by running live smoke tests through the public `Scenario` API.

### CRIT-1 — `Scenario.connect()` never propagates the constraint to `World`

**Location:** `src/aerislab/api/scenario.py:100-137` and `src/aerislab/core/simulation.py:363-404`.

**Symptom:** Every parachute simulation built via the `Scenario` API is unconstrained. The payload and parachute fall as **independent bodies**, producing physically wrong results. The user sees the simulation run without errors. The CSV and plots look plausible. The result is silently incorrect.

**Mechanism:**

```python
# Scenario.add_system() — first
def add_system(self, components, name="main_system"):
    sys = System(name=name)
    for comp in components:
        sys.add_component(comp)
    self.systems.append(sys)
    self.current_system = sys
    self.world.add_system(sys)        # registers bodies + (currently empty) constraints
    return self

# World.add_system() — captures snapshot, not a live reference
def add_system(self, system):
    self.systems.append(system)
    for component in system.components:
        self.add_body(component.body)
    for constraint in system.constraints:
        self.add_constraint(constraint)   # ← reads system.constraints AT THIS MOMENT (empty)

# Scenario.connect() — runs AFTER add_system
def connect(self, comp1, comp2, type="tether", length=0.0):
    ...
    constraint = joint.attach(self.current_system.get_bodies())
    self.current_system.add_constraint(constraint)   # ← only added to System, never to World
    return self
```

The constraint is added to `system.constraints` after the `World.add_system()` call has already finished iterating that list. `World.constraints` therefore stays empty, the KKT solver sees zero constraint rows, and the tether is effectively absent.

**Direct verification (executed on this branch):**

```python
sc = (Scenario("confirm")
      .add_system([payload, chute])
      .connect(payload, chute, type="tether", length=10.0))
print(len(sc.current_system.constraints))   # → 1
print(len(sc.world.constraints))            # → 0
```

**Smoke test:** `examples/scenarios/02_parachute_system.py` (a 50 kg payload + 12 m parachute on a 10 m tether, deploying at 1.5 km) terminates with the payload at touchdown velocity ≈ -56 m/s — which is the **terminal velocity of the bare payload**, not of the payload + parachute system. The parachute does its own thing and the payload falls unattached.

**Severity:** Equal to the previously-fixed double-gravity bug. Affects every example in `examples/scenarios/` that uses `connect()` (i.e., 02 and 03), and every user simulation built through the documented public API.

**Fix sketch:** Either make `World.add_system()` keep a live reference and re-scan on use, or make `Scenario.connect()` add the constraint to both `current_system` *and* `self.world`. The first is cleaner and prevents future occurrences of the same bug class. The right architectural answer is to remove `world.constraints` as a separate list entirely and have it be derived from `[c for s in world.systems for c in s.constraints]` at solver-assembly time.

### CRIT-2 — `CSVLogger` silently drops force-category columns

**Location:** `src/aerislab/logger.py:131-216` (header generation and row writing).

**Symptom:** Force breakdown data (drag, parachute drag, constraint forces) is dropped from the CSV unless the corresponding `force_categories` keys exist on the body **at the moment of the very first `log()` call**. Plots that depend on these columns silently show empty or partial data. There is no warning.

**Mechanism:** `_write_header` is called once on the first `log()` invocation and freezes the column set in `self._force_columns`. Subsequent `log()` calls only write values for the categories captured then. New categories appearing later in the simulation are silently discarded.

In the **fixed-step path**, `World.run()` calls `self.logger.log(self)` *before* the first `step()` (line 530-531 in `simulation.py`), so no forces have been applied yet, `force_categories` is empty for every body, and **no force-category columns ever exist in the CSV**. Only the plain `f_x/y/z` and `tau_*` columns get written — i.e., the force breakdown feature is entirely dead in the fixed-step path.

In the **IVP path**, the first log occurs after a full RHS evaluation, so categories from forces that were active at `t=0` get captured. But any force whose `apply()` early-returns at low velocity (e.g., `Drag` skips when `|v| < ε`, line 158 `forces.py`) won't have populated its category yet. In the smoke test of `examples/scenarios/01_simple_drop.py`, the CSV header contains only `cap_body.f_gravity_x/y/z` — no aerodynamic columns, despite the drag force being active for ~30 s.

**Direct verification (smoke test):**

```
$ head -1 output/.../simulation.csv | tr , '\n' | grep '^cap_body.f_'
cap_body.f_x
cap_body.f_y
cap_body.f_z
cap_body.f_gravity_x
cap_body.f_gravity_y
cap_body.f_gravity_z
```

No `f_aerodynamics_*` even though drag was applied throughout.

**Severity:** High. This is a data-correctness bug for analysis (the force-breakdown plot is the most diagnostic plot), not a physics bug. But for a research tool that produces "the plot the user looks at," it is bad.

**Fix sketch:** Two viable options:
- (a) Build the header lazily on a synthetic "warmup" log call after the first step (cheapest fix).
- (b) Switch the on-disk format to one that doesn't require a fixed schema: write each row as a JSON object, or use Parquet with sparse-column writes. The right long-term answer is (b), with Parquet — see §11.

### CRIT-3 — Stateful per-instance derivatives in advanced parachute models corrupt under IVP

**Location:** `src/aerislab/models/aerodynamics/parachute_models.py`:
- `_force_added_mass` (lines 696-792) caches `prev_velocity`, `prev_time`, `prev_added_mass` between `apply()` calls and computes `dV/dt` and `dm_added/dt` by finite-differencing successive calls.
- `_force_mass_flow_balance` (lines 522-585) accumulates `self._state.air_mass_inside += dm_dt * dt` between calls, with a **hardcoded `dt=0.01`** unrelated to the actual integrator step.

**Symptom under IVP:** The IVP solver (Radau, BDF, RK45) evaluates the RHS many times per accepted step — Radau IIA does ≥3 stages per step, plus failed/rejected steps. The per-instance "previous" values are overwritten by every RHS call, including:
- intermediate stages of one accepted step (multiple per step),
- rejected attempts the integrator throws away,
- backwards-time evaluations (some methods do this).

So the finite differences compute garbage. The accumulated `air_mass_inside` integrates the wrong quantity at the wrong rate. The peak-load prediction (the entire reason these advanced models exist) is silently incorrect.

The **fixed-step path** `HybridSolver.step()` calls each force's `apply()` exactly once per accepted step with the right ordering, so these models work as intended there.

**Severity:** High for any quantitative claim. These are exactly the models a researcher would pick because they are "physically complete." A user comparing the six inflation models in `parachute_models.py` under the default Scenario (which uses the IVP solver) is comparing two valid models and four corrupted ones.

**Why it slipped through tests:** `tests/test_parachute_models.py` (24 tests) tests the models with synthetic body states in isolation, calling `compute_force()` directly with monotonically advancing time stamps — i.e., the test is structured exactly the way the fixed-step solver would call it, never the way the IVP solver does.

**Fix sketch:** Move the time-derivative calculation into the integrator's framework. The cleanest fix is to extend the state vector with the auxiliary quantities (`m_added`, `m_inside`) and add their ODEs to the RHS so the integrator handles them coherently. Equivalently, you can mark these models as "fixed-step only" in the model registry and refuse to instantiate them under the IVP solver (defensive but inelegant). The right long-term answer is the state-vector approach — see §5 and §11.

---

## 4. Standing Bugs from Prior Docs (Status Update)

| ID | Prior doc | Status today | Notes |
|---|---|---|---|
| Double gravity in Scenario | `scenario.py:36-38` | **FIXED** | Only one `Gravity` is added (line 36) |
| 11 failing tests | various | **FIXED** | 115/115 pass |
| Version mismatch | `pyproject.toml` vs `__init__.py` | **FIXED** | Both 0.2.0 |
| `Constraint` not real ABC | `constraints.py:21-26` | **STILL TRUE** | Uses `raise NotImplementedError` instead of `@abstractmethod` |
| `World` too large (765 LOC) | `simulation.py` | **STILL TRUE** | 764 LOC; orchestration + logging + plots + termination + energy + output mgmt all in one |
| `components/standard` duplication | `components/{payload,parachute,standard}.py` | **STILL TRUE** | Two `Payload`, two `Parachute`; all `examples/` use `standard.*` |
| IVP re-applies forces for logging | `solver.py:594-631` | **STILL TRUE** | After integration, re-runs full force application + KKT for each logged sample (correctness-fragile and slow) |
| `hasattr(fb, "last_force")` duck-typing | `simulation.py:455, 462` | **STILL TRUE** | The `last_force` attribute is never actually set on any force class — so the breakdown captured here is permanently empty |
| `Scenario.connect` guesses attach points from `body.radius` | `scenario.py:118-123` | **STILL TRUE** | And now compounded by CRIT-1: even when guessed correctly, the constraint is dropped |
| `print()` everywhere | World, Scenario, Parachute | **STILL TRUE** | No `logging` module use anywhere in `src/` |
| No atmosphere model | hardcoded `rho=1.225` | **STILL TRUE** | |
| No wind model | drag uses `body.v` | **STILL TRUE** | |
| `ParachuteDrag.activation_status` mutable across runs | `forces.py:243` | **STILL TRUE** | No `reset()` method |
| Mutable default args (`B006`) | 2 occurrences | **STILL TRUE** | Per ruff stats |
| Whitespace hygiene | ~245 W293 | **STILL TRUE** | 248 W293 + 13 W291; auto-fixable |
| Mypy errors (20) | `parachute_models.py`, `plotting.py`, `scenario.py` | **STILL TRUE** | 17 in `parachute_models.py` (all `Optional[float]` not narrowed before arithmetic on `S0`) |
| Cluttered repo (`examples/Old/`, `scripts/`, `simulation.csv`, committed `.ipynb` outputs) | repo root | **STILL TRUE** | |
| No CI for ruff/mypy/multi-Python | `.github/workflows/test.yml` | **WORSE** | The single CI workflow is broken (HIGH-3 below) |

Plus three new findings I want to escalate from "code smell" to "fix soon":

### HIGH-1 — `standard.Parachute` bypasses `DeploymentState`

Confirms the prior doc, but worth restating because **every `examples/scenarios/` file imports from `standard`**, not from the canonical `components`. So there is no example demonstrating the deployment state machine in action. Tests cover the canonical class in isolation; users see only the bypassed path.

### HIGH-2 — `_check_deployment_complete` makes the state machine 1-tick

`components/parachute.py:201-208` checks whether `_compute_effective_area()` has reached 99% of `self.area`. But `_compute_effective_area()` lines 233-249 returns `self.area` immediately upon transitioning to `DEPLOYING`. So one tick after deployment, the state machine transitions to `DEPLOYED`. The "DEPLOYING" state is an instantaneous fiction. The actual smooth area transition lives in `ParachuteDrag._eval_smooth_area()`'s tanh gate, not in the state machine. The two are not coupled.

### HIGH-3 — CI is silently broken

`.github/workflows/test.yml`:

```yaml
- run: pip install -e ".[dev]"
- run: pytest
```

`pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0", "pytest-cov>=3.0", "pytest-benchmark>=4.0",
    "ruff>=0.1.0", "mypy>=1.0",
    "types-numpy",   # ← does not exist on PyPI
]
```

The first step fails before `pytest` is even invoked. There is no recent green build. Either remove `types-numpy` (numpy now ships type stubs natively since 1.20) or replace with the package that does provide them.

### HIGH-4 — Hardcoded z-up axis assumption

Several places assume the third coordinate is altitude:
- `parachute_models.py:336` `altitude = body.p[2]`
- `components/parachute.py:189` `altitude = float(self.body.p[2])`
- `core/solver.py:559` (touchdown event) `z = y[13 * world.payload_index + 2]`
- `core/simulation.py:469-487` (ground termination) `z_pre = payload.p[2]`
- `core/simulation.py:757` (energy) uses `np.array([0, 0, ground_z])`

This precludes ECEF/inertial frames, tilted-launch scenarios, or any orbital/ballistic problem where "altitude" is a curvilinear function of position. For the parachute use case it's fine. For the long-term aerospace/space scope (§12), it must be abstracted into a `Frame` or `Geodesy` model.

---

## 5. Architectural Assessment

The user has decided to keep the in-house solver — no rebase onto Project Chrono / MuJoCo / MBDyn. I'll flag the trade-offs for completeness, then commit to "Path A" and recommend the best version of it.

### 5.1 Why the build/buy debate matters (briefly)

Mature multibody libraries (PyChrono, MuJoCo MJX, Drake, MBDyn) have:
- richer constraint solvers (Krylov-accelerated, projection methods, complementarity for contact),
- mature joint catalogues (revolute, prismatic, universal, ball, gimbal, gear, screw, …),
- contact and friction models,
- broadphase collision detection,
- existing FEA / flexible-body coupling,
- in some cases, GPU / autodiff backends (MuJoCo MJX, Brax).

For a thesis whose novelty is **FSI-trained ML aerodynamic surrogates coupled into recovery-system simulations**, the multibody substrate is not where the contribution lives. Building it from scratch costs months that could be spent on the actual research.

You've consciously chosen to write your own anyway — that's a defensible decision (full control, pedagogy, no dependency lock-in, the engine itself becomes an artifact that demonstrates the methodology end-to-end). Given that, the rest of this document is calibrated to making the in-house engine **as good as the parts it would be measured against**, where that measurement matters.

### 5.2 The five structural moves the in-house engine needs

Ordered by leverage. None of these is "rewrite from scratch." All are bounded refactors of existing modules.

#### Move 1 — Explicit state-vector boundary between `World` and `Solver`

Today `HybridIVPSolver.integrate()` mutates `world.bodies` *inside* the RHS function (`_unpack_to_world`, `quat_normalize`, `clear_forces`, `apply`, `assemble_system`, `solve_kkt`, ...). This is convenient but carries three problems:

1. **Inability to plug in new integrators cleanly.** Adding RK4-with-substepping, position-Verlet, or an exponential integrator means rewriting the whole RHS construction.
2. **The double-RHS-for-logging anti-pattern.** Because the RHS mutates state, the logger has to re-run the whole pipeline for each logged sample (`solver.py:594-631`). This is both 2× redundant and a correctness-bug nucleus (and it's exactly what corrupts CRIT-3).
3. **No clean entry point for ML surrogates** that need the *full* state to query the network.

The fix is a thin protocol:

```python
class World:
    def state(self) -> NDArray:                    # pack y from all bodies
    def set_state(self, y: NDArray) -> None:       # unpack y back into bodies
    def state_derivative(self, t, y) -> NDArray:   # the RHS — pure function of (t, y)

class Solver(Protocol):
    def integrate(self, world: World, t_end: float) -> Trajectory: ...
```

Once this exists:
- New integrators are ~80 LOC each.
- Logging stores `(t, y)` and replays force application *once* (or zero times if you also store `(t, y, F, λ)` tuples in the RHS — the right answer).
- ML surrogates plug into `state_derivative` like any other force.

This is the single highest-leverage change. It's a 1–2 week refactor and unblocks half the rest of the roadmap.

#### Move 2 — `Model` layer between `Force` and physics

Today `Force` classes (`Drag`, `ParachuteDrag`, `AdvancedParachute`) directly contain inflation logic, drag-coefficient logic, area logic, density logic. This is fine for one model; it doesn't compose.

Introduce model providers:

```
models/
├── atmosphere/        # ISA, exponential, NRLMSISE-00 wrapper later
│   ├── base.py        #   class AtmosphereModel(Protocol): density(p, t), temperature(p, t), pressure(p, t)
│   └── isa.py
├── wind/              # Constant, AltitudeProfile, DrydenTurbulence later
├── aerodynamics/
│   ├── drag_coefficient.py    # Constant, CdTable(M, Re), CdNeural(...)
│   ├── inflation.py           # the 6 existing models, but as pure InflationModel objects
│   └── added_mass.py
├── geodesy/           # FlatEarth, WGS84, ECEF (for §12 space scope)
└── ground/            # NoBounce, Restitution, Friction, Soil
```

A single `AeroForce` then composes them:

```python
AeroForce(
    drag_coefficient=CdConstant(0.85),
    area=InflationModel.continuous(diameter=10.0),
    atmosphere=ISA(),
    wind=Wind.constant([5, 0, 0]),
)
```

Properties unlocked for free:
- altitude-dependent density (no more 20-40 % drag error above sea level),
- wind in five lines (current `body.v` becomes `body.v - wind.velocity(p, t)`),
- ML drag coefficients are a `CdNeural` swap,
- the same `AeroForce` serves a sphere, a wing, a parachute.

#### Move 3 — Resolve the `components` / `standard` duplication

Two parallel APIs is the highest-effort source of confusion. The decision:

- **Keep the `Component` ABC + the `DeploymentState` machine + the `AdvancedParachute` model library.**
- **Promote the convenience constructors from `standard.py` into `@classmethod`s on the canonical classes** (e.g., `Parachute.from_basic(name, mass, diameter, model="knacke", ...)`).
- **Delete `standard.py`.** Migrate the three example scripts in one commit.

Concretely, `components.Parachute.__init__` should accept *either* a `body=` (current) or `mass=`, `diameter=`, `position=` (auto-build), gated through the classmethod. It should also accept `model: InflationModel | str` so `model="knacke"` resolves to `AdvancedParachute(model_type=KNACKE)` internally — bridging the deployment state machine to the rich inflation models.

#### Move 4 — Decompose `World`

`World` (764 LOC) does orchestration + logging + plotting + output dirs + termination + energy + progress printing. Split:

- `World` (~200 LOC): bodies, constraints, systems, time, state.
- `Simulator` / `Runner` (~150 LOC): the `run()` and `integrate_to()` loops, progress, log_interval, termination.
- `OutputManager` (~150 LOC): output dir creation, manifest writing, plot generation.
- `TerminationPolicy` (~60 LOC): callback chain (ground, time, custom). The hardcoded z-axis check becomes one strategy among many.
- `Diagnostics` (~80 LOC): `get_energy`, future `get_constraint_violation`, `get_momentum`.

This is 4 small PRs; the test suite rarely depends on internal `World` shape.

#### Move 5 — Single source of truth for the constraint list

Eliminate the `world.constraints` list entirely. Have `World.constraints_iter()` yield from `[c for s in self.systems for c in s.constraints] + self._free_constraints` at solver-assembly time. This makes CRIT-1 architecturally impossible to reintroduce, and aligns with how `World.bodies` could similarly become derived (`[b for s in systems for b in s.bodies()]`).

### 5.3 Architecture moves I am explicitly NOT recommending

- **Dataclass-frozen state.** Keep the mutable `RigidBody6DOF` with `__slots__`. The simulation-time hot path benefits from in-place updates. Introduce immutability only at the I/O / logging boundary.
- **A general-purpose ECS (Entity-Component-System).** Tempting because Component already has the word, but ECS is overkill at <100 bodies and adds indirection costs.
- **Symbolic codegen (e.g., SymPy → C / Cython).** A common research move; it pays off for thousands of bodies, not for the 2–10 you'll actually have. Skip.
- **Switching to JAX wholesale.** Differentiable simulation is great post-thesis, but a JAX rewrite is a 6-month detour. Keep numpy at the core; build a *parallel* JAX kernel later for the inner loop if needed.
- **Adding pluggable backends now.** Don't define a `Backend` interface "in case." Wait until you actually have a second backend.

---

## 6. Subsystem Deep-Dive

Per-module assessment, **not** line-by-line. Verdicts: ✅ keep / 🟡 refactor in place / 🔴 needs structural change.

### 6.1 `dynamics/body.py` — `RigidBody6DOF` (464 LOC) — ✅ keep, minor improvements

Solid: quaternion (scalar-last to match scipy), `__slots__`, validation at construction, both Euler and exponential-map integrators, kinetic-energy diagnostic, well-documented.

Issues:
- `inv_mass_matrix_world()` is recomputed every assembly call (every RHS in IVP). Cache it; invalidate on `q` change. Combined with caching `inertia_world()`, this is a measurable perf win for IVP.
- `force_categories` accumulator pattern is fine, but the convention for category labels is informal — add a small enum or constant module (`forces.LABEL_GRAVITY`, etc.) to prevent typos that silently produce empty plots.
- Quaternion normalization inside the IVP RHS is non-smooth and bothers stiff solvers. Long-term, treat `||q|| = 1` as a holonomic constraint enforced by KKT (production multibody codes do this). Short-term, this is acceptable.

### 6.2 `dynamics/forces.py` (462 LOC) — 🟡 refactor in place

The `Force` Protocol is good. `Gravity`, `Drag`, `ParachuteDrag`, `Spring` are all reasonable.

Issues:
- The `last_force` attribute that `simulation.py:455` checks for via `hasattr` is **never actually set anywhere in this file**. So the force-breakdown capture in `World.step()` collects nothing. Either implement it on each force, or delete the `hasattr` branch.
- `ParachuteDrag` mixes drag + activation logic + smooth-area interpolation — the things the `Model` layer (Move 2) should pull apart.
- `ParachuteDrag.activation_status` is mutable per-instance state. Add `.reset()`. Also add a class-level `__repr__` that includes activation status.
- `Spring.apply_pair` does not implement the `Force` protocol — it has a different method signature. The handling in `World` (separate `interaction_forces` list) works but is asymmetric. Promote pairwise forces into the same machinery (a `BodyPair` parameter on the Force protocol) for one less special case.

### 6.3 `dynamics/constraints.py` (135 LOC) — 🟡 refactor in place

Distance and PointWeld are correctly formulated (Jacobians match the velocity-level `Ċ = J v_g` form; I checked the `r × d` and `-skew(r)` patterns). The `_geom()` helper recomputes geometry on every `evaluate()` and `jacobian()` — fine for current sizes, would matter at scale.

Issues:
- `Constraint` is not a real ABC. `from abc import ABC, abstractmethod` is one import + four decorators away. Do it.
- No constraint catalogue. For the long-term scope (§12), add at minimum `RevoluteJoint`, `PrismaticJoint`, `BallJoint`, `UniversalJoint`, `Hinge`. Each is ~30 LOC — the KKT machinery doesn't change.

### 6.4 `dynamics/joints.py` (49 LOC) — ✅ keep

Thin facades. Fine. Will grow as the joint catalogue does.

### 6.5 `core/solver.py` (633 LOC) — 🔴 needs structural change

Two integrators in one file is fine; the structural problem is the IVP path's coupling to `World` mutation (Move 1) and the redundant logging path (CRIT-2 amplifier). Specific issues:

- **IVP `integrate()` mutates `world.bodies` inside the RHS.** Move 1 in §5.2 fixes this.
- **`_unpack_to_world` runs a Python `for` loop per RHS call.** Replace with a single `np.frombuffer` / structured-array view or vectorise across bodies.
- **`assemble_system()` is called once per RHS evaluation.** For ML-surrogate calls that take 1–10 ms, this matters. Cache `Minv` (which depends only on `q`, not `v`) and invalidate selectively.
- **`HybridSolver.step()` and the IVP path apply constraint forces "for logging" as a fake `apply_force(...)` call.** This conflates KKT-output Lagrange multipliers with physical force application. Logging should consume the multipliers directly.
- **The chunked `integrate_to()` returns only the *last* chunk's `OdeResult`** (`final_sol`), not a merged trajectory. Document this or fix it. If a downstream consumer treats `final_sol.t` as the full trajectory, they get only the last chunk.
- **Touchdown event is hardcoded to z-axis on `payload_index`.** Generalise via the `TerminationPolicy` of Move 4.
- **No constraint-violation logging.** Baumgarte stabilisation reduces drift but doesn't eliminate it. Logging `||C(q)||` over time is one of the most useful debug aids; add it as a diagnostic column in the CSV (or whatever replaces it).
- **Fixed Baumgarte α/β with no auto-tuning.** Adding `Solver.diagnose()` that reports the equivalent constraint oscillation frequency and damping ratio for a given α, β, and the natural timescales of the system would help users pick values. Worth a short methodology note in the thesis.

### 6.6 `core/simulation.py` — `World` (764 LOC) — 🔴 needs structural change

Already covered: split per Move 4. Add: `World.constraints` should become derived (Move 5).

Specific bugs in addition to those already listed:
- `with_logging()` defaults `auto_save_plots=True` but `Scenario.__init__()` overrides to `False` and "handles plots explicitly" — leading to the awkward `getattr(self, '_show_plots', False)` dance at line 176-178 of `scenario.py`. Pick one ownership model.
- `_auto_save_plots` is a private attribute that `Scenario` reaches into (`self.world._auto_save_plots = ...`). Either make it public or expose a setter.
- The chunked IVP `integrate_to()` checks `sol.status != 0` to detect touchdown but `solve_ivp` returns `status=1` for terminal events *or* `status=2` for max-iteration failure — the chunk loop conflates them. Use the explicit `t_events` check it already does for the print, but for the loop break too.
- Energy accounting only handles uniform `Gravity`. Long-term, factor potential-energy contributions back into individual forces (a `force.potential_energy(body)` method).

### 6.7 `components/` (base, payload, parachute, system, standard) — 🔴 resolve duplication

Already covered in Move 3. Additional notes:
- `Component.update_state(t, dt)` is called with `dt=0.0` from the IVP path (`solver.py:517`). Components that look at `dt` (none today, but inevitable) will see a non-physical zero. Either pass an estimate, or pass `None` and require components to handle that explicitly.
- `Component._state` is a `dict`. Fine for prototyping; fragile for typed code. Promote to a TypedDict (per-component class) when the API stabilises.
- `System` is a thin grouping. With Move 5 (derived `World.constraints`), `System.add_constraint` becomes the single source of truth and many of the bookkeeping awkwardnesses go away.

### 6.8 `models/aerodynamics/parachute_models.py` (1,223 LOC) — 🟡 split + fix

- **270 LOC of `example_*` functions live in this source file** (lines 954-1222). Move them to `examples/parachute_models/` as standalone scripts. They inflate LOC counts, drag down coverage, and ship in the wheel.
- The 6 inflation models are good domain content. After Move 2, they live in `models/aerodynamics/inflation.py` as small `InflationModel` subclasses; `AdvancedParachute` becomes a thin `AeroForce` configurator.
- **CRIT-3** (mass-flow + added-mass under IVP) — already discussed.
- **17 of the 20 mypy errors are here**, mostly because `ParachuteGeometry.S0: float | None = None` is computed in `__post_init__` to be a `float`, but mypy can't see that. Either annotate `S0: float = field(init=False)` and make `S0_input` the constructor arg, or add `assert self.S0 is not None` calls at every use site. The first is cleaner.
- The activation logic (`_check_activation`) reads `body.p[2]` for altitude — couples the model to the z-up assumption. After Move 2, this becomes `atmosphere.altitude(body.p)`.

### 6.9 `api/scenario.py` (180 LOC) — 🔴 fix CRIT-1, then redesign

- CRIT-1 is the headline bug — fix immediately (§3).
- Coverage 22 % — the public, documented API is the least tested surface.
- `connect()` accepts `type="tether"` only; `type="weld"`, `type="soft"` are advertised in joint facades (`joints.py`) but not exposed.
- `connect()` guesses attachment points from `body.radius`. After CRIT-1 is fixed, this is the next-most-important UX issue: silently produces wrong tether geometry for non-spherical bodies. Either require explicit `attach_a=`, `attach_b=`, or remove the `radius` heuristic and require explicit specification.
- `set_initial_state(altitude=...)` shifts all bodies by `Δz`. Reasonable for vertical drops; meaningless for inclined / horizontal launches. Generalise to "shift entire system to a target payload state" or remove.
- Typed `**kwargs: dict[str, object]` is the source of the 2 mypy errors here. Fix by typing as `dict[str, Any]` or by using a `SolverConfig` dataclass.
- The fluent API is nice but only works if the user remembers to call methods in order. Document the order, or move to a builder pattern with a final `.build()` returning an immutable `Simulation` (the runtime separation of "configuring" and "running" is worth the small extra code).

### 6.10 `visualization/plotting.py` (517 LOC) — 🟡 refactor

- Reasonable functions, but the contract with the CSV header is fragile (CRIT-2 silently makes plots blank).
- One mypy error (`plt.cm.tab10`) — known matplotlib stub gap; suppress with `# type: ignore[attr-defined]` and move on.
- Long-term, plot from a stable in-memory data structure (`Trajectory` dataclass), not from CSV columns. The CSV becomes a serialisation choice, not an internal API.

### 6.11 `logger.py` (236 LOC) — 🔴 redesign

- CRIT-2 is the headline bug.
- The fundamental design (CSV with fixed header captured on first row) is wrong for simulations whose output schema can change mid-run. The fix is either:
  - (a) Buffer all rows, infer the union of columns at flush time (works for moderate runs).
  - (b) Switch to a self-describing format (Parquet, JSON-Lines, HDF5).
- The right long-term answer is **Parquet** via `pyarrow`: 5–10× smaller files, columnar (downstream pandas / polars / DuckDB analysis is fast), schema is per-file metadata, sparse / late-binding columns are handled cleanly.
- Keep CSV as an option for human-readability and Excel users.

### 6.12 `utils/` (validation.py, io.py) — 🟡 use or delete

Both at 0 % coverage. Either use `validate_*` functions everywhere (replacing inline validation in `RigidBody6DOF`, etc.), or delete them.

---

## 7. Numerical & Physics Correctness

### 7.1 What is right

These are real strengths and should not be lost during refactoring.

- **KKT formulation with Schur complement** is the appropriate solver for this problem (few constraints relative to DOFs).
- **Baumgarte stabilisation** is implemented correctly (rhs = -(1+β)·J·v - α·C).
- **Condition-number monitoring** with least-squares fallback for singular constraint systems.
- **Quaternion orientation** with both Euler and exponential-map integration.
- **Gyroscopic torque** −ω × (Iω) is correctly added to the generalised force in `assemble_system` (line 113, `solver.py`). Easy to forget; you didn't.
- **Semi-implicit Euler** is symplectic for unconstrained systems → bounded energy drift in the fixed-step path.
- **Constraint Jacobians** for `DistanceConstraint` and `PointWeldConstraint` follow the right pattern (`r × d` for distance, `-skew(r)` for weld).
- **Verification suite** covers free fall, terminal velocity, pendulum period, spring period, energy conservation, angular-momentum conservation, and the **Dzhanibekov effect** — the last is a real test of 3D rotational integrity.

### 7.2 What needs attention

#### Numerical

- **Quaternion normalisation inside the IVP RHS** (`solver.py:508, 548`) is a non-smooth operation. Stiff solvers using Jacobian estimation see spurious gradients near `||q|| = 1 ± ε`. Long-term: add `||q||² = 1` as a holonomic constraint (production multibody codes do this).
- **Mass matrix recomputed every RHS call.** For Radau (3 stages × N bodies × K steps), this is `3KN` rotation-matrix multiplies. Cache, invalidate on `q` change.
- **No constraint-violation logging.** Drift is bounded by Baumgarte but not eliminated. Add `||C(q)||` to the diagnostic stream.
- **Fixed Baumgarte α, β with no diagnostics.** Add a `Solver.diagnose(world)` returning the equivalent oscillator (ω, ζ) so users can pick values.
- **No convergence-rate verification.** Add tests of the form "halve `dt`, verify error halves" for each integrator. Catches order-of-accuracy regressions immediately. 1 day's work.
- **No literature-replication test.** Reproduce one published parachute case (Apollo drogue, T-10, NASA TN data). Even one such test is enormously valuable for credibility and is publishable evidence that the engine produces real-world numbers.

#### Physical fidelity

- **No atmosphere model.** Hardcoded ρ = 1.225. At 5 km altitude actual ρ ≈ 0.74 → 40 % drag-force error. ISA atmosphere is one day's work and unblocks all quantitative validation.
- **No wind model.** True airspeed is `v − v_wind(p, t)`, not `v`. Five lines once Move 2 is in place.
- **No Mach / Reynolds dependency for Cd.** Acceptable for round parachutes at subsonic speeds; binding for any supersonic drogue or any winged decelerator.
- **Added-mass model is applied as a post-hoc force, not in the mass matrix.** Physically the entrained air modifies the effective inertia (M_eff = M + M_a) and M_a should appear in the LHS. As a force, qualitatively similar peak loads but not formally correct. After Move 2 + the §11 "added mass into M" refactor, the FSI surrogate's natural output (M_a(state) + F_aero(state)) plugs in cleanly.
- **Parachute is rigid.** Constant inertia. Fine for trajectory studies; precludes shape change (skirt collapse, breathing, glide). FSI data will give time-varying inertia and force application points; the architecture must accommodate it.

### 7.3 Verification & validation gaps

The verification suite is unusually good for research code. Gaps:

- No convergence-rate tests (above).
- No literature-replication tests (above).
- No regression baselines for the parachute models. If `_force_continuous_inflation` changes next year, you need a reference force-time curve to detect silent regressions. Easy to add: snapshot `(t, F)` for each model under one canonical scenario; compare future runs to the snapshot with a tolerance. Pytest fixture, ~50 LOC.
- No property-based tests (`hypothesis`). For physical invariants (energy bounded, momentum conserved without external forces, constraint distance preserved), property tests give much stronger coverage than example tests.

---

## 8. Tests, V&V, and Reproducibility

### 8.1 Tests

- 115 collected, 115 pass, 36 s runtime — all green.
- Coverage 75 % overall, but the user-facing path (`scenario.py` 22 %, `standard.py` 0 %) is least covered. **CRIT-1 went undetected because no test runs an end-to-end Scenario with a parachute and asserts that the canopy and payload stay tethered.**
- One file (`test_parachute_models.py`) accounts for 24 of the 115 tests but tests in isolation, not under either solver — this is why CRIT-3 went undetected.
- No `@pytest.mark.integration` marker is used despite being declared in `pyproject.toml`. Integration tests would catch CRIT-1 / CRIT-3 the day they're introduced.

### 8.2 V&V hierarchy (per ASME V&V 10-2006 / NASA-STD-7009A)

Currently you have **verification** (analytical solutions). Add:

- **Code verification by manufactured solutions.** Pick a non-trivial dynamics problem, manufacture an exact solution, demonstrate convergence at the expected order. This is the gold standard for solver verification and a publishable methodology contribution.
- **Calculation verification.** Convergence study at every published result. Plotting `||error||` vs `dt` on a log-log plot, with a slope annotation, makes any reviewer happy.
- **Validation.** Comparison to experimental / flight-test data. Required for any ML claim. Even one wind-tunnel or drop-test dataset reproduced is gold.

### 8.3 Reproducibility

This is non-negotiable for a PhD. Currently absent.

- **Provenance metadata.** Every output directory should contain `manifest.json`: git SHA, package version, Python version, scenario config, solver settings, RNG seed, start/end timestamps, warnings list. ~40 LOC. Without this, reproducing your own thesis figures three years from now is anguish.
- **Seeded RNG.** Currently nothing is stochastic; the moment you add Monte Carlo, atmospheric turbulence, or ML dropout, deterministic seeds become essential. Add a `Scenario.set_seed()` now, propagate through everything that draws random numbers.
- **Pinned dependencies.** `pyproject.toml` says `numpy>=1.21`. For thesis figures, generate a `requirements.lock` per chapter (e.g., via `uv pip compile`).
- **Container per published result.** A Docker / Apptainer image tagged at each release. Expensive but pays off in year 4.

---

## 9. Engineering Hygiene

| Item | State | Fix effort |
|---|---|---|
| `ruff check --fix` | 238 issues auto-fixable | 30 seconds |
| Mypy errors (20) | 17 in `parachute_models.py` (Optional narrowing on `S0`) | 1–2 hours |
| `print()` everywhere | 100 % of status messages | 2 hours: `logger = logging.getLogger(__name__)` and pass --verbose |
| `Constraint` not real ABC | 1-line fix | 5 minutes |
| `types-numpy` in `[dev]` | Breaks CI | 30 seconds (delete it) |
| CI matrix | Single Python 3.11, runs only pytest | 1 hour: 3.10/3.11/3.12 + ruff + mypy + coverage |
| Pre-commit hooks | None | 30 minutes: ruff + mypy + nbstripout |
| Repo cleanup | `examples/Old/`, `scripts/`, `simulation.csv`, ipynb outputs | 30 minutes |
| `.gitignore` audit | OK | already 215 lines |
| README | Sparse (1.4 KB) | Half a day with the new architecture |
| `[project.scripts]` CLI | None | 1 day with typer + a YAML loader |
| Docs site | None (folder empty other than evaluations) | 1 week with mkdocs-material |

The **single highest-leverage hygiene fix** is `ruff check --fix` + adding pre-commit. After that, the codebase looks meaningfully cleaner without changing a single line of logic.

---

## 10. Short-Term Roadmap (next ~4 weeks)

Listed in priority order. Items in **bold** are physics-correctness fixes.

### Week 1 — Stop the bleeding

1. **Fix CRIT-1** (`Scenario.connect` constraint not propagated). One-line fix as a hotpatch; the architectural fix is Move 5 in §5.2.
2. **Fix CRIT-2** (logger header timing). Lazy-initialise the header on the first row that has any non-empty `force_categories`, *or* infer columns at flush time.
3. **Fix CRIT-3** (mass-flow + added-mass under IVP). Short-term: refuse to instantiate these two models with the IVP solver and raise a clear error. Long-term: §11 / Move 1.
4. Fix **HIGH-3** (CI broken — delete `types-numpy`).
5. Fix **HIGH-2** (deployment state machine 1-tick collapse). Use the smooth-area gate as the source of truth for `effective_area`.

### Week 2 — Hygiene + tooling

6. `ruff check --fix` everywhere.
7. Fix the 20 mypy errors (`Optional[float]` narrowing in `parachute_models.py`; minor others).
8. Add `.pre-commit-config.yaml` (ruff + mypy + nbstripout).
9. Expand CI matrix: 3.10/3.11/3.12 + ruff + mypy + pytest + coverage upload.
10. Delete `examples/Old/`, `scripts/`, `simulation.csv` from repo root. Add `output/`, `simulation.csv` to `.gitignore` (already there for `output/`?).
11. `nbstripout --install` + clean notebook diffs.
12. Replace every `print(...)` in `src/` with `logger.{info,warning,error}`. Examples can keep `print`.
13. Convert `Constraint` to true ABC.
14. Move the 270 LOC of `example_*` functions out of `parachute_models.py` into `examples/`.

### Week 3 — Atmosphere + duplication

15. Implement `models.atmosphere.ISA` (one day; well-known formulas).
16. Implement `models.wind.Constant` (one day).
17. Refactor `Drag` / `ParachuteDrag` to consume `AtmosphereModel` + `WindModel`. Keep backwards-compatible `rho=` for now (deprecation warning).
18. Resolve `components/standard` duplication (Move 3): promote convenience constructors as classmethods, delete `standard.py`, migrate examples.

### Week 4 — Diagnostics + reproducibility

19. Add constraint-violation logging.
20. Add `manifest.json` provenance writer to `OutputManager` (or to `World` as a transitional home).
21. Add convergence-rate verification tests (one per integrator).
22. Add custom exceptions: `ConstraintSingularError`, `IntegrationFailedError`, `DeploymentError`.

After these four weeks, the codebase is **scientifically usable** above sea level for the first time, and the bug-detection holes that hid CRIT-1 / CRIT-3 are closed.

---

## 11. Medium-Term Refactors (3–6 months)

These are the structural changes that pay off across the rest of the thesis.

### 11.1 The `Model` layer (Move 2)

Already described. Concrete deliverables:

```
models/
├── atmosphere/{base,isa,exponential}.py
├── wind/{base,constant,altitude_profile,dryden}.py
├── aerodynamics/
│   ├── base.py                 # AeroForce composing drag_coefficient + area + atmosphere + wind
│   ├── drag_coefficient.py     # Constant, CdTable(M, Re), CdNeural
│   ├── inflation.py            # Knacke, Continuous, MassFlow, FrenchHuckins, Porosity, AddedMass
│   └── added_mass.py
├── geodesy/
│   ├── flat_earth.py
│   └── wgs84.py
└── ground/
    ├── no_bounce.py
    ├── restitution.py
    └── friction.py
```

This unblocks every downstream feature.

### 11.2 The state-vector boundary (Move 1)

Already described. Concretely:

- Add `World.state(...)`, `World.set_state(...)`, `World.state_derivative(...)`.
- Refactor `HybridIVPSolver` to call only the new interface.
- The IVP logging path stops re-running the pipeline; it stores `(t, y, F, λ)` tuples emitted from inside the RHS.
- Add an `RK4` integrator (one weekend) as the third one — proves the interface generalises.

### 11.3 Decompose `World` (Move 4)

Already described. 4 small PRs.

### 11.4 Output: Parquet + manifest

- Add `ParquetLogger` as the new default; keep `CSVLogger` as a fallback flag.
- Manifest already added in week 4 of §10; expand it to include the full scenario config.
- Add a small `aerislab.io.read_run(path)` helper that returns a `Trajectory` dataclass with positions, velocities, forces (broken out), Lagrange multipliers, constraint violations, and metadata.

### 11.5 First-class configuration

```yaml
# scenario.yaml
name: knacke_main_chute
duration: 60.0
solver:
  preset: stiff
  rtol: 1e-7
  alpha: 5.0
  beta: 1.0
atmosphere: { model: isa, ground_temp: 288.15 }
wind: { model: constant, velocity: [5, 0, 0] }
components:
  - !Payload   { name: capsule, mass: 50, radius: 0.4, position: [0,0,2000] }
  - !Parachute { name: main, mass: 5, diameter: 12, model: continuous_inflation,
                 activation_altitude: 1500, attach_local: [0,0,-0.5] }
connections:
  - { from: capsule, to: main, type: tether, length: 10.0,
      attach_a_local: [0, 0, 0.4], attach_b_local: [0, 0, -0.5] }
output: { format: parquet, manifest: true }
seed: 42
```

`pydantic v2` for typed schemas + a `Scenario.from_yaml(path)` classmethod. This is the bedrock for everything in §13 (Monte Carlo, parameter sweeps, ML training scenarios).

### 11.6 Joint catalogue

- `RevoluteJoint` (1 DOF — for hinges, control surfaces).
- `PrismaticJoint` (1 DOF — sliders).
- `UniversalJoint` (2 DOF).
- `BallJoint` (3 DOF rotational).
- `Hinge` (revolute with limits).

Each is ~30 LOC; the KKT solver doesn't change. Tests come for free from existing constraint test patterns.

### 11.7 Performance baseline

- `pytest-benchmark` is already in dev deps but unused. Add benchmarks for:
  - 2-body parachute system, 60 s, IVP Radau → steps/s.
  - 10-body multibody chain, fixed-step → ms/step.
  - 1000-step KKT solve isolated → ms/solve.
- Profile with `py-spy` and save the SVG flame graph in `docs/perf/`. Re-run after each major refactor.

### 11.8 CLI

```bash
$ aerislab run scenario.yaml
$ aerislab plot output/run_xyz/
$ aerislab compare output/run_a/ output/run_b/
```

`typer` + the YAML loader from 11.5. ~1 day of work.

---

## 12. Long-Term Capabilities & Aerospace/Space Expansion

This is the answer to your "what to add, what could be beneficial, what about more aerospace problems (space things etc)" question. Some of these are clear next steps; others are speculative — flagged accordingly.

### 12.1 Recovery-system enrichment (clear, near-term)

Things any serious recovery-system tool needs that AerisLab doesn't have:

- **Multi-stage parachutes.** Drogue → main, with sequencer logic (altitude or time triggers).
- **Reefing.** Multi-stage area expansion of a single canopy. Already implied as a "future extension" in `Parachute` docstrings.
- **Suspension-line network.** Today the parachute is one rigid body connected to the payload by a single tether. Real systems have N suspension lines from a confluence point. Model as a tree of point-mass lumps with stiff Spring + damping, or as a small KKT subsystem. The latter gives correct line-tension distributions.
- **Risers and bridles.** Same machinery as suspension lines.
- **Disreefing dynamics.** Time-varying reefing-line length.
- **Opening shock detail.** Time-resolved load profile with peak, plateau, settling.
- **Canopy oscillation / pendulum modes.** Treat canopy + payload as a coupled pendulum; FSI surrogate provides the moment that excites/damps the swing.
- **Cluster parachute systems.** N canopies on one payload. Topologically straightforward once joint catalogue exists; aerodynamically nontrivial (interference effects).
- **Glide / steerable parachutes.** Asymmetric Cd/Cl from the canopy → lateral force component. ML surrogate is the natural home for this.

### 12.2 Beyond rigid bodies (medium-term, choose carefully)

- **Reduced-order modal canopy.** Keep `RigidBody6DOF` for gross motion; add modal coordinates for canopy deformation (a few modes). Modal forces come from FSI training data — same pipeline as the ML aerodynamic surrogate.
- **Particle-system suspension lines.** Stiff Spring + damping tree, optionally with axial elasticity (Hooke). Solve at a sub-step inside the integrator.
- **Lumped-parameter cloth / membrane.** Mass-spring-damper grid for low-fidelity canopy shape; expensive but valuable for FSI surrogate validation.

### 12.3 Control & actuation (medium-term)

- **`Controller` Protocol.** Consumes `World` state, writes torques/forces:
  ```python
  class Controller(Protocol):
      def update(self, t: float, dt: float, world: World) -> None: ...
  ```
- **PID, LQR, MPC** as concrete controllers.
- **RL policy adapter.** Wraps a trained policy (`stable-baselines3`, `RLLib`) as a Controller. Useful for steerable-parachute guidance research.

### 12.4 Aerospace expansion: launch / ascent / atmospheric flight (longer-term)

- **Variable-mass rocket.** `RigidBody6DOF` with `mass(t)` and a `MassFlow` model. Adds a (ṁ × v_exhaust) thrust force and a (ṁ) inertia derivative.
- **Thrust-vector control.** Force application point as a function of gimbal angle.
- **Aerodynamic model expansion.** Lift, side force, pitching/yawing/rolling moments. Coefficients as `Cl(α, β, M, Re)`, `Cm(α, ...)` etc. Tabular today, ML-derived later.
- **Autopilot integration.** Same `Controller` Protocol.

### 12.5 Space dynamics (your "very hypothetical" — let me give it more credit than that)

The space scope is reachable from the current architecture with bounded effort, *provided* Move 1 (state-vector boundary) and the `Frame`/`Geodesy` abstractions exist. The hard parts:

- **Gravitational model.** `Gravity` becomes a `GravityField` model:
  - `UniformGravity(g)` (current).
  - `PointMassGravity(GM, center)` (Keplerian two-body).
  - `J2Gravity` (Earth oblateness; one term, sufficient for most LEO).
  - `SphericalHarmonicGravity(coeffs)` (high-fidelity Earth; standard EGM coefficients freely available).
  - `NBodyGravity(perturbers)` (Sun, Moon, planets).
- **Reference frames and frame transforms.** ECI ↔ ECEF ↔ topocentric. Time systems (UT1, UTC, TAI, TT). Use `astropy` (heavy) or implement the minimum (ITRF/ICRF rotation + UT1-UTC offset) yourself.
- **Atmospheric drag at altitude.** NRLMSISE-00 wrapper or equivalent (`pymsis` exists). Drives orbital decay simulations.
- **Solar radiation pressure.** Tiny but mission-relevant for high area-to-mass.
- **Attitude dynamics with reaction wheels / magnetorquers.** A `RigidBody6DOF` with internal angular-momentum exchange — requires generalising `inertia_world` to `inertia_world + Σ Iw`.
- **Orbit propagation as a special case.** With `PointMassGravity` and no atmosphere, the existing IVP solver (Radau or DOP853) is already a valid orbit propagator. Validate against `poliastro` or `Skyfield` for a known trajectory.
- **Trajectory targeting / Lambert / impulsive maneuvers.** Once orbit propagation works.
- **Deployable structures, solar panels, antennae.** Joint catalogue + flexible-body extension (12.2) covers most cases.
- **Atmospheric entry.** Combines launch-phase aero (12.4), variable atmosphere, hypersonic Cd (Mach + Knudsen), and recovery-system deployment. Natural endpoint of all three threads.

The cleanest packaging is to **keep `aerislab` core as the constrained-multibody + force/model framework**, and add **`aerislab.aerospace.atmospheric`** and **`aerislab.aerospace.orbital`** as feature packages that compose on top. Don't pollute the core with orbital concepts; let them live in their own namespace and reuse the state-vector / model machinery.

What you should **not** try to do in AerisLab: full mission planning (use GMAT), high-fidelity CFD (you'll be running CFD externally and ingesting the data into the ML surrogate), or attitude determination from sensors (write a separate package, AerisLab is the dynamics, not the GNC stack).

### 12.6 Multiphysics couplings (long-term)

- **One-way coupling: CFD → multibody.** Already covered by the ML surrogate path (§13).
- **Co-simulation: external solver in lockstep.** Once Move 1 is in place, the `World.state_derivative(t, y)` boundary is the single clean call site. Couple to FEniCS / OpenFOAM / your FSI solver with a small TCP / FMI-2 adapter.
- **Two-way coupling.** Substantially harder; defer beyond thesis.

### 12.7 Differentiable simulation (post-thesis)

If gradient-based design optimisation becomes interesting:

- A separate JAX-backed kernel for the inner loop (`assemble_system`, `solve_kkt`, `state_derivative`) — keep numpy as the default.
- `torch.func` or `jax.grad` over the JAX backend gives free gradients of trajectory metrics (terminal velocity, opening-shock peak, miss distance) with respect to design parameters (canopy diameter, tether length, deployment altitude).
- Useful for inverse design and Bayesian optimisation. Genuinely publishable. Beyond thesis scope.

---

## 13. ML / FSI Integration Plan (Refined)

Building on the prior doc's plan, with corrections after the architectural moves above.

### 13.1 The dependency chain

The ML surrogate work depends on, in order:

1. **Move 1** (state-vector boundary) — so the surrogate is a normal force, not a special case.
2. **Move 2** (Model layer) — so `CdNeural` and `AreaNeural` are drop-in replacements for analytical models.
3. **Atmosphere + wind models** — without them, the surrogate is fitting noise from a wrong baseline.
4. **Validation case** (literature replication, §7.3 / §10) — so surrogate errors are measurable against ground truth.

Skip any step and the ML work either doesn't compose with the rest of the engine (1, 2) or produces results that no reviewer will trust (3, 4).

### 13.2 Surrogate API

```python
class AeroSurrogate(Protocol):
    """Maps state → aerodynamic quantities, body frame."""
    def predict(self, state: AeroState) -> AeroOutput: ...

@dataclass(frozen=True)
class AeroState:
    v_air_body: np.ndarray            # (3,) airspeed in body frame
    rho: float
    T: float                          # temperature
    deployment_phase: float           # 0..1
    inflation_progress: float         # 0..1
    history: np.ndarray | None        # for sequence models

@dataclass(frozen=True)
class AeroOutput:
    F_body: np.ndarray                # (3,) aerodynamic force
    M_body: np.ndarray                # (3,) moment
    M_added_body: np.ndarray | None   # (3,3) added-mass tensor
    Cd_eff: float                     # diagnostic
    confidence: float | None          # OOD / uncertainty
```

Three reference implementations:

1. **`AnalyticalSurrogate(model=AdvancedParachute(...))`** — wraps existing analytical models in the protocol. Lets you exercise the new code path with the existing test suite.
2. **`OnnxSurrogate(model_path="parachute.onnx")`** — production. ONNX Runtime is 5–10× faster than `torch` for inference, no torch import.
3. **`TorchSurrogate(model=...)`** — development. Allows gradient propagation if you eventually want differentiable simulation.

### 13.3 Engineering pitfalls (in order of "I have personally watched people fall into this")

- **Frame inconsistency.** FSI runs in canopy body frame; the surrogate must output body-frame quantities. Apply `body.rotation_world() @ F_body` only at the boundary. Train rotation-equivariant or add explicit augmentations.
- **Wrong non-dimensionalisation.** Train on `(v/v_ref, ρ/ρ₀, A/A₀)`, never on raw SI. Otherwise the network does not generalise across scales/altitudes.
- **History dependency.** Parachute opening shock has memory. A pure state-conditioned MLP will be wrong during inflation. Either include `(t − t_deploy)` and `dA/dt` as features, or use an LSTM/GRU/Transformer.
- **OOD silent extrapolation.** A neural net asked for `(v, ρ)` outside training envelope will *not* error — it returns garbage smoothly. Train an OOD detector (Mahalanobis distance, density estimator, calibrated-uncertainty ensemble) and either warn-and-fall-back-to-analytical or refuse to evaluate.
- **Per-RHS-call inference cost.** Naive PyTorch from inside a Radau RHS is 1–10 ms × 3 stages × N steps × N bodies. For a 60 s sim that can be 100–1000 s of pure inference time. Mitigations:
  - Batch in time when fixed-step.
  - ONNX Runtime instead of torch.
  - `torch.compile` / TensorRT.
  - Cache by quantised state (e.g., velocity rounded to nearest 0.1 m/s) for the slowly-varying parts.
- **Random sample splits.** Hold out **trajectories**, not random samples. A network that interpolates between t=1.00 s and t=1.05 s of the same FSI run will look great on paper and fail in production.
- **Pointwise vs trajectory metrics.** Report both. The end-user metric is descent prediction, not pointwise drag.

### 13.4 Validation & uncertainty

- **Deep ensembles** (5–10 networks with different seeds): cheap, well-calibrated, easy to integrate. Output uncertainty bands on terminal velocity, opening time, peak load.
- **Calibration plots** at every milestone.
- **Comparison plots** vs. each analytical inflation model on held-out FSI cases.
- **Worst-case analysis.** What's the maximum trajectory error across the test set? This single number matters more for safety claims than mean error.

### 13.5 Training infrastructure (`aerislab.ml/`)

```
aerislab/ml/
├── data/
│   ├── fsi_loader.py     # Read FSI sim outputs (CFD-format-agnostic via a thin adapter)
│   ├── featurise.py      # state → (X, y) tensors with non-dim
│   └── augment.py        # symmetry augmentation
├── models/
│   ├── mlp.py
│   ├── lstm.py
│   └── transformer.py
├── train.py              # Hydra/Typer config-driven
├── evaluate.py           # held-out FSI cases, plots vs analytical
├── infer.py              # ONNX/TorchScript export
└── ood.py                # OOD detector
```

Use **Hydra** or **Typer** for config (matches the YAML scenario approach in §11.5). Use **MLflow** or **Weights & Biases** for experiment tracking — important for the PhD because every run becomes reproducible and citable.

### 13.6 Why this is publishable

Coupling FSI-derived ML aerodynamic surrogates to constrained multibody recovery-system dynamics in an open, scriptable Python package, validated end-to-end against drop-test data, is a thesis-grade contribution. AIAA, *Aerospace Science & Technology*, *Journal of Computational Physics*, JOSS for the software side. Don't underweight the JOSS paper — a citable DOI for the software is what gets you cited by people running your tool, which is the audience that matters most for a research code's afterlife.

---

## 14. Risks, Trade-offs, and What to Avoid

### 14.1 Risks

- **Refactor budget.** The §10 + §11 work is roughly 4–5 months part-time. Sequence ruthlessly: the items that *unblock scientific contributions* (Move 1, Model layer, atmosphere) must precede anything that just makes the code prettier (whitespace, docstring polish).
- **Premature generalisation.** Don't build a "general physics engine" before there's a clear second use case. Build the second physics (e.g., a winged decelerator or a launch vehicle) first; let the abstractions emerge from real demand. The §12 list is a menu, not a backlog — pick what your thesis actually needs.
- **ML before physics is wrong.** Do not start training surrogates until atmosphere + wind + a validation case are in place. Without them, you cannot interpret the surrogate's errors. Pure ML on a wrong baseline is the most common failure mode in CFD-ML papers.
- **Compatibility breakage.** The Move 1 / 2 / 3 refactors will break user code. There is no user code yet (except your own examples). Do them now, before publication. After v1.0, semver discipline becomes necessary.
- **Test holes.** CRIT-1 and CRIT-3 went undetected for at least two evaluation cycles because nothing exercised the user-facing path under realistic conditions. The single highest-leverage testing fix is a small set of **end-to-end smoke tests** that run an example scenario and assert physical sanity (touchdown velocity in expected range, parachute reduces velocity, energy doesn't explode). 5–10 such tests catch the entire bug class.

### 14.2 Things to actively avoid

- **Don't switch the core to JAX wholesale "for autodiff."** It's a 6-month detour. Build a parallel JAX kernel later if you actually need gradients.
- **Don't add a plugin/backend interface speculatively.** It's a tax until you have a second backend.
- **Don't reach for `numba` / `cython` to make Python faster** unless profiling has identified a specific hot loop. The current bottleneck is *redundant work*, not slow Python — caching the inverse mass matrix (Move 1 follow-on) will give a 3–5× speedup with no language change.
- **Don't build a GUI / web UI.** Leave that to downstream tools (Streamlit dashboards on top of the Parquet output, when needed).
- **Don't try to support every parachute geometry / material in the analytical models.** ML is the right tool for that distribution; analytical models cover canonical cases (round, ribbon, ringsail) and you don't need 20 of them.
- **Don't write your own CFD.** It's the rest of someone else's PhD.

### 14.3 What is genuinely worth keeping if you ever feel like rewriting from scratch

- The `Component` HAS-A `RigidBody6DOF` composition pattern.
- The verification suite (especially Dzhanibekov, energy conservation, pendulum period).
- The 6 inflation models as a domain-knowledge artifact.
- The KKT formulation with Schur complement and Baumgarte stabilisation.
- The Scenario fluent API design (after CRIT-1 is fixed and `connect()` is generalised).
- The dual fixed-step + IVP solver design.
- The output-directory organisation (logs/ + plots/ + manifest/).

The rest is replaceable.

---

## 15. Action Checklist

Compact, copy-pasteable. Roughly ordered.

### Critical (this week)

- [ ] **Fix CRIT-1**: `Scenario.connect` propagates constraint to `World`.
- [ ] **Fix CRIT-2**: `CSVLogger` header captures all force categories that appear.
- [ ] **Fix CRIT-3**: Either disable mass-flow + added-mass models under IVP or move their auxiliary state into the ODE state vector.
- [ ] **Fix HIGH-3**: Remove `types-numpy` from `pyproject.toml` `[dev]` extras; verify CI is green.
- [ ] **Fix HIGH-2**: Make `_compute_effective_area` use the smooth-area gate as the truth.

### Hygiene (next 2 weeks)

- [ ] `ruff check --fix`.
- [ ] Fix 20 mypy errors.
- [ ] Add `.pre-commit-config.yaml` (ruff + mypy + nbstripout).
- [ ] Expand CI matrix to 3.10 / 3.11 / 3.12 + ruff + mypy + coverage upload.
- [ ] Replace every `print(...)` in `src/` with `logging.getLogger(__name__).{info,warning,error}`.
- [ ] Convert `Constraint` to `@abstractmethod` ABC.
- [ ] Delete `examples/Old/`, `scripts/`, `simulation.csv`.
- [ ] `nbstripout --install` and clean notebook diffs.
- [ ] Move 270 LOC of `example_*` from `parachute_models.py` into `examples/`.
- [ ] Add 5–10 end-to-end smoke tests that run `examples/scenarios/*.py` and assert physical sanity.

### Physics (next month)

- [ ] Implement `models.atmosphere.ISA`.
- [ ] Implement `models.wind.Constant`.
- [ ] Refactor `Drag` / `ParachuteDrag` to consume `AtmosphereModel` / `WindModel`.
- [ ] Add convergence-rate verification tests (one per integrator).
- [ ] Add constraint-violation logging.
- [ ] Reproduce one published parachute case (e.g., Apollo drogue) for literature validation.
- [ ] Add `manifest.json` provenance to every output dir.

### Architecture (next 3–6 months)

- [ ] Introduce `World.state` / `set_state` / `state_derivative` (Move 1).
- [ ] Refactor `HybridIVPSolver` to use the state-vector interface; eliminate the double-RHS-for-logging path.
- [ ] Cache inverse mass matrix; invalidate on quaternion change.
- [ ] Make added mass enter the LHS (`M + M_a`), not as a force.
- [ ] Extract `OutputManager`, `Simulator`, `TerminationPolicy`, `Diagnostics` from `World`.
- [ ] Make `World.constraints` derived from systems (Move 5) — kills CRIT-1's bug class.
- [ ] Build the `Model` layer (atmosphere/wind/aero/geodesy/ground).
- [ ] Resolve `components/standard` duplication; promote convenience constructors as classmethods (Move 3).
- [ ] Add joint catalogue (revolute, prismatic, ball, universal, hinge).
- [ ] Add Parquet output + `Trajectory` reader.
- [ ] Add YAML scenario configuration (`Scenario.from_yaml`).
- [ ] Add CLI (`aerislab run`, `aerislab plot`, `aerislab compare`).
- [ ] Add `pytest-benchmark` baselines for IVP and fixed-step.
- [ ] Define `AeroSurrogate` Protocol; implement `AnalyticalSurrogate` first.

### Long-term (towards thesis)

- [ ] Implement `OnnxSurrogate` and `TorchSurrogate`.
- [ ] Build FSI → training data pipeline.
- [ ] Add OOD detector for surrogates.
- [ ] Add deep ensembles for uncertainty.
- [ ] Validate surrogates against held-out FSI cases.
- [ ] Reduced-order modal canopy.
- [ ] Suspension-line network.
- [ ] Multi-stage parachute / reefing.
- [ ] Add `Controller` Protocol; implement PID and an RL adapter.
- [ ] Atmospheric model upgrades: NRLMSISE-00 wrapper.
- [ ] Optional: orbital extension (`aerislab.aerospace.orbital`) with PointMass / J2 / SH gravity, ECI/ECEF frames.
- [ ] Sphinx or mkdocs-material site with theory + tutorials + API.
- [ ] JOSS / Software Impacts paper.
- [ ] Zenodo DOI per release.
- [ ] Methodology paper (FSI-ML × multibody recovery dynamics).

---

## Appendix A — Tooling commands used for ground-truth verification

For reproducibility of this evaluation:

```bash
# From repo root, after installing into venv/
venv/bin/pip install -e . pytest pytest-cov pytest-benchmark ruff mypy

venv/bin/pytest --tb=short                          # 115 / 115 pass, 36 s
venv/bin/pytest --cov=aerislab --cov-report=term-missing -q
venv/bin/ruff check src/ --statistics               # 283 issues
venv/bin/mypy src/aerislab/                         # 20 errors

# CRIT-1 confirmation
venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from aerislab.api.scenario import Scenario
from aerislab.components.standard import Payload, Parachute
p = Payload(name='cap', mass=50.0, radius=0.4, position=[0,0,2000])
c = Parachute(name='main', mass=5.0, diameter=12.0, model='knacke',
              activation_altitude=1500, position=[0,0,2000.5])
sc = (Scenario(name='confirm').add_system([p, c])
      .connect(p, c, type='tether', length=10.0))
print('System constraints:', len(sc.current_system.constraints))
print('World constraints: ', len(sc.world.constraints))
"
# Expected output:
# System constraints: 1
# World constraints:  0
```

## Appendix B — File-by-file quick verdict

| File | LOC | Coverage | Verdict |
|---|---|---|---|
| `dynamics/body.py` | 463 | 83 % | ✅ keep, cache inertia/mass matrix |
| `dynamics/forces.py` | 462 | 89 % | 🟡 split into Force + Model layers |
| `dynamics/constraints.py` | 135 | 97 % | 🟡 promote to real ABC, expand catalogue |
| `dynamics/joints.py` | 49 | 57 % | ✅ keep, will grow |
| `core/solver.py` | 633 | 79 % | 🔴 state-vector boundary, fix IVP logging path |
| `core/simulation.py` | 764 | 92 % | 🔴 decompose; fix CRIT-1 by deriving constraints |
| `components/base.py` | 170 | 91 % | ✅ keep |
| `components/payload.py` | 116 | 84 % | 🟡 absorb `standard.Payload` as classmethod |
| `components/parachute.py` | 269 | 83 % | 🔴 fix HIGH-2; absorb `standard.Parachute`; accept `InflationModel` |
| `components/standard.py` | 122 | 0 % | 🔴 delete after consolidation |
| `components/system.py` | 208 | 71 % | ✅ keep; becomes the real owner of constraints (Move 5) |
| `models/aerodynamics/parachute_models.py` | 1223 | 76 % | 🟡 split: examples → `examples/`; models → `models/aerodynamics/inflation.py` |
| `api/scenario.py` | 180 | 22 % | 🔴 fix CRIT-1 + redesign `connect`; raise coverage |
| `visualization/plotting.py` | 517 | 70 % | 🟡 plot from `Trajectory` dataclass, not raw CSV |
| `logger.py` | 236 | 82 % | 🔴 fix CRIT-2; add Parquet backend |
| `utils/validation.py` | 134 | 0 % | 🟡 use it everywhere or delete |
| `utils/io.py` | 30 | 0 % | 🟡 use it or delete |

---

*Evaluation prepared for: Štěpán Kaspar (kaspar.stepan.cz@gmail.com), VUT Brno PhD project.*
*Reviewer: independent re-evaluation against `ClaudeHelp` @ `8ef845a`, 2026-05-13.*
