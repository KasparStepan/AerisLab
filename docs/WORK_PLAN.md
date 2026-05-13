# AerisLab — Work Plan

**Companion to:** `docs/AERISLAB_REVIEW.md`
**Owner:** Štěpán Kaspar
**Created:** 2026-05-13
**Branch reviewed:** `ClaudeHelp` @ `8ef845a`

This file is the executable to-do list derived from the review. It is sequenced — earlier work unblocks later work — and grouped into phases tied to suggested release versions. Each task lists the **files touched**, **effort estimate**, **dependencies**, and **acceptance criteria** so you can pick up the next item cold.

---

## How to Use This Document

- **Work top-to-bottom.** Phases are ordered so that completing Phase N makes Phase N+1 straightforward. Within a phase, dependencies are flagged but most items are independent.
- **One task per branch.** Branch name = task ID (e.g., `crit-1-scenario-connect`). Merge via PR even though you're solo — the diff review is for *you* a week later.
- **Mark done in this file.** Change `[ ]` to `[x]` and (optionally) add the commit SHA in `<…>`. Keep the file under version control so progress is auditable.
- **Acceptance criteria are non-negotiable.** Don't merge a task whose acceptance check doesn't pass. If the criterion turns out to be wrong, update the criterion in a separate commit before doing the work.
- **Verification commands** assume you're in the repo root with `venv/` active (or use `venv/bin/python`, `venv/bin/pytest`, etc.).

### Branching strategy (suggested)

```
main                      ← only green, releasable
├── ClaudeHelp            ← current dev branch, keep going here
│   ├── crit-1-…          ← short-lived, one task each
│   ├── hygiene-ruff-fix
│   └── …
└── (release tags: v0.2.1, v0.3.0, …)
```

Tag a release at the end of each phase. Use semver loosely: patch for bugfixes (`0.2.0 → 0.2.1`), minor for breaking changes you're free to make pre-1.0 (`0.2.1 → 0.3.0`), major reserved for v1.0.

### Effort estimate scale

- **XS** = ≤1 hour
- **S** = a few hours / half a day
- **M** = 1–2 days
- **L** = ~1 week
- **XL** = >1 week

These assume part-time PhD pace. Halve if you sit down for an uninterrupted day.

### Phase summary

| Phase | Target version | Theme | Calendar | Effort |
|---|---|---|---|---|
| 0 | v0.2.1 | Hotfix critical bugs | 1 week | ~3 days |
| 1 | v0.3.0 | Hygiene + duplication + atmosphere | 3–4 weeks | ~2 weeks |
| 2 | v0.4.0 | Architecture: state-vector, Model layer, decompose World | 2–3 months | ~6 weeks |
| 3 | v0.5.0 | Capabilities: joints, Parquet, YAML, CLI | 1–2 months | ~3 weeks |
| 4 | v0.6.0 | ML/FSI surrogate framework | 3–6 months | ~10 weeks |
| 5 | v0.7+ | Aerospace expansion + thesis features | open | open |

---

# Phase 0 — Hotfix (target: v0.2.1)

**Goal:** stop the bleeding. The three CRIT bugs make the public API silently wrong; the broken CI hides regressions. None of these requires architectural change.

**Exit criteria:** all CRIT-* and HIGH-* bugs from the review are closed, CI is green, no new tests fail.

---

### P0-T1 — Fix CRIT-1: `Scenario.connect` constraint propagation

- **Why:** `Scenario.connect()` adds the constraint to `current_system.constraints` after `world.add_system()` has already snapshotted that list. The KKT solver sees zero constraints. Every parachute scenario through `Scenario` runs unconstrained.
- **Files:**
  - `src/aerislab/api/scenario.py` (`connect`, lines 100–137)
  - **OR** `src/aerislab/core/simulation.py` (`add_system`, `add_constraint`)
- **Effort:** XS (one-line hotpatch) — the architectural fix (Move 5) lands in Phase 2.
- **Depends on:** —
- **Implementation note:** simplest hotpatch — at the end of `Scenario.connect`, also call `self.world.add_constraint(constraint)`. Add a comment pointing to P2-T11 (the structural fix).
- **Acceptance criteria:**
  - `len(scenario.world.constraints) == len(scenario.current_system.constraints)` after every `connect()` call.
  - Smoke test of `examples/scenarios/02_parachute_system.py` shows touchdown velocity `< -10 m/s` (reduced by parachute) instead of the bare-payload `~-56 m/s`.
- **Verification:**
  ```bash
  venv/bin/python -c "
  import sys; sys.path.insert(0, 'src')
  from aerislab.api.scenario import Scenario
  from aerislab.components.standard import Payload, Parachute
  p = Payload(name='cap', mass=50.0, radius=0.4, position=[0,0,2000])
  c = Parachute(name='main', mass=5.0, diameter=12.0, model='knacke',
                activation_altitude=1500, position=[0,0,2000.5])
  sc = (Scenario(name='check').add_system([p, c])
        .connect(p, c, type='tether', length=10.0))
  assert len(sc.world.constraints) == 1, 'CRIT-1 not fixed'
  print('OK')
  "
  ```
- [ ] Done — commit `<sha>`

---

### P0-T2 — Fix CRIT-2: `CSVLogger` header captures all force categories

- **Why:** Header is frozen on first `log()`; in the fixed-step path that's before any forces are applied (empty header), in the IVP path it captures only categories active at `t=0`. Force-breakdown columns are silently absent.
- **Files:** `src/aerislab/logger.py`
- **Effort:** S
- **Depends on:** —
- **Implementation note:** Two acceptable approaches:
  - (a) Buffer all rows in memory; resolve the union of `force_categories` keys at flush time; write header then. Easy, correct, but uses memory.
  - (b) Detect on every `log()` whether new categories appeared; if so, raise a `LoggerSchemaChangedError` and tell the user to re-run with `buffer_size=large_enough`. Defensive but ugly.
  - (c) Switch to JSON-Lines (one JSON per row) — schema-free, slower to read but trivially correct.
  - **Recommended:** (a) for now; the Parquet migration in P3-T6 makes this a transient fix.
- **Acceptance criteria:**
  - After running `examples/scenarios/01_simple_drop.py`, `head -1 .../simulation.csv` includes `cap_body.f_aerodynamics_x/y/z`.
  - All existing tests still pass.
- **Verification:**
  ```bash
  venv/bin/python examples/scenarios/01_simple_drop.py >/dev/null 2>&1
  head -1 output/01_simple_drop_*/logs/simulation.csv | tr , '\n' | grep -q 'aerodynamics' && echo OK
  ```
- [ ] Done — commit `<sha>`

---

### P0-T3 — Fix CRIT-3: mass-flow + added-mass parachute models under IVP

- **Why:** These two models cache `prev_velocity` / `prev_time` between `apply()` calls and accumulate state with a hardcoded `dt=0.01`. Under Radau (≥3 RHS evals per step + rejected steps) those values are corrupted. Silently produce wrong peak loads.
- **Files:** `src/aerislab/models/aerodynamics/parachute_models.py` (`_force_mass_flow_balance`, `_force_added_mass`)
- **Effort:** S (defensive guard) + L (proper fix, deferred to P2)
- **Depends on:** —
- **Implementation note (defensive guard, do this now):** Add a class-level flag `IVP_SAFE: bool = False` on these two models; check it in `HybridIVPSolver.integrate()` before the run starts and raise `ValueError("Model X is not safe under IVP solver. Use HybridSolver, or wait for v0.4.")`. Keep the models usable in fixed-step.
- **Implementation note (proper fix, P2):** Move the auxiliary state (`m_added`, `m_inside`) into the ODE state vector so the integrator handles it.
- **Acceptance criteria:**
  - Trying to use `model="mass_flow_balance"` or `model="added_mass"` with `Scenario` (which uses IVP) raises a clear error pointing to the docs.
  - Both models still work and pass their existing tests under `HybridSolver`.
- [ ] Done — commit `<sha>`

---

### P0-T4 — Fix HIGH-3: CI broken (`types-numpy` doesn't exist)

- **Why:** `pip install -e ".[dev]"` fails because `types-numpy` is not on PyPI. CI hasn't run successfully in some time. Remove the offending line; numpy ships type stubs natively.
- **Files:** `pyproject.toml`
- **Effort:** XS
- **Depends on:** —
- **Acceptance criteria:**
  - `pip install -e ".[dev]"` succeeds in a clean venv.
  - GitHub Actions workflow runs to completion (push to a branch and check).
- **Verification:**
  ```bash
  python -m venv /tmp/ci-check && /tmp/ci-check/bin/pip install -e ".[dev]" 2>&1 | tail -3
  ```
- [ ] Done — commit `<sha>`

---

### P0-T5 — Fix HIGH-2: deployment state machine 1-tick collapse

- **Why:** `_compute_effective_area()` returns `self.area` immediately after deployment, so `_check_deployment_complete()` transitions DEPLOYING → DEPLOYED on the next tick. The smooth area transition lives in `ParachuteDrag._eval_smooth_area`'s tanh gate and is decoupled from the state machine.
- **Files:** `src/aerislab/components/parachute.py` (`_compute_effective_area`, `_check_deployment_complete`)
- **Effort:** S
- **Depends on:** —
- **Implementation note:** Have `_compute_effective_area()` query the underlying `ParachuteDrag._eval_smooth_area(t, body)` (read the gate value at current time). The state machine then transitions DEPLOYING → DEPLOYED when the gate value reaches 0.99.
- **Acceptance criteria:**
  - `Parachute.is_deploying` is true for at least 0.5 s after deployment in a typical scenario.
  - A new test: deploy a parachute at t=10s, assert `is_deploying` at t=10.05s and `is_deployed` at t=11.0s.
- [ ] Done — commit `<sha>`

---

### P0-T6 — Tag release v0.2.1

- **Files:** `pyproject.toml`, `src/aerislab/__init__.py`, `CHANGELOG.md` (new)
- **Effort:** XS
- **Depends on:** P0-T1 … P0-T5
- **Acceptance criteria:**
  - Both version strings = `0.2.1`.
  - `CHANGELOG.md` lists the 5 fixed bugs with brief descriptions.
  - Git tag `v0.2.1` pushed.
- [ ] Done — commit `<sha>`

---

# Phase 1 — Stabilize (target: v0.3.0)

**Goal:** clean codebase, working CI, atmosphere + wind, no more API duplication. After this phase, the engine produces scientifically interpretable results above sea level for the first time.

**Exit criteria:** ruff/mypy clean, CI matrix runs, atmosphere + wind models exist and are wired into `Drag`/`ParachuteDrag`, `components/standard.py` deleted, end-to-end smoke tests in place.

---

### P1-T1 — `ruff check --fix` and pre-commit hooks

- **Files:** all of `src/`, plus new `.pre-commit-config.yaml`
- **Effort:** XS
- **Depends on:** —
- **Implementation note:** Run `ruff check --fix src/`. Inspect the diff (mostly whitespace + import sorting). Then add `.pre-commit-config.yaml` with ruff, mypy, and `nbstripout`. Run `pre-commit install`.
- **Acceptance criteria:**
  - `ruff check src/` reports 0 issues (or only the un-auto-fixable ones, with each justified by a `# noqa: <code>` comment).
  - `pre-commit run --all-files` exits 0.
- [ ] Done — commit `<sha>`

---

### P1-T2 — Fix all 20 mypy errors

- **Files:**
  - `src/aerislab/models/aerodynamics/parachute_models.py` (17 errors — `Optional[float]` narrowing on `S0`)
  - `src/aerislab/visualization/plotting.py` (1 error — `plt.cm.tab10` stub)
  - `src/aerislab/api/scenario.py` (2 errors — `dict[str, object]` kwargs)
- **Effort:** S
- **Depends on:** —
- **Implementation note:** For `parachute_models.py`, change `ParachuteGeometry`:
  ```python
  D0: float
  S0: float = field(init=False)   # always computed in __post_init__
  S0_input: float | None = None   # optional override
  ```
  For `plotting.py`, suppress with `# type: ignore[attr-defined]`. For `scenario.py`, type kwargs as `Any` or introduce a `SolverConfig` dataclass.
- **Acceptance criteria:** `venv/bin/mypy src/aerislab/` reports `Success: no issues found`.
- [ ] Done — commit `<sha>`

---

### P1-T3 — Convert `Constraint` to real ABC

- **Files:** `src/aerislab/dynamics/constraints.py`
- **Effort:** XS
- **Depends on:** —
- **Implementation note:** Add `from abc import ABC, abstractmethod`. Make `Constraint(ABC)`, decorate `rows`, `index_map`, `evaluate`, `jacobian` with `@abstractmethod`. Remove the `raise NotImplementedError` bodies.
- **Acceptance criteria:**
  - Instantiating a bare `Constraint()` raises `TypeError: Can't instantiate abstract class`.
  - Existing tests pass.
- [ ] Done — commit `<sha>`

---

### P1-T4 — Replace `print()` with `logging`

- **Files:** all of `src/` (use `grep -rn '\bprint(' src/`)
- **Effort:** S
- **Depends on:** —
- **Implementation note:** At the top of each file: `import logging; logger = logging.getLogger(__name__)`. Replace `print(...)` with `logger.info(...)`, `logger.warning(...)`, etc. Configure a default handler in `aerislab/__init__.py` so messages still appear by default. Do **not** touch `examples/` or `scripts/`.
- **Acceptance criteria:**
  - `grep -rn '\bprint(' src/` returns nothing.
  - Running `examples/scenarios/01_simple_drop.py` still shows progress messages on stdout.
  - Setting `logging.getLogger("aerislab").setLevel(logging.WARNING)` silences the per-step progress.
- [ ] Done — commit `<sha>`

---

### P1-T5 — Repo cleanup

- **Files:** delete `examples/Old/`, `scripts/`, `simulation.csv`. Audit notebooks.
- **Effort:** XS
- **Depends on:** —
- **Implementation note:**
  - `git rm -r examples/Old/ scripts/`
  - `git rm simulation.csv`
  - Add `simulation.csv` and `output/` to `.gitignore` (if not already).
  - `pip install nbstripout && nbstripout --install` then `git add` the cleaned notebooks.
  - Optionally move salvageable plot scripts from `scripts/` into `examples/scripts/` first.
- **Acceptance criteria:**
  - `git status` clean after the changes.
  - No `*.ipynb` shows output cells in `git diff`.
- [ ] Done — commit `<sha>`

---

### P1-T6 — Move `example_*` functions out of `parachute_models.py`

- **Files:**
  - Remove from `src/aerislab/models/aerodynamics/parachute_models.py` (lines ~954–1222).
  - New: `examples/parachute_models/compare_models.py`, `examples/parachute_models/quick_start.py`, `examples/parachute_models/all_models.py`.
- **Effort:** S
- **Depends on:** —
- **Acceptance criteria:**
  - `parachute_models.py` is `<1000 LOC`.
  - Coverage of `parachute_models.py` improves (the dead `example_*` lines were dragging it down).
  - Each extracted example runs without error.
- [ ] Done — commit `<sha>`

---

### P1-T7 — Resolve `components/standard` duplication (Move 3)

- **Files:**
  - `src/aerislab/components/payload.py` — add `Payload.from_basic(...)` classmethod absorbing `standard.Payload`'s convenience constructor.
  - `src/aerislab/components/parachute.py` — add `Parachute.from_basic(name, mass, diameter, model="knacke", ...)` classmethod; accept `model: InflationModel | str`; if string, resolve to `AdvancedParachute(model_type=...)`.
  - `src/aerislab/components/__init__.py` — re-export only canonical classes.
  - **DELETE:** `src/aerislab/components/standard.py`
  - Update all `examples/scenarios/*.py` to use `from aerislab.components import Payload, Parachute` and the `.from_basic(...)` constructors.
  - Migrate any tests that imported from `standard` (probably none — coverage is 0%).
- **Effort:** M
- **Depends on:** —
- **Acceptance criteria:**
  - `grep -r 'components.standard' src/ tests/ examples/` returns nothing.
  - All examples run.
  - `Parachute.from_basic(model="knacke")` exposes the deployment state machine (i.e., printed deployment messages appear in logs).
- [ ] Done — commit `<sha>`

---

### P1-T8 — Add end-to-end smoke tests

- **Why:** CRIT-1 went undetected because no test runs an example end-to-end. This is the single highest-leverage testing fix.
- **Files:** new `tests/test_examples_smoke.py`
- **Effort:** S
- **Depends on:** P0-T1 (CRIT-1 fixed)
- **Implementation note:** For each script in `examples/scenarios/*.py`, write a test that:
  - Runs the example in a `tmp_path` working directory.
  - Asserts that an output CSV exists.
  - Asserts physical sanity:
    - For `01_simple_drop`: terminal velocity of bare body matches `sqrt(2mg/(ρ Cd A))` within 5%.
    - For `02_parachute_system`: payload touchdown velocity is `< 30 m/s` (parachute reduces it from ~56).
    - For all: total simulation time `> 0`, no NaN in output, no warnings about constraint singularity.
  - Mark with `@pytest.mark.slow` so they can be skipped during fast iteration.
- **Acceptance criteria:**
  - All smoke tests pass.
  - `pytest -m "not slow"` skips them; `pytest` runs them.
- [ ] Done — commit `<sha>`

---

### P1-T9 — Implement `models.atmosphere.ISA`

- **Files:**
  - New: `src/aerislab/models/atmosphere/__init__.py`
  - New: `src/aerislab/models/atmosphere/base.py` (Protocol: `density(p, t)`, `temperature(p, t)`, `pressure(p, t)`, `speed_of_sound(p, t)`)
  - New: `src/aerislab/models/atmosphere/isa.py` (US 1976 / ICAO standard atmosphere; well-known piecewise formulas)
  - New: `src/aerislab/models/atmosphere/exponential.py` (simple `ρ = ρ₀ exp(-z/H)` for sanity / education)
  - New: `tests/models/test_atmosphere.py` (verify ISA against tabulated values at sea level, 11 km, 20 km, 32 km, 47 km)
- **Effort:** M (one day for ISA done right; references abound)
- **Depends on:** —
- **Acceptance criteria:**
  - ISA density at 0 m matches 1.225 kg/m³ within 0.1%.
  - ISA density at 11 km matches 0.3639 kg/m³ within 0.5%.
  - ISA density at 20 km matches 0.0880 kg/m³ within 0.5%.
- [ ] Done — commit `<sha>`

---

### P1-T10 — Implement `models.wind`

- **Files:**
  - New: `src/aerislab/models/wind/{__init__,base,constant,altitude_profile}.py`
  - `base.py`: Protocol `velocity(p, t) -> NDArray[(3,)]`
  - `constant.py`: `Constant(velocity)`, returns the same vector everywhere
  - `altitude_profile.py`: piecewise-linear interpolation over `(altitude, vx, vy)` table; useful for wind-tunnel-style profiles
  - New: `tests/models/test_wind.py`
- **Effort:** S
- **Depends on:** —
- **Acceptance criteria:**
  - Both classes have correct outputs on trivial test cases.
- [ ] Done — commit `<sha>`

---

### P1-T11 — Wire atmosphere + wind into `Drag` and `ParachuteDrag`

- **Files:** `src/aerislab/dynamics/forces.py`
- **Effort:** S
- **Depends on:** P1-T9, P1-T10
- **Implementation note:** Backwards-compatible — accept `atmosphere: AtmosphereModel | None = None` and `wind: WindModel | None = None`; if `None`, fall back to the old `rho` constant and zero wind. Use altitude `body.p[2]` for now (z-up assumption stays until P2).
- **Acceptance criteria:**
  - Existing tests pass.
  - A new test: bare-body terminal velocity at 5 km altitude with `atmosphere=ISA()` is significantly different from the same scenario at sea level (because ρ is ~50% lower).
  - A new test: drag with `wind=Constant([10, 0, 0])` produces a horizontal force component on a vertically-falling body.
- [ ] Done — commit `<sha>`

---

### P1-T12 — Expand CI matrix

- **Files:** `.github/workflows/test.yml`
- **Effort:** S
- **Depends on:** P0-T4
- **Implementation note:**
  ```yaml
  strategy:
    matrix:
      python-version: ["3.10", "3.11", "3.12"]
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - run: pip install -e ".[dev]"
    - run: ruff check src/
    - run: mypy src/aerislab/
    - run: pytest --cov=aerislab --cov-report=xml
    - uses: codecov/codecov-action@v4   # optional
  ```
- **Acceptance criteria:** All three Python versions go green on a push.
- [ ] Done — commit `<sha>`

---

### P1-T13 — Add constraint-violation diagnostic

- **Files:**
  - `src/aerislab/core/solver.py` (compute `||C(q)||` after each step)
  - `src/aerislab/core/simulation.py` (`Diagnostics.constraint_violation()` or equivalent; add to logger output)
- **Effort:** S
- **Depends on:** —
- **Acceptance criteria:**
  - CSV has a `constraint_violation` column when any constraint is present.
  - For a 60 s pendulum simulation, `max(constraint_violation) < 1e-3`.
- [ ] Done — commit `<sha>`

---

### P1-T14 — Add `manifest.json` provenance

- **Files:** `src/aerislab/core/simulation.py` (`World.enable_logging` writes manifest)
- **Effort:** S
- **Depends on:** —
- **Implementation note:** Write to `output_path / "manifest.json"`:
  ```json
  {
    "git_sha": "...",
    "package_version": "0.3.0",
    "python": "3.11.4",
    "scenario_name": "...",
    "started": "2026-05-13T10:00:00Z",
    "rng_seed": 42,
    "platform": "linux"
  }
  ```
  Use `subprocess.check_output(["git", "rev-parse", "HEAD"])` for the SHA; fall back to `"unknown"` if not in a git repo.
- **Acceptance criteria:** Every output dir has `manifest.json` with the expected fields.
- [ ] Done — commit `<sha>`

---

### P1-T15 — Add convergence-rate verification tests

- **Files:** new `tests/verification/test_convergence.py`
- **Effort:** M
- **Depends on:** —
- **Implementation note:** For each integrator (semi-implicit Euler, Radau, RK45):
  - Run a free-fall problem with `dt = {0.1, 0.05, 0.025, 0.0125}` (or for IVP, `rtol = {1e-3, 1e-4, 1e-5, 1e-6}`).
  - Compute error vs. analytical solution at `t = 1.0 s`.
  - Fit `log(error) vs log(dt)`; assert slope ≈ expected order (semi-implicit Euler: 1, Radau IIA order 5, etc.) within tolerance.
- **Acceptance criteria:** Tests pass and detect a regression if integrator order is wrong.
- [ ] Done — commit `<sha>`

---

### P1-T16 — Tag release v0.3.0

- **Files:** `pyproject.toml`, `src/aerislab/__init__.py`, `CHANGELOG.md`
- **Effort:** XS
- **Depends on:** P1-T1 … P1-T15
- [ ] Done — commit `<sha>`

---

# Phase 2 — Architecture (target: v0.4.0)

**Goal:** the five structural moves from the review §5.2. After this phase the engine is internally clean enough that ML, additional joints, and atmospheric flight are straightforward additions instead of fights with the existing structure.

**Exit criteria:** `World.state/set_state/state_derivative` exists and IVP uses it; `Model` layer composes; `World` decomposed into ≤4 collaborating objects; `World.constraints` derived; CRIT-3's proper fix lands.

This phase is the longest and most risky. Take it in small PRs.

---

### P2-T1 — Introduce `World.state / set_state / state_derivative`

- **Files:**
  - `src/aerislab/core/simulation.py` — add three methods
  - `tests/test_state_vector.py` — new test asserting round-trip `state(...) → set_state(state())` is identity
- **Effort:** M
- **Depends on:** —
- **Implementation note:** Pack order: `[p(3), q(4), v(3), w(3)]` per body, then auxiliary state from any model that registered it (see P2-T2). Initially no auxiliary state. Make `state_derivative(t, y)` a pure function: take `y`, build a temporary local view (don't mutate `world.bodies`), assemble KKT, return `ydot`. Keep the old IVP path working alongside; delete it in P2-T3.
- **Acceptance criteria:**
  - Round-trip test passes.
  - `state_derivative` produces the same `ydot` as the existing IVP RHS (compare numerically on a 2-body problem).
- [ ] Done — commit `<sha>`

---

### P2-T2 — Auxiliary-state registration for stateful models

- **Files:**
  - `src/aerislab/dynamics/forces.py` — add optional `Force.aux_state_size() -> int` and `Force.aux_state_derivative(state, t, body) -> NDArray` methods (default 0 / empty)
  - `src/aerislab/core/simulation.py` — wire into `World.state` packing
- **Effort:** M
- **Depends on:** P2-T1
- **Acceptance criteria:**
  - A test `Force` with `aux_state_size() == 1` participates correctly in the integrator.
  - State vector size = `13 * n_bodies + Σ aux_state_sizes`.
- [ ] Done — commit `<sha>`

---

### P2-T3 — Refactor `HybridIVPSolver` onto the state-vector interface

- **Files:** `src/aerislab/core/solver.py`
- **Effort:** M
- **Depends on:** P2-T1, P2-T2
- **Implementation note:** Inside `integrate()`, replace `_unpack_to_world` mutation with `world.state_derivative(t, y)`. After integration, walk through `sol.t, sol.y` and call `world.set_state(y_k)` once per logged sample, then read forces / constraints / etc. directly without re-running the pipeline. Eliminates the double-RHS bug nucleus.
- **Acceptance criteria:**
  - All existing tests pass.
  - Profile (`py-spy`) confirms force application is run once per logged sample, not twice.
  - The IVP CSV log no longer has any "constraint force from second-pass KKT" weirdness.
- [ ] Done — commit `<sha>`

---

### P2-T4 — Add RK4 integrator (validates the boundary)

- **Files:**
  - New `src/aerislab/core/integrators/rk4.py` (or expand `solver.py`)
  - `tests/test_integrators.py` — RK4 produces 4th-order convergence on free fall
- **Effort:** S
- **Depends on:** P2-T1
- **Why:** Adding a third integrator after Move 1 should be ~80 LOC. If it isn't, the boundary isn't right yet — fix before moving on.
- **Acceptance criteria:**
  - RK4 integrator works on the existing test suite (use it for the free-fall test).
  - 4th-order convergence demonstrated.
- [ ] Done — commit `<sha>`

---

### P2-T5 — Cache inverse mass matrix; invalidate on `q` change

- **Files:** `src/aerislab/dynamics/body.py`, `src/aerislab/core/solver.py`
- **Effort:** S
- **Depends on:** P2-T1 (state-vector boundary makes invalidation tractable)
- **Implementation note:** Store `_q_at_last_inertia_compute` and the cached `_inv_mass_matrix_world`. Recompute only if `np.array_equal(q, _q_at_last_inertia_compute)` is False. Combined with Move 1, this is the biggest single perf win.
- **Acceptance criteria:**
  - `pytest-benchmark` baseline shows ≥2× speedup on the parachute scenario.
  - All tests pass.
- [ ] Done — commit `<sha>`

---

### P2-T6 — Resolve CRIT-3 properly: stateful inflation models in state vector

- **Files:** `src/aerislab/models/aerodynamics/parachute_models.py` (`_force_mass_flow_balance`, `_force_added_mass`)
- **Effort:** M
- **Depends on:** P2-T2 (aux state registration), P0-T3 (defensive guard already in place)
- **Implementation note:** Use the `aux_state_size()` / `aux_state_derivative(...)` mechanism to put `m_inside`, `m_added`, `prev_velocity` into the integrator's state. Remove the per-instance caches. Remove the defensive guard from P0-T3.
- **Acceptance criteria:**
  - Both models produce the same peak-load curve under fixed-step (dt=0.001) and IVP (Radau, rtol=1e-6) within 1%.
  - Defensive guard removed.
- [ ] Done — commit `<sha>`

---

### P2-T7 — Decompose `World` (Move 4)

- **Files:**
  - `src/aerislab/core/simulation.py` (slim `World` to ~250 LOC)
  - New `src/aerislab/core/runner.py` (`Simulator` / `Runner`: `run`, `integrate_to`, progress, log_interval, termination loop)
  - New `src/aerislab/core/output.py` (`OutputManager`: dir creation, manifest, plot generation)
  - New `src/aerislab/core/termination.py` (`TerminationPolicy`: callback chain, ground, time, custom)
  - New `src/aerislab/core/diagnostics.py` (`get_energy`, future `momentum`, `constraint_violation`)
- **Effort:** L (4 sub-PRs, one per extraction)
- **Depends on:** P2-T1, P2-T3
- **Acceptance criteria:**
  - `World` is ≤300 LOC.
  - All existing tests pass without modification.
  - The `Scenario` API is unchanged for users.
- [ ] Done — commit `<sha>`

---

### P2-T8 — `Model` layer for aerodynamics (Move 2)

- **Files:**
  - New `src/aerislab/models/aerodynamics/base.py` (`AeroForce` composing `drag_coefficient + area + atmosphere + wind`)
  - New `src/aerislab/models/aerodynamics/drag_coefficient.py` (`Constant`, `CdTable`, `CdNeural` placeholder)
  - New `src/aerislab/models/aerodynamics/inflation.py` (the 6 inflation models as small `InflationModel` subclasses, no longer mixed with force application)
  - Refactor `src/aerislab/dynamics/forces.py::Drag` and `ParachuteDrag` as thin wrappers around `AeroForce` (or deprecate them in favour of `AeroForce`)
- **Effort:** L
- **Depends on:** P1-T9, P1-T10
- **Acceptance criteria:**
  - The 6 inflation models still produce their expected force curves (regression-test against snapshots from before the refactor).
  - A single `AeroForce` can be configured with any `(drag_coefficient, area, atmosphere, wind)` combination.
- [ ] Done — commit `<sha>`

---

### P2-T9 — `Geodesy` / `Frame` abstraction for z-up assumption

- **Files:**
  - New `src/aerislab/models/geodesy/{base,flat_earth}.py`
  - `base.py` Protocol: `altitude(p) -> float`, `down_vector(p) -> NDArray[(3,)]`, `gravity_at(p) -> NDArray[(3,)]`
  - Update `World`, `Parachute._check_deployment_trigger`, `AdvancedParachute._check_activation`, touchdown event to consume the geodesy model.
  - Default to `FlatEarth(z_up=True)` for backwards compatibility.
- **Effort:** M
- **Depends on:** P2-T7
- **Acceptance criteria:**
  - All existing tests pass with the default `FlatEarth`.
  - `grep -rn 'p\[2\]' src/aerislab/` returns only references inside `geodesy/` and `body.py` (where it's local-frame).
- [ ] Done — commit `<sha>`

---

### P2-T10 — Make added mass enter the LHS

- **Files:** `src/aerislab/core/solver.py` (assemble_system), `src/aerislab/dynamics/body.py` (per-body added mass), or new `src/aerislab/models/aerodynamics/added_mass.py`
- **Effort:** M
- **Depends on:** P2-T8
- **Why:** Currently added-mass is applied as a force; physically it modifies the inertia matrix `M_eff = M + M_a`. After this fix, an FSI surrogate's `M_a(state)` output plugs in cleanly.
- **Acceptance criteria:**
  - Added-mass test compares LHS-form vs current force-form on a canonical inflation scenario; results agree qualitatively but the LHS form is more numerically stable (smaller dt allowed before instability).
- [ ] Done — commit `<sha>`

---

### P2-T11 — Make `World.constraints` derived (Move 5)

- **Files:** `src/aerislab/core/simulation.py`
- **Effort:** S
- **Depends on:** P2-T7
- **Implementation note:** Replace `self.constraints: list[Constraint] = []` with a property that yields from `self.systems` plus a private `_free_constraints` list (for legacy add). This makes CRIT-1 architecturally impossible — the system *is* the source of truth.
- **Acceptance criteria:**
  - CRIT-1 hotpatch (P0-T1) can be reverted; the structural fix takes over.
  - All tests pass.
  - `add_constraint(c)` still works (appends to `_free_constraints`).
- [ ] Done — commit `<sha>`

---

### P2-T12 — Tag release v0.4.0

- **Files:** version bump + `CHANGELOG.md`
- **Effort:** XS
- **Depends on:** P2-T1 … P2-T11
- [ ] Done — commit `<sha>`

---

# Phase 3 — Capabilities (target: v0.5.0)

**Goal:** the user-facing capabilities that make AerisLab a tool, not just an engine. Joint catalogue, Parquet output, YAML config, CLI, performance baseline.

---

### P3-T1 — Custom exceptions

- **Files:** new `src/aerislab/exceptions.py`
- **Effort:** XS
- **Depends on:** —
- **Implementation note:** `ConstraintSingularError`, `IntegrationFailedError`, `DeploymentError`, `ConfigurationError`. Subclass appropriately. Replace the `ValueError`/`RuntimeError` calls throughout `src/`.
- **Acceptance criteria:** Tests can `pytest.raises(ConstraintSingularError)` instead of `RuntimeError`.
- [ ] Done — commit `<sha>`

---

### P3-T2 — Joint catalogue

- **Files:**
  - `src/aerislab/dynamics/constraints.py` — add `RevoluteJoint`, `PrismaticJoint`, `BallJoint`, `UniversalJoint`, `Hinge` (revolute with angular limits)
  - `src/aerislab/dynamics/joints.py` — facade for each
  - `tests/verification/test_joints.py` — verification per joint type
- **Effort:** L
- **Depends on:** P1-T3 (real ABC)
- **Acceptance criteria:**
  - Each joint has a verification test (e.g., revolute: pendulum motion in a single plane).
  - Two-link chain with revolutes matches analytical double-pendulum behaviour.
- [ ] Done — commit `<sha>`

---

### P3-T3 — Suspension-line network

- **Files:**
  - New `src/aerislab/components/lines.py` — `SuspensionLineNetwork` component (tree of point-mass lumps + stiff springs, or KKT subsystem)
- **Effort:** L
- **Depends on:** P3-T2
- **Acceptance criteria:**
  - A 4-line riser between payload and canopy correctly distributes load across lines under asymmetric loading.
- [ ] Done — commit `<sha>`

---

### P3-T4 — Multi-stage parachute / reefing

- **Files:**
  - `src/aerislab/components/parachute.py` — extend `Parachute` to accept reefing schedule
  - or new `src/aerislab/components/multi_stage.py` — `MultiStageParachute` orchestrating drogue → main
- **Effort:** L
- **Depends on:** P0-T5, P2-T8
- **Acceptance criteria:**
  - Drogue → main scenario reproduces qualitatively correct two-peak deceleration profile.
- [ ] Done — commit `<sha>`

---

### P3-T5 — `Trajectory` dataclass + reader

- **Files:** new `src/aerislab/io/trajectory.py`
- **Effort:** M
- **Depends on:** —
- **Implementation note:** A typed dataclass holding `t`, `state`, `forces` (per body, per category), `lambdas`, `constraint_violation`, `metadata`. Loadable from CSV (legacy) and Parquet (new in P3-T6). Plotting (P3-T7) consumes this dataclass, not raw CSV columns.
- **Acceptance criteria:**
  - `traj = Trajectory.from_csv(path)` and `traj = Trajectory.from_parquet(path)` both work.
  - Existing plotting tests pass with the dataclass-fed plot functions.
- [ ] Done — commit `<sha>`

---

### P3-T6 — Parquet output backend

- **Files:**
  - New `src/aerislab/io/parquet_logger.py`
  - `src/aerislab/core/simulation.py` — accept `output_format: Literal["csv", "parquet"] = "parquet"` (parquet becomes default)
  - `pyproject.toml` — add `pyarrow` to runtime deps (or as optional `[parquet]` extra)
- **Effort:** M
- **Depends on:** P3-T5
- **Acceptance criteria:**
  - Same simulation produces equivalent results in CSV and Parquet (round-trip check).
  - Parquet file is ≥3× smaller.
  - All plot tests work with Parquet input.
- [ ] Done — commit `<sha>`

---

### P3-T7 — Refactor plotting to consume `Trajectory`

- **Files:** `src/aerislab/visualization/plotting.py`
- **Effort:** M
- **Depends on:** P3-T5
- **Acceptance criteria:** Plot functions accept `Trajectory` (or path); CSV/Parquet path becomes a thin adapter.
- [ ] Done — commit `<sha>`

---

### P3-T8 — YAML scenario configuration

- **Files:**
  - New `src/aerislab/config/{__init__,schema,loader}.py`
  - Use `pydantic` v2 for typed schemas
  - `src/aerislab/api/scenario.py` — add `Scenario.from_yaml(path)` classmethod
- **Effort:** M
- **Depends on:** P1-T7 (clean `from_basic` constructors)
- **Implementation note:** Schema in `schema.py` with `pydantic.BaseModel` for each scenario element. The example `scenario.yaml` from review §11.5 is the target spec.
- **Acceptance criteria:**
  - All `examples/scenarios/*.py` have an equivalent `*.yaml` next to them; both produce the same simulation output.
  - Schema validation rejects malformed YAML with helpful error messages.
- [ ] Done — commit `<sha>`

---

### P3-T9 — CLI

- **Files:**
  - New `src/aerislab/cli/{__init__,main,run,plot,compare}.py`
  - `pyproject.toml` — add `[project.scripts] aerislab = "aerislab.cli.main:app"`
  - `pyproject.toml` — add `typer` to runtime deps
- **Effort:** M
- **Depends on:** P3-T8
- **Implementation note:**
  ```bash
  aerislab run scenario.yaml [--seed 42] [--output dir/]
  aerislab plot output/run_xyz/ [--bodies payload,canopy]
  aerislab compare output/run_a/ output/run_b/
  aerislab info  # print version, dependencies, available solvers
  ```
- **Acceptance criteria:**
  - All four subcommands work.
  - `aerislab --help` shows usable help.
- [ ] Done — commit `<sha>`

---

### P3-T10 — Performance baseline with `pytest-benchmark`

- **Files:**
  - New `tests/perf/test_benchmark.py`
  - New `docs/perf/` (output directory)
- **Effort:** S
- **Depends on:** —
- **Implementation note:** Three benchmarks:
  1. 2-body parachute system, 60 s, IVP Radau → steps/s.
  2. 10-body multibody chain, fixed-step → ms/step.
  3. Isolated KKT solve at 50 bodies × 50 constraints → ms/solve.
  Save flame graph SVGs from `py-spy` in `docs/perf/`.
- **Acceptance criteria:**
  - Benchmarks run with `pytest tests/perf/ --benchmark-only`.
  - CI fails if a benchmark regresses by ≥20% (use `pytest-benchmark --benchmark-compare-fail=mean:20%`).
- [ ] Done — commit `<sha>`

---

### P3-T11 — Reproduce a published parachute case

- **Files:**
  - New `examples/validation/apollo_drogue.py` (or T-10, or whichever is best documented)
  - New `tests/validation/test_published_cases.py`
- **Effort:** L (the work is in finding the data and matching parameters, not in coding)
- **Depends on:** P1-T9 (atmosphere), P2-T8 (Model layer)
- **Why:** Single biggest credibility move for a research code. Even one case reproduced is a publishable software artifact.
- **Acceptance criteria:**
  - Simulation reproduces published opening-time and peak-load within 15%.
  - Discrepancy is documented (modelling assumptions, data uncertainty).
- [ ] Done — commit `<sha>`

---

### P3-T12 — Documentation site (mkdocs-material or sphinx)

- **Files:**
  - New `docs/_site/` (or wherever your chosen tool prefers)
  - Existing `docs/AERISLAB_REVIEW.md`, `docs/WORK_PLAN.md` etc. become source pages
  - GitHub Actions workflow to build + deploy to GitHub Pages
- **Effort:** L
- **Depends on:** —
- **Implementation note:** mkdocs-material is faster to set up; sphinx is more powerful (good if you want auto-API docs from docstrings — and you should, since the docstrings are NumPy-style and high quality). Suggest sphinx + `myst-parser` + `sphinx-autodoc2`.
- **Acceptance criteria:**
  - `aerislab.readthedocs.io` (or GH-Pages equivalent) serves the docs.
  - API reference generated automatically from docstrings.
  - Tutorial: "your first parachute simulation" works end-to-end.
- [ ] Done — commit `<sha>`

---

### P3-T13 — Tag release v0.5.0

- [ ] Done — commit `<sha>`

---

# Phase 4 — ML / FSI integration (target: v0.6.0)

**Goal:** the thesis's central novel contribution. After this phase, FSI-trained surrogates are first-class citizens in AerisLab simulations.

**Prerequisites met after Phase 2-3:** state-vector boundary (P2-T1), Model layer (P2-T8), atmosphere/wind (P1-T9/T10), validation case (P3-T11). Without these, ML errors are uninterpretable.

---

### P4-T1 — `AeroSurrogate` Protocol and reference implementations

- **Files:**
  - New `src/aerislab/ml/__init__.py`
  - New `src/aerislab/ml/surrogate/{base,analytical,onnx,torch}.py`
- **Effort:** M
- **Depends on:** P2-T8
- **Implementation note:** API per review §13.2:
  ```python
  class AeroSurrogate(Protocol):
      def predict(self, state: AeroState) -> AeroOutput: ...
  ```
  - `AnalyticalSurrogate(model=AdvancedParachute(...))` — wraps existing models. Lets you exercise the new code path with the existing test suite.
  - `OnnxSurrogate(model_path)` — production.
  - `TorchSurrogate(model)` — development.
- **Acceptance criteria:**
  - `AnalyticalSurrogate` produces identical results to the underlying analytical model (regression test).
  - `OnnxSurrogate` loads and runs a trivial ONNX model (e.g., a saved 2-layer MLP).
- [ ] Done — commit `<sha>`

---

### P4-T2 — `NeuralAeroForce`

- **Files:** new `src/aerislab/ml/force.py`
- **Effort:** S
- **Depends on:** P4-T1
- **Implementation note:** A `Force` that consumes an `AeroSurrogate` and an `AtmosphereModel`/`WindModel` (composes with the Phase 2 Model layer naturally).
- **Acceptance criteria:**
  - A simulation with `NeuralAeroForce(AnalyticalSurrogate(...))` produces identical trajectories to the equivalent analytical simulation.
- [ ] Done — commit `<sha>`

---

### P4-T3 — FSI data ingestion + featurisation

- **Files:**
  - New `src/aerislab/ml/data/{fsi_loader,featurise,augment}.py`
- **Effort:** L
- **Depends on:** —
- **Implementation note:** FSI loader is format-agnostic via a thin adapter — write the first adapter for whatever your FSI tool actually outputs (OpenFOAM? in-house? VTK? HDF5?). Featurise per review §13.3: non-dimensionalise, body-frame, include `(t − t_deploy)`, `dA/dt`. Augment with rotational symmetries.
- **Acceptance criteria:**
  - `fsi_loader.load(path) → list[FSIRun]` works on at least one real dataset.
  - `featurise.to_training(runs) → (X, y)` produces tensors ready for training.
- [ ] Done — commit `<sha>`

---

### P4-T4 — Training pipeline

- **Files:** new `src/aerislab/ml/train.py`, `src/aerislab/ml/models/{mlp,lstm,transformer}.py`
- **Effort:** L
- **Depends on:** P4-T3
- **Implementation note:** Hydra or Typer for config-driven training (matches the YAML scenario approach). Weights & Biases or MLflow for experiment tracking.
- **Acceptance criteria:**
  - Trains and saves a model on a small synthetic dataset.
  - All hyperparameters configurable from YAML.
- [ ] Done — commit `<sha>`

---

### P4-T5 — OOD detector

- **Files:** new `src/aerislab/ml/ood.py`
- **Effort:** M
- **Depends on:** P4-T4
- **Implementation note:** Mahalanobis distance to training-set features is the cheapest start. Deep ensembles or calibrated uncertainty as a follow-on. The surrogate should expose `confidence` in its `AeroOutput`.
- **Acceptance criteria:**
  - On in-distribution inputs, confidence > 0.9.
  - On out-of-distribution inputs (e.g., 10× training velocity), confidence < 0.3.
- [ ] Done — commit `<sha>`

---

### P4-T6 — Deep ensembles for uncertainty bands

- **Files:** `src/aerislab/ml/ensemble.py`
- **Effort:** M
- **Depends on:** P4-T4
- **Acceptance criteria:**
  - 5-member ensemble produces uncertainty bands on terminal velocity, opening time, peak load.
  - Bands cover ground truth on held-out FSI cases at the 95% level.
- [ ] Done — commit `<sha>`

---

### P4-T7 — Validation against held-out FSI cases

- **Files:** `tests/ml/test_surrogate_validation.py`, plotting helpers
- **Effort:** L
- **Depends on:** P4-T4, P4-T6, P3-T11
- **Acceptance criteria:**
  - Pointwise drag error: ≤10% on held-out trajectories.
  - Trajectory error (terminal velocity, opening time, peak load): ≤5%.
  - Comparison plots vs. each analytical inflation model on the same scenarios.
- [ ] Done — commit `<sha>`

---

### P4-T8 — Tag release v0.6.0

- [ ] Done — commit `<sha>`

---

# Phase 5 — Aerospace expansion + thesis features (post-v0.6, ongoing)

**Goal:** scope the engine beyond recovery systems, in the directions that match your thesis or post-thesis interests. Pick what your thesis actually needs; the rest is a menu.

These items are deliberately less detailed because their priority depends on which research direction you commit to.

---

### P5-T1 — Reduced-order modal canopy (deformable body)

- **Why:** Lets the canopy change shape; couples directly with FSI-trained modal forces.
- **Effort:** XL
- **Depends on:** P2-T2 (aux state), P4-T1 (surrogate)
- [ ] Done

### P5-T2 — `Controller` Protocol + PID + RL adapter

- **Why:** Steerable parachutes, attitude control, RL-policy guidance research.
- **Effort:** L
- [ ] Done

### P5-T3 — NRLMSISE-00 atmospheric upgrade

- **Why:** High-altitude / orbital atmospheric drag.
- **Effort:** S (wrap `pymsis`)
- [ ] Done

### P5-T4 — Variable-mass rocket / thrust-vector control

- **Why:** Launch-phase / ascent simulations.
- **Effort:** M
- [ ] Done

### P5-T5 — Aerodynamic-coefficient expansion (lift, side force, moments)

- **Why:** Winged decelerators, lifting bodies.
- **Effort:** M
- [ ] Done

### P5-T6 — `aerislab.aerospace.orbital` package

- **Why:** Two-body / J2 / spherical-harmonic gravity → orbital propagation.
- **Files:**
  - New `src/aerislab/aerospace/orbital/{gravity,frames,ephemeris}.py`
  - New `src/aerislab/models/geodesy/wgs84.py`
- **Effort:** XL
- **Depends on:** P2-T9 (geodesy abstraction)
- **Implementation note:** Validate `PointMassGravity` orbit against `poliastro` or `Skyfield` for a known LEO trajectory.
- [ ] Done

### P5-T7 — Atmospheric entry simulation

- **Why:** Combines orbital + variable atmosphere + recovery deployment. Natural endpoint of the engine's scope.
- **Effort:** XL
- **Depends on:** P5-T3, P5-T4, P5-T6
- [ ] Done

### P5-T8 — Monte Carlo / parameter sweep framework

- **Files:** `src/aerislab/study/monte_carlo.py`
- **Effort:** M
- **Depends on:** P3-T8 (YAML), P1-T14 (manifest with seeds)
- **Implementation note:** `multiprocessing.Pool` is sufficient until N > 10⁴. Each run gets its own seed and manifest; results aggregated into a Parquet file.
- [ ] Done

### P5-T9 — Differentiable simulation backend (post-thesis stretch)

- **Why:** Inverse design, gradient-based optimisation.
- **Effort:** XL
- **Depends on:** P2-T1
- **Implementation note:** Parallel JAX backend for `assemble_system`, `solve_kkt`, `state_derivative`. Keep numpy as default.
- [ ] Done

---

# Methodology, milestones, and what to write up

Each phase end is also a candidate writeup point.

| Phase | Release | Possible thesis chapter / paper |
|---|---|---|
| 0 | v0.2.1 | — (hotfix) |
| 1 | v0.3.0 | "Software architecture" chapter |
| 2 | v0.4.0 | "Numerical methods" chapter — KKT, Baumgarte, integrators, convergence |
| 3 | v0.5.0 | JOSS / Software Impacts paper (citable software contribution) |
| 4 | v0.6.0 | "ML-FSI surrogate methodology" chapter — main novel contribution |
| 5 | v0.7+ | Application chapters (drop-test campaign, validation, design study) |

---

# Quick triage if budget is tight

If you only have time for **one week** of work, do P0-T1 through P0-T6 (Phase 0). The engine becomes correct on the user-facing path.

If you have **one month**, add P1-T1, P1-T7, P1-T8, P1-T9, P1-T11. The engine becomes scientifically usable.

If you have **three months**, add Phase 2 (architecture). The engine becomes a platform for everything else.

Do not start Phase 4 (ML) until Phase 2 is done. The cost of the wrong baseline is months of debugging surrogate errors that are actually engine errors.

---

*This work plan is a living document. Update it as priorities shift, and as new bugs / opportunities surface during the work itself. The phase boundaries are guidelines, not contracts — if a Phase 2 item turns out to be straightforward, do it during Phase 1.*
