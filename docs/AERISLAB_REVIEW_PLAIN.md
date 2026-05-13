# AerisLab — Plain-English Review

**What this is:** the same review as `AERISLAB_REVIEW.md`, but written in plainer language. Same findings, fewer technical assumptions. If you find a sentence here too hand-wavy, the precise technical version is the sibling file `AERISLAB_REVIEW.md`.

**Codebase version:** 0.2.0 on branch `ClaudeHelp` at commit `8ef845a`
**Date:** 2026-05-13
**Companion:** `docs/WORK_PLAN.md` is the step-by-step to-do list. Read this for *why*; read the work plan for *what*.

---

## Table of Contents

1. [The 30-second summary](#1-the-30-second-summary)
2. [What I actually checked](#2-what-i-actually-checked)
3. [Three serious bugs you didn't know you had](#3-three-serious-bugs-you-didnt-know-you-had)
4. [Old known problems — still there or now fixed?](#4-old-known-problems--still-there-or-now-fixed)
5. [The big architecture question (and what I think you should do)](#5-the-big-architecture-question-and-what-i-think-you-should-do)
6. [Module-by-module verdict](#6-module-by-module-verdict)
7. [Is the physics right?](#7-is-the-physics-right)
8. [Tests, validation, and reproducibility](#8-tests-validation-and-reproducibility)
9. [Day-to-day code quality](#9-day-to-day-code-quality)
10. [What to do next month](#10-what-to-do-next-month)
11. [What to do over the next year](#11-what-to-do-over-the-next-year)
12. [Where you could take this long-term](#12-where-you-could-take-this-long-term)
13. [About the ML/FSI plan](#13-about-the-mlfsi-plan)
14. [Things I'd specifically warn you against](#14-things-id-specifically-warn-you-against)
15. [Glossary](#15-glossary)

---

## 1. The 30-second summary

**Good news:**
- The bug that was secretly doubling gravity is fixed.
- All 115 tests pass.
- The version-number mismatch is resolved.
- The way you've organized the code (layers stacked on each other; components that *contain* a rigid body rather than *being* one) is genuinely good design. Don't lose this when you refactor.

**Bad news (new findings):**
- Three serious bugs make the easy-to-use `Scenario` API silently produce wrong physics. The simulation runs, no errors appear, plots come out — but the numbers are wrong. Section 3 explains all three.
- A few old "code smell" issues are still there.
- The continuous integration (the automatic test-run on GitHub) has been broken for a while, so you haven't been getting any safety net from it.

**The grade:** roughly 7 out of 10. It didn't go up from the previous review even though you fixed real things, because the new bugs I found offset the fixes. With the work in section 10, it becomes a solid 8/10 and a credible v0.3 release.

**The big question:** the engine is yours to own — you've said you don't want to switch to using someone else's multibody library underneath. That's fine and totally defensible. So this review focuses on making *your* engine the best version of itself, not on replacing it.

---

## 2. What I actually checked

I ran the standard tools (`pytest`, `ruff`, `mypy`, `coverage`) on a fresh clone, read every important source file, and ran two of your example scripts to see what they actually do. Here are the numbers.

| What I measured | Number | Comment |
|---|---|---|
| Tests collected | 115 | All passing |
| Test runtime | 36 seconds | Reasonable |
| Source code lines | ~5,900 | Manageable |
| Test code lines | ~3,600 | Healthy ratio (lots of tests per line of code) |
| Code coverage | 75% overall | But the user-facing path (`scenario.py`, `standard.py`) has the *worst* coverage — that's why the new bugs went undetected |
| Largest source file | `parachute_models.py` (1,223 lines) | About 270 of those lines are example functions that don't belong inside the source — they should be in `examples/` |
| Second-largest | `simulation.py` (764 lines) | This is the `World` class, doing too much |
| Lint warnings | 283 | Mostly trivial whitespace; ~38 are real issues |
| Type-checker errors | 20 | None are real bugs, but they signal sloppy types |
| Continuous integration | **Broken** | See HIGH-3 in section 3 |

A "bug" in this document means: the simulation produces physically wrong numbers without telling you. A "code smell" means: the code works, but it's set up in a way that *will* cause a bug eventually.

---

## 3. Three serious bugs you didn't know you had

These three are the most important findings in the entire review. They make your `Scenario` API silently produce wrong results. Fix these first.

### Bug #1 — When you connect a parachute to a payload with a tether, the tether is silently ignored

**The symptom:** You write what looks like a perfectly good simulation:

```python
sc = Scenario(name="my_drop").add_system([payload, parachute]) \
     .connect(payload, parachute, type="tether", length=10.0)
```

The simulation runs. No errors. Plots come out. But — the parachute and payload are actually falling **as two completely separate objects**, not connected by a tether. The parachute drifts off doing its own thing while the payload falls at the speed it would fall at *without* any parachute attached.

**How I confirmed it:** I ran your `examples/scenarios/02_parachute_system.py` (a 50 kg payload + 12 m parachute supposedly tethered together). The payload reached the ground at -56 m/s. That's the bare-payload terminal velocity (the speed it falls at with just air resistance and no parachute). If the tether had been working, the payload would have been doing something like -8 m/s.

**Why it happens (in plain words):** Your `Scenario.connect()` method puts the tether constraint into the `System` object that holds the components. But the `System` was already given to the `World` *before* the constraint was added, and the `World` only made a copy of the constraints that existed at that moment. So the `World` (which is what actually runs the physics solver) never sees the tether. It's like writing a guest's name on a clipboard *after* the guest list has already been printed and handed to the doorman.

**Severity:** This is exactly as bad as the old "double gravity" bug that was fixed before — every parachute simulation through the `Scenario` API is wrong.

**Fix difficulty:** A one-line patch will fix it. The proper architectural fix (which prevents the same kind of bug from ever coming back) is one of the bigger refactors in the work plan.

---

### Bug #2 — The CSV log is missing most of the force data

**The symptom:** When you open the simulation CSV in Excel or pandas, you'll see columns for total force (`f_x`, `f_y`, `f_z`) and the gravity breakdown (`f_gravity_x/y/z`). But you *won't* see columns for aerodynamic drag, parachute drag, or constraint forces. The plots that try to show "what forces are acting on this body" will be missing data.

**Why it happens (in plain words):** Your `CSVLogger` decides what columns to write the *very first time* you log a data point. It looks at the body and asks "what kinds of force have you accumulated so far?" Then it locks in those column names forever.

The trouble is, on that first call, the body usually has no aerodynamic force on it yet — either because the simulation hasn't started moving (so air-drag returns zero) or because the parachute hasn't deployed. So those columns never get created, and they never get added later either, even when those forces *do* start being applied. The data is silently thrown away.

**How I confirmed it:** I ran the simple drop example and looked at the CSV header — only `f_gravity_*` appeared, no `f_aerodynamics_*`, even though air drag was active for the entire 30 seconds.

**Severity:** Medium-high. The simulation itself is right; the *recording* of what happened is incomplete. So your trajectory plots are right, but the "force breakdown" plots are wrong.

**Fix difficulty:** Small to medium. Either buffer all rows in memory and figure out the columns at the end, or switch to a file format (Parquet) that doesn't need a fixed column list upfront. The work plan recommends both — small fix now, real fix later.

---

### Bug #3 — Two of the six parachute models give wrong results when used with the recommended solver

**The symptom:** You have six different parachute inflation models (Knacke, continuous inflation, mass-flow balance, French-Huckins, porosity-corrected, added mass). Two of them — **mass-flow balance** and **added mass** — silently give wrong peak-load numbers when run with the default IVP solver (the one that `Scenario` uses).

The other four models work correctly with both solvers. The two broken ones work correctly with the simpler fixed-step solver.

**Why it happens (in plain words):** Both broken models compute things like "how fast is the velocity changing" by remembering the previous velocity from the last function call and subtracting. This is fine if the function gets called once per time step, in order, with predictable spacing — which is what the fixed-step solver does.

But the IVP solver (Radau, the recommended one for stiff problems like parachute opening) calls the function *multiple times per step* to figure out the answer, including for steps it later throws away. So the "previous velocity" the model remembers might be from a step that didn't actually happen, or from a probe at a slightly different time. The arithmetic gets garbled.

There's also a hardcoded number `dt = 0.01` inside one of the models, which has nothing to do with the actual time step the IVP solver is using.

**How it slipped past your tests:** You have 24 tests for these models, but they call them in a loop with predictable timing — exactly the way the fixed-step solver would. Nothing tests them under the IVP solver.

**Severity:** High for any quantitative claim. These two models exist precisely because they're "more physically complete" — so the user comparing models would naturally pick them, and silently get wrong numbers.

**Fix difficulty:** Short-term: tell the IVP solver to refuse these two models and raise a clear error message — that's small work. Long-term fix: rework these models so the integrator handles their internal state properly. The work plan lays this out.

---

### Plus two more issues worth fixing right away

#### HIGH-3 — Your continuous integration is broken

When you push to GitHub, an automated workflow is supposed to install the package and run the tests. That workflow has been failing because your `pyproject.toml` lists a development dependency called `types-numpy`, which doesn't actually exist on PyPI (the public Python package registry). The install step fails before the tests are even attempted. So you've been getting no automated safety net for some time.

**Fix:** Delete one line. Modern numpy ships type stubs natively.

#### HIGH-2 — The "deploying" state of a parachute lasts exactly one tick

Your `Parachute` component has a deployment state machine that goes `STOWED → DEPLOYING → DEPLOYED`. The intent is that "DEPLOYING" represents the period when the canopy is opening — half a second to a few seconds. But the way the code is written, the moment the parachute deploys, the helper function that's supposed to report "how open is it now" returns "fully open." So the state machine immediately decides "ok, we're done deploying" on the very next time step.

The actual smooth-area-opening behavior is happening, but it lives somewhere else (inside `ParachuteDrag`'s tanh smoothing function), not connected to the state machine.

**Fix:** Have the state machine read the actual smooth-area value, so DEPLOYING lasts as long as the canopy is genuinely opening.

---

## 4. Old known problems — still there or now fixed?

The two earlier review documents flagged a list of issues. Here's their status today.

| Issue | Status | Notes |
|---|---|---|
| Double gravity bug | ✅ **Fixed** | |
| 11 failing tests | ✅ **Fixed** | All 115 pass now |
| Version mismatch (0.1.0 vs 0.2.0) | ✅ **Fixed** | Both say 0.2.0 |
| `Constraint` is not a real abstract base class | ⚠️ Still there | Tiny fix — five minutes |
| `World` class is too big (765 lines) | ⚠️ Still there | Doing too many jobs at once |
| Two parallel `Payload`/`Parachute` classes (`components/` vs `components/standard/`) | ⚠️ Still there | All your examples use the `standard` ones, which bypass the deployment state machine — confusing |
| IVP solver runs the entire force pipeline twice (once for physics, once for logging) | ⚠️ Still there | Wastes time and creates room for bugs |
| `print()` everywhere instead of proper logging | ⚠️ Still there | |
| No atmosphere model (air density hardcoded at sea level) | ⚠️ Still there | Above 2 km, drag is wrong by 20-40% |
| No wind model | ⚠️ Still there | Five-line fix once the structure is right |
| Whitespace cruft in source files | ⚠️ Still there | Auto-fixable in 30 seconds |
| 20 type-checker complaints | ⚠️ Still there | |
| `examples/Old/` and `scripts/` clutter | ⚠️ Still there | |
| `simulation.csv` accidentally committed at repo root | ⚠️ Still there | |
| Notebooks with output cells committed | ⚠️ Still there | Makes diffs huge |
| CI is minimal (only Python 3.11, only pytest) | ❌ **Worse** | And it's broken — see HIGH-3 |

The pattern: real progress on the bugs, no progress on the architectural issues. That's normal for a research code at this stage.

---

## 5. The big architecture question (and what I think you should do)

You've made it clear you want to keep your own solvers — you don't want to rebase the engine on top of Project Chrono, MuJoCo, or some other existing multibody library. That's a defensible decision. Reasons it's defensible:
- you keep full control over what the code does;
- you learn the math by writing it;
- you're not at the mercy of someone else's API decisions or bugs;
- the engine itself becomes a research artifact you can write up.

The trade-off is that you're going to spend some PhD time maintaining a multibody integrator that other people have already written. As long as you're aware of that trade-off, fine.

So this review is calibrated to making your in-house engine the best version of itself, not to replacing it. Here are the **five structural changes** that, in my opinion, will pay off the most. None of these is a rewrite — they're all targeted refactors of existing modules.

### Move 1 — Clean separation between "the world's state" and "the solver"

**Right now:** the IVP solver reaches into the `World` and modifies bodies' positions, velocities, and quaternions while it's solving. This is convenient but tangles things up. It also means that when you want to log what happened, the solver has to re-run the entire force pipeline a second time (which is where some of the wasted time and bug risk comes from).

**The change:** introduce three methods on `World`: `state()` (pack everything into one big array), `set_state(y)` (unpack a big array back into bodies), and `state_derivative(t, y)` (the physics — pure function, doesn't change `World`).

**Why this matters:** once this is in place, adding new integrators is trivial (the boundary is clean). Adding a neural-network surrogate for aerodynamics is trivial (it plugs into `state_derivative` like any other force). Logging becomes one-pass. This is the single biggest leverage refactor — it unblocks the rest.

### Move 2 — A "Model" layer between forces and the actual physics

**Right now:** your `Drag` class hardcodes air density. Your `ParachuteDrag` class hardcodes the inflation logic *and* the activation conditions *and* the area smoothing *and* the application to the body. Everything's mashed together, so adding wind or altitude-dependent density means surgery on every force class.

**The change:** introduce small composable model objects:

```python
AeroForce(
    drag_coefficient = ConstantCd(0.85),
    area = InflationModel.continuous(diameter=10.0),
    atmosphere = ISA(),                # International Standard Atmosphere
    wind = ConstantWind([5, 0, 0]),
)
```

**Why this matters:** atmosphere and wind become five-line additions instead of refactors. Your future neural-network drag-coefficient surrogate just becomes another `drag_coefficient = NeuralCd(...)`. The same `AeroForce` works for spheres, parachutes, wings.

### Move 3 — Pick *one* version of `Payload` and `Parachute`

**Right now:** there's a `Payload` in `components/payload.py` (which takes a body you've already built) and another `Payload` in `components/standard.py` (which builds the body for you). Same for `Parachute`. The `standard` versions are friendlier to use, so all your examples use them — but they bypass the deployment state machine. The "good" versions are only used by tests.

**The change:** keep the `components/` versions as the canonical API. Add convenience constructors (`@classmethod from_basic(...)`) that do what the `standard/` versions do today. Delete `standard.py`. Update the three example scripts.

**Why this matters:** one obvious right way to do something is much better than two ways with subtle differences. And it stops examples from silently bypassing the state machine.

### Move 4 — Break up the `World` class

**Right now:** `World` is 764 lines and does at least seven jobs (orchestrating physics + logging + plotting + output directories + termination conditions + energy diagnostics + status printing).

**The change:** keep `World` to about 250 lines (just bodies, constraints, time, state). Pull the rest into:
- a `Simulator` / `Runner` (the loop, progress reporting, termination)
- an `OutputManager` (directories, manifest, plots)
- a `TerminationPolicy` (the chain of "should we stop now" callbacks)
- a `Diagnostics` (energy, momentum, constraint violation)

**Why this matters:** smaller files are easier to test, easier to reason about, easier to change without breaking unrelated stuff. This is also where you naturally fix the hardcoded "ground is at z=0" assumption.

### Move 5 — Make `World.constraints` derived from systems, not stored separately

**Right now:** the `World` keeps its own list of constraints, separately from the `System`s it contains. Bug #1 above happens because those two lists got out of sync.

**The change:** delete `World.constraints`. Instead, when the solver needs constraints, ask the `World` to give them — and `World` derives the answer by walking through its `Systems`. Single source of truth.

**Why this matters:** Bug #1 becomes structurally impossible. You can't forget to update a list that doesn't exist.

### What I'm explicitly *not* recommending

Just so it's clear:
- **No rewrite from scratch.** Everything above is a refactor of code you already have.
- **No switch to JAX or other GPU framework.** Tempting for "automatic differentiation" but it's a 6-month detour that doesn't help your thesis. Keep numpy as the default; build a parallel JAX kernel later if you ever genuinely need gradients.
- **No "plugin/backend" interface "in case you switch later."** Don't pay that complexity tax until you actually have a second backend.
- **No microservice / web UI.** Save that for downstream tools when you actually need them.

---

## 6. Module-by-module verdict

Quick table. The verdicts mean:
- ✅ keep — well-designed, leave it alone except for small polish
- 🟡 refactor in place — the design is OK but needs improvement
- 🔴 needs structural change — significant work needed (the work plan covers this)

| File | Lines | Coverage | Verdict | One-line summary |
|---|---|---|---|---|
| `dynamics/body.py` | 463 | 83% | ✅ | `RigidBody6DOF` is solid. Add caching for performance. |
| `dynamics/forces.py` | 462 | 89% | 🟡 | Split forces from models (Move 2). The `last_force` attribute that other code looks for is never actually set — dead code. |
| `dynamics/constraints.py` | 135 | 97% | 🟡 | Make `Constraint` a real abstract class. Add more joint types over time. |
| `dynamics/joints.py` | 49 | 57% | ✅ | Thin facades, fine. Will grow with the joint catalogue. |
| `core/solver.py` | 633 | 79% | 🔴 | The IVP path is the most tangled module. Move 1 untangles it. |
| `core/simulation.py` | 764 | 92% | 🔴 | Decompose (Move 4); fix Bug #1 by deriving constraints (Move 5). |
| `components/base.py` | 170 | 91% | ✅ | The composition pattern (HAS-A) is right. |
| `components/payload.py` | 116 | 84% | 🟡 | Absorb `standard.Payload`'s convenience constructor (Move 3). |
| `components/parachute.py` | 269 | 83% | 🔴 | Fix HIGH-2 (1-tick deployment); absorb `standard.Parachute`; accept rich inflation models. |
| `components/standard.py` | 122 | **0%** | 🔴 | Delete after Move 3. |
| `components/system.py` | 208 | 71% | ✅ | Will become the real owner of constraints (Move 5). |
| `models/aerodynamics/parachute_models.py` | 1223 | 76% | 🟡 | Move the 270 lines of example functions out. Fix Bug #3 properly. |
| `api/scenario.py` | 180 | **22%** | 🔴 | Public API has worst coverage. Fix Bug #1; redesign `connect()`. |
| `visualization/plotting.py` | 517 | 70% | 🟡 | Read from a `Trajectory` dataclass, not raw CSV columns. |
| `logger.py` | 236 | 82% | 🔴 | Fix Bug #2; eventually add Parquet output. |
| `utils/validation.py` | 134 | **0%** | 🟡 | Either use these functions everywhere, or delete the file. |
| `utils/io.py` | 30 | **0%** | 🟡 | Same. |

---

## 7. Is the physics right?

### What you got right

These are real strengths. Don't lose them when you refactor.

- **The KKT formulation with Schur-complement solving** (the math used to enforce constraints like "this tether must stay 10 m long") is the right approach for your problem. It's what serious multibody dynamics codes use.
- **Baumgarte stabilization** (the technique for keeping constraint violations from drifting over time) is implemented correctly.
- **You check for ill-conditioned matrices** and fall back to least-squares — graceful degradation rather than crashing.
- **Quaternion orientation** (instead of Euler angles) — the right choice; avoids gimbal lock.
- **Gyroscopic torque** ($-\omega \times (I\omega)$) is correctly added in the assembly. Easy to forget; you didn't.
- **Semi-implicit Euler** (your fixed-step integrator) is *symplectic* for unconstrained systems, meaning energy stays bounded. Nice property.
- **Distance and weld constraint Jacobians** (the math that tells the solver "if I pull on this rod, this is how the bodies move") are correct.
- **The verification suite** is unusually rigorous for a research code: it tests free fall, terminal velocity, pendulum period, spring period, energy conservation, angular momentum, and even the **Dzhanibekov effect** (the weird flipping motion of asymmetric spinning objects in zero gravity). That last one is a real test of 3D rotational integrity.

### What needs work

These don't mean your physics is wrong — they mean accuracy could be better, or the code is set up in a way that will get harder to fix later.

**Numerical:**

- **Quaternion normalization is done inside the IVP solver's right-hand-side function.** This is a "bumpy" operation (mathematically non-smooth) that can confuse stiff solvers. Long-term fix: treat "this quaternion has length 1" as a constraint that the solver enforces.
- **The mass matrix is recomputed on every solver call.** It only depends on orientation, not velocity, so it could be cached. With Move 1 in place this is straightforward. Likely a 2-5× speedup.
- **No constraint-violation logging.** Baumgarte stabilization keeps drift small but doesn't make it zero. You should log "how much is the constraint actually being violated" so you can see when something's drifting.
- **Fixed Baumgarte parameters with no diagnostic.** The `α` and `β` numbers are user-tunable but undocumented for non-trivial systems. A small diagnostic ("for this system, your α and β correspond to a constraint oscillation at X Hz with damping ratio Y") would help users pick values.
- **No convergence-rate tests.** A standard test for a solver is "halve the time step, the error should drop by a known factor." You don't have these. Easy to add — would catch regressions immediately.
- **No replication of a published case.** Even one comparison ("we reproduced the Apollo drogue chute opening shock from NASA TN-XXXX within 10%") would be enormously credibility-boosting and is publishable as a software contribution by itself.

**Physical fidelity:**

- **No atmosphere model.** Air density is hardcoded at 1.225 kg/m³ (sea level). At 5 km altitude actual density is around 0.74 — so your drag forces are off by 40%. ISA atmosphere is one day's work and unblocks everything quantitative.
- **No wind model.** Drag treats body velocity as airspeed. With non-zero wind, true airspeed is `body.v − wind.velocity`. Five lines, after Move 2.
- **No Mach or Reynolds number dependency for drag coefficients.** Fine for round subsonic parachutes; binding the moment you do supersonic drogues or anything with a wing.
- **Added mass is treated as a force, not as inertia.** The physically correct thing: when a parachute is opening, it drags air with it, so its effective inertia goes up. You're modeling this as an extra force instead of as extra mass. Qualitatively similar peak loads, formally not quite right. After Move 2 it's straightforward to fix properly.
- **The parachute is rigid.** Constant inertia, doesn't change shape. Fine for trajectory studies; precludes any work where canopy shape matters (collapse, breathing, glide, asymmetric loading). Your ML/FSI pipeline naturally produces shape-changing data; the engine should be ready to consume it.

---

## 8. Tests, validation, and reproducibility

### Tests

You have 115 of them, all passing. The verification suite is unusually thorough. But there's a hole that explains why Bugs #1 and #3 went undetected:

**Nothing in the test suite runs an end-to-end scenario through the user-facing API and checks that the result is physically sensible.** The tests check pieces in isolation. So when the `Scenario` API wires the pieces together wrong, none of the tests notice.

**Easiest highest-value fix:** add 5-10 "smoke tests" that run each example script and assert simple things — "the touchdown velocity should be less than 30 m/s when there's a parachute," "no NaNs in the output," "the simulation finished without warnings about singular matrices." This kind of test catches almost everything dangerous.

### Validation

You have **verification** (does the simulation match analytical solutions?) but no **validation** (does the simulation match real experimental data?). For a research code, both matter; validation is what reviewers ask about.

Specific things missing:
- **Convergence-rate tests** (mentioned above).
- **Literature replication.** Even one case from a published paper or NASA report.
- **Regression baselines for parachute models.** If you change the inflation math next year, you should have a saved force-vs-time curve from today to compare against, so you immediately notice if behavior changed.
- **Property-based tests.** For physical invariants (energy bounded, momentum conserved without external forces, distance constraints actually maintained), property-based testing libraries like `hypothesis` are stronger than example-based tests.

### Reproducibility

This is non-negotiable for a PhD and currently absent.

Each output directory should contain a `manifest.json` recording:
- the git commit hash of the code that produced it,
- the package version,
- the Python version and platform,
- the full scenario configuration,
- any random seeds used,
- the start/end timestamps,
- any warnings issued during the run.

Without this, "regenerate Figure 4.7 from chapter 4" three years from now becomes painful. With it, it's one command.

You should also start seeding any random number generator from day one (even if nothing is currently random) — Monte Carlo, atmospheric turbulence, ML model dropout will all need it eventually, and retrofitting seeded randomness later is a chore.

---

## 9. Day-to-day code quality

Things that are easy to fix and would make the codebase nicer to live in:

| Item | Where | Effort | Payoff |
|---|---|---|---|
| Run `ruff check --fix` | All of `src/` | 30 seconds | Removes 238 of the 283 lint warnings in one go |
| Fix the 20 type-checker errors | Mostly `parachute_models.py` | 1-2 hours | Makes the type contract honest |
| Replace `print()` with `logging` | All of `src/` | 2 hours | Users can silence verbose output; works with the standard Python ecosystem |
| Make `Constraint` a real abstract class | `dynamics/constraints.py` | 5 minutes | Errors at class-definition time instead of at runtime |
| Delete `types-numpy` from dev dependencies | `pyproject.toml` | 30 seconds | Fixes the broken CI |
| Set up pre-commit hooks | New `.pre-commit-config.yaml` | 30 minutes | Lint and type-check run automatically before each commit |
| Expand CI to test on Python 3.10/3.11/3.12 + run ruff/mypy/coverage | `.github/workflows/test.yml` | 1 hour | Automated safety net |
| Delete `examples/Old/`, `scripts/`, the stray `simulation.csv` | Repo root | 30 minutes | Less clutter |
| Strip output cells from notebooks (`nbstripout --install`) | Notebooks | 30 minutes | Diffs become readable |
| Move 270 lines of example code out of `parachute_models.py` | Source → examples | 1 hour | Source files become focused |
| Write a real README and project documentation | New `docs/_site/` | 1 week | Makes the project usable by others |
| Add a CLI (`aerislab run scenario.yaml`) | New `cli/` | 1 day | Much better story than "run this Python script" |

The single highest-leverage thing is `ruff check --fix` plus pre-commit. It changes nothing functional but the codebase looks meaningfully cleaner.

---

## 10. What to do next month

If you can spend serious time on this in the next four weeks, here's the order. (The companion `WORK_PLAN.md` has every task spelled out with files, effort, and acceptance criteria.)

### Week 1 — Stop the bleeding (Phase 0 in the work plan)

1. Fix Bug #1 (tether ignored) — one-line patch.
2. Fix Bug #2 (logger header timing).
3. Fix Bug #3 (mass-flow + added-mass models under IVP) — defensive fix now, proper fix later.
4. Fix the broken CI (delete `types-numpy`).
5. Fix the 1-tick deployment state machine.

After this week the engine is **correct on the path users actually use**. Tag it as `v0.2.1`.

### Weeks 2-3 — Hygiene + atmosphere

6. Run `ruff check --fix`, set up pre-commit, expand CI.
7. Fix the 20 type-checker errors.
8. Replace `print()` with `logging`.
9. Make `Constraint` a real abstract class.
10. Implement an ISA atmosphere model. Implement a constant wind model. Wire them into your drag forces.
11. Delete the duplicate `components/standard.py`; promote its convenience constructors as classmethods on the canonical components.
12. Add 5-10 end-to-end smoke tests so this kind of bug never sneaks past again.

### Week 4 — Diagnostics + reproducibility

13. Add constraint-violation logging.
14. Add `manifest.json` to every output directory.
15. Add convergence-rate tests for each integrator.
16. Add custom exception types (`ConstraintSingularError`, `IntegrationFailedError`, etc.).

After this month the engine is **scientifically usable above sea level** for the first time. Tag it as `v0.3.0`.

---

## 11. What to do over the next year

This is the bigger structural work. It's where the engine becomes a real platform instead of a prototype. The work plan breaks each of these into individual tasks.

### The 5 architecture moves (Phase 2 — months 2-4)

Already described in §5. In order of dependency:

1. Add `World.state / set_state / state_derivative` — the clean boundary between physics and integration.
2. Refactor the IVP solver to use that boundary; eliminate the double-pipeline-for-logging.
3. Add an RK4 integrator to validate the new boundary (if it isn't ~80 lines, the boundary isn't right yet).
4. Cache the inverse mass matrix.
5. Properly fix Bug #3 by moving model state into the integrator.
6. Decompose `World` into 4-5 smaller objects.
7. Build the Model layer (atmosphere/wind/aero/geodesy/ground).
8. Make `World.constraints` derived from systems, killing Bug #1's class permanently.

### The capabilities (Phase 3 — months 4-6)

9. Add more joint types: revolute (hinge), prismatic (slider), ball, universal, hinge with limits. Each is ~30 lines because the KKT solver doesn't change.
10. Switch from CSV to **Parquet** as the default output format. Smaller files (5-10× compression), faster reads, schema metadata travels with the file. Keep CSV as an option for human-readability.
11. Add a `Trajectory` dataclass — plot functions read from this, not from raw CSV columns. Decouples plotting from on-disk format.
12. Add YAML scenario configuration. Write `Scenario.from_yaml(path)`. This unblocks parameter studies and external collaboration — writing 200 YAML files is much easier than writing 200 Python scripts.
13. Add a CLI (`aerislab run scenario.yaml`, `aerislab plot output/run_xyz/`, etc.).
14. Reproduce one published parachute case (Apollo drogue or similar) — biggest single credibility move you can make.
15. Stand up a documentation site (Sphinx or mkdocs-material).

### The ML/FSI plumbing (Phase 4 — months 6-12)

This is your thesis's central novel contribution. See §13 for the detailed plan. The short version:

16. Define an `AeroSurrogate` interface. Implement three versions: one wrapping your existing analytical models (so the new code path is exercised by the existing test suite), one loading ONNX models (production-friendly), one using PyTorch (development).
17. Build the FSI-data → training-tensor pipeline.
18. Build the training infrastructure (Hydra/Typer + W&B/MLflow).
19. Add an out-of-distribution detector so the surrogate refuses to extrapolate silently.
20. Add deep ensembles for uncertainty bands.
21. Validate end-to-end against held-out FSI cases.

---

## 12. Where you could take this long-term

You mentioned the long-term vision includes more aerospace problems and "space things" hypothetically. Some of this is short-term, some is research-direction-dependent, all of it is a menu, not a backlog.

### Recovery-system enrichment (clear, near-term)

These are things any serious recovery-system simulator needs:

- **Multi-stage parachutes** (drogue → main with a sequencer).
- **Reefing** (multi-stage area expansion of one canopy).
- **Suspension-line networks** — currently your parachute is one rigid body connected to the payload by one tether. Real systems have N suspension lines from a confluence point. Modeling this gives you correct line-tension distributions.
- **Risers and bridles** — same machinery as suspension lines.
- **Disreefing dynamics** (time-varying reefing-line length).
- **Detailed opening shock profiles** (peak, plateau, settling).
- **Canopy oscillation / pendulum modes** — payload swings under canopy. Important for landing accuracy.
- **Cluster parachute systems** (multiple canopies on one payload).
- **Glide / steerable parachutes** — ML surrogate is the natural home for asymmetric Cd/Cl.

### Beyond rigid bodies (medium-term)

If you want to model a canopy that actually changes shape:

- **Reduced-order modal canopy.** Keep the rigid-body approximation for gross motion, add a few "modes" of deformation. Modal forces come from FSI training data (same pipeline as the ML aerodynamic surrogate).
- **Particle-system suspension lines.** Tree of point masses connected by stiff springs. Solve at a sub-step inside the integrator if needed.
- **Lumped-parameter cloth/membrane.** Mass-spring-damper grid for low-fidelity canopy shape.

### Control and actuation (medium-term)

- A `Controller` interface that consumes `World` state and writes torques/forces.
- PID, LQR, MPC as concrete controllers.
- An RL-policy adapter (wrap a trained policy from `stable-baselines3` or similar).

Useful for steerable-parachute guidance research.

### Atmospheric flight beyond recovery (longer-term)

- **Variable-mass rocket** (a body whose mass changes over time as it burns fuel).
- **Thrust-vector control** (force application point as a function of gimbal angle).
- **Lift, side force, pitching/yawing/rolling moments** for winged/lifting bodies.
- **Autopilot integration** (uses the `Controller` interface from above).

### Space dynamics (your "very hypothetical")

This is reachable from your current architecture with bounded effort, **once Move 1 (clean state-vector boundary) and a `Geodesy/Frame` abstraction exist**. The hard pieces:

- **Gravitational models.** Today `Gravity` is a constant vector. Generalize to:
  - `PointMassGravity(GM, center)` — Keplerian two-body.
  - `J2Gravity` — Earth oblateness (one extra term, sufficient for most low Earth orbit work).
  - `SphericalHarmonicGravity` — high-fidelity Earth gravity field.
  - `NBodyGravity` — Sun, Moon, planets as perturbers.
- **Reference frames and time systems.** ECI ↔ ECEF ↔ topocentric. UT1, UTC, TAI. You can wrap `astropy` (heavy) or implement a minimal subset yourself.
- **Atmospheric drag at altitude.** NRLMSISE-00 is the standard model; `pymsis` wraps it.
- **Solar radiation pressure** — tiny but mission-relevant for high area-to-mass spacecraft.
- **Attitude dynamics with reaction wheels / magnetorquers.**
- **Orbit propagation.** With a point-mass gravity model, your existing IVP solver (Radau or DOP853) is already a valid orbit propagator. Validate against `poliastro` or `Skyfield` for a known trajectory.
- **Lambert / impulsive maneuvers** (post-orbit-propagation).
- **Atmospheric entry** combines launch-phase aero, variable atmosphere, hypersonic drag (Mach + Knudsen number), and recovery deployment. The natural endpoint of all three threads.

Suggested packaging: keep `aerislab` core as the constrained-multibody + force/model framework. Add `aerislab.aerospace.atmospheric` and `aerislab.aerospace.orbital` as feature packages on top. Don't pollute the core with orbital concepts — let them live in their own namespace.

What you should **not** try to do in AerisLab: full mission planning (use GMAT), high-fidelity CFD (you'll be running CFD externally and ingesting the data into your ML surrogate), or attitude determination from sensor data (write a separate package).

---

## 13. About the ML/FSI plan

This is the central novel contribution of your thesis. A quick summary of the design and the most common ways such projects go wrong.

### What you're building

A system where, during a multibody simulation, a parachute's instantaneous aerodynamic force vector (and ideally moment, added-mass tensor, and deformation modes) is produced by a neural network that was trained on FSI (Fluid-Structure Interaction) simulation data. The network's input is the current state — velocity, altitude, deployment phase, etc. The network's output replaces or augments your analytical inflation models.

### What it depends on

This work depends, in order, on:

1. The clean `World.state_derivative` boundary (Move 1).
2. The Model layer (Move 2).
3. The atmosphere and wind models (so the surrogate isn't fitting against a wrong baseline).
4. At least one validated reference case (so surrogate errors are measurable against ground truth).

If you skip any of these, ML errors become uninterpretable. The single most common failure mode in CFD-ML papers is "we trained a great surrogate against an underlying physics model that was already wrong by 30%."

### Common pitfalls (in rough order of "I have personally watched people fall into this")

- **Frame inconsistency.** Your FSI simulation runs in canopy body frame. The neural net should output body-frame quantities to remain rotation-equivariant. Apply the rotation to world frame only at the very last step.
- **Wrong non-dimensionalization.** Train on normalized inputs (velocity / reference velocity, density / sea-level density), not raw SI units. Otherwise the network won't generalize across scales and altitudes.
- **History dependency.** Parachute opening has memory — the canopy is filling. A network that only sees the current state will be wrong during inflation. Either add features like "time since deployment" and "rate of area change," or use an LSTM/GRU/Transformer.
- **Silent extrapolation.** A neural net asked for inputs outside its training data does *not* error. It returns plausible-looking garbage. You **must** add an out-of-distribution detector — Mahalanobis distance to training set is the cheapest start, calibrated uncertainty estimators are the gold standard.
- **Inference speed.** A single PyTorch call from inside a stiff solver's right-hand-side is 1-10 ms. Radau evaluates the RHS 3+ times per step over a 60-second simulation at millisecond steps — that's 100-1000 seconds of pure inference time. Mitigations: batch in time when fixed-step, use ONNX Runtime instead of raw torch (5-10× faster), `torch.compile` or TensorRT, cache by quantized state for slowly-varying parts.
- **Random sample splits.** Hold out **whole trajectories**, not random samples. A network that interpolates between t=1.00 s and t=1.05 s of the same FSI run will look brilliant on paper and fail in production.
- **Pointwise vs trajectory metrics.** Report both. The end-user metric is descent prediction, not pointwise drag.

### Things that will save you time

- **Use Hydra or Typer** for config-driven training. Matches the YAML scenario approach you'll have in `aerislab` itself by then.
- **Use Weights & Biases or MLflow** for experiment tracking. Every run becomes reproducible and citable. Important for the PhD.
- **Use deep ensembles** (5-10 networks with different random seeds) for uncertainty bands. Cheap, well-calibrated, integrates easily.

### Why this is publishable

Coupling FSI-derived ML aerodynamic surrogates to constrained multibody recovery-system dynamics in an open Python package, validated end-to-end against drop-test data, is a thesis-grade contribution. AIAA, *Aerospace Science & Technology*, *Journal of Computational Physics*, JOSS for the software side. **Don't underweight the JOSS paper** — a citable DOI for the software is what gets you cited by people *using* your tool, which is the audience that matters most for a research code's afterlife.

---

## 14. Things I'd specifically warn you against

A short list of common traps I've seen in projects like yours.

- **"Let me just rewrite this in JAX so I get autodiff for free."** It's a 6-month detour. Build a parallel JAX kernel for the inner loop *later* if and only if you actually need gradients. Keep numpy as the default for the foreseeable future.
- **"Let me add a backend interface so I can swap in MuJoCo later."** Don't pay the abstraction tax until you actually have a second backend.
- **"Let me make this faster with Numba/Cython/C."** Profile first. Your current bottleneck is *redundant work* (recomputing the same matrix every solver call), not slow Python. Move 1's caching is a 3-5× speedup with no language change.
- **"Let me build a GUI / web UI."** Save it for downstream tools (a Streamlit dashboard on top of Parquet output, when you actually need it).
- **"Let me write my own CFD."** That's the rest of someone else's PhD. Use external CFD; ingest its output as data.
- **"Let me start training the ML surrogate before the atmosphere model is in place."** Pure ML on a wrong baseline is the #1 failure mode in this whole research direction. Get the physics baseline right *first*.
- **"Let me add features for hypothetical future requirements."** Build the next thing you actually need; let the abstractions emerge from real demand. The aerospace/space menu in §12 is a menu, not a backlog.

### Things to keep — even if you ever feel like rewriting from scratch

- The `Component` HAS-A `RigidBody6DOF` composition pattern.
- The verification suite (especially Dzhanibekov, energy conservation, pendulum period).
- The 6 inflation models as a domain-knowledge artifact.
- The KKT formulation with Schur complement and Baumgarte stabilization.
- The Scenario fluent API design (after Bug #1 is fixed and `connect()` is generalized).
- The dual fixed-step + IVP solver design.
- The output-directory organization (logs/ + plots/ + manifest/).

The rest is replaceable.

---

## 15. Glossary

If you ran into any unfamiliar words above, here's what they mean in this context.

- **6-DOF (six degrees of freedom)** — the body can move in three directions (x, y, z) and rotate about three axes. Total of six independent motions to track.
- **Baumgarte stabilization** — a small "spring + damper" style correction that the solver adds when constraints (like a tether's fixed length) start to drift due to numerical error. Keeps the constraint approximately satisfied over time.
- **CI (continuous integration)** — automated build/test that runs every time you push to GitHub.
- **Constraint** — a rule the solver must enforce, like "these two attachment points must stay 10 m apart" or "these two points must coincide."
- **Dzhanibekov effect** — the surprising flipping motion that an asymmetric spinning object makes in zero gravity. Named after the Soviet cosmonaut who noticed it. Hard to reproduce in simulation unless 3D rotational dynamics are correct, so it's a good integrity test.
- **ECEF / ECI** — coordinate frames used in space mechanics. ECEF (Earth-Centered, Earth-Fixed) rotates with the Earth; ECI (Earth-Centered, Inertial) doesn't.
- **FSI (Fluid-Structure Interaction)** — high-fidelity simulation that couples fluid flow (e.g., air around a parachute) with structural deformation (e.g., the canopy bending). Expensive but accurate. Source of training data for your ML surrogate.
- **Holonomic constraint** — a constraint expressible as an equation in positions only (like a tether's length), as opposed to one that involves velocities.
- **ISA (International Standard Atmosphere)** — the standard piecewise model of how air density, temperature, and pressure vary with altitude. Reference for all atmospheric flight.
- **IVP (Initial Value Problem)** — fancy way of saying "given the state at time 0, integrate forward in time." Your `HybridIVPSolver` is built on scipy's adaptive IVP solvers (Radau, BDF, RK45).
- **Jacobian** — the matrix of partial derivatives. In constraints: how the constraint violation changes when each body's position/orientation changes.
- **KKT (Karush-Kuhn-Tucker)** — a mathematical formulation for solving "minimize this with these constraints" problems. In multibody dynamics it shows up as the system you solve to get accelerations consistent with constraints.
- **Lagrange multiplier (λ)** — the magnitude of the constraint force. The KKT solve gives you both accelerations and these multipliers.
- **Mach number** — speed divided by speed of sound. Drag coefficients depend on Mach number, especially near and above Mach 1.
- **Multibody dynamics** — the simulation of multiple rigid (or flexible) bodies connected by joints/constraints. What AerisLab is.
- **NRLMSISE-00** — a well-validated model of upper-atmosphere density, temperature, etc., used for orbital decay simulations.
- **ONNX (Open Neural Network Exchange)** — an open format for storing trained neural networks; ONNX Runtime is much faster than PyTorch for inference-only use.
- **Parquet** — an efficient columnar file format. Like CSV but smaller, faster to read, and carries its schema with it.
- **Quaternion** — a 4-number representation of orientation (x, y, z, w). Better than Euler angles because it doesn't have gimbal lock.
- **Reynolds number** — a number that characterizes fluid flow regime (laminar vs turbulent). Drag coefficients depend on it.
- **RHS (right-hand side)** — when you write the equations of motion as `dy/dt = f(t, y)`, that `f(...)` function is the "RHS." The IVP solver calls it many times per integration step to figure out the answer.
- **Schur complement** — a way of solving the KKT system efficiently when you have many bodies but few constraints.
- **Semi-implicit Euler** — a simple but well-behaved time integrator. Updates velocity first, then uses the new velocity to update position. Symplectic (energy-preserving for unconstrained systems).
- **Stiff (system / solver)** — a system where rapid and slow dynamics happen on very different timescales (parachute opening is stiff). Stiff solvers (Radau, BDF) handle this without taking absurdly small time steps.
- **Surrogate (model)** — a fast approximation of a slow underlying model. Here: a neural network that approximates expensive FSI results.
- **Symplectic integrator** — an integrator with a useful mathematical property that bounds energy drift over long simulations.
- **WGS84** — the standard Earth ellipsoid model used by GPS and most geodesy.

---

*This review is for Štěpán Kaspar, kaspar.stepan.cz@gmail.com, VUT Brno.*
*Independent re-evaluation against branch `ClaudeHelp` at commit `8ef845a`, 2026-05-13.*
