# Orchestration (Simulator / OutputManager / pure-snapshot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **⚠ PROVISIONAL:** This plan targets the codebase **after sub-project 1 is implemented** (`PhysicsWorld` with `build_layout`/`state_slice`/`pack_global_state`/`unpack_global_state`, `HybridIVPSolver` with `_make_rhs` + `quat_stab_k`, `RigidBody6DOF.state_derivative`). Line numbers are intentionally omitted; verify exact code anchors against the real tree before each task and adjust if SP1 landed differently.

**Goal:** Split time-orchestration and I/O out of `World` into a `Simulator` and an `OutputManager`, unify physics evaluation into a single pure `compute_snapshot` used by both the solver and logging (fixing the CRIT-3 double-RHS state corruption), and keep `World` as a thin backward-compatible facade.

**Architecture:** `PhysicsWorld` (from SP1) stays the physics container. `compute_snapshot(physics, t, y, …)` is the one pure evaluation function. `Simulator` owns time/loop/events and drives the solver; `OutputManager` owns the output dir, CSV logger, and plots. `World` composes the three and delegates, so the public API is unchanged.

**Tech Stack:** Python 3, NumPy, SciPy (`solve_ivp`), pytest.

**Reference spec:** [docs/superpowers/specs/2026-05-22-orchestration-simulator-output-design.md](../specs/2026-05-22-orchestration-simulator-output-design.md)

---

## File Structure

| File | Responsibility | New / Modified |
|---|---|---|
| `src/aerislab/core/world.py` | `PhysicsWorld` gains `apply_all_forces` | Modify |
| `src/aerislab/core/snapshot.py` | `compute_snapshot` pure evaluation function | **New** |
| `src/aerislab/core/output.py` | `OutputManager`: dirs, CSV logger, plots | **New** |
| `src/aerislab/core/simulator.py` | `Simulator`: time, loop, events, run/integrate_to | **New** |
| `src/aerislab/core/solver.py` | `HybridIVPSolver` uses `compute_snapshot` | Modify |
| `src/aerislab/core/simulation.py` | `World` becomes thin facade | Modify |
| `tests/test_snapshot.py` | snapshot purity + consistency | **New** |
| `tests/test_output.py` | OutputManager dirs/logging | **New** |
| `tests/test_simulator.py` | Simulator run/integrate regression | **New** |
| `tests/test_world_facade.py` | World facade API unchanged | **New** |

Run the suite with `.venv/bin/pytest`.

---

### Task 1: PhysicsWorld.apply_all_forces

**Files:**
- Modify: `src/aerislab/core/world.py`
- Test: `tests/test_snapshot.py`

Unify the force-application order (systems → per-body → global → interaction) that
is currently duplicated across `World.step`, the IVP rhs, and the logging replay.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_snapshot.py
import numpy as np
from aerislab.core.world import PhysicsWorld
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity


def _world_with_body():
    pw = PhysicsWorld()
    b = RigidBody6DOF("b", 2.0, np.eye(3), np.zeros(3), np.array([0,0,0,1.0]))
    pw.add_body(b)
    pw.add_global_force(Gravity(np.array([0,0,-9.81])))
    return pw, b


def test_apply_all_forces_accumulates_gravity():
    pw, b = _world_with_body()
    b.clear_forces()
    pw.apply_all_forces(0.0)
    assert np.allclose(b.f, [0.0, 0.0, 2.0 * -9.81])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_snapshot.py::test_apply_all_forces_accumulates_gravity -v`
Expected: FAIL with `AttributeError: 'PhysicsWorld' object has no attribute 'apply_all_forces'`

- [ ] **Step 3: Write minimal implementation**

Add to `PhysicsWorld` in `world.py`:

```python
    def apply_all_forces(self, t: float) -> None:
        """Apply every registered force in canonical order (assumes forces cleared)."""
        for system in self.systems:
            system.apply_all_forces(t)
        for b in self.bodies:
            for fb in b.per_body_forces:
                fb.apply(b, t)
        for fg in self.global_forces:
            for b in self.bodies:
                fg.apply(b, t)
        for fpair in self.interaction_forces:
            fpair.apply_pair(t)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_snapshot.py::test_apply_all_forces_accumulates_gravity -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/core/world.py tests/test_snapshot.py
git commit -m "feat(core): PhysicsWorld.apply_all_forces unifies force application"
```

---

### Task 2: compute_snapshot pure evaluation function

**Files:**
- Create: `src/aerislab/core/snapshot.py`
- Test: `tests/test_snapshot.py`

`compute_snapshot` returns `ydot` and, as a benign diagnostic side effect, leaves
`force_categories` and constraint forces populated for the logger. It must not
mutate any model's hidden time history.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_snapshot.py  (append)
from aerislab.core.snapshot import compute_snapshot


def test_compute_snapshot_is_pure():
    pw, b = _world_with_body()
    pw.build_layout()
    y = pw.pack_global_state()
    yd1 = compute_snapshot(pw, 0.3, y.copy(), alpha=0.0, beta=0.0, quat_stab_k=0.0)
    snap = pw.pack_global_state().copy()
    yd2 = compute_snapshot(pw, 0.3, y.copy(), alpha=0.0, beta=0.0, quat_stab_k=0.0)
    assert np.allclose(yd1, yd2)               # deterministic
    assert np.allclose(pw.pack_global_state(), snap)  # no hidden mutation
    # ydot layout for a single body: dp=v, dv=a (gravity only -> [0,0,-9.81])
    assert np.allclose(yd1[7:10], [0.0, 0.0, -9.81])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_snapshot.py::test_compute_snapshot_is_pure -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aerislab.core.snapshot'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aerislab/core/snapshot.py
"""
Single pure physics-evaluation function shared by the IVP solver and logging.

compute_snapshot evaluates all forces and the KKT system at state (t, y) and
returns the global derivative ydot. It populates force_categories and constraint
forces on the bodies as a diagnostic side effect for the logger, but never mutates
any model's hidden time history (the precondition for safe re-evaluation / replay).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aerislab.core.solver import assemble_system, solve_kkt


def compute_snapshot(
    physics,
    t: float,
    y: NDArray[np.float64],
    alpha: float,
    beta: float,
    quat_stab_k: float,
) -> NDArray[np.float64]:
    """Evaluate physics at (t, y); return ydot and cache forces for logging."""
    physics.unpack_global_state(y)

    for b in physics.bodies:
        b.clear_forces()
    physics.apply_all_forces(t)

    Minv, J, F, rhs_v, _ = assemble_system(physics.bodies, physics.constraints, alpha, beta)
    a, lam = solve_kkt(Minv, J, F, rhs_v)

    # Cache constraint forces for the logger (diagnostic only).
    if J.shape[0] > 0:
        F_constraint = J.T @ lam
        for i, b in enumerate(physics.bodies):
            fc = F_constraint[6*i:6*i+6]
            b.apply_force(fc[:3], label="constraint")
            b.apply_torque(fc[3:])

    ydot = np.empty_like(y)
    for i, b in enumerate(physics.bodies):
        lo, hi = physics.state_slice(b)
        ydot[lo:hi] = b.state_derivative(a[6*i:6*i+3], a[6*i+3:6*i+6], quat_stab_k=quat_stab_k)
    for comp in physics.aux_providers:
        lo, hi = physics.state_slice(comp)
        ydot[lo:hi] = comp.compute_derivatives(t)
    return ydot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_snapshot.py::test_compute_snapshot_is_pure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/core/snapshot.py tests/test_snapshot.py
git commit -m "feat(core): compute_snapshot pure physics evaluation"
```

---

### Task 3: HybridIVPSolver uses compute_snapshot

**Files:**
- Modify: `src/aerislab/core/solver.py`
- Test: `tests/test_solver_ivp.py`

Replace the `_make_rhs` inner logic with a thin wrapper over `compute_snapshot`, so
there is exactly one physics implementation.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_solver_ivp.py  (append)
def test_solver_rhs_matches_compute_snapshot():
    import numpy as np
    from aerislab.core import World
    from aerislab.core.solver import HybridIVPSolver
    from aerislab.core.snapshot import compute_snapshot
    from aerislab.dynamics.body import RigidBody6DOF
    from aerislab.dynamics.forces import Gravity

    w = World(ground_z=-1000.0, payload_index=0)
    w.add_body(RigidBody6DOF("b", 1.0, np.eye(3), np.array([0,0,5.0]), np.array([0,0,0,1.0])))
    w.add_global_force(Gravity(np.array([0,0,-9.81])))
    ivp = HybridIVPSolver(method="Radau", alpha=0.0, beta=0.0, quat_stab_k=0.0)
    w.build_layout()
    y = w.pack_global_state()
    rhs = ivp._make_rhs(w.physics if hasattr(w, "physics") else w)
    assert np.allclose(rhs(0.2, y.copy()),
                       compute_snapshot(w.physics if hasattr(w, "physics") else w,
                                        0.2, y.copy(), 0.0, 0.0, 0.0))
```

- [ ] **Step 2: Run test to verify it fails or is inconsistent**

Run: `.venv/bin/pytest tests/test_solver_ivp.py::test_solver_rhs_matches_compute_snapshot -v`
Expected: FAIL (rhs still inlines its own logic / signature differs)

- [ ] **Step 3: Rewrite `_make_rhs` as a wrapper**

In `HybridIVPSolver._make_rhs`, replace the body with:

```python
    def _make_rhs(self, physics):
        from aerislab.core.snapshot import compute_snapshot

        def rhs(t, y):
            return compute_snapshot(physics, t, y, self.alpha, self.beta, self.quat_stab_k)
        return rhs
```

Ensure `integrate` passes the physics container (in SP1 it passed `world`; here pass
`world.physics` if present, else `world`). Keep `build_layout`, `touchdown_event`,
`pack/unpack_global_state` usage intact.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_solver_ivp.py -v`
Expected: PASS (including SP1's free-fall + purity tests)

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/core/solver.py tests/test_solver_ivp.py
git commit -m "refactor(solver): IVP rhs delegates to compute_snapshot"
```

---

### Task 4: OutputManager

**Files:**
- Create: `src/aerislab/core/output.py`
- Test: `tests/test_output.py`

Move directory creation, CSV logger, and plotting out of `World`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_output.py
from pathlib import Path
import numpy as np
from aerislab.core.output import OutputManager
from aerislab.core.world import PhysicsWorld
from aerislab.dynamics.body import RigidBody6DOF


def test_output_manager_creates_dirs_and_logs(tmp_path):
    om = OutputManager("run1", output_dir=tmp_path, auto_timestamp=False)
    assert (tmp_path / "run1" / "logs").is_dir()
    assert (tmp_path / "run1" / "plots").is_dir()

    pw = PhysicsWorld()
    pw.add_body(RigidBody6DOF("b", 1.0, np.eye(3), np.zeros(3), np.array([0,0,0,1.0])))
    om.log(pw, 0.0)
    om.logger.flush()
    csv = tmp_path / "run1" / "logs" / "simulation.csv"
    assert csv.exists() and csv.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_output.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aerislab.core.output'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aerislab/core/output.py
"""OutputManager: owns the output directory, CSV logger, and plot generation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from aerislab.logger import CSVLogger


class OutputManager:
    def __init__(self, name, output_dir=Path("output"), auto_timestamp=True,
                 auto_save_plots=False):
        self.name = name
        self.auto_save_plots = auto_save_plots
        folder = f"{name}_{datetime.now():%Y%m%d_%H%M%S}" if auto_timestamp else name
        self.output_path = Path(output_dir) / folder
        self.logs_dir = self.output_path / "logs"
        self.plots_dir = self.output_path / "plots"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logger = CSVLogger(str(self.logs_dir / "simulation.csv"))

    def log(self, physics, t) -> None:
        # CSVLogger.log currently takes the World; it reads .bodies and .t.
        # Provide a lightweight adapter so it can log a PhysicsWorld at time t.
        physics.t = t  # transient attribute for the logger's timestamp
        self.logger.log(physics)

    def flush(self) -> None:
        self.logger.flush()

    def save_plots(self, physics, bodies=None, show=False) -> None:
        from aerislab.visualization.plotting import (
            plot_forces, plot_trajectory_3d, plot_velocity_and_acceleration,
            plot_force_breakdown,
        )
        csv_path = self.logs_dir / "simulation.csv"
        if bodies is None:
            bodies = [b.name for b in physics.bodies if not getattr(b, "fixed", False)]
        for name in bodies:
            plot_trajectory_3d(str(csv_path), name,
                               save_path=str(self.plots_dir / f"{name}_trajectory_3d.png"), show=show)
            plot_velocity_and_acceleration(str(csv_path), name,
                               save_path=str(self.plots_dir / f"{name}_velocity_acceleration.png"),
                               show=show, magnitude=False)
            plot_forces(str(csv_path), name,
                               save_path=str(self.plots_dir / f"{name}_forces.png"),
                               show=show, magnitude=False)
            plot_force_breakdown(str(csv_path), name,
                               save_path=str(self.plots_dir / f"{name}_force_breakdown.png"), show=show)
```

> **Note:** check `CSVLogger.log`'s real signature when implementing; if it requires
> `.t` on the object, the adapter above suffices, otherwise adjust `log()` to pass `t`
> explicitly.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_output.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/core/output.py tests/test_output.py
git commit -m "feat(core): OutputManager owns dirs, CSV logger, plots"
```

---

### Task 5: Simulator (time, loop, events)

**Files:**
- Create: `src/aerislab/core/simulator.py`
- Test: `tests/test_simulator.py`

Move `run`, `step`, `integrate_to`, termination, and touchdown out of `World`.
Logging goes through `compute_snapshot` + `OutputManager` (pure replay).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulator.py
import numpy as np
from aerislab.core.world import PhysicsWorld
from aerislab.core.simulator import Simulator
from aerislab.core.solver import HybridIVPSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity


def test_simulator_freefall_regression():
    pw = PhysicsWorld()
    z0 = 20.0
    pw.add_body(RigidBody6DOF("b", 1.0, np.eye(3), np.array([0,0,z0]), np.array([0,0,0,1.0])))
    pw.add_global_force(Gravity(np.array([0,0,-9.81])))
    sim = Simulator(pw, output=None, payload_index=0, ground_z=-1000.0)
    ivp = HybridIVPSolver(method="Radau", rtol=1e-8, atol=1e-10, max_step=np.inf)
    sim.integrate_to(ivp, t_end=1.234)
    z_exact = z0 - 0.5 * 9.81 * 1.234**2
    assert abs(pw.bodies[0].p[2] - z_exact) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_simulator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aerislab.core.simulator'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aerislab/core/simulator.py
"""Simulator: advances a PhysicsWorld in time and drives the solver."""
from __future__ import annotations

import numpy as np

from aerislab.core.snapshot import compute_snapshot

EPSILON_GROUND = 1e-9


class Simulator:
    def __init__(self, physics, output=None, payload_index=0, ground_z=0.0):
        self.physics = physics
        self.output = output
        self.payload_index = int(payload_index)
        self.ground_z = float(ground_z)
        self.t = 0.0
        self.t_touchdown = None
        self.termination_callback = None

    def set_termination_callback(self, fn):
        self.termination_callback = fn

    def integrate_to(self, solver, t_end, log_interval=1.0):
        sol = solver.integrate(self, t_end)  # solver reads .physics, .payload_index, .ground_z, .t
        if self.output is not None and sol.t.size > 0:
            for k, tk in enumerate(sol.t):
                compute_snapshot(self.physics, float(tk), sol.y[:, k],
                                 solver.alpha, solver.beta, solver.quat_stab_k)
                self.output.log(self.physics, float(tk))
            self.output.flush()
            if self.output.auto_save_plots:
                self.output.save_plots(self.physics)
        return sol

    # run()/step() for the fixed-step HybridSolver are moved here from World.step
    # verbatim, operating on self.physics.bodies / self.physics.constraints and
    # self.t / self.ground_z. (Reproduce World.step's body here during implementation.)
```

> **Implementation note:** `solver.integrate` in SP1 took a `world` and read
> `world.bodies`, `world.payload_index`, `world.ground_z`, `world.t`, and the logger.
> Refactor `HybridIVPSolver.integrate` to read `sim.physics.*` for bodies/constraints
> and `sim.payload_index/ground_z/t` for termination, and to **stop logging inside the
> solver** (logging now lives in `Simulator.integrate_to`). Move `World.step` and
> `World.run` bodies into `Simulator` unchanged except for `self.bodies` →
> `self.physics.bodies` and logging via `self.output`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_simulator.py -v`
Expected: PASS

- [ ] **Step 5: Full regression checkpoint**

Run: `.venv/bin/pytest -q`
Expected: existing solver/ivp tests still pass (World still works via SP1 until Task 6 reparents it).

- [ ] **Step 6: Commit**

```bash
git add src/aerislab/core/simulator.py src/aerislab/core/solver.py tests/test_simulator.py
git commit -m "feat(core): Simulator owns time/loop/events; pure-replay logging"
```

---

### Task 6: World becomes a thin facade

**Files:**
- Modify: `src/aerislab/core/simulation.py`
- Test: `tests/test_world_facade.py`

`World` composes `PhysicsWorld` + `Simulator` + optional `OutputManager` and
delegates. Every attribute/method existing tests touch stays available.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_world_facade.py
import numpy as np
from aerislab.core import World
from aerislab.core.simulator import Simulator
from aerislab.core.world import PhysicsWorld
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity


def test_world_delegates_to_components():
    w = World(ground_z=-1000.0, payload_index=0)
    assert isinstance(w.physics, PhysicsWorld)
    assert isinstance(w.sim, Simulator)
    idx = w.add_body(RigidBody6DOF("b", 1.0, np.eye(3), np.array([0,0,5.0]), np.array([0,0,0,1.0])))
    assert idx == 0
    assert w.bodies is w.physics.bodies          # same list
    assert w.t == w.sim.t                          # delegated time
    w.add_global_force(Gravity(np.array([0,0,-9.81])))
    from aerislab.core.solver import HybridIVPSolver
    w.integrate_to(HybridIVPSolver(method="Radau", max_step=np.inf), t_end=0.5)
    assert w.t > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_world_facade.py -v`
Expected: FAIL with `AttributeError: 'World' object has no attribute 'physics'`

- [ ] **Step 3: Rewrite World as a facade**

Rewrite `World.__init__` to build the three components and delegate. Replace the
container/orchestration attributes with delegating properties and pass-through
methods:

```python
class World:
    def __init__(self, ground_z=0.0, payload_index=0, simulation_name=None,
                 output_dir=None, auto_timestamp=True, auto_save_plots=False,
                 atmosphere=None):
        self.physics = PhysicsWorld(atmosphere=atmosphere)
        self.output = None
        self._output_cfg = dict(output_dir=output_dir, auto_timestamp=auto_timestamp,
                                auto_save_plots=auto_save_plots)
        self.sim = Simulator(self.physics, None, payload_index, ground_z)
        if simulation_name is not None:
            self.enable_logging(simulation_name)

    # --- container delegation ---
    def add_body(self, b): return self.physics.add_body(b)
    def add_system(self, s): return self.physics.add_system(s)
    def add_global_force(self, f): return self.physics.add_global_force(f)
    def add_interaction_force(self, f): return self.physics.add_interaction_force(f)
    def add_constraint(self, c): return self.physics.add_constraint(c)
    def build_layout(self): return self.physics.build_layout()
    def pack_global_state(self): return self.physics.pack_global_state()
    def validate_constraints(self, **kw): return self.physics.validate_constraints(**kw)
    @property
    def WORLD(self): return self.physics.WORLD
    @property
    def bodies(self): return self.physics.bodies
    @property
    def constraints(self): return self.physics.constraints
    @property
    def systems(self): return self.physics.systems
    @property
    def atmosphere(self): return self.physics.atmosphere
    @property
    def aux_providers(self): return self.physics.aux_providers

    # --- orchestration delegation ---
    def set_termination_callback(self, fn): self.sim.set_termination_callback(fn)
    def run(self, solver, duration, dt, **kw): return self.sim.run(solver, duration, dt, **kw)
    def integrate_to(self, solver, t_end, **kw): return self.sim.integrate_to(solver, t_end, **kw)
    @property
    def t(self): return self.sim.t
    @property
    def t_touchdown(self): return self.sim.t_touchdown
    @property
    def payload_index(self): return self.sim.payload_index
    @property
    def ground_z(self): return self.sim.ground_z

    # --- output delegation ---
    def enable_logging(self, name=None):
        from aerislab.core.output import OutputManager
        self.output = OutputManager(name or self._output_cfg.get("name"),
                                    output_dir=self._output_cfg["output_dir"] or "output",
                                    auto_timestamp=self._output_cfg["auto_timestamp"],
                                    auto_save_plots=self._output_cfg["auto_save_plots"])
        self.sim.output = self.output
        return self.output.output_path
    @property
    def logger(self): return self.output.logger if self.output else None
    @property
    def output_path(self): return self.output.output_path if self.output else None
    def save_plots(self, bodies=None, show=False):
        return self.output.save_plots(self.physics, bodies=bodies, show=show)
    def get_energy(self):
        ...  # move the existing implementation, operating on self.physics
```

> **Implementation note:** move `get_energy` verbatim, swapping `self.bodies` →
> `self.physics.bodies` and `self.global_forces` → `self.physics.global_forces`.
> Keep `with_logging` classmethod delegating to `__init__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_world_facade.py -v`
Expected: PASS

- [ ] **Step 5: Full regression checkpoint (the key gate)**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check src/ && .venv/bin/mypy src/aerislab/`
Expected: ALL existing tests pass — the facade preserved behavior.

- [ ] **Step 6: Commit**

```bash
git add src/aerislab/core/simulation.py tests/test_world_facade.py
git commit -m "refactor(core): World becomes thin facade over PhysicsWorld/Simulator/OutputManager"
```

---

### Task 7: Per-component atol (optional scaling)

**Files:**
- Modify: `src/aerislab/core/simulator.py`, `src/aerislab/core/solver.py`
- Test: `tests/test_simulator.py`

`Simulator`/solver assemble an `atol` array per state slice: scalar default, with a
provider override via an optional `state_atol()` method.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulator.py  (append)
import numpy as np
from aerislab.core.simulator import build_atol_array
from aerislab.core.world import PhysicsWorld
from aerislab.dynamics.body import RigidBody6DOF


class _ScaledAux:
    def num_states(self): return 1
    def pack_state(self, out): out[0] = 0.0
    def unpack_state(self, y): pass
    def compute_derivatives(self, t): return np.zeros(1)
    def state_atol(self): return np.array([1e-3])


def test_build_atol_array_uses_overrides():
    pw = PhysicsWorld()
    pw.add_body(RigidBody6DOF("b", 1.0, np.eye(3), np.zeros(3), np.array([0,0,0,1.0])))
    pw.aux_providers.append(_ScaledAux())
    pw.build_layout()
    atol = build_atol_array(pw, default_atol=1e-9)
    assert atol.shape == (14,)
    assert np.allclose(atol[:13], 1e-9)   # body: default
    assert atol[13] == 1e-3               # aux override
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_simulator.py::test_build_atol_array_uses_overrides -v`
Expected: FAIL with `ImportError: cannot import name 'build_atol_array'`

- [ ] **Step 3: Write minimal implementation**

Add to `simulator.py`:

```python
def build_atol_array(physics, default_atol: float) -> np.ndarray:
    """Per-state absolute tolerance: scalar default, overridden by state_atol()."""
    atol = np.full(physics.num_states(), float(default_atol))
    for p, lo, hi in physics._layout:
        if hasattr(p, "state_atol"):
            atol[lo:hi] = np.asarray(p.state_atol(), dtype=float)
    return atol
```

Then in `HybridIVPSolver.integrate`, build `atol = build_atol_array(physics, self.atol)`
and pass it to `solve_ivp` instead of the scalar (only when any provider overrides;
otherwise keep the scalar to preserve current behavior).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_simulator.py::test_build_atol_array_uses_overrides -v`
Expected: PASS

- [ ] **Step 5: Full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/aerislab/core/simulator.py src/aerislab/core/solver.py tests/test_simulator.py
git commit -m "feat(core): optional per-component atol scaling for mixed state vector"
```

---

## Done criteria

- One physics implementation (`compute_snapshot`); solver and logger both use it.
- IVP logging no longer re-implements force/KKT evaluation and does not corrupt
  pure forces (CRIT-3 resolved for pure forces; stateful legacy models gone in SP3).
- `Simulator` and `OutputManager` are independent, testable units.
- `World` public API and behavior unchanged — full pre-existing suite passes.

## Deferred to sub-project 3

- Retire stateful legacy parachute models (the last remaining source of in-RHS
  mutation); NN model; `SlackTether`.
