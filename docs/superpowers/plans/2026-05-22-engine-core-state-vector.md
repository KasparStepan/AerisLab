# Engine Core — State-Vector Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the hardcoded "13 states per body" assumption from the solver, introduce composable `StateProvider` / `AuxDynamics` / `InertialProvider` protocols, support variable inertia (added mass) on the LHS, and stabilize the quaternion norm with a Baumgarte term — all while keeping existing scenarios bit-compatible.

**Architecture:** A new `PhysicsWorld` base class owns the physical container plus a dynamic global state-vector layout (each `StateProvider` declares `num_states()`; the world assigns contiguous slices). `World` becomes a subclass adding time/logging/orchestration, so the public API is unchanged. `RigidBody6DOF` implements the new protocols; its `state_derivative` carries the quaternion Baumgarte term and its mass-matrix methods optionally include an anisotropic added-mass tensor. `HybridIVPSolver.rhs` is rewritten to drive everything through the layout.

**Tech Stack:** Python 3, NumPy, SciPy (`solve_ivp`), pytest. SI units; scalar-last quaternions `[qx,qy,qz,qw]`.

**Reference spec:** [docs/superpowers/specs/2026-05-22-engine-core-state-vector-design.md](../specs/2026-05-22-engine-core-state-vector-design.md)

---

## File Structure

| File | Responsibility | New / Modified |
|---|---|---|
| `src/aerislab/core/protocols.py` | `StateProvider`, `AuxDynamics`, `InertialProvider` protocols | **New** |
| `src/aerislab/dynamics/body.py` | `RigidBody6DOF` implements protocols; added mass; `state_derivative` with quaternion Baumgarte | Modify |
| `src/aerislab/core/world.py` | `PhysicsWorld` base: container + state-vector layout | **New** |
| `src/aerislab/core/simulation.py` | `World(PhysicsWorld)` keeps orchestration/logging | Modify |
| `src/aerislab/core/solver.py` | `HybridIVPSolver.rhs` generalized via layout; `quat_stab_k` param | Modify |
| `src/aerislab/dynamics/forces.py` | `AddedMassFlux` force (`ṁ_added·v` term, generic) | Modify |
| `tests/test_protocols.py` | protocol conformance + layout roundtrip | **New** |
| `tests/test_body.py` | pack/unpack, `state_derivative`, added mass | Modify |
| `tests/test_solver_ivp.py` | generalized rhs, regression, aux-state integration | Modify |
| `tests/test_forces.py` | `AddedMassFlux` | Modify |

**Phasing (each phase leaves the suite green):**
- **Phase 1 (Tasks 1–4):** protocols + `RigidBody6DOF` conformance + quaternion Baumgarte — purely additive.
- **Phase 2 (Task 5):** added mass on the mass matrix — additive, zero-added-mass is bit-identical.
- **Phase 3 (Tasks 6–7):** `PhysicsWorld` extraction + dynamic layout.
- **Phase 4 (Task 8):** generalized IVP solver.
- **Phase 5 (Task 9):** `ṁ_added·v` force scaffolding.

Run the full suite with `.venv/bin/pytest` (or `pytest` if the venv is active).

---

## Phase 1 — Protocols and RigidBody6DOF conformance

### Task 1: Protocol module

**Files:**
- Create: `src/aerislab/core/protocols.py`
- Test: `tests/test_protocols.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_protocols.py
import numpy as np

from aerislab.core.protocols import StateProvider, AuxDynamics, InertialProvider


class _DummyAux:
    def num_states(self):
        return 2

    def pack_state(self, out):
        out[:] = [1.0, 2.0]

    def unpack_state(self, y):
        self._y = np.asarray(y).copy()

    def compute_derivatives(self, t):
        return np.array([0.0, 0.0])


def test_dummy_satisfies_stateprovider_and_auxdynamics():
    d = _DummyAux()
    assert isinstance(d, StateProvider)
    assert isinstance(d, AuxDynamics)
    assert not isinstance(d, InertialProvider)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_protocols.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aerislab.core.protocols'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aerislab/core/protocols.py
"""
Composable protocols for the dynamics engine.

Each protocol answers one orthogonal question, so any physics object is a
combination of the relevant subset (see spec 2026-05-22-engine-core-state-vector):

- StateProvider   : "Do I carry states that the integrator advances in time?"
- AuxDynamics     : "Do I compute my own derivatives locally (not via the KKT solve)?"
- InertialProvider: "Do I contribute to the left-hand-side mass/inertia?"

The Force protocol (right-hand-side generalized force) lives in dynamics.forces.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class StateProvider(Protocol):
    """Anything that contributes states to the global integration vector y."""

    def num_states(self) -> int:
        """Number of state variables this provider owns."""
        ...

    def pack_state(self, out: NDArray[np.float64]) -> None:
        """Write current states into ``out`` (a view of this provider's slice)."""
        ...

    def unpack_state(self, y: NDArray[np.float64]) -> None:
        """Read states from ``y`` (a view of this provider's slice)."""
        ...


@runtime_checkable
class AuxDynamics(Protocol):
    """A Tier-2 provider that computes its own state derivatives locally."""

    def compute_derivatives(self, t: float) -> NDArray[np.float64]:
        """Return d(state)/dt for this provider's states (length == num_states())."""
        ...


@runtime_checkable
class InertialProvider(Protocol):
    """Anything contributing a 6x6 generalized mass matrix to the LHS."""

    def mass_matrix_world(self) -> NDArray[np.float64]:
        """Return the 6x6 generalized mass matrix in world frame (incl. added mass)."""
        ...

    def inv_mass_matrix_world(self) -> NDArray[np.float64]:
        """Return the inverse 6x6 generalized mass matrix in world frame."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_protocols.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/core/protocols.py tests/test_protocols.py
git commit -m "feat(core): add StateProvider/AuxDynamics/InertialProvider protocols"
```

---

### Task 2: RigidBody6DOF implements StateProvider

**Files:**
- Modify: `src/aerislab/dynamics/body.py` (add three methods to `RigidBody6DOF`)
- Test: `tests/test_body.py`

State order is `[p(3), q(4), v(3), w(3)]` = 13, matching the existing
`HybridIVPSolver._pack` (`[*b.p, *b.q, *b.v, *b.w]`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_body.py  (append)
import numpy as np
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.core.protocols import StateProvider


def _make_body():
    return RigidBody6DOF(
        "b", 2.0, np.diag([1.0, 2.0, 3.0]),
        position=np.array([1.0, 2.0, 3.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity=np.array([4.0, 5.0, 6.0]),
        angular_velocity=np.array([0.1, 0.2, 0.3]),
    )


def test_body_is_state_provider():
    assert isinstance(_make_body(), StateProvider)
    assert _make_body().num_states() == 13


def test_body_pack_unpack_roundtrip():
    b = _make_body()
    out = np.empty(13)
    b.pack_state(out)
    expected = np.concatenate([b.p, b.q, b.v, b.w])
    assert np.allclose(out, expected)

    # Mutate via unpack, then re-pack and compare
    new = np.arange(13, dtype=float) + 0.5
    b.unpack_state(new)
    out2 = np.empty(13)
    b.pack_state(out2)
    assert np.allclose(out2, new)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_body.py::test_body_is_state_provider tests/test_body.py::test_body_pack_unpack_roundtrip -v`
Expected: FAIL with `AttributeError: 'RigidBody6DOF' object has no attribute 'num_states'`

- [ ] **Step 3: Write minimal implementation**

Add to `RigidBody6DOF` (in `src/aerislab/dynamics/body.py`), after `clear_forces`:

```python
    # --- StateProvider protocol ---------------------------------------------

    def num_states(self) -> int:
        """13 IVP states: [p(3), q(4), v(3), w(3)]."""
        return 13

    def pack_state(self, out: NDArray[np.float64]) -> None:
        """Write [p, q, v, w] into the 13-element view ``out``."""
        out[0:3] = self.p
        out[3:7] = self.q
        out[7:10] = self.v
        out[10:13] = self.w

    def unpack_state(self, y: NDArray[np.float64]) -> None:
        """Read [p, q, v, w] from the 13-element view ``y``."""
        self.p[:] = y[0:3]
        self.q[:] = y[3:7]
        self.v[:] = y[7:10]
        self.w[:] = y[10:13]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_body.py::test_body_is_state_provider tests/test_body.py::test_body_pack_unpack_roundtrip -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/dynamics/body.py tests/test_body.py
git commit -m "feat(body): RigidBody6DOF implements StateProvider (pack/unpack)"
```

---

### Task 3: state_derivative with quaternion Baumgarte stabilization

**Files:**
- Modify: `src/aerislab/dynamics/body.py`
- Test: `tests/test_body.py`

`state_derivative` returns the 13-element `ydot` slice given the body's
generalized acceleration from the KKT solve. The quaternion derivative carries a
Baumgarte term `−k·(qᵀq−1)·q` so the norm error relaxes as `ṡ ≈ −2k·s` for
`s = |q|²−1`. With `k=0` it reduces to the plain kinematic derivative.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_body.py  (append)
from aerislab.dynamics.body import quat_derivative


def test_state_derivative_layout_and_plain_quat():
    b = _make_body()
    a_lin = np.array([7.0, 8.0, 9.0])
    a_ang = np.array([0.4, 0.5, 0.6])
    yd = b.state_derivative(a_lin, a_ang, quat_stab_k=0.0)
    assert yd.shape == (13,)
    assert np.allclose(yd[0:3], b.v)                       # dp/dt = v
    assert np.allclose(yd[3:7], quat_derivative(b.q, b.w)) # plain qdot when k=0
    assert np.allclose(yd[7:10], a_lin)
    assert np.allclose(yd[10:13], a_ang)


def test_quaternion_baumgarte_pulls_norm_toward_one():
    # Deliberately de-normalized quaternion (norm^2 = 1.21)
    b = RigidBody6DOF("b", 1.0, np.eye(3),
                      position=np.zeros(3),
                      orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    b.q[:] = np.array([0.0, 0.0, 0.0, 1.1])  # bypass __init__ normalization
    k = 10.0
    yd = b.state_derivative(np.zeros(3), np.zeros(3), quat_stab_k=k)
    s = float(b.q @ b.q - 1.0)                 # current norm error (= 0.21)
    sdot = float(2.0 * b.q @ yd[3:7])          # d/dt(qᵀq)
    # With w = 0, sdot should equal -2 k s (1 + s); sign must be corrective (<0)
    assert sdot < 0.0
    assert np.isclose(sdot, -2.0 * k * s * (1.0 + s))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_body.py::test_state_derivative_layout_and_plain_quat tests/test_body.py::test_quaternion_baumgarte_pulls_norm_toward_one -v`
Expected: FAIL with `AttributeError: 'RigidBody6DOF' object has no attribute 'state_derivative'`

- [ ] **Step 3: Write minimal implementation**

Add to `RigidBody6DOF` (after `unpack_state`):

```python
    def state_derivative(
        self,
        a_lin: NDArray[np.float64],
        a_ang: NDArray[np.float64],
        quat_stab_k: float = 0.0,
    ) -> NDArray[np.float64]:
        """
        Assemble the 13-element IVP derivative [dp, dq, dv, dw].

        The linear/angular accelerations come from the global KKT solve. The
        quaternion derivative includes a Baumgarte stabilization term that drives
        the unit-norm error s = |q|^2 - 1 toward zero (sdot = -2k s (1+s)),
        keeping |q|=1 without the discontinuity of in-RHS renormalization.

        Parameters
        ----------
        a_lin, a_ang : NDArray[np.float64]
            Linear/angular acceleration in world frame (3,) each.
        quat_stab_k : float
            Quaternion Baumgarte gain [1/s]. 0 disables (plain kinematics).
        """
        ydot = np.empty(13, dtype=np.float64)
        ydot[0:3] = self.v
        qdot = quat_derivative(self.q, self.w)
        if quat_stab_k != 0.0:
            qdot = qdot - quat_stab_k * (self.q @ self.q - 1.0) * self.q
        ydot[3:7] = qdot
        ydot[7:10] = a_lin
        ydot[10:13] = a_ang
        return ydot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_body.py::test_state_derivative_layout_and_plain_quat tests/test_body.py::test_quaternion_baumgarte_pulls_norm_toward_one -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/dynamics/body.py tests/test_body.py
git commit -m "feat(body): state_derivative with quaternion Baumgarte stabilization"
```

---

### Task 4: Confirm InertialProvider conformance (no behavior change)

`RigidBody6DOF` already has `mass_matrix_world` and `inv_mass_matrix_world`, so it
already satisfies `InertialProvider`. This task just locks that in with a test
before Task 5 changes those methods.

**Files:**
- Test: `tests/test_body.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_body.py  (append)
from aerislab.core.protocols import InertialProvider


def test_body_is_inertial_provider():
    assert isinstance(_make_body(), InertialProvider)
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.venv/bin/pytest tests/test_body.py::test_body_is_inertial_provider -v`
Expected: PASS immediately (methods already exist). If it FAILS, the protocol import path is wrong — fix the import, do not add methods.

- [ ] **Step 3: (none — conformance already holds)**

- [ ] **Step 4: Run the full body test file**

Run: `.venv/bin/pytest tests/test_body.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add tests/test_body.py
git commit -m "test(body): lock in InertialProvider conformance"
```

---

## Phase 2 — Added mass on the mass matrix

### Task 5: Anisotropic added mass on RigidBody6DOF

**Files:**
- Modify: `src/aerislab/dynamics/body.py` (`__init__`, `mass_matrix_world`, `inv_mass_matrix_world`, new `set_added_mass`)
- Test: `tests/test_body.py`

Added mass is stored as a per-axis local diagonal `[m_t, m_t, m_a]` (z is the
symmetry axis) plus an optional local diagonal added inertia. The world-frame
effective mass is `R·diag(m+m_axes)·Rᵀ`, inverted analytically (no `np.linalg.inv`
on the translational block). **When no added mass is set, both methods must return
exactly the current values** (regression guarantee), so the zero case takes the
existing code path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_body.py  (append)
def test_added_mass_zero_matches_baseline():
    b = _make_body()
    # No added mass set -> identical to plain block-diagonal forms
    M = b.mass_matrix_world()
    assert np.allclose(M[0:3, 0:3], b.mass * np.eye(3))
    W = b.inv_mass_matrix_world()
    assert np.allclose(W[0:3, 0:3], (1.0 / b.mass) * np.eye(3))


def test_added_mass_identity_orientation_translational_block():
    b = RigidBody6DOF("b", 2.0, np.eye(3),
                      position=np.zeros(3),
                      orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    b.set_added_mass(m_transverse=1.0, m_axial=8.0)  # local diag [1,1,8]
    M = b.mass_matrix_world()
    assert np.allclose(np.diag(M[0:3, 0:3]), [3.0, 3.0, 10.0])  # 2 + [1,1,8]
    W = b.inv_mass_matrix_world()
    assert np.allclose(np.diag(W[0:3, 0:3]), [1/3.0, 1/3.0, 1/10.0])
    # M @ W should be identity on the translational block
    assert np.allclose(M[0:3, 0:3] @ W[0:3, 0:3], np.eye(3))


def test_added_mass_rotated_is_consistent_inverse():
    b = RigidBody6DOF("b", 2.0, np.diag([1.0, 1.0, 2.0]),
                      position=np.zeros(3),
                      orientation=np.array([0.0, 0.0, np.sin(0.3), np.cos(0.3)]))
    b.set_added_mass(m_transverse=1.0, m_axial=8.0)
    M = b.mass_matrix_world()
    W = b.inv_mass_matrix_world()
    assert np.allclose(M @ W, np.eye(6), atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_body.py::test_added_mass_identity_orientation_translational_block -v`
Expected: FAIL with `AttributeError: 'RigidBody6DOF' object has no attribute 'set_added_mass'`

- [ ] **Step 3: Write minimal implementation**

In `RigidBody6DOF.__init__`, after the force accumulators block, add:

```python
        # Added (apparent) mass, anisotropic, in the body/local frame.
        # _added_mass_local = [m_transverse, m_transverse, m_axial]; z = symmetry axis.
        self._has_added_mass: bool = False
        self._added_mass_local = np.zeros(3, dtype=np.float64)
        self._added_inertia_local = np.zeros(3, dtype=np.float64)
```

Add the setter (after `clear_forces`):

```python
    def set_added_mass(
        self,
        m_transverse: float = 0.0,
        m_axial: float = 0.0,
        added_inertia_local: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Set anisotropic added (apparent) mass in the local frame.

        Parameters
        ----------
        m_transverse : float
            Added mass on the two transverse axes (local x, y) [kg].
        m_axial : float
            Added mass on the symmetry axis (local z) [kg].
        added_inertia_local : NDArray[np.float64] | None
            Optional local diagonal added inertia (3,) [kg·m²].
        """
        self._added_mass_local[:] = [m_transverse, m_transverse, m_axial]
        if added_inertia_local is not None:
            self._added_inertia_local[:] = np.asarray(added_inertia_local, dtype=np.float64)
        self._has_added_mass = bool(
            np.any(self._added_mass_local) or np.any(self._added_inertia_local)
        )
```

Replace `mass_matrix_world` and `inv_mass_matrix_world` bodies:

```python
    def mass_matrix_world(self) -> NDArray[np.float64]:
        """6x6 generalized mass matrix in world frame, including added mass."""
        M = np.zeros((6, 6), dtype=np.float64)
        R = self.rotation_world()
        if self._has_added_mass:
            m_axes = self.mass + self._added_mass_local            # (3,)
            M[0:3, 0:3] = R @ np.diag(m_axes) @ R.T
            I_eff_local = self.I_body + np.diag(self._added_inertia_local)
            M[3:6, 3:6] = R @ I_eff_local @ R.T
        else:
            M[0:3, 0:3] = self.mass * np.eye(3)
            M[3:6, 3:6] = R @ self.I_body @ R.T
        return M

    def inv_mass_matrix_world(self) -> NDArray[np.float64]:
        """Inverse 6x6 generalized mass matrix in world frame, including added mass."""
        W = np.zeros((6, 6), dtype=np.float64)
        R = self.rotation_world()
        # Fixed / negligible-mass anchors keep zero inverse (no acceleration).
        if self._has_added_mass and not self.fixed and self.inv_mass != 0.0:
            m_axes = self.mass + self._added_mass_local
            W[0:3, 0:3] = R @ np.diag(1.0 / m_axes) @ R.T
            I_eff_local = self.I_body + np.diag(self._added_inertia_local)
            W[3:6, 3:6] = R @ np.linalg.inv(I_eff_local) @ R.T
        else:
            W[0:3, 0:3] = self.inv_mass * np.eye(3)
            W[3:6, 3:6] = R @ self.I_body_inv @ R.T
        return W
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_body.py -v`
Expected: PASS (all, including the regression tests from Task 4)

- [ ] **Step 5: Run the full suite (regression checkpoint)**

Run: `.venv/bin/pytest -q`
Expected: PASS — no existing test changes behavior (added mass defaults off).

- [ ] **Step 6: Commit**

```bash
git add src/aerislab/dynamics/body.py tests/test_body.py
git commit -m "feat(body): anisotropic added mass on world mass matrix"
```

---

## Phase 3 — PhysicsWorld extraction and dynamic state-vector layout

### Task 6: Extract PhysicsWorld base class

**Files:**
- Create: `src/aerislab/core/world.py`
- Modify: `src/aerislab/core/simulation.py` (make `World` subclass `PhysicsWorld`)
- Test: `tests/test_simulation.py`

This is a **mechanical move**, not a rewrite. Create `PhysicsWorld` holding the
physical-container responsibilities; `World` keeps orchestration/logging by
subclassing it. The public `World` API and behavior must be unchanged.

Move the following from `World` (in `simulation.py`) **into** `PhysicsWorld` (in
`world.py`), verbatim:

- the container attributes initialized in `__init__`: `bodies`, `global_forces`,
  `interaction_forces`, `constraints`, `_world_index`, `force_breakdown`,
  `systems`, `atmosphere`;
- methods: `WORLD` (property), `add_body`, `add_global_force`,
  `add_interaction_force`, `add_constraint`, `add_system`, `_constraint_label`,
  `validate_constraints`.

`World.__init__` calls `super().__init__(atmosphere=atmosphere)` then sets its own
orchestration/logging attributes (`payload_index`, `ground_z`, `t`,
`t_touchdown`, `termination_callback`, output config, `logger`, etc.) and the
logging methods, `run`, `step`, `integrate_to`, `save_plots`, `get_energy` stay on
`World`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulation.py  (append)
import numpy as np
from aerislab.core.world import PhysicsWorld
from aerislab.core.simulation import World
from aerislab.dynamics.body import RigidBody6DOF


def test_world_is_physicsworld_subclass():
    assert issubclass(World, PhysicsWorld)


def test_physicsworld_holds_container():
    pw = PhysicsWorld()
    b = RigidBody6DOF("b", 1.0, np.eye(3), np.zeros(3), np.array([0,0,0,1.0]))
    idx = pw.add_body(b)
    assert idx == 0
    assert pw.bodies[0] is b
    assert pw.atmosphere is not None  # defaults to FastISA
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_simulation.py::test_world_is_physicsworld_subclass -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aerislab.core.world'`

- [ ] **Step 3: Create PhysicsWorld and reparent World**

Create `src/aerislab/core/world.py`:

```python
"""
PhysicsWorld: the pure physical container for a multi-body simulation.

Owns bodies, systems, forces, constraints, and the atmosphere service, plus the
dynamic global state-vector layout. Knows nothing about time, logging, or output —
those live in core.simulation.World, which subclasses this.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import Constraint
from aerislab.models.atmosphere.isa import FastISA, AtmosphereModel

if TYPE_CHECKING:
    from aerislab.components.system import System as SystemType


class PhysicsWorld:
    """Container for bodies, systems, forces, constraints, and atmosphere."""

    def __init__(self, atmosphere: AtmosphereModel = None) -> None:
        self.bodies: list[RigidBody6DOF] = []
        self.global_forces: list = []
        self.interaction_forces: list = []
        self.constraints: list[Constraint] = []
        self.systems: list[SystemType] = []
        self.aux_providers: list = []          # Tier-2 StateProviders (AuxDynamics)
        self._world_index: int | None = None
        self.force_breakdown: dict[str, NDArray] = {}
        self.atmosphere = atmosphere if atmosphere is not None else FastISA()
        self._layout: list[tuple[object, int, int]] = []
        self._slice_of: dict[int, tuple[int, int]] = {}
        self._n_states: int = 0
```

Then **move** the following methods from `World` into `PhysicsWorld` verbatim
(cut from `simulation.py`, paste into `world.py`, fix imports as needed):
`WORLD`, `add_body`, `add_global_force`, `add_interaction_force`,
`add_constraint`, `add_system`, `_constraint_label`, `validate_constraints`.

In `src/aerislab/core/simulation.py`:
- add `from aerislab.core.world import PhysicsWorld`;
- change the class declaration to `class World(PhysicsWorld):`;
- delete the moved methods and the moved `__init__` container lines;
- at the top of `World.__init__`, call `super().__init__(atmosphere=atmosphere)`
  before setting orchestration attributes, and remove the now-duplicated
  container initializations (`self.bodies = []`, ..., `self.atmosphere = ...`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_simulation.py -v`
Expected: PASS

- [ ] **Step 5: Full regression checkpoint**

Run: `.venv/bin/pytest -q`
Expected: PASS — `World` behaves identically; only its base class changed.

- [ ] **Step 6: Commit**

```bash
git add src/aerislab/core/world.py src/aerislab/core/simulation.py tests/test_simulation.py
git commit -m "refactor(core): extract PhysicsWorld container base from World"
```

---

### Task 7: Dynamic state-vector layout on PhysicsWorld

**Files:**
- Modify: `src/aerislab/core/world.py`
- Test: `tests/test_protocols.py`

`state_providers` = bodies first (so KKT body indices stay aligned), then
`aux_providers`. `_build_layout` assigns contiguous slices; `pack/unpack` delegate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_protocols.py  (append)
from aerislab.core.world import PhysicsWorld
from aerislab.dynamics.body import RigidBody6DOF


def _body(name):
    return RigidBody6DOF(name, 1.0, np.eye(3), np.zeros(3), np.array([0,0,0,1.0]))


def test_layout_sizes_and_slices():
    pw = PhysicsWorld()
    pw.add_body(_body("a"))
    pw.add_body(_body("b"))
    pw.aux_providers.append(_DummyAux())   # 2 states
    pw.build_layout()
    assert pw.num_states() == 13 + 13 + 2
    assert pw.state_slice(pw.bodies[0]) == (0, 13)
    assert pw.state_slice(pw.bodies[1]) == (13, 26)
    assert pw.state_slice(pw.aux_providers[0]) == (26, 28)


def test_pack_unpack_global_roundtrip():
    pw = PhysicsWorld()
    pw.add_body(_body("a"))
    pw.aux_providers.append(_DummyAux())
    pw.build_layout()
    y = pw.pack_global_state()
    assert y.shape == (15,)
    y2 = np.arange(15, dtype=float) + 0.25
    pw.unpack_global_state(y2)
    assert np.allclose(pw.pack_global_state(), y2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_protocols.py::test_layout_sizes_and_slices -v`
Expected: FAIL with `AttributeError: 'PhysicsWorld' object has no attribute 'build_layout'`

- [ ] **Step 3: Write minimal implementation**

Add to `PhysicsWorld` (in `world.py`):

```python
    @property
    def state_providers(self) -> list:
        """All StateProviders: bodies first (KKT-aligned), then aux providers."""
        return [*self.bodies, *self.aux_providers]

    def build_layout(self) -> None:
        """(Re)build the global state-vector layout from current providers."""
        self._layout = []
        self._slice_of = {}
        offset = 0
        for p in self.state_providers:
            n = int(p.num_states())
            self._layout.append((p, offset, offset + n))
            self._slice_of[id(p)] = (offset, offset + n)
            offset += n
        self._n_states = offset

    def num_states(self) -> int:
        """Total length of the global state vector y."""
        return self._n_states

    def state_slice(self, provider) -> tuple[int, int]:
        """Return (lo, hi) slice bounds for a provider in the global vector."""
        return self._slice_of[id(provider)]

    def pack_global_state(self) -> NDArray[np.float64]:
        """Assemble the global state vector y from all providers."""
        y = np.empty(self._n_states, dtype=np.float64)
        for p, lo, hi in self._layout:
            p.pack_state(y[lo:hi])
        return y

    def unpack_global_state(self, y: NDArray[np.float64]) -> None:
        """Distribute y back into all providers."""
        for p, lo, hi in self._layout:
            p.unpack_state(y[lo:hi])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_protocols.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/core/world.py tests/test_protocols.py
git commit -m "feat(core): dynamic global state-vector layout on PhysicsWorld"
```

---

## Phase 4 — Generalized IVP solver

### Task 8: Rewrite HybridIVPSolver to use the layout

**Files:**
- Modify: `src/aerislab/core/solver.py` (`HybridIVPSolver.__init__`, `integrate`)
- Test: `tests/test_solver_ivp.py`

Add a `quat_stab_k` parameter. Replace the `13*k` packing/unpacking and the
in-RHS `quat_normalize` calls with layout-driven pack/unpack and
`state_derivative`. The old `_pack`/`_unpack_to_world` are replaced by the world's
pack/unpack; keep `touchdown_event` and logging working (logging is reworked fully
in sub-project 2 — here just keep it functional and behavior-preserving).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_solver_ivp.py  (append)
import numpy as np
import pytest
from aerislab.core import World
from aerislab.core.solver import HybridIVPSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity
from aerislab.core.protocols import AuxDynamics


def test_freefall_regression_with_generalized_solver():
    z0 = 20.0
    w = World(ground_z=-1000.0, payload_index=0)
    b = RigidBody6DOF("b", 1.0, np.eye(3), np.array([0,0,z0]), np.array([0,0,0,1.0]))
    w.add_body(b)
    w.add_global_force(Gravity(np.array([0,0,-9.81])))
    ivp = HybridIVPSolver(method="Radau", rtol=1e-8, atol=1e-10, max_step=np.inf)
    w.integrate_to(ivp, t_end=1.234)
    z_exact = z0 - 0.5 * 9.81 * 1.234**2
    assert abs(w.bodies[0].p[2] - z_exact) < 1e-5


class _DecayState:
    """Aux provider with one state x, dx/dt = -x  (analytic: x(t)=x0 e^-t)."""
    def __init__(self, x0):
        self.x = float(x0)
    def num_states(self): return 1
    def pack_state(self, out): out[0] = self.x
    def unpack_state(self, y): self.x = float(y[0])
    def compute_derivatives(self, t): return np.array([-self.x])


def test_aux_state_integrates_alongside_bodies():
    w = World(ground_z=-1000.0, payload_index=0)
    b = RigidBody6DOF("b", 1.0, np.eye(3), np.array([0,0,5.0]), np.array([0,0,0,1.0]))
    w.add_body(b)
    w.add_global_force(Gravity(np.array([0,0,-9.81])))
    aux = _DecayState(2.0)
    assert isinstance(aux, AuxDynamics)
    w.aux_providers.append(aux)
    ivp = HybridIVPSolver(method="Radau", rtol=1e-9, atol=1e-11, max_step=np.inf)
    w.integrate_to(ivp, t_end=1.0)
    assert np.isclose(aux.x, 2.0 * np.exp(-1.0), rtol=1e-5)


def test_rhs_is_pure_no_state_mutation():
    w = World(ground_z=-1000.0, payload_index=0)
    b = RigidBody6DOF("b", 1.0, np.eye(3), np.array([0,0,5.0]), np.array([0,0,0,1.0]))
    w.add_body(b)
    w.add_global_force(Gravity(np.array([0,0,-9.81])))
    ivp = HybridIVPSolver(method="Radau", alpha=0.0, beta=0.0)
    w.build_layout()
    rhs = ivp._make_rhs(w)
    y = w.pack_global_state()
    yd1 = rhs(0.3, y.copy())
    snapshot = w.pack_global_state().copy()
    yd2 = rhs(0.3, y.copy())
    assert np.allclose(yd1, yd2)                          # deterministic
    assert np.allclose(w.pack_global_state(), snapshot)   # no hidden mutation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_solver_ivp.py::test_aux_state_integrates_alongside_bodies tests/test_solver_ivp.py::test_rhs_is_pure_no_state_mutation -v`
Expected: FAIL (`_make_rhs` missing / aux states not integrated).

- [ ] **Step 3: Write the implementation**

In `HybridIVPSolver.__init__`, add the parameter (after `beta`):

```python
        quat_stab_k: float = 0.0,
```
and store it:
```python
        self.quat_stab_k = float(quat_stab_k)
```

Add a `_make_rhs` method that builds the layout-driven RHS (extracted so it is
unit-testable), and call it from `integrate`:

```python
    def _make_rhs(self, world):
        """Build the layout-driven RHS closure for solve_ivp (pure: no state mutation)."""
        bodies = world.bodies
        constraints = world.constraints

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            world.unpack_global_state(y)

            for b in bodies:
                b.clear_forces()
            for system in world.systems:
                system.update_all_states(t, 0.0)
                system.apply_all_forces(t)
            for b in bodies:
                for fb in b.per_body_forces:
                    fb.apply(b, t)
            for fg in world.global_forces:
                for b in bodies:
                    fg.apply(b, t)
            for fpair in world.interaction_forces:
                fpair.apply_pair(t)

            Minv, J, F, rhs_v, _ = assemble_system(
                bodies, constraints, self.alpha, self.beta
            )
            a, _ = solve_kkt(Minv, J, F, rhs_v)

            ydot = np.empty_like(y)
            for i, b in enumerate(bodies):
                lo, hi = world.state_slice(b)
                ydot[lo:hi] = b.state_derivative(
                    a[6*i:6*i+3], a[6*i+3:6*i+6], quat_stab_k=self.quat_stab_k
                )
            for comp in world.aux_providers:
                lo, hi = world.state_slice(comp)
                ydot[lo:hi] = comp.compute_derivatives(t)
            return ydot

        return rhs
```

In `integrate`, replace the body of the method so it: builds the layout, makes the
RHS, packs `y0` from the world, and defines `touchdown_event` against the layout
slice. Replace the old `_pack(bodies)` with `world.pack_global_state()` and the old
final `_unpack_to_world(sol.y[:, -1], bodies)` with
`world.unpack_global_state(sol.y[:, -1])`:

```python
        world.build_layout()
        rhs = self._make_rhs(world)

        payload = world.bodies[world.payload_index]
        z_lo, _ = world.state_slice(payload)  # payload position z lives at z_lo + 2

        def touchdown_event(t: float, y: NDArray[np.float64]) -> float:
            return float(y[z_lo + 2] - world.ground_z)
        touchdown_event.terminal = True   # type: ignore[attr-defined]
        touchdown_event.direction = -1.0  # type: ignore[attr-defined]

        y0 = world.pack_global_state()
```

Keep the existing `solve_ivp_kwargs` block, the `solve_ivp(...)` call, and the
"update world with final state" / `t_touchdown` lines, but change the final unpack
to `world.unpack_global_state(sol.y[:, -1])`. The existing logging replay block
(lines that re-apply forces per `sol.t` sample) stays for now — it is replaced
wholesale in sub-project 2; only swap its `self._unpack_to_world(sol.y[:, k], bodies)`
calls for `world.unpack_global_state(sol.y[:, k])`.

Delete `_pack` and `_unpack_to_world` once no longer referenced.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_solver_ivp.py -v`
Expected: PASS (regression free-fall + aux state + purity).

- [ ] **Step 5: Full regression checkpoint**

Run: `.venv/bin/pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aerislab/core/solver.py tests/test_solver_ivp.py
git commit -m "feat(solver): generalized layout-driven IVP rhs + quaternion Baumgarte"
```

---

## Phase 5 — Added-mass momentum-flux force

### Task 9: AddedMassFlux force (`ṁ_added·v` term)

**Files:**
- Modify: `src/aerislab/dynamics/forces.py`
- Test: `tests/test_forces.py`

The LHS `M_added·a` term is handled by Task 5. This task adds the RHS reaction
`F = −ṁ_added·v`, where `ṁ_added` is supplied by a callable (in sub-project 3 this
will read `V̇_air` from the NN — here it is a generic source so the mechanism is
testable in isolation). Direction opposes velocity; zero at zero `ṁ`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forces.py  (append)
import numpy as np
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import AddedMassFlux


def _body_with_velocity(v):
    b = RigidBody6DOF("b", 1.0, np.eye(3), np.zeros(3), np.array([0,0,0,1.0]),
                      linear_velocity=np.array(v, dtype=float))
    return b


def test_added_mass_flux_opposes_velocity():
    b = _body_with_velocity([3.0, 0.0, 0.0])
    f = AddedMassFlux(mdot_func=lambda t, body: 2.0)  # ṁ = 2 kg/s
    f.apply(b, t=0.0)
    # F = -ṁ v = -[6,0,0]
    assert np.allclose(b.f, [-6.0, 0.0, 0.0])


def test_added_mass_flux_zero_rate_no_force():
    b = _body_with_velocity([3.0, 0.0, 0.0])
    AddedMassFlux(mdot_func=lambda t, body: 0.0).apply(b, t=0.0)
    assert np.allclose(b.f, [0.0, 0.0, 0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_forces.py::test_added_mass_flux_opposes_velocity -v`
Expected: FAIL with `ImportError: cannot import name 'AddedMassFlux'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/aerislab/dynamics/forces.py`:

```python
class AddedMassFlux:
    """
    Momentum-flux reaction from a time-varying added mass: F = -ṁ_added · v.

    This is the right-hand-side half of the added-mass force. The acceleration-
    proportional half (m_added · a) is carried on the left-hand side by the body's
    mass matrix (see RigidBody6DOF.set_added_mass), so the two halves share a single
    source of ṁ_added.

    Parameters
    ----------
    mdot_func : Callable[[float, RigidBody6DOF], float]
        Returns the added-mass rate ṁ_added [kg/s] at time t for the body.
        (In the parachute model this derives from V̇_air; here it is generic.)
    """
    def __init__(self, mdot_func: Callable[[float, RigidBody6DOF], float]) -> None:
        self.mdot_func = mdot_func

    def apply(self, body: RigidBody6DOF, t: float | None = None) -> None:
        tval = 0.0 if t is None else float(t)
        mdot = float(self.mdot_func(tval, body))
        if mdot == 0.0:
            return
        body.apply_force(-mdot * body.v, label="added_mass_flux")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_forces.py -v`
Expected: PASS

- [ ] **Step 5: Full suite + lint + type check**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check src/ && .venv/bin/mypy src/aerislab/`
Expected: PASS / no new errors.

- [ ] **Step 6: Commit**

```bash
git add src/aerislab/dynamics/forces.py tests/test_forces.py
git commit -m "feat(forces): AddedMassFlux momentum-flux reaction (ṁ_added·v)"
```

---

## Done criteria

- Solver contains no hardcoded `13`; `num_states()` drives the layout.
- A body plus an arbitrary `AuxDynamics` provider integrate together (verified by
  the analytic decay test).
- `rhs` is pure: calling it twice gives identical output and mutates no hidden state.
- Added mass appears on the LHS (anisotropic, analytic inverse) and its `ṁ·v`
  reaction on the RHS, both from one source of `ṁ`.
- Quaternion norm error decays via the Baumgarte term; no in-RHS renormalization.
- `World` public API unchanged; all pre-existing tests pass.

## Deferred to later sub-projects

- **Sub-project 2:** `Simulator`/`OutputManager` split; "pure snapshot replay"
  logging (removes the double-RHS replay block left intact in Task 8);
  per-component `atol` scaling.
- **Sub-project 3:** retire legacy parachute models (keep one analytic baseline);
  NN drag + `V_air` model wiring `set_added_mass` and `AddedMassFlux` to real
  `V̇_air`; anisotropic tensor from canopy geometry; slack-tether lines.
