# Parachute NN Model / Analytic Baseline / Slack Tether Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **⚠ PROVISIONAL:** Targets the codebase **after sub-projects 1 and 2**. Depends on: `RigidBody6DOF.set_added_mass`, `AddedMassFlux` (SP1); `StateProvider`/`AuxDynamics` protocols, `PhysicsWorld.aux_providers`, `compute_snapshot` purity (SP1/SP2). Line numbers omitted; verify anchors before each task.

**Goal:** Replace the stateful legacy parachute models with a pure, state-vector model that exposes `V̇_air` + `Cd`, drives added mass from `V_air`, ships an analytic baseline sharing the same interface, and uses a tension-only `SlackTether` for lines instead of a rigid `DistanceConstraint`.

**Architecture:** A `ParachuteModel` interface (`StateProvider` + `AuxDynamics` + drag `Force` + added-mass wiring). `AnalyticParachute` implements it in closed form (baseline + V&V reference); `NeuralParachute` implements it via ONNX. Geometry helpers map scalar `V_air` to projected area and an anisotropic added-mass tensor. `SlackTether` is a one-sided force.

**Tech Stack:** Python 3, NumPy, SciPy, pytest; optional `onnxruntime` for the NN backend.

**Reference spec:** [docs/superpowers/specs/2026-05-22-parachute-nn-model-design.md](../specs/2026-05-22-parachute-nn-model-design.md)

---

## File Structure

| File | Responsibility | New / Modified |
|---|---|---|
| `src/aerislab/dynamics/forces.py` | `SlackTether` tension-only force | Modify |
| `src/aerislab/models/aerodynamics/parachute_geometry.py` | `V_air` → area, added-mass tensor | **New** |
| `src/aerislab/models/aerodynamics/parachute_base.py` | `ParachuteModel` interface + shared drag/added-mass | **New** |
| `src/aerislab/models/aerodynamics/analytic_parachute.py` | closed-form baseline | **New** |
| `src/aerislab/models/aerodynamics/neural_parachute.py` | ONNX backend | **New** |
| `src/aerislab/components/parachute.py` | compose model into component; register aux_provider | Modify |
| `src/aerislab/models/aerodynamics/not_in_use/` | move legacy `parachute_models.py` here | Move |
| `tests/test_slack_tether.py`, `tests/models/test_parachute_model.py` | tests | **New** |

---

### Task 1: SlackTether force (tension-only)

**Files:**
- Modify: `src/aerislab/dynamics/forces.py`
- Test: `tests/test_slack_tether.py`

Force law: `L ≤ L0` → zero; `L > L0` → `F = -(k·ΔL + k3·ΔL³)·d̂ - c·v_rel_line·d̂`,
applied at attachment points (like `Spring`). Continuous at `L = L0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slack_tether.py
import numpy as np
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import SlackTether


def _two_bodies(dz):
    a = RigidBody6DOF("a", 1.0, np.eye(3), np.array([0,0,dz]), np.array([0,0,0,1.0]))
    b = RigidBody6DOF("b", 1.0, np.eye(3), np.array([0,0,0.0]), np.array([0,0,0,1.0]))
    return a, b


def test_slack_no_force_when_below_rest_length():
    a, b = _two_bodies(dz=3.0)  # distance 3 < L0=5
    t = SlackTether(a, b, np.zeros(3), np.zeros(3), k=100.0, k3=0.0, c=0.0, rest_length=5.0)
    a.clear_forces(); b.clear_forces()
    t.apply_pair(0.0)
    assert np.allclose(a.f, 0.0) and np.allclose(b.f, 0.0)


def test_slack_tension_when_stretched():
    a, b = _two_bodies(dz=7.0)  # distance 7 > L0=5, dL=2
    t = SlackTether(a, b, np.zeros(3), np.zeros(3), k=100.0, k3=0.0, c=0.0, rest_length=5.0)
    a.clear_forces(); b.clear_forces()
    t.apply_pair(0.0)
    # F = -k*dL*d_hat on A; d_hat = +z (A above B) -> A pulled down
    assert np.allclose(a.f, [0, 0, -200.0])
    assert np.allclose(b.f, [0, 0, +200.0])


def test_slack_continuous_at_rest_length():
    a, b = _two_bodies(dz=5.0)  # exactly L0
    t = SlackTether(a, b, np.zeros(3), np.zeros(3), k=100.0, k3=5.0, c=10.0, rest_length=5.0)
    a.clear_forces(); b.clear_forces()
    t.apply_pair(0.0)
    assert np.allclose(a.f, 0.0)  # zero at the seam
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_slack_tether.py -v`
Expected: FAIL with `ImportError: cannot import name 'SlackTether'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/aerislab/dynamics/forces.py`:

```python
class SlackTether:
    """
    Tension-only line between two bodies (no compression force).

    L <= rest_length : F = 0           (line is slack)
    L  > rest_length : F = -(k*dL + k3*dL**3) * d_hat - c * v_rel_line * d_hat
                       dL = L - rest_length

    Applied at the attachment points (generates torques if offset from CoM).
    """
    def __init__(self, body_a, body_b, attach_a_local, attach_b_local,
                 k, k3, c, rest_length):
        self.a, self.b = body_a, body_b
        self.ra_local = np.asarray(attach_a_local, dtype=np.float64)
        self.rb_local = np.asarray(attach_b_local, dtype=np.float64)
        self.k, self.k3, self.c, self.L0 = float(k), float(k3), float(c), float(rest_length)

    def apply_pair(self, t=None):
        ra_w = self.a.rotation_world() @ self.ra_local
        rb_w = self.b.rotation_world() @ self.rb_local
        pa, pb = self.a.p + ra_w, self.b.p + rb_w
        d = pa - pb
        dist = np.linalg.norm(d)
        if dist <= self.L0 or dist < EPSILON_DISTANCE:
            return  # slack -> no force
        d_hat = d / dist
        dL = dist - self.L0
        va = self.a.v + np.cross(self.a.w, ra_w)
        vb = self.b.v + np.cross(self.b.w, rb_w)
        vrel_line = np.dot(va - vb, d_hat)
        f = -(self.k * dL + self.k3 * dL**3) * d_hat - self.c * vrel_line * d_hat
        self.a.apply_force(+f, point_world=pa, label="tether")
        self.b.apply_force(-f, point_world=pb, label="tether")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_slack_tether.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/dynamics/forces.py tests/test_slack_tether.py
git commit -m "feat(forces): SlackTether tension-only line force"
```

---

### Task 2: Parachute geometry helpers

**Files:**
- Create: `src/aerislab/models/aerodynamics/parachute_geometry.py`
- Test: `tests/models/test_parachute_model.py`

Map scalar `V_air` to projected area and to an anisotropic added-mass tensor
(local `[m_t, m_t, m_a]`), using configurable coefficients.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_parachute_model.py
import numpy as np
from aerislab.models.aerodynamics.parachute_geometry import (
    area_from_volume, added_mass_local,
)


def test_area_from_volume_monotonic_and_positive():
    a1 = area_from_volume(1.0, k_vol=1.0)
    a2 = area_from_volume(8.0, k_vol=1.0)
    assert a1 > 0 and a2 > a1


def test_added_mass_local_anisotropic():
    m_t, m_a = added_mass_local(V_air=2.0, rho=1.2, k_axial=1.0, k_transverse=0.15)
    assert np.isclose(m_a, 1.0 * 1.2 * 2.0)
    assert np.isclose(m_t, 0.15 * 1.2 * 2.0)
    assert m_a > m_t
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/models/test_parachute_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aerislab/models/aerodynamics/parachute_geometry.py
"""Geometry mapping: scalar enclosed air volume -> projected area & added-mass tensor."""
from __future__ import annotations

import numpy as np


def area_from_volume(V_air: float, k_vol: float) -> float:
    """Projected area from enclosed volume (hemispherical canopy: V ~ k_vol*A^1.5/sqrt(pi))."""
    if V_air <= 0.0:
        return 0.0
    return float((V_air * np.sqrt(np.pi) / k_vol) ** (2.0 / 3.0))


def added_mass_local(V_air: float, rho: float, k_axial: float, k_transverse: float):
    """Return (m_transverse, m_axial) added mass [kg] from enclosed volume."""
    m_axial = k_axial * rho * V_air
    m_transverse = k_transverse * rho * V_air
    return float(m_transverse), float(m_axial)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/models/test_parachute_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/models/aerodynamics/parachute_geometry.py tests/models/test_parachute_model.py
git commit -m "feat(parachute): geometry mapping V_air -> area & added-mass tensor"
```

---

### Task 3: AnalyticParachute baseline (StateProvider + AuxDynamics)

**Files:**
- Create: `src/aerislab/models/aerodynamics/parachute_base.py` (shared mixin)
- Create: `src/aerislab/models/aerodynamics/analytic_parachute.py`
- Test: `tests/models/test_parachute_model.py`

State is `V_air` (1 state). `compute_derivatives` returns `V̇_air` from a closed-form
exponential approach to a target volume — **no accumulator, no prev_time** (pure).

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_parachute_model.py  (append)
from aerislab.models.aerodynamics.analytic_parachute import AnalyticParachute
from aerislab.core.protocols import StateProvider, AuxDynamics


def test_analytic_parachute_protocols_and_purity():
    p = AnalyticParachute(V_target=4.0, tau=0.5, rho=1.2)
    assert isinstance(p, StateProvider) and isinstance(p, AuxDynamics)
    assert p.num_states() == 1
    out = np.empty(1); p.pack_state(out)
    assert out[0] == 0.0                       # starts empty
    p.unpack_state(np.array([1.0]))
    d1 = p.compute_derivatives(0.0)
    d2 = p.compute_derivatives(0.0)
    assert np.allclose(d1, d2)                 # pure: no mutation
    # dV/dt = (V_target - V)/tau = (4-1)/0.5 = 6
    assert np.isclose(d1[0], 6.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/models/test_parachute_model.py::test_analytic_parachute_protocols_and_purity -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aerislab/models/aerodynamics/analytic_parachute.py
"""Closed-form parachute inflation model: pure baseline and V&V reference."""
from __future__ import annotations

import numpy as np


class AnalyticParachute:
    """
    Single-state (V_air) inflation via first-order relaxation to V_target:
        dV_air/dt = (V_target - V_air) / tau
    Pure: derivatives are a function of the current state only (no hidden history).
    """
    def __init__(self, V_target, tau, rho, Cd=1.5):
        self.V_target = float(V_target)
        self.tau = float(tau)
        self.rho = float(rho)
        self.Cd = float(Cd)
        self.V_air = 0.0

    # StateProvider
    def num_states(self): return 1
    def pack_state(self, out): out[0] = self.V_air
    def unpack_state(self, y): self.V_air = float(y[0])

    # AuxDynamics
    def compute_derivatives(self, t):
        return np.array([(self.V_target - self.V_air) / self.tau])

    def cd(self, t): return self.Cd
```

(Place a thin shared mixin in `parachute_base.py` only if `NeuralParachute` later
reuses code; otherwise keep models independent. Do not pre-abstract — YAGNI.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/models/test_parachute_model.py::test_analytic_parachute_protocols_and_purity -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/models/aerodynamics/analytic_parachute.py tests/models/test_parachute_model.py
git commit -m "feat(parachute): AnalyticParachute pure closed-form baseline"
```

---

### Task 4: Drag force + added-mass wiring

**Files:**
- Modify: `src/aerislab/models/aerodynamics/analytic_parachute.py`
- Test: `tests/models/test_parachute_model.py`

The model produces (a) a drag `Force` from `Cd`, `q`, `A(V_air)`; (b) sets the
body's added mass from `V_air`; (c) reports `ṁ_added` from `V̇_air` for
`AddedMassFlux`. All read the single source `V_air`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_parachute_model.py  (append)
from aerislab.dynamics.body import RigidBody6DOF


def test_update_added_mass_sets_body_tensor():
    p = AnalyticParachute(V_target=4.0, tau=0.5, rho=1.2)
    p.k_axial, p.k_transverse, p.k_vol = 1.0, 0.15, 1.0
    p.unpack_state(np.array([2.0]))
    b = RigidBody6DOF("c", 1.0, np.eye(3), np.zeros(3), np.array([0,0,0,1.0]))
    p.update_added_mass(b)
    assert b._has_added_mass
    assert np.isclose(b._added_mass_local[2], 1.0 * 1.2 * 2.0)   # axial
    assert np.isclose(b._added_mass_local[0], 0.15 * 1.2 * 2.0)  # transverse


def test_added_mass_rate_from_volume_rate():
    p = AnalyticParachute(V_target=4.0, tau=0.5, rho=1.2)
    p.k_axial, p.k_transverse, p.k_vol = 1.0, 0.15, 1.0
    p.unpack_state(np.array([1.0]))
    b = RigidBody6DOF("c", 1.0, np.eye(3), np.zeros(3), np.array([0,0,0,1.0]))
    # mdot_axial = k_axial*rho*Vdot ; Vdot = (4-1)/0.5 = 6
    assert np.isclose(p.added_mass_rate(0.0, b), 1.0 * 1.2 * 6.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/models/test_parachute_model.py::test_update_added_mass_sets_body_tensor -v`
Expected: FAIL with `AttributeError: ... has no attribute 'update_added_mass'`

- [ ] **Step 3: Write minimal implementation**

Add coefficient defaults to `AnalyticParachute.__init__`:

```python
        self.k_vol = 1.0
        self.k_axial = 1.0
        self.k_transverse = 0.15
        self.area_collapsed = 1e-3
```

Add methods:

```python
    def projected_area(self):
        from aerislab.models.aerodynamics.parachute_geometry import area_from_volume
        return max(self.area_collapsed, area_from_volume(self.V_air, self.k_vol))

    def update_added_mass(self, body):
        from aerislab.models.aerodynamics.parachute_geometry import added_mass_local
        m_t, m_a = added_mass_local(self.V_air, self.rho, self.k_axial, self.k_transverse)
        body.set_added_mass(m_transverse=m_t, m_axial=m_a)

    def added_mass_rate(self, t, body):
        Vdot = float(self.compute_derivatives(t)[0])
        return self.k_axial * self.rho * Vdot

    def drag_force(self):
        model = self
        class _Drag:
            def apply(self, body, t=None):
                v = body.v
                speed = np.linalg.norm(v)
                if speed < 1e-12:
                    return
                q = 0.5 * model.rho * speed * speed
                f = -model.cd(t or 0.0) * q * model.projected_area() * (v / speed)
                body.apply_force(f, label="aerodynamics")
        return _Drag()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/models/test_parachute_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/models/aerodynamics/analytic_parachute.py tests/models/test_parachute_model.py
git commit -m "feat(parachute): drag force + added-mass wiring from V_air"
```

---

### Task 5: Parachute component composition + registration

**Files:**
- Modify: `src/aerislab/components/parachute.py`
- Test: `tests/models/test_parachute_model.py`

`Parachute` holds `body` + a `ParachuteModel`. On registration it adds the model as
an `aux_provider`, adds the drag and `AddedMassFlux` forces to the body, and calls
`update_added_mass` each evaluation.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/models/test_parachute_model.py  (append)
def test_parachute_inflates_and_added_mass_grows_under_integration():
    import numpy as np
    from aerislab.core import World
    from aerislab.core.solver import HybridIVPSolver
    from aerislab.dynamics.body import RigidBody6DOF
    from aerislab.dynamics.forces import Gravity, AddedMassFlux
    from aerislab.models.aerodynamics.analytic_parachute import AnalyticParachute

    w = World(ground_z=-1e4, payload_index=0)
    canopy = RigidBody6DOF("canopy", 1.0, np.eye(3), np.array([0,0,3000.0]),
                           np.array([0,0,0,1.0]), linear_velocity=np.array([0,0,-50.0]))
    w.add_body(canopy)
    model = AnalyticParachute(V_target=10.0, tau=1.0, rho=1.0)
    w.aux_providers.append(model)
    canopy.per_body_forces.append(model.drag_force())
    canopy.per_body_forces.append(AddedMassFlux(model.added_mass_rate))
    canopy.per_body_forces.append(_UpdateAddedMass(model))  # tiny adapter applying update each call
    w.add_global_force(Gravity(np.array([0,0,-9.81])))

    ivp = HybridIVPSolver(method="Radau", rtol=1e-7, atol=1e-9, max_step=0.1)
    w.integrate_to(ivp, t_end=2.0)
    assert model.V_air > 5.0                    # inflated toward target
    assert canopy._has_added_mass               # added mass active


class _UpdateAddedMass:
    """Force-protocol adapter that refreshes the body's added mass each evaluation."""
    def __init__(self, model): self.model = model
    def apply(self, body, t=None): self.model.update_added_mass(body)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest "tests/models/test_parachute_model.py::test_parachute_inflates_and_added_mass_grows_under_integration" -v`
Expected: FAIL (initially `_UpdateAddedMass`/wiring or `Parachute` API mismatch)

- [ ] **Step 3: Implement Parachute composition**

Update `components/parachute.py` so `Parachute(name, body, model)` exposes a
`register(world)` that performs the four wiring steps shown in the test (add aux
provider, drag force, `AddedMassFlux`, and an internal update adapter). Keep the
adapter inside the component rather than the test once green.

```python
# components/parachute.py (sketch)
from aerislab.dynamics.forces import AddedMassFlux

class Parachute:
    def __init__(self, name, body, model):
        self.name, self.body, self.model = name, body, model
    def register(self, world):
        world.aux_providers.append(self.model)
        self.body.per_body_forces.append(self.model.drag_force())
        self.body.per_body_forces.append(AddedMassFlux(self.model.added_mass_rate))
        self.body.per_body_forces.append(_UpdateAddedMass(self.model))

class _UpdateAddedMass:
    def __init__(self, model): self.model = model
    def apply(self, body, t=None): self.model.update_added_mass(body)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest "tests/models/test_parachute_model.py::test_parachute_inflates_and_added_mass_grows_under_integration" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/components/parachute.py tests/models/test_parachute_model.py
git commit -m "feat(parachute): compose model into Parachute component + registration"
```

---

### Task 6: NeuralParachute ONNX backend (optional dependency)

**Files:**
- Create: `src/aerislab/models/aerodynamics/neural_parachute.py`
- Test: `tests/models/test_parachute_model.py`

Same interface as `AnalyticParachute`, but `V̇_air` and `Cd` come from ONNX
inference. Test skips when `onnxruntime` is absent; logic is tested with a stub
session so it runs in CI without a model file.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_parachute_model.py  (append)
def test_neural_parachute_uses_inference_outputs():
    import numpy as np
    from aerislab.models.aerodynamics.neural_parachute import NeuralParachute

    class _StubSession:
        def infer(self, features):  # returns [Vdot, Cd]
            return np.array([3.0, 1.4])

    p = NeuralParachute(session=_StubSession(), rho=1.2)
    p.unpack_state(np.array([1.0]))

    class _Body:
        v = np.array([0.0, 0.0, -40.0])
        p_pos = np.array([0.0, 0.0, 1000.0])
        def to_body(self, vec): return vec
        @property
        def p(self): return self.p_pos

    d = p.compute_derivatives(0.0, body=_Body(), rho=1.2)
    assert np.isclose(d[0], 3.0)        # Vdot from inference
    assert np.isclose(p.cd(0.0), 1.4)   # Cd cached for drag
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/models/test_parachute_model.py::test_neural_parachute_uses_inference_outputs -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aerislab/models/aerodynamics/neural_parachute.py
"""Neural ODE parachute backend (ONNX). Outputs V̇_air and Cd from inference."""
from __future__ import annotations

import numpy as np


class NeuralParachute:
    """
    Same interface as AnalyticParachute, but derivatives/Cd come from a session
    whose .infer(features) -> [Vdot_air, Cd]. The session wraps onnxruntime (or a
    stub in tests). Pure: no hidden state beyond V_air (carried in the state vector).
    """
    def __init__(self, session, rho, k_vol=1.0, k_axial=1.0, k_transverse=0.15):
        self.session = session
        self.rho = float(rho)
        self.k_vol, self.k_axial, self.k_transverse = k_vol, k_axial, k_transverse
        self.area_collapsed = 1e-3
        self.V_air = 0.0
        self._cd = 0.0

    def num_states(self): return 1
    def pack_state(self, out): out[0] = self.V_air
    def unpack_state(self, y): self.V_air = float(y[0])

    def _features(self, body, rho):
        v_body = body.to_body(body.v)
        return np.array([*v_body, rho, self.V_air], dtype=np.float64)

    def compute_derivatives(self, t, body=None, rho=None):
        rho = self.rho if rho is None else rho
        vdot, cd = self.session.infer(self._features(body, rho))
        self._cd = float(cd)
        return np.array([float(vdot)])

    def cd(self, t): return self._cd
    # projected_area / update_added_mass / added_mass_rate / drag_force:
    # identical to AnalyticParachute (extract a shared mixin in parachute_base.py
    # only now that two models share them — see Task 3 note).
```

> **Note:** the engine's `AuxDynamics.compute_derivatives(t)` takes only `t`. Bridge
> this by having the `Parachute` component pass `body`/`rho` via a closure or by the
> component calling `model.compute_derivatives(t)` after stashing `body`/`rho` from
> `update_added_mass`. Decide the cleaner bridge during implementation and keep it
> consistent with `AnalyticParachute`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/models/test_parachute_model.py::test_neural_parachute_uses_inference_outputs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerislab/models/aerodynamics/neural_parachute.py tests/models/test_parachute_model.py
git commit -m "feat(parachute): NeuralParachute ONNX backend (stub-testable)"
```

---

### Task 7: Retire legacy stateful models

**Files:**
- Move: `src/aerislab/models/aerodynamics/parachute_models.py` → `.../not_in_use/`
- Modify: any imports of `AdvancedParachute`; `tests/test_parachute_models.py`

The stateful `AdvancedParachute` is the last source of in-RHS mutation. Move it out
of the active path, keep `AnalyticParachute` as the baseline.

- [ ] **Step 1: Find references**

Run: `grep -rn "parachute_models\|AdvancedParachute" src/ tests/ examples/`
Expected: a list of import sites to update or remove.

- [ ] **Step 2: Move the file and update imports**

```bash
git mv src/aerislab/models/aerodynamics/parachute_models.py \
       src/aerislab/models/aerodynamics/not_in_use/parachute_models.py
```
Update or delete each reference found in Step 1. For `tests/test_parachute_models.py`,
either move it alongside (mark `@pytest.mark.skip(reason="legacy, retired")`) or
delete if fully superseded by `tests/models/test_parachute_model.py`.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (no active code imports the retired module)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(parachute): retire stateful legacy models to not_in_use"
```

---

### Task 8: Migrate a parachute scenario to SlackTether + new model

**Files:**
- Modify: one example under `examples/scenarios/` (pick the simplest parachute drop)
- Test: `tests/test_integration_example.py` (add a smoke run)

Demonstrate the full stack: `Parachute` component + `SlackTether` line + payload,
integrated with `HybridIVPSolver`, descending under gravity with growing added mass
and no snatch spike.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_integration_example.py  (append)
def test_parachute_payload_descends_with_slack_tether():
    import numpy as np
    from aerislab.core import World
    from aerislab.core.solver import HybridIVPSolver
    from aerislab.dynamics.body import RigidBody6DOF
    from aerislab.dynamics.forces import Gravity, SlackTether
    from aerislab.models.aerodynamics.analytic_parachute import AnalyticParachute
    from aerislab.components.parachute import Parachute

    w = World(ground_z=0.0, payload_index=0)
    payload = RigidBody6DOF("payload", 80.0, np.eye(3), np.array([0,0,1000.0]),
                            np.array([0,0,0,1.0]), linear_velocity=np.array([0,0,-50.0]))
    canopy = RigidBody6DOF("canopy", 3.0, np.eye(3), np.array([0,0,1005.0]),
                           np.array([0,0,0,1.0]), linear_velocity=np.array([0,0,-50.0]))
    w.add_body(payload); w.add_body(canopy)
    Parachute("para", canopy, AnalyticParachute(V_target=20.0, tau=1.5, rho=1.1)).register(w)
    w.add_interaction_force(SlackTether(canopy, payload, np.zeros(3), np.zeros(3),
                                        k=5000.0, k3=200.0, c=300.0, rest_length=5.0))
    w.add_global_force(Gravity(np.array([0,0,-9.81])))

    w.integrate_to(HybridIVPSolver(method="Radau", rtol=1e-6, atol=1e-8, max_step=0.05),
                   t_end=5.0)
    assert payload.p[2] < 1000.0          # descended
    assert payload.v[2] > -50.0           # decelerated by the canopy
```

- [ ] **Step 2: Run test to verify it fails (or runs)**

Run: `.venv/bin/pytest tests/test_integration_example.py::test_parachute_payload_descends_with_slack_tether -v`
Expected: FAIL until the scenario wiring/imports are correct.

- [ ] **Step 3: Update the example script**

Port the simplest existing parachute example under `examples/scenarios/` to this new
stack (Parachute component + SlackTether), removing the old `DistanceConstraint` and
`AdvancedParachute` usage.

- [ ] **Step 4: Run test + full suite**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check src/ && .venv/bin/mypy src/aerislab/`
Expected: PASS / no new errors.

- [ ] **Step 5: Commit**

```bash
git add examples/ tests/test_integration_example.py
git commit -m "feat(examples): parachute drop on new model + SlackTether"
```

---

## Done criteria

- Parachute model is pure (state-vector `V_air`); no in-RHS mutation remains, so
  CRIT-3 is fully resolved end-to-end.
- Added mass is driven from `V_air` (LHS) with its `ṁ·v` reaction (RHS); both from
  one source.
- `AnalyticParachute` and `NeuralParachute` share one interface; swapping them needs
  no scenario change.
- `SlackTether` replaces rigid lines in parachute scenarios (no snatch spike).
- Legacy stateful models are out of the active path.

## V&V follow-ups (post-plan)

- Opening-shock validation against FSI / flight reference.
- Analytic Jacobian for the NN into `solve_ivp(jac=…)`.
- Geometry coefficient calibration (`k_vol`, `k_axial`, `k_transverse`).
