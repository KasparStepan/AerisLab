# HybridSim — Modular Hybrid Multibody Dynamics (No Contact)

**What it is:** A compact, extensible Python ≥3.10 framework for simulating 3D rigid-body multibody systems with rigid constraints via a KKT solve, two solver modes (fixed-step and SciPy IVP), and **no** contact modeling. If a designated **payload** hits the ground plane (z ≤ `ground_z`), the simulation halts immediately.

## Features
- 6-DoF rigid bodies with quaternions (scalar-first), correct world inertia, off-center force torques.
- Forces: gravity, drag (linear/quadratic, runtime-tunable), optional soft spring (two-body tether).
- Constraints: distance (1 eq) and point-weld (3 eq) with explicit Jacobians.
- KKT solve:
  \[
    \begin{bmatrix} M & J^\top \\ J & 0 \end{bmatrix}
    \begin{bmatrix} a \\ \lambda \end{bmatrix} =
    \begin{bmatrix} F \\ -Jv - \alpha C - \beta \dot C \end{bmatrix}
  \]
  Scatter accelerations to bodies; integrate (semi-implicit Euler).
- Two solver paths:
  - **Fixed-step** `HybridSolver`: per-step KKT + symplectic Euler.
  - **IVP** `HybridIVPSolver`: `solve_ivp` (Radau/BDF) computes `y' = [v, ½ q⊗[0,ω], a_lin, a_ang]` with accelerations from the KKT at `(t,y)`. Terminal event stops at touchdown.
- **Termination on ground only:** no contact forces/impulses; fixed-step uses linear interpolation for touchdown time.
- Clean API, `__slots__` on hot objects, float64, minimal heap churn.

## Install & Run
```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pytest
# Optional for IVP:
pip install scipy

### Visualizing results

After running an example (e.g., `examples/parachute_fixed.py`) you’ll get a CSV log at `logs/*.csv`.

Plot trajectory, velocity, acceleration, and forces:

```bash
python examples/plot_parachute_logs.py


---

## Notes & tips

- **Acceleration source:** since we don’t currently log accelerations, we compute them by differentiating the logged velocity with `np.gradient`, which handles uneven time steps reasonably.
- **Multiple bodies:** just change `body = "canopy"` to inspect the other body, or call the plotting functions twice for both.
- **Saving vs showing:** all plotting functions accept `save_path` and `show` flags so you can render images in CI or notebooks without popping windows.

If you want real-time plotting during a fixed-step run, we can add a tiny “live plot” hook that updates every N steps—just say the word.

