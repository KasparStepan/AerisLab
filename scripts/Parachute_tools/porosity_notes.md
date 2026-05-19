# Parachute Porosity Evaluation — Notes

Working notes from developing `porosity_evaluation.py` for fitting MIL-C-7020-III canopy fabric permeability data and extracting Darcy–Forchheimer parameters for LS-DYNA ICFD.

---

## 1. Original script review

Original code fit `v = a·ΔP² + b·ΔP + c` to permeability data and printed coefficients. Issues identified:

- `__main__` used as function name (dunder reserved for module identity) — should be `main`.
- Relative CSV path breaks when script is run from a different directory — use `Path(__file__).parent`.
- `.values` is legacy pandas — prefer `.to_numpy()`.
- No fit-quality output (R² etc.).
- Coefficient ordering opaque in print statement.
- **Physical prior ignored**: permeability at zero ΔP must be zero; unconstrained quadratic produces unphysical intercept.
- No type hints, no docstrings, no `if __name__` guard on a properly named function.

---

## 2. Math behind `fit_quadratic(..., through_origin=True)`

### Model

$$y = a\,x^2 + b\,x$$

Intercept dropped on physical grounds (`v(ΔP=0) = 0`). Two unknowns `(a, b)`, N data points.

### Design matrix

Stack N equations `yᵢ ≈ a·xᵢ² + b·xᵢ`:

$$
\begin{bmatrix} y_1 \\ \vdots \\ y_N \end{bmatrix} \approx
\underbrace{\begin{bmatrix} x_1^2 & x_1 \\ \vdots & \vdots \\ x_N^2 & x_N \end{bmatrix}}_{A}
\begin{bmatrix} a \\ b \end{bmatrix}
$$

Each column is a basis function (`x²` and `x`) evaluated at the data sites. The model is **linear in the parameters** even though quadratic in `x` — that's what makes OLS applicable.

### Least-squares objective

System is overdetermined (N ≫ 2), so no exact solution. Minimize sum of squared residuals:

$$
\min_{a,b}\; S(a,b) = \sum_i (y_i - a x_i^2 - b x_i)^2 = \|y - A[a,b]^T\|_2^2
$$

Geometrically: project `y` orthogonally onto the 2-D column space of `A` in ℝᴺ.

### Normal equations

Setting `∇S = 0`:

$$
A^{\!\top}\!A \begin{bmatrix} a \\ b \end{bmatrix} = A^{\!\top} y
$$

Closed-form solution: `[a, b]ᵀ = (AᵀA)⁻¹ Aᵀ y` (the Moore–Penrose pseudoinverse `A⁺` applied to `y`).

### Why `np.linalg.lstsq` instead of normal equations

Forming `AᵀA` **squares the condition number** of `A`, hurting numerical precision when columns are correlated (which they are for `x` and `x²` over a positive range). `np.linalg.lstsq` uses QR or SVD on `A` directly, avoiding `AᵀA` entirely.

### Statistical interpretation

Under iid zero-mean equal-variance residuals, the OLS estimator is the **best linear unbiased estimator (BLUE)** of `(a, b)` — Gauss–Markov theorem. Minimum variance among unbiased linear estimators.

---

## 3. Interpreting R² = 0.977

R² = 0.977 sounds high, but for **48 clean, lab-instrumented data points along a smooth monotonic curve**, expected R² is > 0.999. The 2.3% missing variance is systematic, not noise — fingerprint of model misspecification.

### Quick check that quadratic is wrong

Endpoints: ΔP grows ×111, v grows only ×18.

- If quadratic (`v ∝ ΔP²`): v should grow ×12 000.
- If linear (`v ∝ ΔP`): v should grow ×111.
- Empirical exponent: `n = log(18) / log(111) ≈ 0.61`.

So the data is **sub-linear**, not super-linear. The quadratic is geometrically wrong.

---

## 4. The correct physical model: Darcy–Forchheimer

Standard form for flow through porous media:

$$
\Delta P = \underbrace{\frac{\mu\,t}{K}}_{\alpha}\,v + \underbrace{\rho\,t\,\beta_{\text{phys}}}_{\beta}\,v^2
$$

- `α·v` — viscous (Darcy) term, dominant at low velocity.
- `β·v²` — inertial (Forchheimer) term, dominant at high velocity.
- Both coefficients are **positive** (drag always resists flow).

### Why fitting `v = a·ΔP² + b·ΔP` gave a negative quadratic coefficient

Inverting `ΔP = α·v + β·v²` (Taylor-expand for small β·ΔP/α²):

$$
v \approx \frac{1}{\alpha}\Delta P - \frac{\beta}{\alpha^3}\Delta P^2 + O(\Delta P^3)
$$

So when data follows Forchheimer:

- `b ≈ 1/α` — **positive** ✓
- `a ≈ -β/α³` — **negative** ✓ (signature of sub-linear curve)

A negative `a` is exactly what you should get when fitting v(ΔP) for genuinely Forchheimer data. The problem isn't the fit; the problem is calling `a` a "Forchheimer factor". It's `-β/α³`, which inherits the sign.

### Original `permeability` formula was also inverted

From the Darcy limit: `b = K/(μ·t)`, so `K = b·μ·t`.

Original code had `K = μ·t / b`. Dimensionally inconsistent:

- `μ·t/b` gives units kg²/(m²·s²), not m². ✗
- `b·μ·t` gives m². ✓

---

## 5. Fix: fit in the physical direction

```python
def fit_forchheimer(v, dP):
    """Fit ΔP = α·v + β·v² through origin."""
    A = np.column_stack([v, v**2])
    alpha, beta = np.linalg.lstsq(A, dP, rcond=None)[0]
    return alpha, beta
```

Then:

```python
K   = AIR_DYNAMIC_VISCOSITY * FABRIC_THICKNESS / alpha             # [m²]
C_F = beta * np.sqrt(K) / (AIR_DENSITY * FABRIC_THICKNESS)         # [-]
```

Derivation of `C_F`: from `β = ρ·t·C_F/√K`, rearrange.

---

## 6. Results on MIL-C-7020-III data

```
Quadratic fit (v = a·ΔP² + b·ΔP):         R² = 0.977
Linear fit (v = m·ΔP):                    R² = 0.899
Cubic fit (v = p·ΔP³ + q·ΔP² + r·ΔP):     R² = 0.993
Darcy–Forchheimer fit (ΔP = α·v + β·v²):  R² = 0.998
```

Forchheimer fit clearly wins. Final values:

| Parameter | Value | Units |
|---|---|---|
| α (viscous coef) | 254.4 | Pa·s/m |
| β (inertial coef) | 37.6 | Pa·s²/m² |
| **Permeability K** | **4.268 × 10⁻¹²** | **m²** |
| **Forchheimer C_F** | **1.057** | **dimensionless** |

Both K and C_F are positive and sit in the expected ranges for low-permeability woven nylon (K ∈ 10⁻¹²–10⁻¹⁰ m², C_F ∈ 0.1–5).

---

## 7. Plotting the Forchheimer curve

Plot has v on y-axis vs ΔP on x-axis, so the Forchheimer fit (which is ΔP(v)) must be inverted:

$$
v(\Delta P) = \frac{-\alpha + \sqrt{\alpha^2 + 4\beta\,\Delta P}}{2\beta}
$$

(positive root of the quadratic in v).

In code:
```python
y_df = (-alpha + np.sqrt(alpha**2 + 4 * beta * x_fit)) / (2 * beta)
```

Use log-y scale to make the spread visible across decades of ΔP.

---

## 8. Porosity ε — what to put in LS-DYNA

### Three meanings of "porosity" that get confused

1. **Coefficient in a drag formula** — not actually part of Darcy–Forchheimer; don't bake into K or C_F.
2. **Fluid-volume fraction (volumetric porous-medium formulation)** — `ε = void volume / total volume`. Real input the solver uses.
3. **Open-area fraction of the weave** — geometric property of the fabric, ~0.05 for tight nylon.

### "Parachute porosity" in literature

Often refers to **effective porosity** — a dimensionless permeability ratio at standard ΔP — NOT a volume fraction. For MIL-C-7020-III this is reported as ~3.5–5.5%. If you read this from a datasheet and plug it into ε, you'd be double-counting what K already encodes.

For the geometric open-area fraction (what LS-DYNA actually wants), 3.5–5.5% happens to also be roughly right for this fabric, but the values mean different things even when numerically similar.

### LS-DYNA's formulation (from manual)

The manual shows the **volumetric porous-media formulation**:

$$
\varepsilon = \frac{\text{void volume}}{\text{total volume}}, \quad u_i = \varepsilon u_{if}
$$

- `u_i` = volume-averaged (superficial) velocity — what the permeability test measures.
- `u_{if}` = intrinsic / pore velocity.

The momentum equation has ε in the denominator everywhere. **This means ε is a load-bearing physical input, not optional, even when the fabric is meshed as a shell** — LS-DYNA extrudes the shell into a thin volumetric porous region internally.

### Final answer for shell-meshed parachute

| Field | Value | Source |
|---|---|---|
| Permeability K | **4.27 × 10⁻¹² m²** | fitted α |
| Forchheimer C_F | **1.06** | fitted β |
| Porosity ε | **0.045** (sweep 0.035–0.055) | MIL spec, weave open-area |
| Fabric thickness t | **6 × 10⁻⁵ m** | measured |

(Earlier I suggested ε = 1 for a thin-shell pressure-jump formulation. That's wrong for this card — the manual confirms LS-DYNA solves the full volumetric porous-medium PDE, so ε is the real physical volume fraction.)

### Calibration / validation run

Pick one data point from the CSV, e.g. ΔP = 1000 Pa → v ≈ 2.06 m/s. Set up a minimal LS-DYNA case with these inputs:

- Flat canopy patch, ΔP = 1000 Pa applied across it, run to steady state.
- Measure through-flow.
- Expect ≈ 2.06 m/s within a few %.
- If wildly off → check unit slots, ε interpretation, K/C_F swap.

After calibration passes, run a small **sensitivity sweep** ε ∈ {0.035, 0.045, 0.055} to see if parachute dynamics depend strongly on ε. Bulk drag should barely move (K and C_F dominate); transient response is where the convective term `∂(u_i u_j/ε)/∂x_j` will show.

---

## 9. Final values to enter in LS-DYNA ICFD card (SI units)

```
K   = 4.268357e-12   [m²]
C_F = 1.056553       [-]
ε   = 0.045          [-]
t   = 6e-5           [m]
```

(LS-DYNA operates in basic SI — all values above are SI-consistent.)

---

## Appendix: Other recommendations from review

- Drop `E_POROSITY` from the K/C_F derivation entirely (it was a bug in the original Forchheimer factor formula).
- Add residual plot (residuals of Forchheimer fit vs v on linear-y axis) for a visual goodness-of-fit check.
- Consider log-log plot `log ΔP vs log v` — slope reveals the flow regime (slope 1 = Darcy, slope 2 = full Forchheimer).
- If matplotlib warns about `FigureCanvasAgg is non-interactive` in WSL2: use `plt.savefig()` instead of `plt.show()`, or install `PyQt5` for an interactive backend.
- Confirm `FABRIC_THICKNESS = 6e-5 m` (60 µm) against a micrometer measurement; K is linear in t, so errors there propagate directly.
- Low-ΔP coverage: dataset starts at 101 Pa. The viscous slope α is mostly constrained by the lowest few points; sub-50-Pa points would improve K accuracy if the rig supports it.
