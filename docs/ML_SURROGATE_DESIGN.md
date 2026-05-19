# ML Surrogate Design — Parachute Aerodynamics for AerisLab

**Context:** notes from a working session on 2026-05-14, working through the ML approach for the FSI-trained parachute surrogate that will plug into AerisLab. Synthesized from a conversation that started with "should I use 3 models?" and converged on a single-model neural-ODE design as the right approach for this thesis.

For Štěpán: PhD on parachute recovery systems, has FSI for inflation + steady descent, plans to do parameter studies (payload mass × activation velocity → peak snatch force).

---

## Table of Contents

1. [The use case in one paragraph](#1-the-use-case-in-one-paragraph)
2. [Final recommended approach](#2-final-recommended-approach)
3. [Why this approach (vs. the alternatives we considered)](#3-why-this-approach-vs-the-alternatives-we-considered)
4. [The ML framework stack](#4-the-ml-framework-stack)
5. [Specific PyTorch building blocks you'll use](#5-specific-pytorch-building-blocks-youll-use)
6. [Architecture details with code](#6-architecture-details-with-code)
7. [Training procedure](#7-training-procedure)
8. [How it integrates with AerisLab](#8-how-it-integrates-with-aerislab)
9. [Validation strategy](#9-validation-strategy)
10. [Honest risks and how to handle them](#10-honest-risks-and-how-to-handle-them)
11. [Concrete first steps](#11-concrete-first-steps)
12. [How to defend this in the thesis](#12-how-to-defend-this-in-the-thesis)

---

## 1. The use case in one paragraph

You have a fluid-structure-interaction (FSI) simulator that produces detailed parachute aerodynamics. Running it inside a multi-body simulator for every design study is computationally infeasible. So: **train a neural-network surrogate of the FSI**, plug it into AerisLab, and use the resulting fast simulator to do parameter studies — specifically: at what `(payload_mass, activation_velocity)` combinations do you stay below a target peak snatch force?

The training data:
- **Inflation**: infinite-mass FSI runs (velocity held fixed during the run) at 10-20 representative velocities, per chute geometry.
- **Steady descent**: infinite-mass FSI runs at 10-20 representative velocities in the post-inflation regime.
- For each FSI run, time-series outputs: `force(t)`, `projected_area(t)`, `side_projected_area(t)`, `mass_of_air_in_canopy(t)`.

Because the FSI gives you *internal canopy state* (area, mass of air), not just force, you have richer training data than typical surrogate work. This enables a much cleaner architecture than the alternatives.

---

## 2. Final recommended approach

**One neural network, trained as a "neural ODE" — predicts both the instantaneous aerodynamic force AND the rate-of-change of the canopy's internal state.**

### Inputs (what the network sees at each call)
- `v_air` — relative airspeed (vector or magnitude)
- `ρ` — air density
- `A` — current projected area
- `A_side` — current side area
- `m_air` — current mass of air in canopy

### Outputs (what the network predicts)
- `F` — aerodynamic force vector (3 components, body frame)
- `dA/dt` — rate of projected-area change
- `dA_side/dt` — rate of side-area change
- `dm_air/dt` — rate of air-mass change

### Why this is one model, not three

In the inflation regime, the canopy state changes rapidly — the network outputs large `dA/dt`, `dm_air/dt`, and the force is the inflation force.

In the steady-descent regime, the canopy state has reached equilibrium — the network outputs `dA/dt ≈ 0`, `dm_air/dt ≈ 0`, the area is fully open, and the force is the steady drag.

**The phase transition is automatic.** As the canopy approaches `area_ratio = 99%` and the rates approach zero, the network smoothly transitions from "predicting growth" to "predicting equilibrium." No gating, no switching, no discontinuity. The IVP solver inside AerisLab integrates the canopy state alongside the body state and everything stays smooth.

This is dramatically better than the original 3-model + transition design:
- ✅ One model, one training procedure, one validation procedure, one paragraph in the methods chapter.
- ✅ Smooth derivatives — friendly to AerisLab's stiff IVP solver.
- ✅ History dependence handled cleanly via the auxiliary state, not via fancy ML (LSTMs, transformers).
- ✅ Maps perfectly onto AerisLab's architecture (Move 1: state-vector boundary, Move 2: Model layer, work plan task P2-T2: auxiliary state registration).

---

## 3. Why this approach (vs. the alternatives we considered)

### Alternative 1: 3 separate models (inflation, stabilization, steady) with a switch

What we initially considered. Problems:
- Phase boundaries are where bugs live.
- Hard switches cause discontinuities that break stiff ODE solvers.
- Smooth blending requires a hand-designed gate function, more code, more validation.
- Each model needs its own training data, hyperparameters, validation, methods chapter section.
- More moving parts to maintain and write up.

### Alternative 2: Single model, force-only output (no internal state)

A simpler design: input `(v, ρ, time_since_deploy)` → output `F`. But:
- Has the **history dependence problem**: at instant `t`, the force depends on the whole velocity history (because the canopy filling depended on velocity). A purely current-state model can't capture this without some proxy for history.
- Requires either lagged features or a recurrent model (LSTM/GRU) — more complex, harder to validate.
- Can't be physics-checked (no internal state to compare against).

### Alternative 3: Force-only output, conditioned on phase indicator

A middle ground: input `(v, ρ, area_ratio, dA/dt, time_since_deploy)` → output `F`. Better than Alternative 2 because the inputs partially capture history. But:
- Still doesn't tell you `area_ratio`, `dA/dt` at simulation time — you have to predict them somehow.
- Doesn't leverage the rich FSI data you have.
- Less elegant than the full neural-ODE formulation.

### Why the neural-ODE approach wins

Because you have the canopy state from FSI (`A`, `A_side`, `m_air`), you can train the network to learn **the state evolution** explicitly. Then in AerisLab, the canopy state is integrated alongside the body state — the IVP solver does what it's good at. The network just provides a constitutive relation: *"given the current physical state, here's the rate of change and the force."*

This is how the physics actually works (the canopy state is a continuous dynamical variable, not a categorical "phase"), so the methodology is principled, not just engineering pragmatism.

---

## 4. The ML framework stack

The word "framework" gets used for several different things. You'll need tools from each category:

| Category | What it does | What you'll use |
|---|---|---|
| **Training framework** | The library that builds and trains the neural network | **PyTorch** |
| **Training-loop wrapper** | Removes boilerplate (checkpoints, logging, etc.) | **PyTorch Lightning** |
| **Experiment tracker** | Records every run: hyperparameters, metrics, plots | **Weights & Biases** |
| **Configuration manager** | Hyperparameters in YAML, swap at command line | **Hydra** |
| **Inference runtime** | Runs the trained model fast inside AerisLab | **ONNX Runtime** |
| **Classical-ML baseline** | "Boring" baseline to compare against | **scikit-learn** |

You don't need all of these on day 1. Day-1 essentials: **PyTorch + ONNX Runtime + scikit-learn**. The others (Lightning, W&B, Hydra) save weeks once you start serious experimentation.

### Why PyTorch (not TensorFlow, not JAX)

- **PyTorch** — what most modern ML research uses. Largest ecosystem, best documentation, best Stack Overflow coverage. The mental model is "tensors are like numpy arrays that also remember how they were computed, so you can compute gradients automatically." If you can think in numpy, you can think in PyTorch.
- **TensorFlow / Keras** — older, more "enterprise-y." Don't pick this for new research code in 2026.
- **JAX** — newer, faster, more elegant for advanced math. But smaller community, harder to debug. Worth considering only if you eventually want **differentiable simulation** (gradient-based optimization through your simulator). For your thesis, the switching cost isn't worth it.

**Verdict: PyTorch.** This is what 90% of the ML research field uses; you'll find help easily and your work will be reproducible by others.

---

## 5. Specific PyTorch building blocks you'll use

PyTorch is huge. You'll use a small fraction of it. Here's what you actually need.

### `torch.nn.Module` — the base class for any model

Every neural network you build will inherit from `nn.Module`:

```python
import torch
import torch.nn as nn

class CanopyDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers here
    
    def forward(self, x):
        # define how data flows through layers
        return output
```

Two methods: `__init__` (what layers exist), `forward` (how they connect). That's the recipe for every network you'll build.

### `torch.nn.Linear` — the dense layer

The simplest layer: takes a vector in, multiplies by a learned matrix, adds a learned bias, returns a vector out.

```python
self.layer1 = nn.Linear(in_features=5, out_features=128)
```

This says: "take a vector of length 5, output a vector of length 128." The 5×128 matrix and 128-element bias are the parameters that get learned during training.

### Activation functions — `nn.GELU`, `nn.ReLU`, `nn.Tanh`

After every Linear layer (except the last) you put a non-linear activation. Without these, the whole network would collapse into a single matrix multiplication and could only learn linear relationships.

- `nn.ReLU()` — `max(0, x)`. Most common; fast.
- `nn.GELU()` — smoother variant. **Recommended for your case** — smooth derivatives matter when output gets fed into a stiff ODE solver.
- `nn.Tanh()` — bounds output to `[-1, 1]`; useful for some physics tasks.

### `torch.nn.Sequential` — stack of layers

```python
self.net = nn.Sequential(
    nn.Linear(5, 128),
    nn.GELU(),
    nn.Linear(128, 128),
    nn.GELU(),
    nn.Linear(128, 6),     # output: F(3) + 3 state derivatives
)
```

That's a complete simple network. Three Linear layers, two activations between them. Maybe 50,000 parameters total. **For your problem this might be the entire model.**

### `torch.optim.Adam` — the optimizer

Adam is the default modern optimizer. You don't need to know how it works internally; you need:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

`lr` is the learning rate. `1e-3` is a sensible default.

### `torch.nn.MSELoss` — the loss function

You're doing regression (predicting continuous values), so the loss is mean squared error:

```python
loss_fn = nn.MSELoss()
loss = loss_fn(predicted, target)
```

For multi-output regression with different scales, you'll want to weight different outputs (see §7 on training).

### `torch.utils.data.Dataset` and `DataLoader` — data feeders

Your FSI data needs to be loaded in batches. PyTorch handles this:

```python
from torch.utils.data import Dataset, DataLoader

class FSIDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, i):
        return self.features[i], self.targets[i]

loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

You write the `Dataset`; PyTorch handles batching, shuffling, and parallel loading.

### `torch.compile` — "make it faster"

One line, can give 2-5× speedup at inference time:

```python
model = torch.compile(model)
```

Worth using once your model is finalized.

### `torch.onnx.export` — package for deployment

Once trained, export to ONNX format so you can run it in AerisLab without dragging in PyTorch:

```python
torch.onnx.export(model, dummy_input, "parachute_surrogate.onnx")
```

Then in AerisLab:

```python
import onnxruntime as ort
session = ort.InferenceSession("parachute_surrogate.onnx")
output = session.run(None, {"input": state_vector})
```

ONNX Runtime is 5-10× faster than PyTorch for small-model inference. **Critical** for putting the model inside an ODE solver loop that may call it 100,000+ times per simulation.

### What you DON'T need

- `torch.distributed` (multi-GPU) — your model is small, single GPU/CPU is fine.
- `torch.cuda.amp` (mixed precision) — won't help meaningfully for small models.
- Convolutional layers — you don't have image-like data.
- Transformers, attention mechanisms — overkill for low-dimensional regression.
- LSTMs, GRUs — only if your simple MLP fails on history-dependent dynamics; in our design the auxiliary state captures history, so you don't need recurrence.

---

## 6. Architecture details with code

The complete model:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CanopyDynamics(nn.Module):
    """
    Neural-ODE surrogate for parachute aerodynamics.
    
    Predicts both the instantaneous aerodynamic force and the rate-of-change
    of the canopy's internal state. Trained on infinite-mass FSI data;
    handles inflation through steady descent in a single model.
    """
    
    def __init__(self, hidden=128, depth=4):
        super().__init__()
        # 5 inputs:  v, rho, A, A_side, m_air
        # 6 outputs: F(3), dA/dt, dA_side/dt, dm_air/dt
        layers = [nn.Linear(5, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 6)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.net(x)
        force      = out[..., 0:3]
        dA_dt      = F.softplus(out[..., 3])    # non-negative during inflation
        dA_side_dt = out[..., 4]                 # can be either sign
        dm_dt      = F.softplus(out[..., 5])    # non-negative
        return force, dA_dt, dA_side_dt, dm_dt
```

### Design choices explained

- **`softplus` on `dA/dt` and `dm_air/dt`** enforces non-negativity (canopy doesn't shrink during inflation; air doesn't leave faster than it enters). This is a free physics constraint that helps the network train and behave physically. In steady state, `dA/dt` will naturally approach zero from above.
- **No softplus on `dA_side/dt`** because side area can change sign (canopy oscillates).
- **4 hidden layers of 128 units = ~67,000 parameters.** Plenty of capacity; nowhere near overfitting risk with your data volume.
- **GELU activation** for smooth derivatives.
- **Vector velocity input?** You can use `|v|` (scalar) for a chute-aligned model, or include the full `v_air` vector (3 components) if you want the model to capture asymmetric aerodynamics. Start with scalar; add vector if needed for the steerable case later.

### If side-area dynamics aren't important

Drop them. The architecture trivially scales down to 4 inputs + 5 outputs.

### If the model needs more capacity later

Increase `hidden` and `depth`. With your data volume you can comfortably go up to hidden=512, depth=6 (~1.6M parameters) before risking overfitting. But start small.

---

## 7. Training procedure

### Step 1: extract training pairs from FSI runs

For each FSI run at velocity `v_run`, density `ρ_run`:

```python
# From the FSI output time series:
A_t      = ...   # array of A(t)
A_side_t = ...
m_air_t  = ...
F_t      = ...

# Compute finite-difference derivatives
dt = ...  # FSI timestep
dA_dt      = np.gradient(A_t, dt)
dA_side_dt = np.gradient(A_side_t, dt)
dm_dt      = np.gradient(m_air_t, dt)

# Each FSI timestep becomes one training pair
features = np.column_stack([
    np.full_like(A_t, v_run),
    np.full_like(A_t, rho_run),
    A_t, A_side_t, m_air_t,
])
targets = np.column_stack([F_t, dA_dt, dA_side_dt, dm_dt])
```

Each FSI run produces ~thousands of training pairs (one per FSI timestep). 35 FSI runs × ~5000 timesteps each = ~175,000 training pairs. **Plenty for the 67k-parameter model.**

### Step 2: split into train / validation / test

Crucially: split by **trajectories**, not by random samples. A network that interpolates between t=1.00s and t=1.05s of the same FSI run will look great on paper and fail in production.

```python
# Out of 20 inflation velocities + 15 steady-state velocities:
train_velocities = [v0, v2, v3, v5, v6, ...]   # ~70%
val_velocities   = [v1, v4, ...]                # ~15% (used for hyperparameter tuning)
test_velocities  = [v7, ...]                    # ~15% (NEVER touched until final paper)
```

Or use **k-fold cross-validation** with k=5 to make the most of your small dataset (recommended given you only have 35 FSI runs).

### Step 3: normalize features

Compute mean and std of each input/output on the training set, normalize:

```python
features_normalized = (features - features_mean) / features_std
targets_normalized  = (targets  - targets_mean)  / targets_std
```

Train on normalized data, denormalize at inference time. **Important** — without this, the loss is dominated by whichever output has the largest scale (probably force), and the network ignores the others.

### Step 4: weighted loss

Different outputs have different scales. Weight them so each contributes equally:

```python
def loss_fn(pred, target):
    F_p, dA_p, dAs_p, dm_p = pred
    F_t, dA_t, dAs_t, dm_t = target
    
    return ( w_F * mse(F_p, F_t)
           + w_dA * mse(dA_p, dA_t)
           + w_dAs * mse(dAs_p, dAs_t)
           + w_dm * mse(dm_p, dm_t) )
```

A common trick: divide each MSE by the variance of that output in the training data, so each contributes equally. Or just start with `w_F = 1.0` and `w_dA = w_dAs = w_dm = 0.1` and tune.

### Step 5: training loop

The full training loop is ~30 lines without Lightning, ~10 lines with Lightning. PyTorch Lightning is recommended once you're past the initial prototype.

```python
# Without Lightning (raw PyTorch):
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    for features, targets in train_loader:
        optimizer.zero_grad()
        pred = model(features)
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
    
    # Validation
    val_loss = ...   # compute on validation set
    print(f"Epoch {epoch}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
```

### Step 6: sanity checks during training

Watch for:
- **Train loss decreases** but **val loss stops improving / increases** → overfitting. Reduce model capacity, add regularization, or get more data.
- **Train loss doesn't decrease** → learning rate too low, or model can't represent the function. Try `lr=1e-2` or larger model.
- **Loss explodes (NaN)** → learning rate too high. Try `lr=1e-4`.
- **Force prediction is good, but state-derivative prediction is poor** → adjust loss weights.

### Always train a sklearn baseline

Before trusting the neural network, train a baseline:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

baseline = MultiOutputRegressor(GradientBoostingRegressor())
baseline.fit(X_train, y_train)
print(f"Baseline R²: {baseline.score(X_test, y_test)}")
```

**If your fancy MLP can't beat this 4-line baseline, something is wrong.** Do this check first — it can save you weeks of debugging fancy architectures that secretly don't beat trees.

---

## 8. How it integrates with AerisLab

This is where the design pays off. The canopy state `(A, A_side, m_air)` becomes part of the simulation's state vector. AerisLab's IVP solver evolves it alongside the body's `(p, q, v, ω)`.

### State vector extension

Currently in AerisLab, each rigid body has 13 state variables: `[p, q, v, ω]`.

With the surrogate parachute, the parachute body has those 13 plus 3 more:
```
parachute_state = [p, q, v, ω, A, A_side, m_air]   (16 elements)
```

### Force application during integration

At each ODE step:

1. The IVP solver asks `world.state_derivative(t, y)` for the time derivatives of all state variables.
2. For each parachute body:
   - Extract current `[v_body, ρ(altitude), A, A_side, m_air]`.
   - Pass through ONNX surrogate.
   - Get back `[F, dA/dt, dA_side/dt, dm_air/dt]`.
   - Use `F` as the aerodynamic force on the body.
   - Use `dA/dt`, `dA_side/dt`, `dm_air/dt` directly as the time derivatives of the canopy state.
3. The body translational and rotational dynamics use `F` to compute `dv/dt`, `dω/dt`, etc., as normal.
4. The integrator integrates all 16 state variables together.

### Connection to the work plan

This design **maps perfectly onto the architectural refactors in WORK_PLAN.md**:

- **Move 1 (state-vector boundary)** is what makes this clean. The neural-ODE auxiliary state lives naturally in `World.state()`.
- **Move 2 (Model layer)** gives you the place where the surrogate lives — as an `AeroForce` configured with a `NeuralCd`-equivalent.
- **Task P2-T2 (auxiliary state registration for stateful models)** is exactly the plumbing this design needs. You may want to do this task first, even before all of Move 1, since the surrogate work depends on it.

You can scope the architectural work specifically: the minimum is *enough of Move 1 + P2-T2* to support extra state. You don't need all of Move 2 or Move 5 for the thesis.

---

## 9. Validation strategy

Validation has multiple layers. Each catches different failure modes.

### Layer 1: surrogate accuracy on held-out FSI runs

Train on (say) 17 of 20 inflation velocities; test on the other 3. Plot:
- Force-time curves: surrogate vs FSI for each held-out velocity.
- Area-time curves: same.
- Mass-time curves: same.

Quantitative metrics:
- R² per output channel.
- Maximum absolute error.
- Maximum relative error.

This tells you: "the surrogate has learned the constitutive relation."

### Layer 2: out-of-distribution behavior at envelope edges

- Set v = 0 → force should be ≈ 0.
- Set v = max trained velocity + 10% → check the force is sensible (some extrapolation expected).
- Set the canopy to fully deployed → watch dA/dt → 0.
- Set the canopy partially deployed at a velocity it wasn't trained at → spot-check against physics intuition.

This tells you: "the surrogate behaves sanely at boundary conditions."

### Layer 3: end-to-end coupling with conservation laws

Run a complete AerisLab simulation with the surrogate-driven parachute. Check:

- **Momentum impulse**: change in body momentum should equal the integral of the surrogate's force vector over time. `Δp_body ≈ ∫F dt`. Easy to compute; if it doesn't match within numerical tolerance, something is broken.
- **Energy budget**: the kinetic + potential energy lost should equal the work done by the surrogate plus losses to drag. Approximately: `ΔE ≈ ∫F·v dt`. Big mismatches indicate either solver bugs or an unphysical surrogate.
- **Sanity of magnitudes**: peak opening force should be in the right order of magnitude (compare against simple Cd estimates).

This tells you: "the coupled system is internally consistent."

### Layer 4: finite-mass FSI comparison (the gold standard)

Even though you train only on infinite-mass FSI, run **2-3 finite-mass FSI cases as held-out validation**. These are not for training — they're for checking that the methodology transfers from training conditions (infinite mass) to deployment conditions (finite mass).

Compare the AerisLab+surrogate trajectory against the finite-mass FSI trajectory:
- Velocity vs time during deployment.
- Peak opening force.
- Time to peak.
- Deceleration profile.

If they agree within reasonable tolerance, **your methodology is validated end-to-end**. This is the most important plot in the thesis.

If they disagree, you have a domain shift problem and need richer features (e.g., add `dv/dt` as an input, or include lagged velocity).

### Layer 5: published-case reproduction

Reproduce one published parachute case (work plan task P3-T11). Even within ~15% of published peak load is a strong validation; documented mismatches with explanations are scientifically respectable.

### Layer 6: OOD checking during the parameter sweep

For your design study (`payload_mass × activation_velocity`), log: "did any query during this simulation go outside the training envelope?"

```python
class OODChecker:
    def __init__(self, train_min, train_max):
        self.train_min, self.train_max = train_min, train_max
        self.violations = []
    
    def check(self, features):
        outside = (features < self.train_min) | (features > self.train_max)
        if outside.any():
            self.violations.append(features.copy())
        return outside.any()
```

In the thesis results, color the parameter-sweep heatmap by "fraction of time inside training envelope" — gray out cells that left the envelope. This is honest and actually scientifically interesting.

---

## 10. Honest risks and how to handle them

### Risk 1: history dependence beyond your features

The surrogate assumes that `(v, ρ, A, A_side, m_air)` is sufficient state — that the force at any instant depends only on these. This is *probably* true but not certain.

**Detection**: validation Layer 4 catches this. If finite-mass FSI agrees with surrogate predictions, the assumption holds.

**Mitigation if it fails**: add features like `dv/dt`, `time_since_deploy`, or include a small recurrent (GRU) component. Don't add these speculatively — wait until validation shows you need them.

### Risk 2: small training-data extrapolation

10-20 velocities per phase is sparse. The model will be reliable inside the trained range but extrapolate badly outside it.

**Mitigation**: 
- Pick velocities to cover the range the parameter sweep needs (not just deployment range — also the velocities the canopy will see *during* deployment as the load decelerates).
- OOD checking during the sweep.
- Augment training with cheap synthetic data based on physics (e.g., quasi-steady drag at intermediate velocities) where possible.

### Risk 3: FSI ground truth uncertainty

Your training labels aren't perfect — FSI itself has modeling assumptions, mesh effects, time-step effects. The surrogate can't be more accurate than the FSI it learns from.

**Mitigation**: Document this in the thesis. Quantify FSI uncertainty (compare two mesh resolutions if possible). Bound surrogate accuracy claims by FSI accuracy.

### Risk 4: cross-chute generalization

A surrogate trained on chute A will probably fail on chute B.

**Mitigation**: Frame the thesis as "demonstration of the methodology on chute X," not "general parachute surrogate." If you want generalization, train on multiple chutes and include chute geometry as input features — but this is a follow-up paper, not the thesis.

### Risk 5: training-time bottleneck

The whole pipeline depends on having FSI data. If FSI runs are slow, you're bottlenecked.

**Mitigation**: Start FSI runs **immediately** after the proposal defense. Don't wait for the ML pipeline to be ready. The training of the network takes hours; the FSI generation takes weeks. Parallelize.

### Risk 6: over-engineering the ML side

Tempting to build deep ensembles, transformers, fancy uncertainty quantification, hyperparameter sweeps, etc.

**Resist**. The thesis claim is "the methodology works," not "we built the world's best parachute surrogate." Ship the simple version first; sophistication comes in follow-up papers.

---

## 11. Concrete first steps

For the next 2-3 work sessions, before even starting on the ML proper:

### Session 1: data inspection

1. Run **one** FSI inflation case at a single velocity.
2. Plot all four time series: `A(t)`, `A_side(t)`, `m_air(t)`, `F(t)`. Verify they're smooth, monotonic where they should be, and the finite differences are well-behaved.
3. Eyeball-check that the data has the structure the surrogate needs.

### Session 2: ML toolchain setup

1. Install PyTorch in your venv: `pip install torch onnxruntime scikit-learn`.
2. Try the official PyTorch 60-minute tutorial. Get familiar with `nn.Module`, `nn.Linear`, the training loop pattern.
3. Train a tiny MLP on synthetic data (e.g., learn `f(x, y) = x² + sin(y)`). Get a feel for the workflow end-to-end.

### Session 3: steady-state surrogate (the easy case)

1. Take your steady-state FSI data (10-15 velocities).
2. Train a small MLP: input `(v, ρ)` → output `F` for steady descent only. **Skip inflation for now.**
3. This is the simplest possible version. Validates the entire pipeline (data extraction, model training, inference) without the complexity of inflation dynamics.
4. Compare to a sklearn baseline.

### Sessions 4+: full pipeline

Once steady-state works, extend to the full model:
1. Add inflation FSI data.
2. Add area + air-mass features.
3. Add state-derivative outputs.
4. Train, validate, integrate into AerisLab.

**Don't try to do all of this in one go.** Build up; validate at each step.

### FSI campaign in parallel

Concurrently with the ML setup, plan your FSI runs:
- ~20 inflation velocities per chute, picked carefully (cluster around expected operating range).
- ~15 steady descent velocities per chute.
- ~3-5 finite-mass cases held out for validation Layer 4.
- Estimate total compute time. Schedule. Start as soon as proposal is defended.

---

## 12. How to defend this in the thesis

### Methods chapter framing

> *"We propose a neural-ODE-style surrogate for parachute aerodynamics: a single neural network learns both the canopy state evolution (projected area, side area, mass of internal air) and the instantaneous aerodynamic force, trained on infinite-mass fluid-structure-interaction (FSI) simulations at a sweep of representative velocities. The surrogate is integrated into a constrained multibody dynamics (MBD) solver, where the canopy state evolves as auxiliary dynamical variables alongside the body's six-degree-of-freedom motion. This formulation handles inflation, transition, and steady descent within a single model, without phase switching or hand-designed gating functions. The framework enables design studies — specifically, sweeps of payload mass and deployment velocity to identify configurations meeting peak-opening-shock constraints — that would be computationally infeasible using full FSI for each design point."*

### Why this is publishable

- **Methodology contribution**: combining FSI-trained surrogates with constrained multibody dynamics in a state-vector-coupled formulation is genuinely novel.
- **Validation rigor**: multi-layer validation strategy (surrogate accuracy + conservation laws + finite-mass FSI + published case) addresses the standard reviewer concerns about ML methods.
- **Engineering application**: the parameter-study results (your "money plot") demonstrate practical value, not just a benchmark on contrived data.
- **Open-source artifact**: AerisLab + the trained models become a citable contribution in their own right (JOSS).

### Likely venues

- AIAA Journal of Aircraft / AIAA Journal of Spacecraft and Rockets (methodology + application)
- Aerospace Science and Technology
- Journal of Computational Physics (if framed around the ML/physics coupling)
- Journal of Open Source Software (for the AerisLab code)
- Software Impacts (for the toolchain as a whole)

Two papers is realistic from this thesis: one methodology paper (the surrogate framework) and one applications paper (the parameter-study results). Combined with your existing FSI paper, that's a strong thesis output.

---

## Short version

- **One ML model**, predicts force AND state derivatives — neural-ODE style.
- **Input**: `v`, `ρ`, `A`, `A_side`, `m_air`. **Output**: `F`, `dA/dt`, `dA_side/dt`, `dm_air/dt`.
- **PyTorch + ONNX Runtime + scikit-learn** to start. Lightning + W&B + Hydra later.
- **~67k parameter MLP** is plenty. GELU activations, softplus on non-negative outputs.
- **Train on infinite-mass FSI**, validate end-to-end against finite-mass FSI on a few held-out cases.
- **Maps cleanly onto AerisLab's planned architecture** (Move 1, Move 2, P2-T2).
- **Start the FSI campaign immediately** after proposal — it's the bottleneck, not the ML.
- **Don't over-engineer**. Simple model that works > sophisticated model that doesn't ship.
- **Frame the thesis around the parameter-study application**, not just the surrogate methodology.

---

*Captured 2026-05-14 from working session. May be revised as the work proceeds.*
