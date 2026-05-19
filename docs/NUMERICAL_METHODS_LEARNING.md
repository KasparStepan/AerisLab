# Numerical Methods Literacy — A Beginner's Path

**Context:** notes captured from a working session on 2026-05-14, on what numerical methods literacy actually means for a computational scientist (specifically: an aerospace PhD student doing constrained multibody dynamics + planned ML/FSI surrogates), and how to build that literacy from a near-zero starting point.

---

## Why this matters

Numerical methods literacy is the difference between "I trust my simulation" and "I understand my simulation." A good aerospace numerical engineer can look at a force-time plot and immediately suspect: "that high-frequency oscillation is probably my time step too close to the constraint stiffness," or "that gradual drift is symplectic damping," or "that NaN is a divide-by-zero in the inflation gate when speed = 0." Each of those diagnoses comes from having *seen the problem before*.

You build that library of diagnostic patterns by working on this project, debugging each weirdness as it surfaces, and writing down what you learned each time.

This document has two parts:
1. **What kinds of problems bite scientific code** — the diagnostic vocabulary.
2. **The beginner's curriculum** — a 12-18 month part-time path from "I know almost nothing" to "competent for my domain."

---

# Part 1 — The categories of problems that bite scientific code

These are the failure modes you'll see in your own work, in roughly increasing subtlety.

## 1. Floating-point gotchas
- `0.1 + 0.2 != 0.3` — the same trap appears in `if t == 1.0: ...` after 100 integration steps where t is now `0.99999999998`.
- **Catastrophic cancellation:** `a - b` when `a ≈ b`. You get `1.0 - 0.99999 = 0.00001`, but you've lost 5 of your 16 digits of precision in one operation. Happens silently.
- **Order of operations matters numerically.** `(a + b) + c` and `a + (b + c)` give different answers when the magnitudes differ a lot.
- **Subnormal numbers and underflow** when computing very small quantities.

## 2. ODE integration problems (your bread and butter)
- **Stiffness:** parachute opening is the textbook stiff problem. Explicit solvers (RK45, classic RK4) need impossibly small steps to stay stable. Implicit solvers (Radau, BDF) handle it but cost more per step. Knowing *why* matters more than knowing *how* to implement them.
- **Order of accuracy regression:** after a refactor, your second-order method silently becomes first-order because of a bug in the time-stepping logic. The simulation still runs; the error just grows linearly with dt instead of quadratically. Catches: convergence-rate tests (work plan task P1-T15).
- **Energy drift over long simulations:** non-symplectic integrators bleed energy in a long-running orbit / pendulum / rotation. Symplectic integrators (semi-implicit Euler, Verlet, leapfrog, Stormer) preserve a "nearby" energy bounded forever — your fixed-step solver does this for the unconstrained part.
- **Quaternion drift:** ‖q‖ = 1 is an algebraic constraint that pure ODE integration violates over time. Three solutions: normalize after each step (what AerisLab does — non-smooth, bothers stiff solvers); use the exponential map (what the codebase has as an option); add `‖q‖² = 1` as a real holonomic constraint enforced by the KKT solver (what production multibody codes do).
- **Adaptive step size oscillation:** scipy's adaptive solvers can take huge steps then tiny ones then huge again if your RHS has discontinuities (parachute deployment is a discontinuity!). Sometimes the fix is smoothing the RHS, sometimes adding `events`.

## 3. Linear algebra numerics
- **Ill-conditioned matrices:** the system you're solving might be solvable in theory but numerically garbage. Condition number κ(A) tells you how much error in the input becomes error in the output. The KKT solver in AerisLab already monitors this.
- **Singular matrices:** redundant constraints make the constraint Jacobian rank-deficient. AerisLab's least-squares fallback is the textbook treatment.
- **Picking the right solver:** `np.linalg.solve` is fine for general systems. `np.linalg.cholesky` is faster *if* the matrix is symmetric positive definite. Specialized solvers (sparse, banded, conjugate gradient) matter only at scale you don't have yet.

## 4. Constrained dynamics specifically
- **Constraint drift:** position-level constraints aren't preserved exactly by velocity-level enforcement. Baumgarte stabilization is the standard fix; you're using it.
- **DAE index issues:** constrained ODEs are technically Differential-Algebraic Equations (DAEs). DAEs of high "index" are numerically nasty. Multibody dynamics is index-3 in position formulation, index-1 after Baumgarte. (You don't need to know the math, but the term will appear in papers.)
- **Lagrange multiplier accuracy:** the multiplier λ from the KKT solve *is* the constraint force. Logging it is how you know what your tether is actually doing.

## 5. Long-time-integration phenomena
- **Symmetry breaking:** if your system conserves something analytically (energy, angular momentum, linear momentum, a Casimir invariant), check that it conserves numerically. If it doesn't, your method is wrong for the physics.
- **Resonance and aliasing:** time-stepping can produce spurious oscillations. The Nyquist rule (you need at least 2 samples per period to resolve a frequency) applies to time integration too.

## 6. Random numbers (when you start doing Monte Carlo)
- **Seeded vs. unseeded:** never use `np.random.rand()` in research code. Use `rng = np.random.default_rng(seed)` and track the seed.
- **Pseudo-random vs quasi-random (Sobol, Halton):** for Monte Carlo with low dimensionality (<20), quasi-random sequences converge much faster.

---

# Part 2 — The beginner's curriculum (from near-zero)

Calibrated to: someone with mech-eng undergrad math (calculus, linear algebra, differential equations from classes some years ago) who hasn't formally studied numerical analysis as a discipline.

You're not starting from zero — you have the math foundation, you just haven't seen how it gets translated into computation. That's a much more bridgeable gap than starting from scratch.

## Phase 0 — Refresh the math foundation (4-6 weeks)

Before any "numerical methods" study, get fluent in the underlying math. The single best resource for this is free:

**3Blue1Brown YouTube series** — watch in this order:
1. *Essence of Linear Algebra* (15 episodes, ~3 hours total). The single best learning resource on the internet for linear algebra intuition. It builds the *visual* understanding of what matrices, eigenvalues, determinants, transformations actually *mean*. You'll never look at a matrix the same way again. Watch one episode per evening; don't skip.
2. *Essence of Calculus* (12 episodes). Same author, same approach, for calculus. Refresher even if you remember the formulas.
3. Optional but excellent: *Differential Equations* series.

This is the foundation. Without intuition for what these objects *do*, all the numerical methods writing about them will feel like incantations.

**Cost: ~6 hours of video over a month, plus paper-and-pencil to work through one example per episode.**

## Phase 1 — Floating-point + Python numerics (2-3 weeks)

You now know what math operations *mean*. Time to learn what computers actually do with them.

**Read:** David Goldberg, *"What Every Computer Scientist Should Know About Floating-Point Arithmetic"* (1991, free online). The first 10 pages are the must-read. Skip the technical parts; absorb the intuition: floats are not real numbers, comparisons need tolerance, subtraction can lose precision, NaN propagates.

**Practice in Python:**
```python
>>> 0.1 + 0.2 == 0.3       # False! Why?
>>> 0.1 + 0.2               # 0.30000000000000004
>>> import numpy as np
>>> np.allclose(0.1 + 0.2, 0.3)   # True — the right way to compare
```

Spend an evening playing in the Python REPL with this kind of thing. Get a feel for how floats actually behave.

**Also read:** the early chapters of *Python Data Science Handbook* by Jake VanderPlas (free online). Just the parts on numpy basics. By the end you should be comfortable with `np.array`, slicing, broadcasting, `@` for matrix multiplication.

## Phase 2 — Linear algebra numerically (1-2 months)

Now you connect the math you understand visually to the code that does it.

**Read:** Michael Heath, *Scientific Computing: An Introductory Survey* (3rd edition is fine). This book is the **gentlest, most pedagogical introduction** to numerical methods I know. Written for engineers, not for mathematicians. Use it as your spine for the next year.

**Read just chapters 1-3** in this phase:
- Chapter 1: scientific computing in general (read once, lightly).
- Chapter 2: solving linear systems Ax = b (Gauss elimination, LU, Cholesky, condition number). The most important chapter for you.
- Chapter 3: linear least squares (when you have more equations than unknowns).

**Practice:** for each major method in chapter 2, write your own implementation in Python. Compare to scipy. Don't move on until you understand *why* your version gives the same (or close) answer.

## Phase 3 — ODE methods (2-3 months) — directly relevant to AerisLab

This is where it really pays off for your project.

**Read Heath chapters 9 and 10** (ODEs, both initial-value and boundary-value problems).

**Practice:**
1. Implement Euler's method by hand. Use it to solve free fall. Compare to the analytical answer. Plot the error vs dt — you should see it grow linearly.
2. Implement *improved Euler* (RK2). Compare. Error grows as dt².
3. Implement RK4. Compare. Error grows as dt⁴.
4. Take a *stiff* ODE (e.g., `dy/dt = -1000*y + 999*exp(-t)`). Try RK45 (scipy). Watch it crawl. Try Radau. Watch it fly. Now you understand "stiffness" not as a word but as an experience.

This phase is when you start being able to read AerisLab's `solver.py` and not feel lost. It also makes the work plan task P1-T15 (convergence-rate tests) feel obvious.

## Phase 4 — Constrained dynamics (2-3 months)

Now Featherstone's *Rigid Body Dynamics Algorithms* becomes accessible. Before this phase it would have been like reading Greek; after Phases 1-3 it'll just be hard.

Read the first 5 chapters slowly. Each chapter, find the equivalent code in AerisLab and trace the math through it. That cross-reference is what moves the math from "understood when reading" to "understood when implementing."

Companion: read Baumgarte's 1972 paper on constraint stabilization. It's only ~16 pages.

## Phase 5 — When you start the ML/FSI work

Add a different learning track:
- Bishop's *Pattern Recognition and Machine Learning* OR Murphy's *Probabilistic Machine Learning* — pick one, read selectively as you need each topic.
- For practical PyTorch: just dive in via the official tutorials.

This phase is 1-2 years out. Don't worry about it now.

---

## Honest summary timeline

| Phase | Topic | Time at 1-2 days/week |
|---|---|---|
| 0 | Math intuition refresh (3Blue1Brown + practice) | 4-6 weeks |
| 1 | Floating-point + numpy | 2-3 weeks |
| 2 | Linear algebra numerically | 1-2 months |
| 3 | ODE methods | 2-3 months |
| 4 | Constrained dynamics + AerisLab math | 2-3 months |
| 5 | ML, when needed | later |

**Total to "competent in numerical methods for your domain": roughly 9-12 months of steady-but-light study, alongside your actual coding work.**

---

# Tiered priorities (if you want a different lens)

If the phased curriculum above is too linear, here's the same content as priority tiers — what to learn in what order if your time is limited.

## Tier 1 — must-know (next 6 months)
1. **Floating-point fundamentals.** Goldberg essay.
2. **ODE methods overview.** Why Euler is bad, why RK4 is OK, why Radau exists, what "stiff" means.
3. **Convergence-rate analysis.** What "second-order accurate" means, how to verify experimentally.
4. **Linear algebra essentials.** Solving Ax=b (LU, Cholesky), condition number, what a singular matrix is.

## Tier 2 — important (next year)
5. **Constrained dynamics math.** Lagrange multipliers, the KKT system, Baumgarte derivation. Featherstone first 5 chapters.
6. **Symplectic / structure-preserving integration.** Why semi-implicit Euler conserves "nearby" energy.
7. **Root finding and optimization basics.** Newton, secant, bisection, gradient descent.
8. **Numerical differentiation and integration (quadrature).** Why finite differences are noisy, why Gauss quadrature is more accurate than Simpson's.

## Tier 3 — useful for ML/FSI later (2+ years)
9. **Optimization for ML.** SGD, Adam, momentum.
10. **Probability and information theory.** Bishop / Murphy.
11. **Numerical methods for PDEs** — only if you ingest CFD output deeply.
12. **Differentiable simulation, auto-differentiation theory** — only if you eventually go to JAX-backed gradient-based optimization.

## What to skip
- High-performance linear algebra at very large scale (BLAS/LAPACK internals, GPU programming).
- Multigrid methods, domain decomposition.
- Sparse matrix theory beyond the basics.
- Detailed analysis of specific solver families you'll never implement.

---

# Practical resources, ranked

| Topic | Resource | Time |
|---|---|---|
| Math intuition refresh | 3Blue1Brown YouTube | ~6 hours over a month |
| Floating-point | Goldberg "What Every CS Should Know..." (free PDF) | 1 afternoon |
| Numpy basics | VanderPlas *Python Data Science Handbook* (free online) | dip in |
| Numerical analysis overview | Heath *Scientific Computing* (textbook) | dip in over a year |
| ODE methods, applied | scipy.integrate docs + Hairer/Wanner Vol. I | as needed |
| Symplectic integration | Hairer/Lubich/Wanner *Geometric Numerical Integration* | selective chapters |
| Linear algebra, numerical | Trefethen & Bau *Numerical Linear Algebra* | best in class |
| Multibody dynamics | Featherstone *Rigid Body Dynamics Algorithms* | first 5 chapters |
| ML for engineers | Murphy *Probabilistic Machine Learning* | when you start ML |

---

# How to actually study this

The mistake most engineers make: read three textbooks cover-to-cover, retain nothing. Don't.

The right pattern is **just-in-time learning anchored to a real problem**:

1. **You hit a problem in your code** (e.g., "why does my pendulum simulation slowly gain energy over 60 seconds?").
2. **You ask the right question** ("is my integrator symplectic?").
3. **You read the *one chapter*** that answers it.
4. **You implement and verify.**
5. **You remember it** because it solved a real problem you cared about.

Reading without a question to answer = forgotten in a week. Reading with a question = retained forever.

The work plan gives you these prompts naturally. Tasks like "convergence-rate verification tests" (P1-T15) and "literature replication" (P3-T11) and "make added mass enter the LHS" (P2-T10) are *exactly* the kind of problems that motivate learning the relevant numerical methods. **Don't try to learn ahead of need; let the need drive the learning.**

## Study habits that work

- **Pick ONE book and stick with it.** Don't dabble in five textbooks at once. Heath is your spine. When something there doesn't click, *then* look at another book on the same topic — but only that topic.
- **Code every chapter.** Every numerical method you read about, implement in Python. Don't wait. The reading without the implementation evaporates within a week. Tiny implementations (10-30 lines each) are fine.
- **Tie it to AerisLab.** Whenever a topic is relevant to a piece of AerisLab code (e.g., when reading about LU decomposition, find where `np.linalg.solve` is called in your codebase), trace the connection. The dual-purpose learning is far stickier than abstract study.
- **Don't try to be rigorous.** You're not training to be a numerical analyst. You're training to be a competent user of numerical methods who can recognize when one is misbehaving. Skip the proofs; absorb the intuition.
- **Trust the timeline.** A year of part-time study to be genuinely literate in numerical methods is *fast*. Most engineers never get there at all. Don't try to compress it.
- **Ask AI as a tutor.** When Heath says something you don't follow, paste the paragraph to Claude with "explain this to me, I'm a mechanical engineer with rusty linear algebra." This is exactly the right use of AI in learning.

---

# A simple checkpoint

You'll know you're succeeding when, six months in, you can:
- Look at a force-time plot from your simulation and immediately suspect what kind of numerical issue is showing up.
- Read a scipy docstring (e.g., `solve_ivp`) and understand all the parameters without looking them up.
- Read a numerical methods paper at the level of "I get the gist, even if the details are hard."
- Look at AerisLab's `assemble_system()` function and explain what it's doing in plain words.

That's the real win. From there, the depth comes naturally as you hit harder problems.

---

# The meta-skill (the real prize)

The most useful thing isn't memorizing methods. It's developing **diagnostic intuition**: "this output looks weird — is it a physics bug, an integration bug, a precision bug, or a constraint drift bug?" That intuition only builds by doing.

After two years of working through this curriculum *while doing* the AerisLab work plan, you'll be in the small group of aerospace engineers who genuinely *understand* their numerical tools instead of just trusting them. That's the real prize, and it pays back for the rest of your career.
