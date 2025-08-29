# HybridSim

Minimal, modular 3D multibody simulator (Python ≥ 3.10) for **rigid bodies** with **rigid equality constraints** solved via a **KKT** system. Two solver paths:

- **Fixed-step**: semi-implicit Euler + KKT each step
- **Variable-step** (optional): `scipy.integrate.solve_ivp` (Radau/BDF) with terminal ground event

> **No contact modeling.** A designated “payload” body stops the simulation when it reaches the ground plane (`z ≤ ground_z`).

## Features

- 6-DoF rigid bodies (p, unit quaternion q, linear/ang. velocity)
- Forces: gravity, drag (linear/quadratic), optional soft spring (not part of DAE)
- Constraints: distance (rigid tether), point weld; easy to add more
- KKT solve:  
  \[
  \begin{bmatrix} M & J^T \\ J & 0 \end{bmatrix}
  \begin{bmatrix} a \\ \lambda \end{bmatrix} =
  \begin{bmatrix} Q \\ -Jv - (\alpha C + \beta \dot C) \end{bmatrix}
  \]
- Quaternion integration (unit renormalization)
- Clean API, type hints, docstrings, tests, examples, simple CSV logger

## Install

```bash
python -m venv .venv && . .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -U pip numpy
# optional for IVP solver:
pip install scipy
pip install -U pytest
