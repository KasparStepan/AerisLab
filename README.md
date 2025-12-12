# AerisLab: Aerospace & Engineering Research Integrated Simulator

**AerisLab** is a high-fidelity, Python-based physics engine designed for the simulation of 6-DOF rigid body dynamics subject to holonomic constraints. It is developed specifically for aerospace engineering applications—such as parachute-payload systems—where accurate modeling of stiff linkages, aerodynamics, and variable-mass systems is critical.

The software implements a hybrid solving architecture, offering both fixed-step semi-implicit integrators for rapid visualization and variable-step stiff integrators (Radau IIA) for precision scientific analysis.

## 1. Theoretical Foundation

AerisLab treats multi-body systems as Differential-Algebraic Equations (DAEs) of Index 3. Unlike penalty-based methods common in game engines, AerisLab solves for constraint forces explicitly using Lagrange multipliers.

### 1.1 Constrained Equations of Motion
The dynamics are governed by the Newton-Euler equations augmented with constraint forces:

$$
\begin{align}
\mathbf{M} \dot{\mathbf{v}} + \mathbf{J}^T \boldsymbol{\lambda} &= \mathbf{F}_{ext} \\
\mathbf{J} \dot{\mathbf{v}} &= \boldsymbol{\gamma}
\end{align}
$$

Where:
* $\mathbf{M} \in \mathbb{R}^{6n \times 6n}$ is the block-diagonal mass and inertia matrix.
* $\mathbf{J} \in \mathbb{R}^{m \times 6n}$ is the constraint Jacobian ($\partial \mathbf{C} / \partial \mathbf{q}$).
* $\boldsymbol{\lambda} \in \mathbb{R}^m$ are the Lagrange multipliers (constraint forces).
* $\boldsymbol{\gamma}$ is the stabilization term.

### 1.2 Numerical Solution (KKT System)
To prevent drift in the constraint manifold $\mathbf{C}(\mathbf{q}, t) = 0$, the system solves the following Karush–Kuhn–Tucker (KKT) system at the acceleration level:

$$
\begin{bmatrix}
\mathbf{M} & \mathbf{J}^T \\\\
\mathbf{J} & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\mathbf{a} \\\\
\boldsymbol{\lambda}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{F}_{ext} \\\\
\mathbf{b}_{stab}
\end{bmatrix}
$$

**Stabilization Strategy:**
AerisLab employs Baumgarte stabilization to damp constraint violations:
$$\mathbf{b}_{stab} = -\mathbf{J}\mathbf{v} - 2\alpha \dot{\mathbf{C}} - \beta^2 \mathbf{C}$$
This ensures that numerical errors in position ($\mathbf{C}$) and velocity ($\dot{\mathbf{C}}$) decay exponentially over time.

## 2. Key Features

* **Rigid Body Dynamics**: Full 6-DOF simulation utilizing quaternion algebra for orientation stability.
* **Hybrid Solver Architecture**:
    * **Fixed-Step**: Semi-implicit Euler with direct KKT resolution. Ideal for real-time approximations.
    * **Variable-Step (IVP)**: Wraps `scipy.integrate.solve_ivp` (Radau/BDF methods) to handle stiff systems typical of high-tension tethers and high-speed aerodynamics.
* **Aerodynamics Module**: 
    * Standard drag models ($F_d \propto v^2$).
    * Specialized parachute inflation logic based on activation velocity and dynamic pressure.
* **Instrumentation**: Built-in `CSVLogger` for high-frequency state recording and a post-processing suite for generating trajectory, velocity, and force analysis plots.

## 3. Installation

AerisLab requires Python 3.8+ and standard scientific computing libraries.

```bash
git clone [https://github.com/kasparstepan/AerisLab.git](https://github.com/kasparstepan/AerisLab.git)
cd AerisLab
pip install numpy scipy matplotlib
