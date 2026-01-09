# AerisLab: Aerospace & Engineering Research Integrated Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AerisLab** is a high-fidelity, Python-based physics engine for simulating 6-DOF rigid body dynamics with holonomic constraints. Designed specifically for aerospace engineering applications—such as parachute-payload systems, multi-body aerial systems, and trajectory analysis.

## 🚀 Features

- **6-DOF Rigid Body Dynamics** with quaternion orientation
- **Holonomic Constraints** via KKT solver with Baumgarte stabilization
- **Dual Integration Methods**:
  - Fixed-step semi-implicit Euler (fast, stable)
  - Adaptive stiff IVP solvers (high accuracy, automatic stepping)
- **Aerodynamic Forces**: Gravity, drag, parachute deployment
- **Automatic Output Organization** with timestamped results
- **Built-in Visualization** for trajectories, velocities, forces
- **Extensive Test Suite** with >90% coverage

## 📦 Installation

### From Source (Development)

```bash
# Clone repository
git clone https://github.com/KasparStepan/aerislab.git
cd aerislab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
