"""
Verification Test Suite for AerisLab.

These tests compare simulation results against analytical solutions
to validate the physics engine implementation.

Test Categories:
- Kinematic: Free fall, constant velocity, constant acceleration
- Rotational: Torque-free spin, Dzhanibekov effect
- Constraints: Pendulum, double pendulum, spring oscillator
- Energy: Conservation checks
- Aerodynamic: Terminal velocity, drag deceleration

References:
- NASA-STD-7009A: Models & Simulations Standard
- ASME V&V 10-2006: Verification and Validation
"""

import numpy as np
import pytest

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity


# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------

# Tolerances for analytical comparisons
POSITION_TOLERANCE = 1e-3  # meters
VELOCITY_TOLERANCE = 1e-3  # m/s
ENERGY_TOLERANCE = 1e-4  # relative error
PERIOD_TOLERANCE = 0.02  # 2% for oscillation periods


# -----------------------------------------------------------------------------
# Common Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def gravity():
    """Standard Earth gravity vector."""
    return np.array([0.0, 0.0, -9.81])


@pytest.fixture
def unit_sphere():
    """Unit sphere: mass=1kg, radius=1m, uniform density."""
    I = (2/5) * 1.0 * 1.0**2  # I = 2/5 * m * r²
    return {
        "mass": 1.0,
        "inertia": I * np.eye(3),
        "radius": 1.0,
    }


@pytest.fixture
def asymmetric_body():
    """Asymmetric body for Dzhanibekov effect testing."""
    # I1 < I2 < I3 (intermediate axis instability)
    return {
        "mass": 1.0,
        "inertia": np.diag([1.0, 2.0, 3.0]),
    }


@pytest.fixture
def solver():
    """Default fixed-step solver."""
    return HybridSolver(alpha=5.0, beta=1.0)


@pytest.fixture
def high_precision_solver():
    """High-precision solver for sensitive tests."""
    return HybridSolver(alpha=10.0, beta=2.0)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def create_body(
    name: str,
    mass: float,
    inertia: np.ndarray,
    position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    angular_velocity: np.ndarray | None = None,
) -> RigidBody6DOF:
    """Create a rigid body with default parameters."""
    return RigidBody6DOF(
        name=name,
        mass=mass,
        inertia_tensor_body=inertia,
        position=position if position is not None else np.zeros(3),
        orientation=np.array([0, 0, 0, 1]),
        linear_velocity=velocity if velocity is not None else np.zeros(3),
        angular_velocity=angular_velocity if angular_velocity is not None else np.zeros(3),
    )


def run_simulation(world: World, solver: HybridSolver, duration: float, dt: float = 0.001):
    """Run simulation and return final state."""
    world.set_termination_callback(lambda w: False)  # Don't terminate early
    n_steps = int(duration / dt)
    for _ in range(n_steps):
        world.step(solver, dt)
    return world


def relative_error(computed: float, analytical: float) -> float:
    """Compute relative error, handling zero case."""
    if abs(analytical) < 1e-12:
        return abs(computed - analytical)
    return abs(computed - analytical) / abs(analytical)
