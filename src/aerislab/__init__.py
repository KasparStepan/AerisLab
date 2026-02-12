"""
AerisLab - High-fidelity 6-DOF physics engine for aerospace simulations.

Core Components
---------------
World : Simulation orchestrator
HybridSolver : Fixed-step constrained solver
HybridIVPSolver : Variable-step IVP solver
RigidBody6DOF : 6 degree-of-freedom rigid body

Component Architecture (v0.2+)
------------------------------
Component : Base class for aerospace components
Payload : Simple payload with drag
Parachute : Parachute with deployment state machine
System : Multi-component assembly manager

Examples
--------
>>> from aerislab import World, HybridSolver, RigidBody6DOF
>>> from aerislab.components import Payload, Parachute, System
"""

__version__ = "0.2.0"

# Core simulation classes
# Component architecture
from aerislab.components import (
    Component,
    DeploymentState,
    Parachute,
    Payload,
    System,
)
from aerislab.core.simulation import World
from aerislab.core.solver import HybridIVPSolver, HybridSolver
from aerislab.dynamics.body import RigidBody6DOF

# Constraints
from aerislab.dynamics.constraints import (
    Constraint,
    DistanceConstraint,
    PointWeldConstraint,
)

# Forces
from aerislab.dynamics.forces import Drag, Gravity, ParachuteDrag, Spring

# Logging
from aerislab.logger import CSVLogger
from aerislab.api.scenario import Scenario

__all__ = [
    # Version
    "__version__",
    # Core
    "World",
    "HybridSolver",
    "HybridIVPSolver",
    "RigidBody6DOF",
    # Forces
    "Gravity",
    "Drag",
    "ParachuteDrag",
    "Spring",
    # Constraints
    "Constraint",
    "DistanceConstraint",
    "PointWeldConstraint",
    # Components
    "Component",
    "Payload",
    "Parachute",
    "DeploymentState",
    "System",
    # Logging
    # Logging
    "CSVLogger",
    # API
    "Scenario",
]
