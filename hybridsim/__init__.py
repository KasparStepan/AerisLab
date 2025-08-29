"""
HybridSim: minimal 3D multibody dynamics with rigid constraints (DAE via Lagrange multipliers).
- Fixed-step: semi-implicit Euler + KKT each step
- Variable-step (optional): SciPy solve_ivp (Radau/BDF) with terminal ground event
"""
from .mathutil import (
    q_normalize, q_mul, q_to_R, omega_to_qdot, skew,
)
from .body import RigidBody6DOF
from .forces import Force, Gravity, Drag, Spring
from .constraints import Constraint, DistanceConstraint, PointWeldConstraint
from .joints import RigidTetherJoint, WeldJoint, SoftTetherJoint
from .solver import HybridSolver, HybridIVPSolver
from .world import World
from .logger import CSVLogger

__all__ = [
    "q_normalize", "q_mul", "q_to_R", "omega_to_qdot", "skew",
    "RigidBody6DOF",
    "Force", "Gravity", "Drag", "Spring",
    "Constraint", "DistanceConstraint", "PointWeldConstraint",
    "RigidTetherJoint", "WeldJoint", "SoftTetherJoint",
    "HybridSolver", "HybridIVPSolver",
    "World", "CSVLogger",
]
