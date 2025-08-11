"""
Hybrid multibody dynamics (DAE + ODE) with rigid constraints, soft/rigid joints,
and unilateral ground contact. Variable-step stiff integration via SciPy.

Public API re-exports for convenience.
"""
from .mathutil import (
    normalize_quaternion, quaternion_multiply, quaternion_to_rotation_matrix,
    euler_to_quaternion, quaternion_to_euler, skew,
)
from .body import RigidBody6DOF
from .forces import Force, Gravity, Drag, Spring
from .constraints import Constraint, DistanceConstraint, PointWeldConstraint
from .contact import GroundProjection, GroundPenalty, GroundImpulse
from .solver import (
    HybridSolver, SolverSettings,
    HybridIVPSolver, IVPSettings,
)
from .world import World
from .logger import CSVLogger
from .joints import (
    Joint, SoftTetherJoint, RigidTetherJoint, WeldJoint
)

__all__ = [
    "RigidBody6DOF",
    "Force", "Gravity", "Drag", "Spring",
    "Constraint", "DistanceConstraint", "PointWeldConstraint",
    "GroundProjection", "GroundPenalty", "GroundImpulse",
    "HybridSolver", "SolverSettings",
    "HybridIVPSolver", "IVPSettings",
    "World",
    "CSVLogger",
    "Joint", "SoftTetherJoint", "RigidTetherJoint", "WeldJoint",
    "normalize_quaternion", "quaternion_multiply", "quaternion_to_rotation_matrix",
    "euler_to_quaternion", "quaternion_to_euler", "skew",
]
