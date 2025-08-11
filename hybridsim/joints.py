from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .forces import Spring
from .constraints import DistanceConstraint, PointWeldConstraint

Array = np.ndarray

class Joint:
    """
    High-level joint facade. Concrete joints either:
      - register a Constraint into the World (rigid)
      - register a Spring into the World (soft)
    """
    def attach(self, world: "World") -> None:  # pragma: no cover
        raise NotImplementedError

@dataclass
class SoftTetherJoint(Joint):
    """
    Soft rope-like joint: spring-damper between attachment points.
    """
    body_a: "RigidBody6DOF"
    body_b: "RigidBody6DOF"
    r_a_local: Array
    r_b_local: Array
    rest_length: float
    stiffness: float
    damping: float

    def attach(self, world: "World") -> None:
        spring = Spring(self.body_a, self.body_b, self.r_a_local, self.r_b_local,
                        self.rest_length, self.stiffness, self.damping)
        world.add_interaction_force(spring)

@dataclass
class RigidTetherJoint(Joint):
    """
    Rigid rope of fixed length (non-extensible) via DistanceConstraint.
    """
    body_a: "RigidBody6DOF"
    body_b: "RigidBody6DOF"
    r_a_local: Array
    r_b_local: Array
    length: float

    def attach(self, world: "World") -> None:
        c = DistanceConstraint(self.body_a, self.body_b, self.r_a_local, self.r_b_local, self.length)
        world.add_constraint(c)

@dataclass
class WeldJoint(Joint):
    """
    Weld two points (no relative translation at those points) via PointWeldConstraint.
    """
    body_a: "RigidBody6DOF"
    body_b: "RigidBody6DOF"
    r_a_local: Array
    r_b_local: Array

    def attach(self, world: "World") -> None:
        c = PointWeldConstraint(self.body_a, self.body_b, self.r_a_local, self.r_b_local)
        world.add_constraint(c)
