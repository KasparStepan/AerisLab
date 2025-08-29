from __future__ import annotations
import numpy as np
from .constraints import DistanceConstraint, PointWeldConstraint
from .forces import Spring

class RigidTetherJoint:
    """Facade that registers a rigid DistanceConstraint."""
    def __init__(self, body_i_idx: int, body_j_idx: int,
                 r_i_b: np.ndarray, r_j_b: np.ndarray, length: float) -> None:
        self.i = body_i_idx
        self.j = body_j_idx
        self.r_i_b = np.array(r_i_b, dtype=np.float64)
        self.r_j_b = np.array(r_j_b, dtype=np.float64)
        self.L = float(length)

    def attach(self, world) -> None:
        world.constraints.append(
            DistanceConstraint(self.i, self.j, self.r_i_b, self.r_j_b, self.L)
        )

class WeldJoint:
    """Facade that registers a 3-DOF PointWeldConstraint."""
    def __init__(self, body_i_idx: int, body_j_idx: int,
                 r_i_b: np.ndarray, r_j_b: np.ndarray) -> None:
        self.i = body_i_idx
        self.j = body_j_idx
        self.r_i_b = np.array(r_i_b, dtype=np.float64)
        self.r_j_b = np.array(r_j_b, dtype=np.float64)

    def attach(self, world) -> None:
        world.constraints.append(
            PointWeldConstraint(self.i, self.j, self.r_i_b, self.r_j_b)
        )

class SoftTetherJoint:
    """Facade that registers a Spring (soft link)."""
    def __init__(self, body_i, body_j, r_i_b, r_j_b, k: float, c: float, L0: float) -> None:
        self.spring = Spring(body_i, body_j, r_i_b, r_j_b, k, c, L0)

    def attach(self, world) -> None:
        world.interaction_forces.append(self.spring)
