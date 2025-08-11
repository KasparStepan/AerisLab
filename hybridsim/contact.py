from __future__ import annotations
import numpy as np

Array = np.ndarray

class ContactModel:
    """Base class for unilateral ground contact models."""
    def post_integrate(self, world: "World") -> None:  # pragma: no cover
        raise NotImplementedError

class GroundProjection(ContactModel):
    """Hard stop: clamp z to ground and zero downward vz."""
    def __init__(self, ground_z: float = 0.0):
        self.ground_z = float(ground_z)
    def post_integrate(self, world: "World") -> None:
        for b in world.bodies:
            if b.position[2] < self.ground_z:
                b.position[2] = self.ground_z
                if b.linear_velocity[2] < 0.0:
                    b.linear_velocity[2] = 0.0

class GroundPenalty(ContactModel):
    """Soft spring-damper normal force (applied for next step)."""
    def __init__(self, k: float = 1e5, c: float = 2e3, ground_z: float = 0.0):
        self.k = float(k); self.c = float(c); self.ground_z = float(ground_z)
    def post_integrate(self, world: "World") -> None:
        for b in world.bodies:
            z = b.position[2] - self.ground_z
            if z < 0.0:
                fz = -self.k * z - self.c * min(0.0, b.linear_velocity[2])
                b.apply_force(np.array([0.0, 0.0, fz]))

class GroundImpulse(ContactModel):
    """Bounce-like: project z and set vz := -e * vz (elasticity e in [0,1])."""
    def __init__(self, ground_z: float = 0.0, restitution: float = 0.0):
        self.ground_z = float(ground_z); self.e = float(restitution)
    def post_integrate(self, world: "World") -> None:
        for b in world.bodies:
            if b.position[2] < self.ground_z:
                b.position[2] = self.ground_z
                if b.linear_velocity[2] < 0.0:
                    b.linear_velocity[2] = - self.e * b.linear_velocity[2]
